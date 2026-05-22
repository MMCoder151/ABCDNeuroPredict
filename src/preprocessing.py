import pandas as pd
from pathlib import Path
from src.mri_rois import mri_rois
import duckdb
from pcntoolkit import NormativeModel, BLR, Runner
from pcntoolkit.dataio.norm_data import NormData
from tqdm import tqdm

# ---- DATA WRANGLING ----

def _load_fitbit_df(filepath):
    '''
    Helper function meant to deal with inconsistencies in fitbit file naming schemes
    and formats. It loads a fitbit file, finds the correct time column, renames it to "Wear_Time" for consistency, 
    converts it to datetime format, drops rows with invalid or missing time values, and sorts by time.
    '''
    # Define possible names for time column in fitbit files
    FITBIT_TIME_COLUMNS = ("Wear_Time", "ActivityMinute", "Time", "date")
    # Helper function to find the correct time column in the fitbit dataframe
    def _get_fitbit_time_column(columns):
        normalized_columns = {str(column).strip().lower(): column for column in columns}
        for candidate in FITBIT_TIME_COLUMNS:
            match = normalized_columns.get(candidate.lower())
            if match is not None:
                return match
        raise KeyError(f"No known Fitbit time column found. Available columns: {list(columns)}")
    # Load the fitbit file, find the correct time column, rename it to "Wear_Time", convert it to datetime format, drop rows with invalid or missing time values, and sort by time
    fit_df = pd.read_csv(filepath, sep="\t")
    time_col = _get_fitbit_time_column(fit_df.columns)
    if time_col != "Wear_Time":
        fit_df = fit_df.rename(columns={time_col: "Wear_Time"})
    fit_df["Wear_Time"] = pd.to_datetime(fit_df["Wear_Time"], errors="coerce", format="mixed")
    fit_df = fit_df.dropna(subset=["Wear_Time"]).sort_values("Wear_Time")
    return fit_df


def select_subjects(dta_path, test=False):
    '''
    This function selects subjects and time points based on selection criteria (below) and extracts demographic and meta information for fitbit and mri data
        - Only subjects with both "fit" and "scans" files are included (=> subjects with both Fitbit and MRI data)
        - Only timepoints/sessions with both "fit" and "scans" files are included (=> timepoints/sessions with both Fitbit and MRI data)
        - Only subjects/sessions with complete "fit" data (i.e., all 7 "fit" files present) are included (=> complete Fitbit data for included subjects/sessions)
        - Only "scans" files with non-empty "acq_time" column are included (=> valid MRI acquisition date for included sessions)
        - Only subjects/timepoints with more than 7 days of Fitbit data are included (=> sufficient Fitbit data for included sessions)

    Parameters:
        dta_path (Path): Path to the raw data directory
    Returns:
        demo_df (DataFrame): DataFrame containing demographic data for included subjects (sex, age)
        mri_meta_df (DataFrame): DataFrame containing MRI date and  age at MRI scan for included subjects and timepoints/sessions
        fit_meta_df (DataFrame): DataFrame containing filepaths for Fitbit data for included subjects and timepoints/sessions

    Note:
        "fit" files contain fitbit data. Multiple fitbit files exist containing different types of data:
        - Cal1m: calories measured in 1 minute intervals
        - Int1m: intensity measured in 1 minute intervals
        - Stps1m: steps measured in 1 minute intervals
            => These files always are the same length 
        - HR1m: heart rate measured in 1 minute intervals
            => These files are often shorter than the other fitbit files (Reason unknown)
        - Slp1m: sleep detection measured in 1 minute intervals (1 = asleep, 2 = restless, 3 = awake)
        - Slp30s: sleep stage estimates with wake indices in 30s intervals (See ABCD data release 6.0 documentation for more details)
            => These files are typically the shortest. Probably because participants took off the watch during the night to charge.
            Since these files only contain data during the night, not wearing the watch at night at the start or end of recording, 
            ends the files at an earlier date
    '''
    # Read participant information and get subject folders
    subs = pd.read_csv(dta_path / "participants.tsv", sep="\t")
    sub_folders = [f for f in dta_path.iterdir() if f.name in subs["participant_id"].values]
    if test:
        sub_folders = sub_folders[:100] # Use only first 100 subjects for testing

    n_total_subs = len(sub_folders)
    print(f"Total number of subjects made available: {n_total_subs}")

    # GET FITBIT METADATA

    # Find files with "fit" in the name
    fit_files = []

    for sub_folder in sub_folders:
        sub_id = sub_folder.name
        
        # Search recursively for files with "fit" in the name
        for fit_file in sub_folder.rglob("*fit*"):
            if fit_file.is_file():
                # Extract timepoint/session from file path
                parts = fit_file.relative_to(sub_folder).parts
                timepoint = next((p for p in parts if p.startswith("ses-")), "unknown")
                
                fit_files.append({
                    "subject": sub_id,
                    "timepoint": timepoint,
                    "filename": fit_file.name,
                    "filepath": str(fit_file)
                })

    # Convert to DataFrame for easy inspection
    fit_meta_df = pd.DataFrame(fit_files)

    # get unique timepoints per subject for fit files
    n_fit_subs = fit_meta_df["subject"].nunique()
    print(f"Number of subjects with fitbit data: {n_fit_subs}")
    print(f"Average number of timepoints per subject with fitbit data: {fit_meta_df.groupby('subject')['timepoint'].nunique().mean():.2f}")

    # check if all subjects contain the same amount of "fit" files per timepoint
    fit_counts = fit_meta_df.groupby(["subject", "timepoint"]).size().reset_index(name="fit_count")

    # drop timepoints with incomplete fit data
    incomplete_timepoints = fit_counts[fit_counts["fit_count"] != 7][["subject", "timepoint"]]
    fit_meta_df = fit_meta_df.merge(incomplete_timepoints, on=["subject", "timepoint"], how="left", indicator=True)
    fit_meta_df = fit_meta_df[fit_meta_df["_merge"] == "left_only"].drop(columns=["_merge"])
    print(f"Dropped {len(incomplete_timepoints)} timepoints with incomplete fitbit data.")
    print(f"Number of subjects remaining after dropping incomplete timepoints: {fit_meta_df['subject'].nunique()}")
    print(f"Average number of timepoints per subject with fitbit data after dropping incomplete timepoints: {fit_meta_df.groupby('subject')['timepoint'].nunique().mean():.2f}")

    # get recording duration in days for each fit file and drop timepoints with less than 7 days of data
    print("Computing Fitbit recording durations...")
    fit_meta_df["recording_duration_days"] = [
        (lambda df: (df["Wear_Time"].max() - df["Wear_Time"].min()).days)(_load_fitbit_df(x))
        for x in tqdm(fit_meta_df["filepath"], total=len(fit_meta_df["filepath"]), desc="Fitbit files")
    ]
    short_recordings = fit_meta_df[fit_meta_df["recording_duration_days"] < 7][["subject", "timepoint"]].drop_duplicates()
    fit_meta_df = fit_meta_df.merge(short_recordings, on=["subject", "timepoint"], how="left", indicator=True)
    fit_meta_df = fit_meta_df[fit_meta_df["_merge"] == "left_only"].drop(columns=["_merge"])
    print(f"Dropped {len(short_recordings)} timepoints with less than 7 days of fitbit data.")
    print(f"Number of subjects remaining after dropping short recordings: {fit_meta_df['subject'].nunique()}")
    print(f"Average number of timepoints per subject after dropping short recordings: {fit_meta_df.groupby('subject')['timepoint'].nunique().mean():.2f}")

    # GET MRI METADATA

    # Find "scans" files
    scan_files = []

    for sub_folder in tqdm(sub_folders, total=len(sub_folders), desc="Searching MRI subjects"):
        sub_id = sub_folder.name
        
        # Search recursively for files with "scans" in the name
        for scan_file in sub_folder.rglob("*scans*"):
            if scan_file.is_file():
                # check if "acq_time" column is empty in the file, if so, skip the file
                temp_file = pd.read_csv(scan_file, sep="\t")
                if temp_file.empty:
                    continue
                temp_file["acq_time"] = pd.to_datetime(temp_file["acq_time"], errors="coerce")
                if temp_file["acq_time"].isna().all():
                    continue
                # Extract timepoint/session from file path
                parts = scan_file.relative_to(sub_folder).parts
                timepoint = next((p for p in parts if p.startswith("ses-")), "unknown")
                
                scan_files.append({
                    "subject": sub_id,
                    "timepoint": timepoint,
                    "filename": scan_file.name,
                    "filepath": str(scan_file)
                })
    # Convert to DataFrame for easy inspection
    mri_meta_df = pd.DataFrame(scan_files)
    print(mri_meta_df.columns)

    print(f"Number of subjects with 'scans' files: {mri_meta_df['subject'].nunique()}")
    print(f"Average number of timepoints per subject with 'scans' files: {mri_meta_df.groupby('subject')['timepoint'].nunique().mean():.2f}")

    # Get timepoints per subjects with both "fit" and "scans" files
    fit_timepoints = fit_meta_df.groupby("subject")["timepoint"].unique().reset_index()
    scan_timepoints = mri_meta_df.groupby("subject")["timepoint"].unique().reset_index()
    merged_timepoints = pd.merge(fit_timepoints, scan_timepoints, on="subject", how="inner", suffixes=("_fit", "_scan"))
    merged_timepoints["common_timepoints"] = merged_timepoints.apply(lambda row: set(row["timepoint_fit"]) & set(row["timepoint_scan"]), axis=1)

    # Ensure we keep only exact matching (subject, timepoint) pairs that exist in BOTH fit and MRI metadata.
    fit_pairs = fit_meta_df[["subject", "timepoint"]].drop_duplicates()
    mri_pairs = mri_meta_df[["subject", "timepoint"]].drop_duplicates()
    common_pairs = pd.merge(fit_pairs, mri_pairs, on=["subject", "timepoint"], how="inner")
    # Filter both metadata tables to the intersection of pairs
    fit_meta_df = fit_meta_df.merge(common_pairs, on=["subject", "timepoint"], how="inner")
    mri_meta_df = mri_meta_df.merge(common_pairs, on=["subject", "timepoint"], how="inner")
    print(f"Number of subjects with both fitbit and mri files: {fit_meta_df['subject'].nunique()}")

    # Get subjects with multiple timepoints/sessions with both "fit" and "scans" files
    timepoint_counts = fit_meta_df.groupby("subject")["timepoint"].nunique().reset_index(name="timepoint_count")
    subjects_multiple_timepoints = timepoint_counts[timepoint_counts["timepoint_count"] > 1]
    print(f"Number of subjects with multiple timepoints/sessions with both fitbit and mri files: {len(subjects_multiple_timepoints)}")

    # Get subjects with immediate follow-up timepoints (e.g., ses-01A and ses-02A)
    def has_immediate_followup(timepoints):
        timepoint_numbers = [int(tp.split("-")[1][:-1]) for tp in timepoints if tp.startswith("ses-")]
        timepoint_numbers.sort()
        return any((n2 - n1 == 1) for n1, n2 in zip(timepoint_numbers, timepoint_numbers[1:]))
    subjects_immediate_followup = merged_timepoints[merged_timepoints["common_timepoints"].apply(has_immediate_followup)]
    print(f"Number of subjects with immediate follow-up timepoints: {len(subjects_immediate_followup)}")

    # GET DEMOGRAPHIC DATA
    
    # Get list of included subjects
    included_subjects = fit_meta_df["subject"].unique()

    # import static demographic information
    mri_path = dta_path / "phenotype"
    stc_df = pd.read_csv(mri_path / "ab_g_stc.tsv", sep="\t")

    # create dataframe with sex and date of birth for included subjects
    demo_df = subs[subs["participant_id"].isin(included_subjects)][["participant_id", "sex"]].merge(
        stc_df[stc_df["participant_id"].isin(included_subjects)][["participant_id", "ab_g_stc__cohort_dob"]],
        on="participant_id",
        how="left"
    )
    demo_df.rename(columns={"ab_g_stc__cohort_dob": "date_of_birth", "participant_id": "subject"}, inplace=True)

    # Extract MRI acquisition date and add to mri_meta_df
    for file in mri_meta_df["filepath"]:
        temp_file = pd.read_csv(file, sep="\t")
        if temp_file["acq_time"].dtype != "datetime64[ns]":
            temp_file["acq_time"] = pd.to_datetime(temp_file["acq_time"])
        mri_date = temp_file["acq_time"].min()
        mri_meta_df.loc[mri_meta_df["filepath"] == file, "mri_date"] = mri_date

    # add sex and age at MRI scan (rounded to nearest year) to mri_meta_df
    mri_meta_df = mri_meta_df.merge(demo_df[["subject", "sex", "date_of_birth"]], left_on="subject", right_on="subject", how="left")
    mri_meta_df["mri_date"] = pd.to_datetime(mri_meta_df["mri_date"], errors="coerce")
    mri_meta_df["date_of_birth"] = pd.to_datetime(mri_meta_df["date_of_birth"], errors="coerce")
    mri_meta_df["age_at_mri"] = ((mri_meta_df["mri_date"] - mri_meta_df["date_of_birth"]).dt.days / 365.25).round(0).astype("Int64")
    mri_meta_df = mri_meta_df.drop(columns=["date_of_birth"])

    # Check that subject/timepoint PAIRS in mri_meta_df and fit_meta_df match exactly
    pairs_fit = set(map(tuple, fit_meta_df[["subject","timepoint"]].drop_duplicates().values))
    pairs_mri = set(map(tuple, mri_meta_df[["subject","timepoint"]].drop_duplicates().values))
    assert pairs_fit == pairs_mri, "Subject-timepoint pairs in mri_meta_df and fit_meta_df do not match"

    # drop filepath and filename columns from mri_meta_df
    mri_meta_df = mri_meta_df.drop(columns=["filepath", "filename"])

    return demo_df, mri_meta_df, fit_meta_df

def setup_duckdb(dta_path, fit_meta_df, overwrite = False):
    '''
    This function transforms the raw fitbit and MRI data to make it easier to query with DuckDB for downstream analysis
    and sets up a DuckDB connection with views for the transformed fitbit and MRI data.
        - For fitbit data, it combines all fitbit files for each selected subject and timepoint into a single parquet file 
        based on datetime index for easier querying with DuckDB. It adds two columns to each combined parquet file: "subject" and "timepoint", 
        which are extracted from the file paths of the original fitbit files, for easy filtering in DuckDB. 
        The combined parquet file is saved in a new hive-style directory structure at the top of the dta_path: "processed_fitbit_data/subject=SUBJECT_ID/timepoint=TIMEPOINT/combined_fitbit.parquet"
        - For MRI data, it extracts the MRI ROIs specified in mri_rois for each subject and timepoint and saves it in a similar hive-style directory structure 
        at the top of the dta_path: "processed_mri_data/subject=SUBJECT_ID/timepoint=TIMEPOINT/combined_mri.parquet"
    Parameters:
        dta_path (Path): Path to the raw data directory
        fit_meta_df (DataFrame): DataFrame containing metadata for the selected fitbit files (subjects, timepoints, filepaths) -> also used for mri data to get selected subjects
    Returns:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
    '''
    if overwrite == False:
        print("DuckDB setup skipped (overwrite=False). To re-run data transformation and DuckDB setup, set overwrite=True.")
        con = duckdb.connect()
        con.execute(f"CREATE OR REPLACE VIEW fitbit_data AS SELECT * FROM read_parquet('{output_dir_fit}/**/combined_fitbit.parquet', union_by_name => TRUE)")
        con.execute(f"CREATE OR REPLACE VIEW mri_data AS SELECT * FROM read_parquet('{output_dir_mri}/**/combined_mri.parquet', union_by_name => TRUE)")

        return con

    # Create output directories for fitbit and mri data
    output_dir_fit = dta_path / "processed_fitbit_data"
    output_dir_mri = dta_path / "processed_mri_data"
    output_dir_fit.mkdir(parents=True, exist_ok=True)
    output_dir_mri.mkdir(parents=True, exist_ok=True)
    
    # Combine fitbit files for each subject and timepoint into a single parquet file based on datetime index
    for _, row in tqdm(fit_meta_df.iterrows(), total=len(fit_meta_df), desc="Combining Fitbit files"):
        subject = row["subject"]
        timepoint = row["timepoint"]
        filepath = row["filepath"]

        # Read the fitbit file
        fit_df = _load_fitbit_df(filepath)

        value_cols = [
            col for col in fit_df.columns
            if col != "Wear_Time" and fit_df[col].notna().any()
        ]
        if not value_cols:
            continue

        # Extract metric name from filename (e.g., "Cal1m", "HR1m", etc.) and rename value columns to include metric name for easier identification after merging
        stem = Path(filepath).stem
        metric_name = stem.split("task-fitb", 1)[1].split("_", 1)[0]
        fit_df = fit_df[["Wear_Time", *value_cols]].rename(
            columns={col: f"{col}_{metric_name}" for col in value_cols}
        )

        # Add subject and timepoint columns
        fit_df["subject"] = subject
        fit_df["timepoint"] = timepoint

        # Define output path for combined parquet file
        subject_dir = output_dir_fit / f"{subject}"
        timepoint_dir = subject_dir / f"{timepoint}"
        timepoint_dir.mkdir(parents=True, exist_ok=True)
        output_file = timepoint_dir / "combined_fitbit.parquet"

        # Save combined dataframe as parquet file aligned on Wear_Time (overwrites existing files)
        fit_df.to_parquet(output_file, index=False)
    
    # Get MRI ROIs and files to import
    mri_files, mri_rois_dict = mri_rois()

    # Accumulate MRI data for each subject-timepoint across all phenotype files
    mri_data_accumulator = {}  # {(subject, timepoint): {columns from all files}}
    
    # Extract MRI data for each subject and timepoint and accumulate across all files
    for file in tqdm(mri_files, total=len(mri_files), desc="Processing MRI phenotype files"):
        mri_df = pd.read_csv(dta_path / "phenotype" / file, sep="\t")
        # Select only columns that exist in the dataframe
        available_cols = ["participant_id", "session_id"] + [col for col in mri_rois_dict.keys() if col in mri_df.columns]
        mri_df = mri_df[available_cols]
        mri_df = mri_df.merge(
            fit_meta_df[["subject", "timepoint"]].drop_duplicates(),
            left_on=["participant_id", "session_id"],
            right_on=["subject", "timepoint"],
            how="inner",
        ).drop(columns=["participant_id", "session_id"])
        
        # Accumulate this file's data for each subject-timepoint
        for _, row in mri_df.iterrows():
            subject = row["subject"]
            timepoint = row["timepoint"]
            key = (subject, timepoint)
            
            if key not in mri_data_accumulator:
                mri_data_accumulator[key] = {}
            
            # Merge this row's data into the accumulator
            for col in row.index:
                if col not in ["subject", "timepoint"]:
                    mri_data_accumulator[key][col] = row[col]
    
    # Write accumulated MRI data to parquet files
    for (subject, timepoint), data_dict in tqdm(mri_data_accumulator.items(), total=len(mri_data_accumulator), desc="Writing MRI data"):
        subject_dir = output_dir_mri / f"{subject}"
        timepoint_dir = subject_dir / f"{timepoint}"
        timepoint_dir.mkdir(parents=True, exist_ok=True)
        output_file = timepoint_dir / "combined_mri.parquet"
        
        # Add subject and timepoint back in
        data_dict["subject"] = subject
        data_dict["timepoint"] = timepoint
        
        # Convert to a single-row DataFrame and save
        row_df = pd.DataFrame([data_dict])
        row_df.to_parquet(output_file, index=False)
    
    # Setup DuckDB connection to query the combined fitbit and mri data
    con = duckdb.connect()
    # Use read_parquet with union_by_name=True to allow files with differing schemas
    con.execute(f"CREATE OR REPLACE VIEW fitbit_data AS SELECT * FROM read_parquet('{output_dir_fit}/**/combined_fitbit.parquet', union_by_name => TRUE)")
    con.execute(f"CREATE OR REPLACE VIEW mri_data AS SELECT * FROM read_parquet('{output_dir_mri}/**/combined_mri.parquet', union_by_name => TRUE)")

    # Sanity check
    n_fitbit = con.execute("SELECT COUNT(DISTINCT subject) FROM fitbit_data").fetchone()[0]
    n_mri    = con.execute("SELECT COUNT(DISTINCT subject) FROM mri_data").fetchone()[0]
    print(f"✓ DuckDB ready — {n_fitbit} Fitbit subjects, {n_mri} MRI subjects")

    return con

# ---- DATA ANALYSIS ----

def normative_selection(con, mri_meta_df, output_path):
    '''
    This function performs normative modeling and selects subjects based on their composite absolute z-score. 
    It selects the top 10% (based on prevalence) of subjects with the highest cumulative z-score.
    These subjects are considered to have abnormal development in the selected MRI ROIs associated with depression.

    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        mri_meta_df (DataFrame): DataFrame containing MRI metadata
        output_path (str): Path to the output directory where the normative model results will be saved
    Returns:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their MRI ROI data and respective z-scores
        normative_modelling (Folder): Folder containing the normative model, results, and plots created in the output directory
    '''
    # Get MRI ROIs to include in the normative model
    _, mri_rois_dict = mri_rois()
    mri_roi_cols = list(mri_rois_dict.keys())
    
    # Query MRI data for the first timepoint for each subject
    query = f"""
    SELECT *
    FROM mri_data
    WHERE {"timepoint"} = (
        SELECT MIN(m2.{"timepoint"})
        FROM mri_data m2
        WHERE m2.{"subject"} = mri_data.{"subject"}
    )
    """
    mri_df = con.execute(query).df()
    print(f"MRI data loaded: {len(mri_df)} subjects")

    # Merge MRI data with demographic data
    df = mri_df.merge(
    mri_meta_df[["subject", "sex", "age_at_mri"]].drop_duplicates(),
    on="subject",
    how="inner"    
    )

    missing_cols = set(mri_roi_cols) - set(df.columns)
    if missing_cols:
        print(f"Warning: The following MRI ROI columns are missing from the dataframe: {missing_cols}")

    # Encode sex explicitly so F/M map to 1/2 before passing data to NormData.
    sex_map = {'F': 1, 'M': 2}
    if not pd.api.types.is_numeric_dtype(df['sex']):
        sex_values = df['sex'].astype('string').str.strip().str.upper()
        mapped_sex = sex_values.map(sex_map)
        if mapped_sex.isna().any():
            unexpected_values = sorted(sex_values[mapped_sex.isna()].dropna().unique().tolist())
            raise ValueError(
                f"Unmapped sex values found: {unexpected_values}. Expected F/M or numeric input."
            )
        df['sex'] = mapped_sex.astype('Int64')

    df['sex'] = pd.to_numeric(df['sex'], errors='raise').astype(float)
    df['age_at_mri'] = pd.to_numeric(df['age_at_mri'], errors='raise').astype(float)

    # Prepare data for normative modeling
    data = NormData.from_dataframe(
        name="mri_norm",
        dataframe=df,
        covariates=["sex", "age_at_mri"],
        response_vars=mri_roi_cols,
        subject_ids="subject",
        remove_Nan=True,
    )

    normative_output_dir = Path(output_path) / "normative_modelling"
    if not normative_output_dir.exists():
        normative_output_dir.mkdir(parents=True)
    normative_output_dir_str = str(normative_output_dir)

    # setup normative model
    model = NormativeModel(
        BLR(),
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=True,
        # The directory to save the model, results, and plots.
        save_dir=normative_output_dir_str,
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
        )

    model.fit(data)

    # create a runner
    #runner = Runner(cross_validate = True)

    # fit the model 
    #runner.fit(model, data)

    # Read in z-score file from normative modeling 
    centiles_df = pd.read_csv(normative_output_dir / "results" / "Z_mri_norm.csv")
    # Calculate composite absolute z-score across all MRI ROIs for each subject
    centiles_df["composite_z"] = centiles_df[mri_roi_cols].abs().sum(axis=1)
    # Select top 10% of subjects with the highest composite z-score
    threshold = centiles_df["composite_z"].quantile(0.9)
    selected_subjects = centiles_df[centiles_df["composite_z"] >= threshold].copy()
    print(f"Selected {len(selected_subjects)} subjects with composite z-score >= {threshold:.2f} (top 10%)")

    return selected_subjects


    