import pandas as pd
from pathlib import Path
from src.mri_rois import mri_rois
import duckdb
from tqdm import tqdm
import pathlib
from src.mri_rois import mri_rois

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
    fit_df["Wear_Time"] = pd.to_datetime(fit_df["Wear_Time"], errors="coerce", format = "mixed")
    fit_df = fit_df.dropna(subset=["Wear_Time"]).sort_values("Wear_Time")
    return fit_df

def _recode_fitbit_data(fit_df):
    '''
    Recodes Fitbit data according to specific rules from the ABCD Data Release 6.0 documentation:
        - MET1m: Divide values by 10
        - Slp1m: Recode values to binary asleep (1) vs. awake/restless (0), with "unknown" as missing (None)
        and drops unncessary meta columns (pGUID, logId)
    '''

    # drop unnecessary meta columns
    cols_to_drop = ["pGUID", "logId"]
    fit_df = fit_df.drop(columns=[col for col in cols_to_drop if col in fit_df.columns])

    # Recode MET1m values by dividing by 10
    met_cols = [col for col in fit_df.columns if "METs" in col]
    fit_df[met_cols] = fit_df[met_cols] / 10

    # Recode Slp1m values
    slp1m_cols = [
        col
        for col in fit_df.columns
        if any(token in str(col).lower() for token in ("deep", "light", "rem", "restless", "wake", "1", "2", "3"))
    ]
    slp1m_mapping = {
        "asleep": 1,
        "deep": 1,
        "light": 1,
        "rem": 1,
        "2": 0,
        "3": 0,
        "restless": 0,
        "wake": 0,
        "wake": 0,
        "unknown": None
    }
    for col in slp1m_cols:
        fit_df[col] = fit_df[col].replace(slp1m_mapping)
    # Rename "level" column to "value" for consistency with other fitbit files
    fit_df = fit_df.rename(columns={col: col.replace("Level", "value") for col in slp1m_cols})

    #TODO: Find out why "Level_Slp1m" persists

    return fit_df

def filter_subjects(dta_path, test=False, overwrite=True, output_path=pathlib.Path("output")):
    '''
    This function selects subjects and time points based on selection criteria (below) and extracts demographic and meta information for fitbit and mri data
        - Only subjects with both "fit" and "scans" files are included (=> subjects with both Fitbit and MRI data)
        - Only timepoints/sessions with both "fit" and "scans" files are included (=> timepoints/sessions with both Fitbit and MRI data)
        - Drops Slp30s files, due to unusable data according to ABCD Data Release 6.0 documentation
        - Only subjects/sessions with complete "fit" data (i.e., all 6 "fit" files present) are included (=> complete Fitbit data for included subjects/sessions)
        - Only "scans" files with non-empty "acq_time" column are included (=> valid MRI acquisition date for included sessions)
    Subjects/timepoints with less than 7 days of actually recorded Fitbit data and less than 60% missings are marked for later filtering
    Parameters:
        dta_path (Path): Path to the raw data directory
        test (bool): Whether to run in test mode (only uses first 100 subjects for faster testing)
        overwrite (bool): Whether to overwrite existing metadata files (if False, will load existing metadata files if they exist and skip the selection process)
        output_path (Path): Path to the output directory to reimport metadata files if overwrite=False
    Returns:
        demo_df (DataFrame): DataFrame containing demographic data for included subjects (sex, age at mri, scan site)
        mri_meta_df (DataFrame): DataFrame containing MRI date and  age at MRI scan for included subjects and timepoints/sessions
        fit_meta_df (DataFrame): DataFrame containing filepaths for Fitbit data for included subjects and timepoints/sessions
        demo_df, mri_meta_df, and fit_meta_df are saved as CSV files in the output directory for easy re-import if overwrite=False
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
    if overwrite == False:
        print("Subject selection skipped (overwrite=False). To re-run subject selection, set overwrite=True.")
        demo_df = pd.read_csv(output_path / "demographics_metadata.csv")
        mri_meta_df = pd.read_csv(output_path / "mri_metadata.csv")
        fit_meta_df = pd.read_csv(output_path / "fitbit_metadata.csv")
        return demo_df, mri_meta_df, fit_meta_df

    # Read participant information and get subject folders
    subs = pd.read_csv(dta_path / "participants.tsv", sep="\t")
    sub_folders = [f for f in dta_path.iterdir() if f.name in subs["participant_id"].values]
    if test:
        sub_folders = sub_folders[:100] # Use only first 100 subjects for testing

    n_total_subs = len(sub_folders)
    print(f"Raw fitbit data available for {n_total_subs} subjects")

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

    # Drop "Slp30s" files from fit_meta_df, since they are unusable
    fit_meta_df = fit_meta_df[~fit_meta_df["filename"].str.contains("Slp30s", case=False)]

    # check if all subjects contain the same amount of "fit" files per timepoint
    fit_counts = fit_meta_df.groupby(["subject", "timepoint"]).size().reset_index(name="fit_count")

    # drop timepoints with incomplete fit data
    incomplete_timepoints = fit_counts[fit_counts["fit_count"] != 6][["subject", "timepoint"]]
    fit_meta_df = fit_meta_df.merge(incomplete_timepoints, on=["subject", "timepoint"], how="left", indicator=True)
    fit_meta_df = fit_meta_df[fit_meta_df["_merge"] == "left_only"].drop(columns=["_merge"])
    print(f"Dropped {len(incomplete_timepoints)} timepoints with incomplete fitbit data.")
    print(f"Number of subjects remaining after dropping incomplete timepoints: {fit_meta_df['subject'].nunique()}")
    print(f"Average number of timepoints per subject with fitbit data after dropping incomplete timepoints: {fit_meta_df.groupby('subject')['timepoint'].nunique().mean():.2f}")

    # get recording duration in days for each fit file and drop timepoints with less than 7 days of data
    print("Computing Fitbit recording durations...")

    for file in tqdm(fit_meta_df["filepath"], total=len(fit_meta_df["filepath"]), desc="Fitbit files"):
        temp_df = _load_fitbit_df(file)
        # Get length of recording
        recording_length = (temp_df["Wear_Time"].max() - temp_df["Wear_Time"].min()).days
        fit_meta_df.loc[fit_meta_df["filepath"] == file, "recording_duration_days"] = recording_length
        # Check amount of actually present days
        actual_days = set(temp_df["Wear_Time"].dt.floor("D").unique())
        recording_duration_days = len(actual_days)
        fit_meta_df.loc[fit_meta_df["filepath"] == file, "present_recording_days"] = recording_duration_days

    present_days = pd.to_numeric(fit_meta_df["present_recording_days"], errors="coerce")
    recording_days = pd.to_numeric(fit_meta_df["recording_duration_days"], errors="coerce")
    fit_meta_df["missing_days_percentage"] = 100.0
    valid_duration = recording_days > 0
    fit_meta_df.loc[valid_duration, "missing_days_percentage"] = (
        1 - (present_days.loc[valid_duration] / recording_days.loc[valid_duration])
    ) * 100
    fit_meta_df["missing_days_percentage"] = fit_meta_df["missing_days_percentage"].clip(lower=0, upper=100)

    # Mark short recordings in binary column in fit_meta_df
    fit_meta_df["short"] = fit_meta_df.apply(lambda row: 1 if (row["recording_duration_days"] < 7) | (row["missing_days_percentage"] > 60) else 0, axis=1)

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
    print(f"Number of subjects with both fitbit and mri files: {common_pairs['subject'].nunique()}")

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

    # import scansite information
    scan_site_df = pd.read_csv(mri_path / "ab_g_dyn.tsv", sep="\t")

    # create dataframe with sex, date of birth, and scan site for included subjects
    demo_df = subs[subs["participant_id"].isin(included_subjects)][["participant_id", "sex"]].merge(
        stc_df[stc_df["participant_id"].isin(included_subjects)][["participant_id", "ab_g_stc__cohort_dob"]],
        on="participant_id",
        how="left"
    )
    demo_df = demo_df.merge(
        scan_site_df[scan_site_df["participant_id"].isin(included_subjects)][["participant_id", "ab_g_dyn__design_site"]],
        on="participant_id",
        how="left"
    )
    demo_df.rename(columns={"ab_g_stc__cohort_dob": "date_of_birth", "participant_id": "subject", "ab_g_dyn__design_site": "scan_site"}, inplace=True)

    # Extract MRI acquisition date and add to mri_meta_df
    for file in mri_meta_df["filepath"]:
        temp_file = pd.read_csv(file, sep="\t")
        if temp_file["acq_time"].dtype != "datetime64[ns]":
            temp_file["acq_time"] = pd.to_datetime(temp_file["acq_time"])
        mri_date = temp_file["acq_time"].min()
        mri_meta_df.loc[mri_meta_df["filepath"] == file, "mri_date"] = mri_date

    # add sex and age at MRI scan (rounded to nearest year) to mri_meta_df
    mri_meta_df = mri_meta_df.merge(demo_df[["subject", "sex", "date_of_birth", "scan_site"]], left_on="subject", right_on="subject", how="left")
    mri_meta_df["mri_date"] = pd.to_datetime(mri_meta_df["mri_date"], errors="coerce")
    mri_meta_df["date_of_birth"] = pd.to_datetime(mri_meta_df["date_of_birth"], errors="coerce")
    mri_meta_df["age_at_mri"] = ((mri_meta_df["mri_date"] - mri_meta_df["date_of_birth"]).dt.days / 365.25).round(0).astype("Int64")
    mri_meta_df = mri_meta_df.drop(columns=["date_of_birth"])

    # Add total intracranial volume (TIV) to mri_meta_df per subject and time point
    subcortical_vol = pd.read_csv(dta_path / "phenotype" / "mr_y_smri__vol__aseg.tsv", sep="\t")
    mri_meta_df = mri_meta_df.merge(subcortical_vol[["participant_id", "session_id", "mr_y_smri__vol__aseg__icv_sum"]], left_on=["subject", "timepoint"], right_on=["participant_id", "session_id"], how="left")

    # Check that subject/timepoint PAIRS in mri_meta_df and fit_meta_df match exactly
    pairs_fit = set(map(tuple, fit_meta_df[["subject","timepoint"]].drop_duplicates().values))
    pairs_mri = set(map(tuple, mri_meta_df[["subject","timepoint"]].drop_duplicates().values))
    assert pairs_fit == pairs_mri, "Subject-timepoint pairs in mri_meta_df and fit_meta_df do not match"

    # drop filepath and filename columns from mri_meta_df
    mri_meta_df = mri_meta_df.drop(columns=["filepath", "filename"])

    # Drop duplicates from mri_meta_df and demo_df
    mri_meta_df.drop_duplicates(subset=["subject", "timepoint"], inplace=True)
    demo_df.drop_duplicates(subset=["subject"], inplace=True)

    # Add age at MRI to dem_df for first timepoint
    demo_df["age_at_first_mri"] = demo_df["subject"].map(mri_meta_df.groupby("subject")["age_at_mri"].min())

    print(f"Final number of subjects included after filtering: {demo_df['subject'].nunique()}")

    # save metadata to csv
    output_path.mkdir(parents=True, exist_ok=True)
    demo_df.to_csv(output_path / "demographics_metadata.csv", index=False)
    mri_meta_df.to_csv(output_path / "mri_metadata.csv", index=False)
    fit_meta_df.to_csv(output_path / "fitbit_metadata.csv", index=False)

    return demo_df, mri_meta_df, fit_meta_df

def setup_duckdb(dta_path, fit_meta_df, overwrite=True):
    '''
    This function transforms the raw fitbit and MRI data to make it easier to query with DuckDB for downstream analysis
    and sets up a DuckDB connection with views for the transformed fitbit and MRI data.
        - For fitbit data, it combines all fitbit files for each selected subject and timepoint into a single parquet file 
        based on datetime index for easier querying with DuckDB. It adds two columns to each combined parquet file: "subject" and "timepoint", 
        which are extracted from the file paths of the original fitbit files, for easy filtering in DuckDB. 
        The combined parquet file is saved in a new hive-style directory structure at the top of the dta_path: "processed_fitbit_data/subject=SUBJECT_ID/timepoint=TIMEPOINT/combined_fitbit.parquet"
        - Also recodes the fitbit data according to specific rules from the ABCD Data Release 6.0 documentation and drops unnecessary meta columns
        - For MRI data, it extracts all MRI ROIs for each selected subject and timepoint across the specified phenotype files and accumulates them into a single parquet file 
        for easier querying with DuckDB. It adds two columns to the combined parquet file: "subject" and "timepoint", which are extracted from the file paths of the original MRI phenotype files, for easy filtering in DuckDB. 
        The combined parquet file is saved in a new directory at the top of the dta_path: "processed_mri_data/all_subjects_combined_mri.parquet"
    Parameters:
        dta_path (Path): Path to the raw data directory
        fit_meta_df (DataFrame): DataFrame containing metadata for the selected fitbit files (subjects, timepoints, filepaths) -> also used for mri data to get selected subjects
    Returns:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
    '''

    # Create output directories for fitbit and mri data
    output_dir_fit = dta_path / "processed_fitbit_data"
    output_dir_mri = dta_path / "processed_mri_data"
    output_dir_fit.mkdir(parents=True, exist_ok=True)
    output_dir_mri.mkdir(parents=True, exist_ok=True)

    if overwrite == False:
        print("DuckDB setup skipped (overwrite=False). To re-run data transformation and DuckDB setup, set overwrite=True.")
        try:
            con = duckdb.connect()
            con.execute(f"CREATE OR REPLACE VIEW fitbit_data AS SELECT * FROM read_parquet('{output_dir_fit}/**/combined_fitbit.parquet', union_by_name => TRUE)")
            con.execute(f"CREATE OR REPLACE VIEW mri_data AS SELECT * FROM read_parquet('{output_dir_mri}/all_subjects_combined_mri.parquet')")
        except Exception as e:
            print(f"Error setting up DuckDB views: {e}")
            print("Please check that the combined parquet files exist in the output directories and are correctly formatted.")
            raise e

        return con
    
    # Combine fitbit files for each INCLUDED subject and timepoint into a single parquet file based on datetime index
    for (subject, timepoint), group in tqdm(
        fit_meta_df.groupby(["subject", "timepoint"]),
        total=fit_meta_df.groupby(["subject", "timepoint"]).ngroups,
        desc="Combining Fitbit files",
        ):
        combined_df = None

        for _, row in group.iterrows():
            filepath = row["filepath"]

            # Read the fitbit file
            fit_df = _load_fitbit_df(filepath)

            # Recode fitbit data
            fit_df = _recode_fitbit_data(fit_df)

            value_cols = [
                col for col in fit_df.columns
                if col != "Wear_Time" and fit_df[col].notna().any()
            ]
            if not value_cols:
                continue

            # Extract metric name from filename (e.g., "Cal1m", "HR1m", etc.) and rename value columns
            # to include metric name for easier identification after merging
            stem = Path(filepath).stem
            metric_name = stem.split("task-fitb", 1)[1].split("_", 1)[0]
            fit_df = fit_df[["Wear_Time", *value_cols]].rename(
                columns={col: f"{col}_{metric_name}" for col in value_cols}
            )

            # Merge into the running combined frame for this subject-timepoint, aligning on Wear_Time.
            # Outer join so metrics with differing timestamp coverage don't drop each other's rows.
            if combined_df is None:
                combined_df = fit_df
            else:
                combined_df = combined_df.merge(fit_df, on="Wear_Time", how="outer")

        # Skip if no file in this group contributed any data
        if combined_df is None:
            continue

        # Add subject and timepoint columns
        combined_df["subject"] = subject
        combined_df["timepoint"] = timepoint

        # Define output path for combined parquet file
        subject_dir = output_dir_fit / f"{subject}"
        timepoint_dir = subject_dir / f"{timepoint}"
        timepoint_dir.mkdir(parents=True, exist_ok=True)
        output_file = timepoint_dir / "combined_fitbit.parquet"

        # Save combined dataframe as parquet file aligned on Wear_Time (overwrites existing files)
        combined_df.sort_values("Wear_Time").to_parquet(output_file, index=False)
        
    # Get MRI ROIs and files to import
    mri_files, mri_rois_dict = mri_rois()

    # Accumulate MRI data for each subject-timepoint across all phenotype files
    mri_data_accumulator = {}  # {(subject, timepoint): {columns from all files}}
    mri_column_source = {}     # {(subject, timepoint): {column: source_file}} - tracks provenance to detect collisions

    # Extract MRI data for each subject and timepoint and accumulate across all files
    for file in tqdm(mri_files, total=len(mri_files), desc="Processing MRI phenotype files"):
        mri_df = pd.read_csv(dta_path / "phenotype" / file, sep="\t")

        # CHANGE: keep all columns from the file (not just mri_rois_dict) for possible future analysis.
        # participant_id/session_id are still needed for the merge below and dropped afterward.
        if "participant_id" not in mri_df.columns or "session_id" not in mri_df.columns:
            print(f"Warning: {file} missing participant_id/session_id — skipping file")
            continue

        merged_df = mri_df.merge(
            fit_meta_df[["subject", "timepoint"]].drop_duplicates(),
            left_on=["participant_id", "session_id"],
            right_on=["subject", "timepoint"],
            how="inner",
        ).drop(columns=["participant_id", "session_id"])

        # Warn if the inner join dropped everything (likely a session_id/timepoint encoding
        # mismatch between this file and fit_meta_df), since that fails silently otherwise.
        if len(mri_df) > 0 and len(merged_df) == 0:
            print(f"Warning: {file} had {len(mri_df)} rows but 0 matched fit_meta_df on subject/timepoint — "
                f"check session_id encoding (e.g. '{mri_df['session_id'].iloc[0]}' vs "
                f"'{fit_meta_df['timepoint'].iloc[0]}')")

        # Accumulate this file's data for each subject-timepoint
        for _, row in merged_df.iterrows():
            subject = row["subject"]
            timepoint = row["timepoint"]
            key = (subject, timepoint)

            if key not in mri_data_accumulator:
                mri_data_accumulator[key] = {}
                mri_column_source[key] = {}

            # Merge this row's data into the accumulator
            for col in row.index:
                if col not in ["subject", "timepoint"]:
                    # Detect collisions before overwriting instead of silently clobbering
                    # an earlier file's value for this column.
                    if col in mri_column_source[key] and mri_column_source[key][col] != file:
                        print(f"Warning: column '{col}' for {key} present in both "
                            f"'{mri_column_source[key][col]}' and '{file}' — keeping value from '{file}'")
                    mri_data_accumulator[key][col] = row[col]
                    mri_column_source[key][col] = file

    # Write accumulated MRI data to one bit parquet file
    all_rows = []
    for (subject, timepoint), data_dict in mri_data_accumulator.items():
        row = dict(data_dict)
        row["subject"] = subject
        row["timepoint"] = timepoint
        all_rows.append(row)

    mri_combined_df = pd.DataFrame(all_rows)

    output_file = output_dir_mri / "all_subjects_combined_mri.parquet"
    mri_combined_df.to_parquet(output_file, index=False)
    
    # Setup DuckDB connection to query the combined fitbit and mri data
    con = duckdb.connect()
    # Use read_parquet with union_by_name=True to allow files with differing schemas
    con.execute(f"CREATE OR REPLACE VIEW fitbit_data AS SELECT * FROM read_parquet('{output_dir_fit}/**/combined_fitbit.parquet', union_by_name => TRUE)")
    con.execute(f"CREATE OR REPLACE VIEW mri_data AS SELECT * FROM read_parquet('{output_dir_mri}/all_subjects_combined_mri.parquet')")

    # Sanity check
    n_fitbit = con.execute("SELECT COUNT(DISTINCT subject) FROM fitbit_data").fetchone()[0]
    n_mri    = con.execute("SELECT COUNT(DISTINCT subject) FROM mri_data").fetchone()[0]
    print(f"✓ DuckDB ready — {n_fitbit} Fitbit subjects, {n_mri} MRI subjects")

    return con

def describe_subjects(fit_meta_df, mri_meta_df):
    '''
    This function prints descriptive statistics about the fitbit data for included subjects at the first timepoint in general and split by file domain (e.g., "fitbInt1m", "fitbCal1m", etc.) to the console.
    Parameters:
        fit_meta_df (DataFrame): DataFrame containing metadata for the selected fitbit files (subjects, timepoints, filepaths, recording durations, missingness percentages, etc.)
        mri_meta_df (DataFrame): DataFrame containing MRI metadata for the subjects
    Returns:
        None (prints descriptive statistics to console)
    '''

    if "missing_days_percentage" in fit_meta_df.columns:
        missingness_series = pd.to_numeric(fit_meta_df["missing_days_percentage"], errors="coerce")
        needs_recompute = missingness_series.isna().any() or (~missingness_series.between(0, 100)).any()
    else:
        needs_recompute = True

    if needs_recompute:
        if {"present_recording_days", "recording_duration_days"}.issubset(fit_meta_df.columns):
            fit_meta_df = fit_meta_df.copy()
            present_days = pd.to_numeric(fit_meta_df["present_recording_days"], errors="coerce")
            recording_days = pd.to_numeric(fit_meta_df["recording_duration_days"], errors="coerce")
            fit_meta_df["missing_days_percentage"] = 100.0
            valid_duration = recording_days > 0
            fit_meta_df.loc[valid_duration, "missing_days_percentage"] = (
                1 - (present_days.loc[valid_duration] / recording_days.loc[valid_duration])
            ) * 100
            fit_meta_df["missing_days_percentage"] = fit_meta_df["missing_days_percentage"].clip(lower=0, upper=100)
        else:
            raise KeyError(
                "fit_meta_df must contain 'missing_days_percentage' or the columns required to derive it: "
                "'present_recording_days' and 'recording_duration_days'."
            )

    # Print demographics of selected subjects with both fitbit and mri data
    print("General demographics:")
    print(f"N: {mri_meta_df['subject'].nunique()}")
    print("\nMRI Age Statistics:")
    print(f"Mean: {mri_meta_df['age_at_mri'].mean()}, Std: {mri_meta_df['age_at_mri'].std()}, Min: {mri_meta_df['age_at_mri'].min()}, Max: {mri_meta_df['age_at_mri'].max()}")
    print("\nSex Distribution:")
    print(mri_meta_df["sex"].value_counts())
    print(f"\nAverage missingness percentage of fitbit data: {fit_meta_df['missing_days_percentage'].mean()}")

    # Print Fitbit completeness at the first available timepoint, split by file domain
    first_timepoint_fit_df = fit_meta_df[["subject", "timepoint", "filename", "short", "missing_days_percentage"]].copy()
    first_timepoint_fit_df["domain"] = "other"
    first_timepoint_fit_df.loc[
        first_timepoint_fit_df["filename"].str.contains(r"fitbInt1m", case=False, regex=True),
        "domain",
    ] = "actigraphy"
    first_timepoint_fit_df.loc[
        first_timepoint_fit_df["filename"].str.contains(r"fitbHR1m", case=False, regex=True),
        "domain",
    ] = "heart_rate"
    first_timepoint_fit_df.loc[
        first_timepoint_fit_df["filename"].str.contains(r"fitbSlp1m", case=False, regex=True),
        "domain",
    ] = "sleep"
    first_timepoint_fit_df = first_timepoint_fit_df[first_timepoint_fit_df["domain"] != "other"]
    first_timepoint_fit_df["timepoint_order"] = first_timepoint_fit_df["timepoint"].str.extract(r"(\d+)")[0].astype(int)

    first_timepoint_subject_df = (
        first_timepoint_fit_df.sort_values(["subject", "timepoint_order", "timepoint"])
        .drop_duplicates(subset=["subject"], keep="first")[["subject", "timepoint"]]
    )

    first_timepoint_fit_df = first_timepoint_fit_df.merge(first_timepoint_subject_df, on=["subject", "timepoint"], how="inner")

    domain_subject_df = (
        first_timepoint_fit_df.groupby(["domain", "subject"], as_index=False)
        .agg(
            n_files=("filename", "size"),
            has_short=("short", lambda s: bool((s == 1).any())),
            has_non_short=("short", lambda s: bool((s == 0).any())),
            mean_missingness=("missing_days_percentage", "mean"),
        )
    )

    print("\nSubject descriptive statistics at the first available timepoint, by domain:")

    for domain in ["actigraphy", "heart_rate", "sleep"]:
        domain_df = domain_subject_df[domain_subject_df["domain"] == domain]
        if domain_df.empty:
            continue

        short_subjects = set(domain_df.loc[domain_df["has_short"], "subject"])
        non_short_subjects = set(domain_df.loc[domain_df["has_non_short"], "subject"])

        print(f"\n{domain.replace('_', ' ').title()}:")
        print(f"Subjects at first timepoint: {domain_df['subject'].nunique()}")
        print(f"Subjects with any short files: {len(short_subjects)}")
        print(f"Subjects with any non-short files: {len(non_short_subjects)}")
        print(f"Average missingness percentage: {domain_df['mean_missingness'].mean()}")

        for label, subject_set in [("short", short_subjects), ("non-short", non_short_subjects)]:
            subject_list = list(subject_set)
            if not subject_list:
                print(f"\n{domain.replace('_', ' ').title()} - {label.title()}:")
                print("N: 0")
                continue

            subject_mri_df = mri_meta_df[mri_meta_df["subject"].isin(subject_list)]

            print(f"\n{domain.replace('_', ' ').title()} - {label.title()}:")
            print(f"N: {len(subject_set)}")
            print("MRI Age Statistics:")
            print(
                f"Mean: {subject_mri_df['age_at_mri'].mean()}, "
                f"Std: {subject_mri_df['age_at_mri'].std()}, "
                f"Min: {subject_mri_df['age_at_mri'].min()}, "
                f"Max: {subject_mri_df['age_at_mri'].max()}"
            )
            print("Sex Distribution:")
            print(subject_mri_df["sex"].value_counts())
            print("Average missingness percentage:")
            print(domain_df[domain_df["subject"].isin(subject_set)]["mean_missingness"].mean())
