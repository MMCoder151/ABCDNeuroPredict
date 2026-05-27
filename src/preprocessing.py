import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.mri_rois import mri_rois
import duckdb
from pcntoolkit import NormativeModel, BLR, Runner
from pcntoolkit.dataio.norm_data import NormData
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import STL
import pathlib
from pyampute.exploration.mcar_statistical_tests import MCARTest
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.mixture import BayesianGaussianMixture
import hdbscan
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score
from pyampute.exploration.mcar_statistical_tests import MCARTest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

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
    # TODO: Extract all column names with non-numeric content

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

    return fit_df

def select_subjects(dta_path, test=False, overwrite=True, output_path=pathlib.Path("output")):
    '''
    This function selects subjects and time points based on selection criteria (below) and extracts demographic and meta information for fitbit and mri data
        - Only subjects with both "fit" and "scans" files are included (=> subjects with both Fitbit and MRI data)
        - Only timepoints/sessions with both "fit" and "scans" files are included (=> timepoints/sessions with both Fitbit and MRI data)
        - Drops Slp30s files, due to unusable data according to ABCD Data Release 6.0 documentation
        - Only subjects/sessions with complete "fit" data (i.e., all 6 "fit" files present) are included (=> complete Fitbit data for included subjects/sessions)
        - Only "scans" files with non-empty "acq_time" column are included (=> valid MRI acquisition date for included sessions)
        - Only subjects/timepoints with more than 7 days of actually recorded Fitbit data and/or <60% missing data are included (=> sufficient Fitbit data for included sessions)
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

    fit_meta_df["missing_days_percentage"] = 100 * (1 - (fit_meta_df["present_recording_days"] / fit_meta_df["recording_duration_days"]))
    short_recordings = fit_meta_df[(fit_meta_df["recording_duration_days"] < 14)][["subject", "timepoint"]].drop_duplicates()
    # short_recordings = fit_meta_df[(fit_meta_df["recording_duration_days"] < 7) | (fit_meta_df["missing_days_percentage"] < 0.6)][["subject", "timepoint"]].drop_duplicates()
    fit_meta_df = fit_meta_df.merge(short_recordings, on=["subject", "timepoint"], how="left", indicator=True)
    fit_meta_df = fit_meta_df[fit_meta_df["_merge"] == "left_only"].drop(columns=["_merge"])
    print(f"Dropped {len(short_recordings)} timepoints with less than 14 days of fitbit data.")
    #print(f"Dropped {len(short_recordings)} timepoints with less than 7 days and/or >60% missings of fitbit data.")
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

    # Check that subject/timepoint PAIRS in mri_meta_df and fit_meta_df match exactly
    pairs_fit = set(map(tuple, fit_meta_df[["subject","timepoint"]].drop_duplicates().values))
    pairs_mri = set(map(tuple, mri_meta_df[["subject","timepoint"]].drop_duplicates().values))
    assert pairs_fit == pairs_mri, "Subject-timepoint pairs in mri_meta_df and fit_meta_df do not match"

    # drop filepath and filename columns from mri_meta_df
    mri_meta_df = mri_meta_df.drop(columns=["filepath", "filename"])

    # Drop duplicates from mri_meta_df and demo_df
    mri_meta_df.drop_duplicates(subset=["subject", "timepoint"], inplace=True)
    demo_df.drop_duplicates(subset=["subject"], inplace=True)

    print(f"Final number of subjects included after selection: {demo_df['subject'].nunique()}")

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
        - For MRI data, it extracts the MRI ROIs specified in mri_rois for each subject and timepoint and saves it in a similar hive-style directory structure 
        at the top of the dta_path: "processed_mri_data/subject=SUBJECT_ID/timepoint=TIMEPOINT/combined_mri.parquet"
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
            con.execute(f"CREATE OR REPLACE VIEW mri_data AS SELECT * FROM read_parquet('{output_dir_mri}/**/combined_mri.parquet', union_by_name => TRUE)")
        except Exception as e:
            print(f"Error setting up DuckDB views: {e}")
            print("Please check that the combined parquet files exist in the output directories and are correctly formatted.")
            raise e

        return con
    
    # Combine fitbit files for each subject and timepoint into a single parquet file based on datetime index
    for _, row in tqdm(fit_meta_df.iterrows(), total=len(fit_meta_df), desc="Combining Fitbit files"):
        subject = row["subject"]
        timepoint = row["timepoint"]
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

def normative_selection(con, mri_meta_df, output_path=pathlib.Path("output"), overwrite=True):
    '''
    This function performs normative modeling and selects subjects based on their composite absolute z-score. 
    It selects the top 10% (based on prevalence) of subjects with the highest cumulative z-score.
    These subjects are considered to have abnormal development in the selected MRI ROIs associated with depression.

    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        mri_meta_df (DataFrame): DataFrame containing MRI metadata
        output_path (pathlib.Path): Path to the output directory where the normative model results will be saved
    Returns:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their MRI ROI data and respective z-scores
        normative_modelling (Folder): Folder containing the normative model, results, and plots created in the output directory
    '''

    if overwrite == False:
        print("Normative modeling and subject selection skipped (overwrite=False). To re-run normative modeling and subject selection, set overwrite=True.")
        try:
            selected_subjects = pd.read_csv(Path(output_path) / "normative_modelling" / "results" / "selected_subjects.csv")
            return selected_subjects
        except Exception as e:
            print(f"Error loading selected subjects: {e}")
            print("Please check that the selected_subjects.csv file exists in the normative_modelling results directory and is correctly formatted.")
            raise e

    # Get MRI ROIs to include in the normative model
    _, mri_rois_dict = mri_rois()
    mri_roi_cols = list(mri_rois_dict.keys())

    # Get subjects to include from mri_meta_df
    included_subjects = mri_meta_df["subject"].unique()
    
    # Query MRI data for the first timepoint for each included subject
    query = f"""
        SELECT *
        FROM mri_data
        WHERE subject IN ({', '.join(f"'{sub}'" for sub in included_subjects)})
        AND timepoint IN (
            SELECT MIN(timepoint)
            FROM mri_data AS sub_mri
            WHERE sub_mri.subject = mri_data.subject
        )
    """
    mri_df = con.execute(query).df()
    print(f"MRI data loaded: {len(mri_df)} subjects")

    # Merge MRI data with demographic data
    df = mri_df.merge(
    mri_meta_df[["subject", "sex", "age_at_mri", "scan_site"]].drop_duplicates(),
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
    df['scan_site'] = pd.to_numeric(df['scan_site'], errors='raise').astype(float)
    df['subject'] = df['subject'].astype(str)

    # Prepare data for normative modeling
    data = NormData.from_dataframe(
        name="mri_norm",
        dataframe=df,
        covariates=["sex", "age_at_mri", "scan_site"],
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
        saveplots=False,
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
    subject_scores = (
        centiles_df[["subject_ids", "composite_z"]]
        .dropna()
        .drop_duplicates(subset=["subject_ids"])
        .sort_values("composite_z")
        .reset_index(drop=True)
    )
    subject_scores["subject_ids"].nunique()
    subject_scores["rank"] = np.arange(len(subject_scores))
    # Select top 10% of subjects with the highest composite z-score based on ranked prevalence
    n_select = int(np.ceil(0.10 * len(subject_scores)))
    selected_subject_ids = subject_scores.nlargest(n_select, "composite_z")["subject_ids"]
    selected_subjects = subject_scores[subject_scores["subject_ids"].isin(selected_subject_ids)]
    
    print(f"Selected {len(selected_subject_ids)} subjects with the highest composite z-scores based on a prevalence threshold of 10%.")
    selected_subjects.to_csv(normative_output_dir / "results" / "selected_subjects.csv", index=False)
    
    # create scatter plot of composite z-scores for all subjects, highlighting selected subjects in a different color
    plot_df = centiles_df[["subject_ids", "composite_z"]].dropna().sort_values("composite_z").reset_index(drop=True)
    plot_df["rank"] = np.arange(len(plot_df))

    plt.figure(figsize=(10, 6))
    plt.scatter(subject_scores["rank"], subject_scores["composite_z"], label="All subjects", alpha=0.5, s=12)
    selected_plot = subject_scores[subject_scores["subject_ids"].isin(selected_subject_ids)]
    plt.scatter(selected_plot["rank"], selected_plot["composite_z"], label="Selected subjects", color="red", s=18)
    plt.xlabel("Subject rank by composite z-score")
    plt.ylabel("Composite Absolute Z-Score")
    plt.title("Composite Absolute Z-Scores for MRI ROIs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(normative_output_dir / "results" / "composite_z_scores.png")
    plt.close()

    # Create summary table with mean, std, min, and max per mri_roi for the selected subjects
    stats_selected = (
        centiles_df[mri_roi_cols]
        .agg(['mean', 'std', 'min', 'max'])
        .transpose()
        .reset_index()
        .rename(columns={"index": "mri_roi"})
    )
    stats_selected.to_csv(normative_output_dir / "results" / "mri_roi_statistics.csv", index=False)

    # Create results summary table with mean, std, min, and max per metric
    stats_df = pd.read_csv(normative_output_dir / "results" / "statistics_mri_norm.csv")
    summary = stats_df.assign(
        mean = stats_df[mri_roi_cols].mean(axis=1),
        std  = stats_df[mri_roi_cols].std(axis=1),
        min  = stats_df[mri_roi_cols].min(axis=1),
        max  = stats_df[mri_roi_cols].max(axis=1),
    )[["statistic", "mean", "std", "min", "max"]]
    summary.to_csv(normative_output_dir / "results" / "statistics_summary.csv", index=False)

    return selected_subjects

def create_mri_composites(con, selected_subjects):
    '''
    This function creates composite scores out of selected subject's z-scores based on 
    variance inflation factors (VIF) to account for multicollinearity between MRI ROIs.

    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their MRI ROI data and respective z-scores
    Returns:
        composite_scores_df (DataFrame): DataFrame containing the selected subjects and their composite scores based on the selected MRI ROIs
        composite_dict (dict): Dictionary mapping composite score names to the MRI ROIs included in each composite
    '''

    # Get MRI ROIs to include in the normative model
    _, mri_rois_dict = mri_rois()
    mri_roi_cols = list(mri_rois_dict.keys())

    # Calculate variance inflation factors (VIF) for the selected MRI ROIs
    vif_data = selected_subjects[mri_roi_cols].dropna()
    vif_df = pd.DataFrame({
        "mri_roi": vif_data.columns,
        "vif": [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
    }).sort_values("vif", ascending=False)
    vif_df.head()

    # Replace variables with a vif above 10 with their average until all variables have a vif below 10
    composite_dict = {}
    while vif_df["vif"].max() > 10:
        # Get the variable with the highest VIF
        high_vif_roi = vif_df.iloc[0]["mri_roi"]
        # Create a composite score by averaging this variable with the variable it is most correlated with
        correlations = vif_data.corr()[high_vif_roi].drop(high_vif_roi).abs()
        most_correlated_roi = correlations.idxmax()
        composite_name = f"composite_{high_vif_roi}_{most_correlated_roi}"
        selected_subjects[composite_name] = selected_subjects[[high_vif_roi, most_correlated_roi]].mean(axis=1)
        composite_dict[composite_name] = [high_vif_roi, most_correlated_roi]
        # Drop the original variable with high VIF from the data used to calculate VIFs in the next iteration
        vif_data = vif_data.drop(columns=[high_vif_roi])
        # Recalculate VIFs
        vif_df = pd.DataFrame({
            "mri_roi": vif_data.columns,
            "vif": [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
        }).sort_values("vif", ascending=False)
    print(f"Composites created: {composite_dict}")

    return selected_subjects, composite_dict

def mri_subtyping(dem_df, selected_subjects):
    '''
    This function performs clustering to identify subtypes of depression based on the selected subjects' MRI ROI data.
    It uses several different clustering algorithms (HBDSCAN and Bayesian Gaussian Mixture Models).
    Clustering stability is assessed using bootstrapping and assessing stability using the Jaccard index.
    Algorithms are compared based on their Silhouette coefficient, Density-Based Clustering Validation (DBCV) score, and Davies-Bouldin Index (DBI).

    Parameters:
        dem_df (DataFrame): DataFrame containing demographic information for the selected subjects
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their MRI ROI z-scores
    Returns:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their assigned cluster labels based on their MRI ROI data

    NOTE: This function is currently broken. Neither clustering algorithms compute properly. This is most likely due to the subjects and ROIs already being
    pre-selected and therefore not showing enough differentiating patterns.
    '''
    output_path = Path(output_path)
    # selected subjects read csv
    selected_subjects = pd.read_csv(os.path.join(output_path, "selected_subjects.csv"))
    selected_subjects.shape
    selected_subjects.drop(columns=["observations", "composite_z"], inplace=True)

    def _align_labels(reference, target):
        '''Aligns cluster labels of the target clustering to the reference clustering using the Hungarian algorithm.'''
        # Compute confusion matrix between reference and target labels
        conf_matrix = confusion_matrix(reference, target)
        # Use Hungarian algorithm to find optimal label alignment
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        # Create a mapping from target labels to reference labels
        label_mapping = {target_label: reference_label for target_label, reference_label in zip(col_ind, row_ind)}
        # Apply the mapping to the target labels
        aligned_target = target.map(label_mapping)
        return aligned_target 
    
    # Create reference clustering on the original data
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    reference_labels_hdbscan = hdbscan_clusterer.fit_predict(selected_subjects.drop(columns=["subject_ids"]))
    gmm_clusterer = BayesianGaussianMixture(n_components=3, random_state=42)
    reference_labels_gmm = gmm_clusterer.fit_predict(selected_subjects.drop(columns=["subject_ids"]))

    # Create mask for non-noise points in HDBSCAN and GMM to use for alignment and scoring (only consider points that are not labeled as noise in either clustering)
    mask = (reference_labels_hdbscan[bootstrap_sample.index] >= 0) & (boot_labels >= 0)

    ref_clean  = reference_labels_hdbscan[bootstrap_sample.index][mask]
    
    jaccard_indices = {
        "hdbscan": [],
        "gmm": []
    }
    
    for i in tqdm(range(100), desc="Bootstrapping for clustering stability"):
        # Resample subjects with replacement
        bootstrap_sample = selected_subjects.sample(frac=1, replace=True, random_state=i)

        # HDBSCAN clustering
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        boot_labels = hdbscan_clusterer.fit_predict(bootstrap_sample.drop(columns=["subject_ids"]))
        boot_clean = boot_labels[mask]
        aligned_boot_labels = _align_labels(ref_clean[bootstrap_sample.index], boot_clean)
        jaccard_hdbscan = jaccard_score(reference_labels_hdbscan[bootstrap_sample.index], aligned_boot_labels, average="macro")
        jaccard_indices["hdbscan"].append(jaccard_hdbscan)

        # Gaussian Mixture Models clustering
        gmm_clusterer = BayesianGaussianMixture(n_components=3, random_state=i)
        boot_labels = gmm_clusterer.fit_predict(bootstrap_sample.drop(columns=["subject_ids"]))
        boot_clean = boot_labels[mask]
        aligned_boot_labels = _align_labels(reference_labels_gmm[bootstrap_sample.index], boot_clean)
        jaccard_gmm = jaccard_score(reference_labels_gmm[bootstrap_sample.index], aligned_boot_labels, average="macro")
        jaccard_indices["gmm"].append(jaccard_gmm)
    print(f"HDBSCAN clustering stability (Jaccard index): mean={np.mean(jaccard_indices['hdbscan']):.4f}, std={np.std(jaccard_indices['hdbscan']):.4f}")
    print(f"GMM clustering stability (Jaccard index): mean={np.mean(jaccard_indices['gmm']):.4f}, std={np.std(jaccard_indices['gmm']):.4f}")

def missingness_analysis(con, fit_meta_df):
    '''
    This function analyzes the missingness patterns in the fitbit data for the selected subjects for MCAR, MAR, or MNAR missingness
    using Littles test from pyampute.
        1. Queries different fitbit domain data separately (i.e. actigraphy-based: Stps1m, METs1m, Int1m, heart-rate: HR1m, Sleep: Slp1m)
        2. For each, creates datetime index (days) with proper missing days based on min and max of the Wear_Time column for each subject and timepoint, 
        and merges with the original data to get missingness patterns
        3. Conducts Little's test for each fitbit domain to assess whether the missingness is MCAR, MAR, or MNAR

    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        fit_meta_df (DataFrame): DataFrame containing the fitbit metadata for the selected subjects
    Returns:
        missingness_df (DataFrame): DataFrame containing the missingness patterns in the fitbit data for the selected subjects.

    NOTE: This function is currently broken. Little's test is not computing properly and I have no fucking clue why. WIP
    '''

    # For each subject and timepoint, create datetime index with proper missing days based on min and max of the Wear_Time column, and merge with original data to get missingness patterns
    def create_missingness_df(df, domain_name):
        missingness_list = []
        grouped = df.groupby(["subject", "timepoint"])
        for (subject, timepoint), group in tqdm(grouped, total=grouped.ngroups, desc=f"Creating missingness patterns for {domain_name}"):
            min_date = group["Wear_Time"].min().floor("D")
            max_date = group["Wear_Time"].max().ceil("D")
            value_cols = [c for c in group.columns if c not in ["subject", "timepoint", "Wear_Time"]]
            daily_means = group.set_index("Wear_Time")[value_cols].resample("D").mean().reset_index()
            date_range = pd.date_range(start=min_date, end=max_date, freq="D")
            date_df = pd.DataFrame({"Wear_Time": date_range})
            merged_df = date_df.merge(daily_means, on="Wear_Time", how="left")
            merged_df["subject"] = subject
            merged_df["timepoint"] = timepoint
            missingness_list.append(merged_df)
        return pd.concat(missingness_list, ignore_index=True)
    
    # Conduct Little's test for each fitbit domain to assess if the missingness is MCAR
    mcar_test = MCARTest(method="little")

    def safe_little_test(df, min_obs_per_col=5, min_cols=2):
        df = df.copy()
        # coerce numeric and drop empty cols/rows
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.loc[:, df.notna().sum() >= min_obs_per_col]
        df = df.dropna(how='all')
        if df.shape[1] < min_cols or df.shape[0] < 2:
            return np.nan, "insufficient data after filtering"
        # compute pj/df quickly (same logic as pyampute)
        vars_ = df.dtypes.index.values
        n_var = df.shape[1]
        r = 1 * df.isnull()
        mdp = np.dot(r, [2**i for i in range(n_var)])
        pj = 0
        for i in np.unique(mdp):
            dataset_temp = df.loc[mdp == i, vars_]
            select_vars = ~dataset_temp.isnull().any()
            pj += np.sum(select_vars)
        df_val = pj - n_var
        if df_val <= 0:
            return np.nan, f"df <= 0 (pj={pj}, n_var={n_var})"
        # try running the test and catch numerical errors
        try:
            pval = MCARTest(method="little").little_mcar_test(df)
        except Exception as e:
            return np.nan, f"test error: {e}"
        return pval, None

    p, err = safe_little_test(actigraphy_missingness_df.drop(columns=["subject","timepoint","Wear_Time"]))
    print("p:", p, "err:", err)

    # Query all fitbit data
    fitbit_df = con.execute(f"SELECT subject, timepoint, Wear_Time, * FROM fitbit_data").df()
    print(f"Creating missingness pattern...")
    fitbit_missingness_df = create_missingness_df(fitbit_df, "fitbit")
    fitbit_df = None  # free memory
    fitbit_mcar = mcar_test.little_mcar_test(fitbit_missingness_df.drop(columns=["subject", "timepoint", "Wear_Time"]))
    fitbit_missingness_df = None  # free memory

    # Query actigraphy columns
    actigraphy_cols = ["Calories_Cal1m", "Steps_Stps1m", "METs_METs1m"]
    actigraphy_df = con.execute(f"SELECT subject, timepoint, Wear_Time, {', '.join(actigraphy_cols)} FROM fitbit_data").df()
    # Create missingness patterns for actigraphy data
    actigraphy_missingness_df = create_missingness_df(actigraphy_df, "actigraphy")
    actigraphy_df = None  # free memory
    # Conduct Little's test for actigraphy data
    actigraphy_mcar = mcar_test.little_mcar_test(actigraphy_missingness_df.drop(columns=["subject", "timepoint", "Wear_Time"]))
    actigraphy_missingness_df = None  # free memory

    actigraphy_df.head()
    actigraphy_df.shape
    actigraphy_missingness_df.head()
    actigraphy_missingness_df.shape
    actigraphy_missingness_df["Wear_Time"].min(), actigraphy_missingness_df["Wear_Time"].max()

    daily_means = actigraphy_df.set_index("Wear_Time")[actigraphy_cols].resample("D").mean().reset_index()
    daily_means.head(10)

    # Query heart rate columns
    heart_rate_df = con.execute(f"SELECT subject, timepoint, Wear_Time, Value_HR1m FROM fitbit_data").df()
    # Create missingness patterns for heart rate data
    heart_rate_missingness_df = create_missingness_df(heart_rate_df, "heart_rate")
    heart_rate_df = None  # free memory
    # Conduct Little's test for heart rate data
    heart_rate_mcar = mcar_test.little_mcar_test(heart_rate_missingness_df.drop(columns=["subject", "timepoint", "Wear_Time"]))
    heart_rate_missingness_df = None  # free memory

    # Query sleep columns
    sleep_df = con.execute(f"SELECT subject, timepoint, Wear_Time, value_Slp1m FROM fitbit_data").df()
    # Create missingness patterns for sleep data
    sleep_missingness_df = create_missingness_df(sleep_df, "sleep")
    sleep_df = None  # free memory
    # Conduct Little's test for sleep data
    sleep_mcar = mcar_test.little_mcar_test(sleep_missingness_df.drop(columns=["subject", "timepoint", "Wear_Time"]))
    sleep_missingness_df = None  # free memory

    print(f"Actigraphy missingness MCAR test: p-value={actigraphy_mcar:.4f}")
    print(f"Heart rate missingness MCAR test: p-value={heart_rate_mcar:.4f}")
    print(f"Sleep missingness MCAR test: p-value={sleep_mcar:.4f}")

# ---- FEATURE EXTRACTION ----

def extr_fitbit_features(con, selected_subjects):
    '''
    This function extracts features from the fitbit data for the selected subjects.
        1. Creates daily mean, std, min, and max for each fitbit metric
        2. Imputes missing days 
        3. Conducts weekly Seasonal Trend Decomposition using Loess (STL) for each daily fitbit metric
        4. Creates mean, stdm, min, and max for the trend, seasonal, and residual components of the STL decomposition for each fitbit metric
    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        selected_subjects (DataFrame): DataFrame containing the subjects to extract fitbit features from 
    Returns:
        fitbit_features_df (DataFrame): DataFrame containing the extracted fitbit features for each subject
    '''
    # Get list of selected subjects
    selected_subjects_list = selected_subjects["subject"].unique().tolist()

    # Query FIRST timepoint for the selected subjects
    query = f"""
    SELECT *
    FROM fitbit_data
    WHERE timepoint = (
        SELECT MIN(timepoint)
        FROM fitbit_data f2
        WHERE f2.subject = fitbit_data.subject
    )
    AND subject IN ({', '.join(map(str, selected_subjects_list))})
    """
    print("Querying fitbit data for feature extraction...")
    fitbit_df = con.execute(query).df()

    # Get fitbit metric columns (exclude subject, timepoint, and Wear_Time)
    fitbit_metric_cols = [col for col in fitbit_df.columns if col not in ["subject", "timepoint", "Wear_Time"]]

    # Coerce to numeric
    for col in fitbit_metric_cols:
        fitbit_df[col] = pd.to_numeric(fitbit_df[col], errors="coerce")

    # Create a dataframe to hold the extracted features
    features_list = []

    # Loop through each subject and timepoint to extract features
    grouped = fitbit_df.groupby(["subject", "timepoint"])
    for (subject, timepoint), group in tqdm(grouped, total=grouped.ngroups, desc="Extracting Fitbit features"):
        feature_dict = {"subject": subject, "timepoint": timepoint}
        for metric in fitbit_metric_cols:
            # Check if the metric column exists in the group
            #if metric in group.columns:
                daily_data = group[["Wear_Time", metric]]#.dropna()
                if not daily_data.empty:
                    # Create daily features (mean, std, min, max)
                    daily_data.set_index("Wear_Time", inplace=True)
                    daily_stats = daily_data.resample("D").agg(['mean', 'std', 'min', 'max'])
                    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
                    # Create datetime index with proper missing days based on the daily resampling range
                    min_date = daily_stats.index.min().floor("D")
                    max_date = daily_stats.index.max().ceil("D")
                    date_range = pd.date_range(start=min_date, end=max_date, freq="D")
                    # Reindex to include missing days and impute missing values with multiple imputation
                    daily_stats = daily_stats.reindex(date_range)
                    #if daily_stats.shape[0] > 1 and daily_stats.notna().sum().sum() > daily_stats.shape[1]:
                    try:
                        imputer = IterativeImputer(random_state=0, max_iter=20)
                        daily_stats = pd.DataFrame(
                            imputer.fit_transform(daily_stats),
                            index=daily_stats.index,
                            columns=daily_stats.columns,
                        )
                    except Exception as e:
                        print(f"Iterative imputation failed for subject {subject}, timepoint {timepoint}, metric {metric}: {e}")
                        daily_stats = daily_stats.ffill().bfill()
                    #else:
                        #daily_stats = daily_stats.ffill().bfill()
                    feature_dict.update(daily_stats.mean().to_dict())
                    # STL decomposition
                    try:
                        stl = STL(daily_data[metric], period=7, robust=True)
                        result = stl.fit()
                        stl_features = {
                            f"{metric}_trend_mean": result.trend.mean(),
                            f"{metric}_trend_std": result.trend.std(),
                            f"{metric}_trend_min": result.trend.min(),
                            f"{metric}_trend_max": result.trend.max(),
                            f"{metric}_seasonal_mean": result.seasonal.mean(),
                            f"{metric}_seasonal_std": result.seasonal.std(),
                            f"{metric}_seasonal_min": result.seasonal.min(),
                            f"{metric}_seasonal_max": result.seasonal.max(),
                            f"{metric}_resid_mean": result.resid.mean(),
                            f"{metric}_resid_std": result.resid.std(),
                            f"{metric}_resid_min": result.resid.min(),
                            f"{metric}_resid_max": result.resid.max(),
                        }
                        feature_dict.update(stl_features)
                    except Exception as e:
                        print(f"STL decomposition failed for subject {subject}, timepoint {timepoint}, metric {metric}: {e}")
        features_list.append(feature_dict)
    fitbit_features_df = pd.DataFrame(features_list)

    return fitbit_features_df
