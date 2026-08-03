import pandas as pd
import numpy as np
import os
from pathlib import Path
import duckdb
from tqdm import tqdm
import pathlib
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from statsmodels.tsa.seasonal import STL

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
    if overwrite == False:
        print("Subject selection skipped (overwrite=False). To re-run subject selection, set overwrite=True.")
        fit_meta_df = pd.read_csv(output_path / "fitbit_metadata.csv")
        return fit_meta_df

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

    # Get subjects with multiple timepoints/sessions with both "fit" and "scans" files
    timepoint_counts = fit_meta_df.groupby("subject")["timepoint"].nunique().reset_index(name="timepoint_count")
    subjects_multiple_timepoints = timepoint_counts[timepoint_counts["timepoint_count"] > 1]
    print(f"Number of subjects with multiple timepoints/sessions with fitbit files: {len(subjects_multiple_timepoints)}")
    print(f"Final number of subjects with fitbit data: {fit_meta_df['subject'].nunique()}")

    # save metadata to csv
    output_path.mkdir(parents=True, exist_ok=True)
    fit_meta_df.to_csv(output_path / "fitbit_metadata.csv", index=False)

    return fit_meta_df

def setup_duckdb(dta_path, fit_meta_df, overwrite=False):
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
    output_dir_fit.mkdir(parents=True, exist_ok=True)

    if overwrite == False:
        print("DuckDB setup skipped (overwrite=False). To re-run data transformation and DuckDB setup, set overwrite=True.")
        try:
            con = duckdb.connect()
            con.execute(f"CREATE OR REPLACE VIEW fitbit_data AS SELECT * FROM read_parquet('{output_dir_fit}/**/combined_fitbit.parquet', union_by_name => TRUE)")

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
    
    # Setup DuckDB connection to query the combined fitbit and mri data
    con = duckdb.connect()
    # Use read_parquet with union_by_name=True to allow files with differing schemas
    con.execute(f"CREATE OR REPLACE VIEW fitbit_data AS SELECT * FROM read_parquet('{output_dir_fit}/**/combined_fitbit.parquet', union_by_name => TRUE)")

    # Sanity check
    n_fitbit = con.execute("SELECT COUNT(DISTINCT subject) FROM fitbit_data").fetchone()[0]
    print(f"✓ DuckDB ready — {n_fitbit} Fitbit subjects available for querying.")

    return con

def create_composites(selected_subjects, vif_threshold=10, overwrite=True, output_path=Path("output"), composite_output = None):
    """
    Creates composite scores out of given variables based on variance inflation factors (VIF) and drops zero variance columns.
 
    Parameters:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their data to be analysed
        vif_threshold (float): VIF value above which a variable triggers compositing
        max_iterations (int): safety cap so a degenerate case can't loop forever
 
    Returns:
        selected_subjects (DataFrame): DataFrame with high-VIF columns replaced by composites
        composite_dict (dict): Maps composite score name -> list of *original* base columns
                                that were averaged together to build it (flattened, not nested)
    """

    if overwrite == False:
        print("Overwrite set to False. Reimporting composites.")
        try:
            if composite_output is not None:
                composite_output_path = Path(output_path) / composite_output
                features_path = composite_output_path / "features_with_composites.csv"
                composite_dict_path = composite_output_path / "composite_dictionary.csv"
            else:
                features_path = Path(output_path) / "fitbit_features_with_composites.csv"
                composite_dict_path = Path(output_path) / "composite_dictionary.csv"

            features_with_composites = pd.read_csv(features_path)
            composite_df = pd.read_csv(composite_dict_path)
            return features_with_composites, composite_df
        except Exception as e:
            print(f"An error occured: {e}")
            return e

    vif_cols = [
        col for col in selected_subjects.columns
        if col not in ["subject", "participant_id", "composite_z", "Wear_Time", "subtype", "group", "observations"]
    ]
 
    # Drop zero-variance columns (VIF undefined for these)
    variances = selected_subjects[vif_cols].var()
    zero_variance_cols = variances[variances == 0].index.tolist()
    if zero_variance_cols:
        print(f"Dropping columns with zero variance: {len(zero_variance_cols)}")
        print(zero_variance_cols)
        vif_cols = [c for c in vif_cols if c not in zero_variance_cols]
 
    vif_data = selected_subjects[vif_cols].dropna()
    selected_subjects = selected_subjects[vif_cols].copy()
 
    # Snapshot of original values, indexed the same as vif_data, used so we can always
    # average from true base columns even after they've been dropped from selected_subjects.
    original_values = vif_data.copy()
 
    # composite_dict maps a SHORT composite id -> flattened list of original base columns.
    # base_members maps current-column-name -> flattened list of original base columns,
    # for every column currently alive in vif_data (whether original or composite).
    composite_dict = {}
    base_members = {col: [col] for col in vif_data.columns}
    composite_counter = 0
 
    def vif_table(df):
        if df.shape[1] == 0:
            return pd.DataFrame(columns=["variable", "vif"])
        if df.shape[1] == 1:
            return pd.DataFrame({"variable": df.columns, "vif": [np.inf]})

        df_with_const = sm.add_constant(df)
        vifs = [
            variance_inflation_factor(df_with_const.values, i)
            for i in range(1, df_with_const.shape[1])  # skip the constant column (index 0)
        ]
        return pd.DataFrame({"variable": df.columns, "vif": vifs}).sort_values("vif", ascending=False)
 
    vif_df = vif_table(vif_data)
 
    while not vif_df.empty and vif_df["vif"].max() > vif_threshold:
        if vif_data.shape[1] < 2:
            print("Stopping composite creation because fewer than two VIF columns remain.")
            break
 
        high_vif_col = vif_df.iloc[0]["variable"]
 
        correlations = vif_data.corr()[high_vif_col].drop(high_vif_col).abs()
        if correlations.empty:
            # Nothing left to pair with — stop rather than crash
            break
        most_correlated_col = correlations.idxmax()
 
        # Short, stable, human-readable name — does NOT concatenate ancestry
        composite_counter += 1
        composite_name = f"composite_{composite_counter}"
 
        # Flattened provenance: original base variables in both parents, deduped,
        # order-preserved
        merged_members = []
        for c in base_members[high_vif_col] + base_members[most_correlated_col]:
            if c not in merged_members:
                merged_members.append(c)
 
        print(
            f"Created {composite_name} from '{high_vif_col}' + '{most_correlated_col}' "
            f"(VIF={vif_df.iloc[0]['vif']:.1f}, corr={correlations.max():.3f}) "
            f"-> {len(merged_members)} base vars: {merged_members}"
        )
 
        # Average from the ORIGINAL base columns (not from intermediate composites),
        # so a variable's influence on the final composite doesn't depend on which
        # merge order it happened to go through. We must read these from the
        # `original_values` snapshot taken before the loop started, because by this
        # point some of `merged_members` may already have been dropped from
        # `selected_subjects` in an earlier iteration (folded into a prior composite).
        # NOTE: `original_values` only has rows that survived the initial dropna() for
        # VIF purposes, so this assignment will introduce NaN in `selected_subjects`
        # for any row that had a NaN in ANY vif_col originally. If you need composite
        # scores for those rows too, compute composites on a per-pair basis from
        # selected_subjects directly instead (see alternative below).
        selected_subjects[composite_name] = original_values[merged_members].mean(axis=1)
 
        composite_dict[composite_name] = merged_members
 
        # Drop the two parent columns, register the new composite
        selected_subjects.drop(columns=[high_vif_col, most_correlated_col], inplace=True, errors="ignore")
        vif_data = vif_data.drop(columns=[high_vif_col, most_correlated_col])
        vif_data[composite_name] = selected_subjects[composite_name]
 
        base_members.pop(high_vif_col, None)
        base_members.pop(most_correlated_col, None)
        base_members[composite_name] = merged_members
 
        vif_df = vif_table(vif_data)
 
    selected_subjects = selected_subjects.copy()  # defragment
 
    # composite_dict can still contain entries for intermediate composites that were
    # absorbed into a later, larger composite (e.g. composite_1 -> [A, B] got folded
    # into composite_2 -> [A, B, C]). The composite_1 *column* is already gone from
    # selected_subjects at this point — it was dropped the moment it got merged — but
    # the dict entry lingers as a bookkeeping artifact. Prune any entry whose member
    # set is a strict subset of another entry's member set, since it no longer
    # corresponds to an actual column and is purely redundant provenance info.
    obsolete = set()
    for name_a, members_a in composite_dict.items():
        set_a = set(members_a)
        for name_b, members_b in composite_dict.items():
            if name_a == name_b:
                continue
            if set_a < set(members_b):  # strict subset
                obsolete.add(name_a)
                break
    for name in obsolete:
        composite_dict.pop(name)
 
    print(f"\nComposites created: {len(composite_dict)}")
    for name, members in composite_dict.items():
        print(f"  {name}: {members}")

    if zero_variance_cols:
        print(f"Dropping columns with zero variance: {len(zero_variance_cols)}")
        print(zero_variance_cols)
        selected_subjects.drop(columns=zero_variance_cols, inplace=True, errors="ignore")

    if composite_output is not None:
        composite_output_path = Path(output_path) / composite_output
        if not composite_output_path.exists():
            composite_output_path.mkdir(parents=True)
        selected_subjects.to_csv(Path(composite_output_path / "features_with_composites.csv"), index=False)
        composite_df = pd.DataFrame({
            "composite_name": list(composite_dict.keys()),
            "features_included": [", ".join(features) for features in composite_dict.values()]
        })
        composite_df.to_csv(Path(composite_output_path / "composite_dictionary.csv"), index=False)
    else:    
        selected_subjects.to_csv(Path(output_path / "features_with_composites.csv"), index=False)
        composite_df = pd.DataFrame({
            "composite_name": list(composite_dict.keys()),
            "features_included": [", ".join(features) for features in composite_dict.values()]
        })
        composite_df.to_csv(Path(output_path / "composite_dictionary.csv"), index=False)
    
    return selected_subjects, composite_dict

def extract_fitbit_features_2(con, subject_timepoint_pairs, output_path=None, overwrite=False):
    if output_path is not None and os.path.exists(output_path) and not overwrite:
        print(f"Fitbit features already exist at {output_path}. Skipping extraction.")
        return pd.read_csv(output_path)

    features_list = []
    for row in tqdm(subject_timepoint_pairs.itertuples(index=False), total=subject_timepoint_pairs.shape[0], desc="Extracting Fitbit features"):
        query = f"""
        SELECT *
        FROM fitbit_data
        WHERE subject = '{row.subject}' AND timepoint = '{row.timepoint}'
        """
        subject_fitbit_df = con.sql(query).fetchdf()
        fitbit_metric_cols = [col for col in subject_fitbit_df.columns if col not in ["subject", "timepoint", "Wear_Time"]]
        for col in fitbit_metric_cols:
            subject_fitbit_df[col] = pd.to_numeric(subject_fitbit_df[col], errors="coerce")
        feature_dict = {"subject": row.subject, "timepoint": row.timepoint}
        for metric in fitbit_metric_cols:
            # Check if the metric column exists in the group
            if metric in subject_fitbit_df.columns:
                daily_data = subject_fitbit_df[["Wear_Time", metric]].dropna()
                if not daily_data.empty:
                    # Create daily features (mean, std, min, max)
                    daily_data.set_index("Wear_Time", inplace=True)
                    daily_stats = daily_data.resample("D").agg(['mean', 'std', 'min', 'max'])
                    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
                    # Create datetime index with proper missing days based on the daily resampling range
                    daily_stats = daily_stats.dropna(how="all")
                    min_date = daily_stats.index.min()
                    max_date = daily_stats.index.max()
                    date_range = pd.date_range(start=min_date, end=max_date, freq="D")
                    # Reindex to include missing days and impute missing values with multiple imputation
                    daily_stats = daily_stats.reindex(date_range)
                    # Get percentage of missing values in the daily_stats DataFrame
                    missing_percentage = daily_stats.isna().mean().mean() * 100
                    if daily_stats.shape[0] > 1 and daily_stats.notna().sum().sum() > daily_stats.shape[1]:
                        imputer = SimpleImputer(strategy="median")
                        daily_stats = pd.DataFrame(
                            imputer.fit_transform(daily_stats),
                            index=daily_stats.index,
                            columns=daily_stats.columns,
                        )
                    else:
                        daily_stats = daily_stats.ffill().bfill()
                    feature_dict.update(daily_stats.mean().to_dict())
                    # STL decomposition on the imputed daily aggregate series.
                    for agg in ["mean", "std", "min", "max"]:
                        try:
                            stl_input = daily_stats[f"{metric}_{agg}"].copy()
                            stl_input.index = pd.to_datetime(stl_input.index)
                            stl_input = stl_input.sort_index().asfreq("D")
                            stl = STL(stl_input, period=7, robust=True)
                            result = stl.fit()
                            stl_features = {
                                f"{metric}_{agg}_trend_mean": result.trend.mean(),
                                f"{metric}_{agg}_trend_std": result.trend.std(),
                                f"{metric}_{agg}_trend_min": result.trend.min(),
                                f"{metric}_{agg}_trend_max": result.trend.max(),
                                f"{metric}_{agg}_seasonal_mean": result.seasonal.mean(),
                                f"{metric}_{agg}_seasonal_std": result.seasonal.std(),
                                f"{metric}_{agg}_seasonal_min": result.seasonal.min(),
                                f"{metric}_{agg}_seasonal_max": result.seasonal.max(),
                                f"{metric}_{agg}_resid_mean": result.resid.mean(),
                                f"{metric}_{agg}_resid_std": result.resid.std(),
                                f"{metric}_{agg}_resid_min": result.resid.min(),
                                f"{metric}_{agg}_resid_max": result.resid.max(),
                            }
                            feature_dict.update(stl_features)
                            feature_dict.update({f"{metric}_missing_percentage": missing_percentage})
                        except Exception as e:
                            print(f"STL decomposition failed for subject {row.subject}, metric {metric}: {e}")
        features_list.append(feature_dict)
    fitbit_features_df = pd.DataFrame(features_list)
    
    fitbit_features_df.to_csv(Path("output")/ "fitbit_features.csv", index=False)

    return fitbit_features_df
