from pathlib import Path
import os
from src.preprocessing import *
from src.feature_extraction import *
from sklearn.model_selection import train_test_split

# Set data directory paths
dta_path = Path.home() / "dairc" / "rawdata"

# Set output path
output_path = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ---- DATA WRANGLING ----

# Select subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = filter_subjects(dta_path, test=False, overwrite=True)

# Print demographics of selected subjects with both fitbit and mri data
print("Demographics of filtered subjects with both fitbit and mri files:")
print(f"N: {mri_meta_df['subject'].nunique()}")
print("\nMRI Age Statistics:")
print(f"Mean: {mri_meta_df['age_at_mri'].mean()}, Std: {mri_meta_df['age_at_mri'].std()}, Min: {mri_meta_df['age_at_mri'].min()}, Max: {mri_meta_df['age_at_mri'].max()}")
print("\nSex Distribution:")
print(dem_df["sex"].value_counts())

# Print demographics of subjects without short recordings ("short" == 0)
non_short_pairs = fit_meta_df[fit_meta_df["short"] == 0][["subject", "timepoint"]]
non_short_subjects = non_short_pairs["subject"].unique()
common_mri_non_short_df = mri_meta_df[
    mri_meta_df[["subject", "timepoint"]].apply(tuple, axis=1).isin(non_short_pairs.apply(tuple, axis=1))
]

print("\nDemographics of subjects without short recordings:")
print(f"N: {common_mri_non_short_df['subject'].nunique()}")
print("\nMRI Age Statistics:")
print(f"Mean: {common_mri_non_short_df['age_at_mri'].mean()}, Std: {common_mri_non_short_df['age_at_mri'].std()}, Min: {common_mri_non_short_df['age_at_mri'].min()}, Max: {common_mri_non_short_df['age_at_mri'].max()}")
print("\nSex Distribution:")
print(dem_df[dem_df["subject"].isin(non_short_subjects)]["sex"].value_counts())

# Print demographics of subjects with short recordings ("short" == 1)
short_pairs = fit_meta_df[fit_meta_df["short"] == 1][["subject", "timepoint"]]
short_subjects = short_pairs["subject"].unique()
common_mri_short_df = mri_meta_df[
    mri_meta_df[["subject", "timepoint"]].apply(tuple, axis=1).isin(short_pairs.apply(tuple, axis=1))
] 
print("\nDemographics of subjects with short recordings:")
print(f"N: {common_mri_short_df['subject'].nunique()}")
print("\nMRI Age Statistics:")
print(f"Mean: {common_mri_short_df['age_at_mri'].mean()}, Std: {common_mri_short_df['age_at_mri'].std()}, Min: {common_mri_short_df['age_at_mri'].min()}, Max: {common_mri_short_df['age_at_mri'].max()}")
print("\nSex Distribution:")
print(dem_df[dem_df["subject"].isin(short_subjects)]["sex"].value_counts())

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df, overwrite=False)

# ---- FEATURE EXTRACTION ----

# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_subjects = normative_selection(con, mri_meta_df, overwrite=False)

# Print demographics of normative selected subjects
print("Selected Subjects MRI Age Statistics:")
print(f"Mean: {mri_meta_df[mri_meta_df['subject'].isin(selected_subjects['subject_ids'])]['age_at_mri'].mean()}, Std: {mri_meta_df[mri_meta_df['subject'].isin(selected_subjects['subject_ids'])]['age_at_mri'].std()}, Min: {mri_meta_df[mri_meta_df['subject'].isin(selected_subjects['subject_ids'])]['age_at_mri'].min()}, Max: {mri_meta_df[mri_meta_df['subject'].isin(selected_subjects['subject_ids'])]['age_at_mri'].max()}")
print("\nSelected Subjects Sex Distribution:")
print(dem_df[dem_df["subject"].isin(selected_subjects["subject_ids"])]["sex"].value_counts())

# Print missing statistics of fitbit data for selected subjects
selected_fit_meta_df = fit_meta_df[fit_meta_df["subject"].isin(selected_subjects["subject_ids"])]
print("\nFitbit Data Missingness Statistics for Normative Selected Subjects:")
print(f"Mean missingness: {selected_fit_meta_df['missingness'].mean()}, Std: {selected_fit_meta_df['missingness'].std()}, Min: {selected_fit_meta_df['missingness'].min()}, Max: {selected_fit_meta_df['missingness'].max()}")
print(f"Number of subjects with short recordings (short == 1): {selected_fit_meta_df[selected_fit_meta_df['short'] == 1]['subject'].nunique()} ({selected_fit_meta_df[selected_fit_meta_df['short'] == 1]['subject'].nunique() / len(selected_fit_meta_df) * 100:.2f}% )")

# Print missingness statistics of fitbit data for non-selected subjects
non_selected_fit_meta_df = fit_meta_df[~fit_meta_df["subject"].isin(selected_subjects["subject_ids"])]
print("\nFitbit Data Missingness Statistics for Normative Non-Selected Subjects:")
print(f"Mean missingness: {non_selected_fit_meta_df['missingness'].mean()}, Std: {non_selected_fit_meta_df['missingness'].std()}, Min: {non_selected_fit_meta_df['missingness'].min()}, Max: {non_selected_fit_meta_df['missingness'].max()}")
print(f"Number of subjects with short recordings (short == 1): {non_selected_fit_meta_df[non_selected_fit_meta_df['short'] == 1]['subject'].nunique()} ({non_selected_fit_meta_df[non_selected_fit_meta_df['short'] == 1]['subject'].nunique() / len(non_selected_fit_meta_df) * 100:.2f}% )")

# Conduct confound analysis pre and post normative modeling
confound_effects_df = analyse_confounds(con, dem_df, mri_meta_df)

# Create composite mri z-scores based on VIF
selected_subjects_composites, composite_dict = create_mri_composites(con, selected_subjects)

# Save selected subjects to CSV
selected_subjects_composites.to_csv(os.path.join(output_path, "selected_subjects_composites.csv"), index=False)

# Add group labels to dem_df based on selected_subjects
dem_df["group"] = dem_df["subject"].apply(lambda x: 1 if x in selected_subjects["subject_ids"].values else 0)

# Train-test split of subjects in dem_df
train_df, test_df = train_test_split(dem_df["subject"], test_size=0.2, stratify=dem_df["group"], random_state=42)

# Extract features from selected subjects fitbit data for train set
train_features = extr_fitbit_features(con, train_df)
train_y = dem_df[dem_df["subject"].isin(train_df)]["group"]

# Extract features from selected subjects fitbit data for test set
test_features = extr_fitbit_features(con, test_df)
test_y = dem_df[dem_df["subject"].isin(test_df)]["group"]

# Save features to CSV
train_features.to_csv(os.path.join(output_path, "train_features.csv"), index=False)
test_features.to_csv(os.path.join(output_path, "test_features.csv"), index=False)
train_y.to_csv(os.path.join(output_path, "train_labels.csv"), index=False)
test_y.to_csv(os.path.join(output_path, "test_labels.csv"), index=False)

# ---- DATA ANALYSIS ----

# Conduct unsupervised clustering of selected subjects' z-scores for subtype discovery
#subject_subtypes = mri_clustering(dem_df, selected_subjects)

# Conduct missingness analysis of fitbit data
#missingness_df = missingness_analysis(con, fit_meta_df)

# ---- MODELING ----

