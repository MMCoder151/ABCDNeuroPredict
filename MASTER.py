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

# Filter subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = filter_subjects(dta_path, test=False, overwrite=False)

# Print descriptive statistics of filtered subjects
describe_subjects(fit_meta_df, mri_meta_df)

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df, overwrite=False)

# ---- FEATURE EXTRACTION ----

# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_subjects = normative_selection(con, mri_meta_df, overwrite=True)

# Conduct confound analysis pre and post normative modeling
confound_effects_df = analyse_confounds(con, dem_df, mri_meta_df)

# Print descriptive statistics of normative selected subjects
selected_fit_meta_df = fit_meta_df[fit_meta_df["subject"].isin(selected_subjects["subject_ids"])]
selected_mri_meta_df = mri_meta_df[mri_meta_df["subject"].isin(selected_subjects["subject_ids"])]
describe_subjects(selected_fit_meta_df, selected_mri_meta_df)

# Print descriptive statistics of non-selected subjects
non_selected_fit_meta_df = fit_meta_df[~fit_meta_df["subject"].isin(selected_subjects["subject_ids"])]
non_selected_mri_meta_df = mri_meta_df[~mri_meta_df["subject"].isin(selected_subjects["subject_ids"])]
describe_subjects(non_selected_fit_meta_df, non_selected_mri_meta_df)

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

# Print number of unique subtypes discovered
#print(f"Number of unique subtypes discovered: {len(set(subject_subtypes.values())) - (1 if -1 in subject_subtypes.values() else 0))}")  # Exclude -1 if it exists, which represents subjects not assigned to any subtype

# Add subtype labels to dem_df based on subject_subtypes
#dem_df["subtype"] = dem_df["subject"].apply(lambda x: subject_subtypes.get(x, -1))  # Assign -1 for subjects not in subject_subtypes

# Conduct missingness analysis of fitbit data
#missingness_df = missingness_analysis(con, fit_meta_df)

# ---- MODELING ----

