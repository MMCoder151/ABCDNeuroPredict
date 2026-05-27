from pathlib import Path
import os
from src.preprocessing import *
from sklearn.model_selection import train_test_split

# Set data directory paths
dta_path = Path.home() / "dairc" / "rawdata"

# Set output path
output_path = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ---- DATA WRANGLING ----

# Select subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = select_subjects(dta_path, test=False, overwrite=True)

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df, overwrite=True)

unique_fit_cols = con.execute("SELECT DISTINCT column_name FROM (DESCRIBE fitbit_data)").fetchall()

# ---- DATA ANALYSIS ----

# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_subjects = normative_selection(con, mri_meta_df, overwrite=True)

# Create composite mri z-scores based on VIF
selected_subjects_composites, composite_dict = create_mri_composites(con, selected_subjects)

# Save selected subjects to CSV
selected_subjects_composites.to_csv(os.path.join(output_path, "selected_subjects_composites.csv"), index=False)

# Conduct unsupervised clustering of selected subjects' z-scores for subtype discovery
#subject_subtypes = mri_subtyping(dem_df, selected_subjects)

# Conduct missingness analysis of fitbit data
#missingness_df = missingness_analysis(con, fit_meta_df)

# ---- FEATURE EXTRACTION ----

# Add group labels to dem_df based on selected_subjects
dem_df["group"] = dem_df["subject"].apply(lambda x: 1 if x in selected_subjects["subject_ids"].values else 0)

# Train-test split of subjects in dem_df
train_df, test_df = train_test_split(dem_df["subject"], test_size=0.2, stratify=dem_df["group"], random_state=42)

# Extract features from selected subjects fitbit data for train set
train_features = extr_fitbit_features(con, train_df)

# Extract features from selected subjects fitbit data for test set
test_features = extr_fitbit_features(con, test_df)