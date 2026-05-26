import pandas as pd
from pathlib import Path
import os
from src.preprocessing import select_subjects, setup_duckdb, normative_selection, create_mri_composites, extr_fitbit_features

# Set data directory paths
dta_path = Path.home() / "dairc" / "rawdata"

# Set output path
output_path = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Select subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = select_subjects(dta_path, test=False, overwrite=True)

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df, overwrite=True)

# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_subjects = normative_selection(con, mri_meta_df, output_path)

# Create composite mri z-scores based on VIF
selected_subjects, composite_dict = create_mri_composites(con, selected_subjects)

# Save selected subjects to CSV
selected_subjects.to_csv(os.path.join(output_path, "selected_subjects.csv"), index=False)

# Extract features from selected subjects fitbit data
fitbit_features_df = extr_fitbit_features(con, selected_subjects)