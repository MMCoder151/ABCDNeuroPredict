from pathlib import Path
import os
import sys

from src.preprocessing import select_subjects, transform_dta

test = True # Set to False to run on full dataset

# Set data directory paths
dta_path = Path.home() / "dairc" / "rawdata"

# Set output path
output_path = os.path.join(os.getcwd(), "output")

# Select subjects based on inclusion criteria and extract metadata
dem_df, fit_meta_df, mri_meta_df = select_subjects(dta_path)

# Transform fitbit data to make it easier to query with DuckDB
transform_dta(dta_path, fit_meta_df)
