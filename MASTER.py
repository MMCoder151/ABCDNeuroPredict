from pathlib import Path
import os
from src.preprocessing import select_subjects, setup_duckdb, normative_selection

test = True # Set to False to run on full dataset

# Set data directory paths
dta_path = Path.home() / "dairc" / "rawdata"

# Set output path
output_path = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Select subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = select_subjects(dta_path, test=test)

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df)

# Select subjects based on normative modeling and composite z-scores
#selected_subjects = normative_selection(con, dem_df)
