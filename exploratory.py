import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.modelling import *
from src.data_analysis import *
from src.preprocessing import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pacmap import PaCMAP
from sklearn.ensemble import IsolationForest
from neuroCombat import neuroCombat
import seaborn as sns
from src.mri_rois import *
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score
import shap
import statsmodels.api as sm
from pcntoolkit import NormativeModel, BLR
from pcntoolkit.dataio.norm_data import NormData
from imblearn.over_sampling import RandomOverSampler

# SETUP WORKSPACE
# Set raw data directory 
dta_path = Path.home() / "dairc" / "rawdata"

# Set tabular data directory 
dta_path_tabular = Path.home() / "dairc" / "abcd" / "rawdata" / "phenotype"

# Set output path
output_path = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Set baseline output directory
baseline_output_path = os.path.join(output_path, "baseline_checks")
os.makedirs(baseline_output_path, exist_ok=True)

# Set output directories for normative modeling
normative_output_path = os.path.join(baseline_output_path, "normative_modeling")
os.makedirs(normative_output_path, exist_ok=True)
normative_output_dir_str_y = os.path.join(normative_output_path, "normative_model_y")
normative_output_dir_str_p = os.path.join(normative_output_path, "normative_model_p")

# IMPORT AND FILTER DATA
# Find all subjects with MRI data and get their depression diagnosis labels
# Read in mri files from src/mri_rois.py
mri_path = dta_path / "phenotype"
mri_files, _ = mri_rois()
mri_data = pd.DataFrame()

dfs = []
for file in mri_files:
    file_path = os.path.join(mri_path, file)
    if not os.path.exists(file_path):
        print(f"Warning: MRI data file {file} not found in {mri_path}. Please check the file path.")
        continue  
    df = pd.read_csv(file_path, sep="\t")
    dfs.append(df)

mri_data = reduce(
    lambda left, right: pd.merge(left, right, on=["participant_id", "session_id"], how="outer"),
    dfs
)

mri_data.shape

# Get number of unique subject timepoint pairs in mri_data
unique_subject_timepoints = mri_data[["participant_id", "session_id"]].drop_duplicates()
num_unique_subject_timepoints = unique_subject_timepoints.shape[0]
print(f"Number of unique subject timepoint pairs with MRI data: {num_unique_subject_timepoints}")

# Read in KSADS questionnaires for parent and youth
youth_directory = dta_path_tabular / "mh_y_ksads__dep.tsv"
parent_directory = dta_path_tabular / "mh_p_ksads__dep.tsv"

ksads_youth = pd.read_csv(youth_directory, sep="\t")
ksads_parent = pd.read_csv(parent_directory, sep="\t")

ksads_mdd_youth = ksads_youth[["participant_id", "session_id", "mh_y_ksads__dep__mdd__pres_dx"]]
ksads_mdd_parent = ksads_parent[["participant_id", "session_id", "mh_p_ksads__dep__mdd__pres_dx"]]

# Filter to only include subjects and timepoint pairs with MRI data
mri_pairs = mri_data[["participant_id", "session_id"]].drop_duplicates()
ksads_mdd_youth_filtered = ksads_mdd_youth.merge(mri_pairs, on=["participant_id", "session_id"], how="inner")
ksads_mdd_parent_filtered = ksads_mdd_parent.merge(mri_pairs, on=["participant_id", "session_id"], how="inner")

# Combine youth and parent KSADS questionnaires into a single dataframe
ksads_mdd_combined = ksads_mdd_youth_filtered.merge(ksads_mdd_parent_filtered, on=["participant_id", "session_id"], how="outer")
ksads_mdd_combined.shape

# Filter to only include subjects and timepoint pairs that exist in both youth and parent KSADS questionnaires
ksads_mdd_combined_both = ksads_mdd_combined.dropna(subset=["mh_y_ksads__dep__mdd__pres_dx", "mh_p_ksads__dep__mdd__pres_dx"])

# Print number of unique subject timepoint pairs with both youth and parent KSADS questionnaires
num_unique_subject_timepoints_both = ksads_mdd_combined_both[["participant_id", "session_id"]].drop_duplicates().shape[0]
print(f"Number of unique subject timepoint pairs with both youth and parent KSADS questionnaires: {num_unique_subject_timepoints_both}")

# For subjects with multiple timepoints, keep only the one with depression diagnosis in either parent or youth KSADS or the first timepoint if no diagnosis is present
for subject in ksads_mdd_combined_both.groupby("participant_id")["participant_id"]:
    subject_id = subject[0]
    subject_data = ksads_mdd_combined_both[ksads_mdd_combined_both["participant_id"] == subject_id]
    subject_data = subject_data.sort_values(by="session_id")
    if subject_data.shape[0] > 1:
        if (subject_data["mh_y_ksads__dep__mdd__pres_dx"] == 1).any() or (subject_data["mh_p_ksads__dep__mdd__pres_dx"] == 1).any():
            ksads_mdd_combined_both = ksads_mdd_combined_both.drop(subject_data.index)
            ksads_mdd_combined_both = pd.concat([ksads_mdd_combined_both, subject_data[(subject_data["mh_y_ksads__dep__mdd__pres_dx"] == 1) | (subject_data["mh_p_ksads__dep__mdd__pres_dx"] == 1)]])
        else:
            ksads_mdd_combined_both = ksads_mdd_combined_both.drop(subject_data.index)
            ksads_mdd_combined_both = pd.concat([ksads_mdd_combined_both, subject_data.iloc[[0]]])

# Keep only the first timepoint for each subject in ksads_mdd_combined_both
ksads_mdd_combined_both = ksads_mdd_combined_both.sort_values(by=["participant_id", "session_id"]).drop_duplicates(subset=["participant_id"], keep="first")

ksads_mdd_combined_both.shape

# Print number of depressed subjects according to youth and parent KSADS questionnaires
num_depressed_youth = ksads_mdd_combined_both["mh_y_ksads__dep__mdd__pres_dx"].sum()
num_depressed_parent = ksads_mdd_combined_both["mh_p_ksads__dep__mdd__pres_dx"].sum()

print(f"Number of depressed (youth): {num_depressed_youth}")
print(f"Number of depressed (parent): {num_depressed_parent}")

# Print number of subjects with depression according to both youth and parent KSADS questionnaires
num_depressed_both = ksads_mdd_combined_both[(ksads_mdd_combined_both["mh_y_ksads__dep__mdd__pres_dx"] == 1) & (ksads_mdd_combined_both["mh_p_ksads__dep__mdd__pres_dx"] == 1)].shape[0]
print(f"Number of depressed (both youth and parent): {num_depressed_both}")

# Filter mri_data to only include subjects and timepoint pairs that also exist in combined KSADS questionnaires
mri_data_filtered = mri_data.merge(ksads_mdd_combined_both[["participant_id", "session_id"]], on=["participant_id", "session_id"], how="inner")

# Add depression diagnosis labels from youth and parent KSADS questionnaires to mri_data_filtered
mri_data_filtered = mri_data_filtered.merge(ksads_mdd_combined_both[["participant_id", "session_id", "mh_y_ksads__dep__mdd__pres_dx"]], on=["participant_id", "session_id"], how="left")
mri_data_filtered = mri_data_filtered.merge(ksads_mdd_combined_both[["participant_id", "session_id", "mh_p_ksads__dep__mdd__pres_dx"]], on=["participant_id", "session_id"], how="left")

# Get number of subjects per unique session_id
num_subjects_per_session = mri_data_filtered.groupby("session_id").size()
print("Number of subjects per unique session_id:")
print(num_subjects_per_session)

# Get confound variables for mri_data_filtered
# Get list of included subject timepoint pairs
included_subject_timepoints = mri_data_filtered[["participant_id", "session_id"]].drop_duplicates()
mri_path = dta_path / "phenotype"

# Read in participant information
subs = pd.read_csv(dta_path / "participants.tsv", sep="\t")

# Import subcortical volume data
subcortical_vol = pd.read_csv(dta_path / "phenotype" / "mr_y_smri__vol__aseg.tsv", sep="\t")

# import dynamic information
stc_dyn_df = pd.read_csv(mri_path / "ab_g_dyn.tsv", sep="\t")

# create dataframe with sex, date of birth, and scan site for included subject timepoint pairs
confound_df = included_subject_timepoints.merge(stc_dyn_df[["participant_id", "session_id", "ab_g_dyn__design_site", "ab_g_dyn__visit_age"]], on=["participant_id", "session_id"], how="left")
confound_df = confound_df.merge(subs[["participant_id", "sex"]], on="participant_id", how="left")
confound_df = confound_df.merge(subcortical_vol[["participant_id", "session_id", "mr_y_smri__vol__aseg__icv_sum"]], on=["participant_id", "session_id"], how="left")
confound_df.rename(columns={"ab_g_dyn__design_site": "scan_site", "ab_g_dyn__visit_age": "visit_age"}, inplace=True)
confound_df["sex"] = confound_df["sex"].map({"M": 1, "F": 0})
confound_df["age_squared"] = confound_df["visit_age"] ** 2
confound_df.columns

# Merge confound variables into mri_data_filtered
mri_data_filtered = mri_data_filtered.merge(confound_df, on=["participant_id", "session_id"], how="left")

# Save final feature space to CSV
mri_data_filtered.to_csv(os.path.join(baseline_output_path, "mri_data_filtered.csv"), index=False)

# Optional: Reimport mri_data_filtered
mri_data_filtered = pd.read_csv(os.path.join(baseline_output_path, "mri_data_filtered.csv"))

# Conduct outlier detection using Isolation Forest on the MRI data of only healthy subjects (no depression diagnosis according to both KSADS questionnaire)
healthy_subjects = mri_data_filtered[(mri_data_filtered["mh_y_ksads__dep__mdd__pres_dx"] == 0) & (mri_data_filtered["mh_p_ksads__dep__mdd__pres_dx"] == 0)]
# Select only numeric columns for outlier detection
numeric_cols = mri_data_filtered.select_dtypes(include=["float64", "int64"]).columns
numeric_data = healthy_subjects[numeric_cols]
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
# Fit the model and predict outliers
outlier_preds = isolation_forest.fit_predict(numeric_data)
# Add outlier predictions to the dataframe
healthy_subjects["outlier"] = outlier_preds
print(f"Number of outliers detected: {(outlier_preds == -1).sum()}")
# Filter out the outliers
outliers = healthy_subjects[healthy_subjects["outlier"] == -1]["participant_id"].tolist()
mri_data_filtered = mri_data_filtered[~mri_data_filtered["participant_id"].isin(outliers)]

# Impute missing values in mri_data_filtered using the median
imputer = SimpleImputer(strategy="median")
numeric_cols = mri_data_filtered.select_dtypes(include=["float64", "int64"]).columns
# Assign the raw numpy array directly — purely positional, no index alignment risk
mri_data_filtered[numeric_cols] = imputer.fit_transform(mri_data_filtered[numeric_cols])

# Conduct site harmonization using ComBat for all MRI features in mri_data_filtered
# Prepare data for ComBat 
feature_cols = mri_data.columns.difference(["participant_id", "session_id"])
mri_data_pre_combat = mri_data_filtered.copy()
mri_data_combat = mri_data_filtered[feature_cols].transpose()  # Transpose to have features as rows and subjects as columns

# Define covariates for ComBat
covars = pd.DataFrame({
    "scan_site": mri_data_filtered["scan_site"],
    "sex": mri_data_filtered["sex"],
    "visit_age": mri_data_filtered["visit_age"],
    "age_squared": mri_data_filtered["visit_age"] ** 2,
    "mr_y_smri__vol__aseg__icv_sum": mri_data_filtered["mr_y_smri__vol__aseg__icv_sum"]
})

categorical_cols = ["sex"]
batch_cols = ["scan_site"]

# Harmonize
mri_data_combat = neuroCombat(
                    dat=mri_data_combat, 
                    covars=covars, 
                    batch_col=batch_cols, 
                    categorical_cols=categorical_cols
                    )

# Transpose back to original shape
mri_data_filtered[feature_cols] = mri_data_combat["data"].transpose()

# Conduct confound analysis pre and post harmonization
base_terms = [
    "bs(visit_age, df=4)",
    "C(sex)",
    "C(scan_site)",
    "mr_y_smri__vol__aseg__icv_sum"
]
confounds = {
    "Age": "bs(visit_age, df=4)",
    "Sex": "C(sex)",
    "Site": "C(scan_site)",
    "TIV": "mr_y_smri__vol__aseg__icv_sum"
}
summary_combat, wilcoxon_results_combat = confound_analysis(mri_data_pre_combat, mri_data_filtered, feature_cols, base_terms, confounds)
summary_combat.to_csv(os.path.join(baseline_output_path, "confound_analysis_combat.csv"), index=False)
wilcoxon_results_combat.to_csv(os.path.join(baseline_output_path, "wilcoxon_results_combat.csv"), index=False)

# Do group difference analysis
mri_dep_y_sig, mri_dep_y_all = exploratory_group_difference_analysis(mri_data_filtered, "mh_y_ksads__dep__mdd__pres_dx", 
                                                                     os.path.join(baseline_output_path, "mri_dep_y_results.csv"), 
                                                                     output_path_sig=os.path.join(baseline_output_path, "mri_dep_y_results_sig.csv"),
                                                                     overwrite=False)
mri_dep_p_sig, mri_dep_p_all = exploratory_group_difference_analysis(mri_data_filtered, "mh_p_ksads__dep__mdd__pres_dx", 
                                                                     os.path.join(baseline_output_path, "mri_dep_p_results.csv"), 
                                                                     output_path_sig=os.path.join(baseline_output_path, "mri_dep_p_results_sig.csv"),
                                                                     overwrite=False)

# Filter significant ROIs to only include those with an absolute effect size >0.2
mri_dep_y_sig_filtered = mri_dep_y_sig[abs(mri_dep_y_sig["effect_size"]) > 0.2]
mri_dep_p_sig_filtered = mri_dep_p_sig[abs(mri_dep_p_sig["effect_size"]) > 0.2]
print(f"Number of significant ROIs after filtering by effect size (youth): {mri_dep_y_sig_filtered.shape[0]}")
print(f"Number of significant ROIs after filtering by effect size (parent): {mri_dep_p_sig_filtered.shape[0]}")

# Print effect sizes and standard deviations for filtered significant ROIs
print("Effect sizes and standard deviations for filtered significant ROIs (youth):")
print(mri_dep_y_sig_filtered[["feature", "effect_size", "effect_size_std"]])
print("Effect sizes and standard deviations for filtered significant ROIs (parent):")
print(mri_dep_p_sig_filtered[["feature", "effect_size", "effect_size_std"]])

# Get the list of filtered significant ROIs for both youth and parent
mri_dep_y_sig_filtered_rois = mri_dep_y_sig_filtered["feature"].tolist()
mri_dep_p_sig_filtered_rois = mri_dep_p_sig_filtered["feature"].tolist()
mri_dep_sig_filtered_overlap = list(set(mri_dep_y_sig_filtered_rois) & set(mri_dep_p_sig_filtered_rois))
print(f"Number of overlapping significant ROIs between youth and parent: {len(mri_dep_sig_filtered_overlap)}")
print(f"Overlapping significant ROIs between youth and parent: {mri_dep_sig_filtered_overlap}")

# Get difference in effect sizes for overlapping significant ROIs between youth and parent
mri_dep_sig_filtered_overlap_df = pd.merge(mri_dep_y_sig_filtered[["feature", "effect_size", "effect_size_std"]], mri_dep_p_sig_filtered[["feature", "effect_size", "effect_size_std"]], on="feature", suffixes=("_youth", "_parent"))
mri_dep_sig_filtered_overlap_df["effect_size_diff"] = mri_dep_sig_filtered_overlap_df["effect_size_youth"] - mri_dep_sig_filtered_overlap_df["effect_size_parent"]

# Print age and sex distribution of all subjects
print("Age and sex distribution of all subjects:")
sex_distribution = mri_data_filtered["sex"].value_counts()
print(sex_distribution)
age_distribution = mri_data_filtered["visit_age"].describe()
print(age_distribution)
age_sex_distribution = mri_data_filtered.groupby("sex")["visit_age"].describe()
print(age_sex_distribution)

# Compute correlation matrix for significant ROIs for youth and parent
mri_dep_y_sig_corr = mri_data_filtered[mri_dep_y_sig_filtered_rois].corr()
mri_dep_p_sig_corr = mri_data_filtered[mri_dep_p_sig_filtered_rois].corr()
mri_dep_y_sig_corr.to_csv(os.path.join(baseline_output_path, "mri_dep_y_sig_corr.csv"))
mri_dep_p_sig_corr.to_csv(os.path.join(baseline_output_path, "mri_dep_p_sig_corr.csv"))

# Create composite scores for youth and parent significant ROIs
mri_dep_y_sig_composite, mri_dep_y_sig_composite_dict = create_composites(mri_data_filtered[mri_dep_y_sig_filtered_rois], overwrite=True, composite_output=os.path.join(baseline_output_path, "mri_dep_y_sig_composite"))
mri_dep_y_sig_composite["participant_id"] = mri_data_filtered["participant_id"]
mri_dep_p_sig_composite, mri_dep_p_sig_composite_dict = create_composites(mri_data_filtered[mri_dep_p_sig_filtered_rois], overwrite=True, composite_output=os.path.join(baseline_output_path, "mri_dep_p_sig_composite"))
mri_dep_p_sig_composite["participant_id"] = mri_data_filtered["participant_id"]

# Create scatter plot of data in 2D PCA space for parent significant ROIs and mark depressed subjects
pca_p = PCA(n_components=2, random_state=42)
mri_dep_p_sig_pca = pca_p.fit_transform(mri_dep_p_sig_composite.drop(columns=["participant_id"]))
mri_dep_p_sig_pca_df = pd.DataFrame(mri_dep_p_sig_pca, columns=["PCA_1", "PCA_2"])
mri_dep_p_sig_pca_df["depressed_parent"] = mri_data_filtered["mh_p_ksads__dep__mdd__pres_dx"]
plt.figure(figsize=(10, 8))
sns.scatterplot(data=mri_dep_p_sig_pca_df, x="PCA_1", y="PCA_2", hue="depressed_parent", palette={0: "blue", 1: "red"}, alpha=0.7)
plt.title("PCA of Parent Significant ROIs (Depressed vs Non-Depressed)")
plt.savefig(os.path.join(baseline_output_path, "mri_dep_p_sig_pca.png"))
plt.close()

# NORMATIVE MODELING
# Prepare data for normative modeling
# Filter to only include healthy subjects (no depression diagnosis according to youth KSADS questionnaire) for normative modeling
healthy_subjects_y = mri_dep_y_sig_composite[mri_data_filtered["mh_y_ksads__dep__mdd__pres_dx"] == 0]
healthy_subjects_p = mri_dep_p_sig_composite[mri_data_filtered["mh_p_ksads__dep__mdd__pres_dx"] == 0]

# Define response variables (significant ROIs) for youth and parent
roi_cols_y = mri_dep_y_sig_composite.drop(columns=["participant_id"]).columns.tolist()
roi_cols_p = mri_dep_p_sig_composite.drop(columns=["participant_id"]).columns.tolist()

# Add confound variables to the healthy subjects dataframes for normative modeling
confound_vars = ["visit_age", "sex", "scan_site", "mr_y_smri__vol__aseg__icv_sum"]
healthy_subjects_y = healthy_subjects_y.merge(mri_data_filtered[["participant_id"] + confound_vars], on="participant_id", how="inner")
healthy_subjects_p = healthy_subjects_p.merge(mri_data_filtered[["participant_id"] + confound_vars], on="participant_id", how="inner")
# calculate age squared for healthy subjects
healthy_subjects_y["age_squared"] = healthy_subjects_y["visit_age"] ** 2
healthy_subjects_p["age_squared"] = healthy_subjects_p["visit_age"] ** 2

# Add confound variables to the full dataset for normative modeling
youth_normative_data = mri_dep_y_sig_composite.merge(mri_data_filtered[["participant_id"] + confound_vars], on="participant_id", how="inner")
parent_normative_data = mri_dep_p_sig_composite.merge(mri_data_filtered[["participant_id"] + confound_vars], on="participant_id", how="inner")
# calculate age squared for full dataset
youth_normative_data["age_squared"] = youth_normative_data["visit_age"] ** 2
parent_normative_data["age_squared"] = parent_normative_data["visit_age"] ** 2

data_reference_y = NormData.from_dataframe(
        name="mri_norm_reference_y",
        dataframe=healthy_subjects_y,
        covariates=["visit_age", "age_squared", "mr_y_smri__vol__aseg__icv_sum"],
        batch_effects=["scan_site", "sex"],
        response_vars=roi_cols_y,
        subject_ids="participant_id",
        remove_Nan=True,
    )

data_reference_p = NormData.from_dataframe(
        name="mri_norm_reference_p",
        dataframe=healthy_subjects_p,
        covariates=["visit_age", "age_squared", "mr_y_smri__vol__aseg__icv_sum"],
        batch_effects=["scan_site", "sex"],
        response_vars=roi_cols_p,
        subject_ids="participant_id",
        remove_Nan=True,
    )

data_full_y = NormData.from_dataframe(
        name="mri_norm_full_y",
        dataframe=youth_normative_data,
        covariates=["visit_age", "age_squared", "mr_y_smri__vol__aseg__icv_sum"],
        batch_effects=["scan_site", "sex"],
        response_vars=roi_cols_y,
        subject_ids="participant_id",
        remove_Nan=True,
    )

data_full_p = NormData.from_dataframe(
        name="mri_norm_full_p",
        dataframe=parent_normative_data,
        covariates=["visit_age", "age_squared", "mr_y_smri__vol__aseg__icv_sum"],
        batch_effects=["scan_site", "sex"],
        response_vars=roi_cols_p,
        subject_ids="participant_id",
        remove_Nan=True,
    )

# Setup normative model
model_y = NormativeModel(
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
        save_dir=normative_output_dir_str_y,
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize"
)

model_p = NormativeModel(
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
        save_dir=normative_output_dir_str_p,
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize"
)

model_y.fit(data_reference_y)
model_y.predict(data_full_y)

model_p.fit(data_reference_p)
model_p.predict(data_full_p)

# Read in z-scores for the full dataset for youth and parent
z_scores_y = pd.read_csv(os.path.join(normative_output_dir_str_y, "results", "Z_mri_norm_full_y.csv"))
z_scores_p = pd.read_csv(os.path.join(normative_output_dir_str_p, "results", "Z_mri_norm_full_p.csv"))
# Note: IF normative modeling is rerun with less subjects, pcntoolkit only overwrites what is there. The file is not created anew!!!
z_scores_y.dropna(subset=["subject_ids"], inplace=True)
z_scores_p.dropna(subset=["subject_ids"], inplace=True)
z_scores_y.shape, z_scores_p.shape
mri_data_filtered.shape

# z-score normalization of significant ROIs for youth and parent
scaler_y = StandardScaler()
mri_y_scaled = pd.DataFrame(
    scaler_y.fit_transform(mri_dep_y_sig_composite[roi_cols_y]),
    columns=roi_cols_y
)
mri_y_scaled["participant_id"] = mri_data_filtered["participant_id"]

scaler_p = StandardScaler()
mri_p_scaled = pd.DataFrame(
    scaler_p.fit_transform(mri_dep_p_sig_composite[roi_cols_p]),
    columns=roi_cols_p
)
mri_p_scaled["participant_id"] = mri_data_filtered["participant_id"]

# Add confound variables to the z-scores for youth and parent for confound analysis
covar_cols = confound_df.columns.difference(["participant_id", "session_id"]).tolist()

covars = mri_data_filtered[["participant_id"] + covar_cols]

mri_y_scaled = mri_y_scaled.merge(covars, on="participant_id", how="inner")
mri_p_scaled = mri_p_scaled.merge(covars, on="participant_id", how="inner")

z_scores_y_confounds = (z_scores_y.drop(columns=["observations"]).merge(covars, left_on="subject_ids", right_on="participant_id", how="inner"))
z_scores_p_confounds = (z_scores_p.drop(columns=["observations"]).merge(covars, left_on="subject_ids", right_on="participant_id", how="inner"))

# Conduct confound analysis on z-scores for youth and parent
base_terms = [
    "bs(visit_age, df=4)",
    "C(sex)",
    "C(scan_site)",
    "mr_y_smri__vol__aseg__icv_sum"
]
confounds = {
    "Age": "bs(visit_age, df=4)",
    "Sex": "C(sex)",
    "Site": "C(scan_site)",
    "TIV": "mr_y_smri__vol__aseg__icv_sum"
}
summary_normative_y, wilcoxon_normative_y = confound_analysis(mri_y_scaled, z_scores_y_confounds, roi_cols_y, base_terms, confounds)
summary_normative_y.to_csv(os.path.join(baseline_output_path, "confound_analysis_normative_y.csv"), index=False)
wilcoxon_normative_y.to_csv(os.path.join(baseline_output_path, "wilcoxon_test_normative_y.csv"), index=False)

summary_normative_p, wilcoxon_normative_p = confound_analysis(mri_p_scaled, z_scores_p_confounds, roi_cols_p, base_terms, confounds)
summary_normative_p.to_csv(os.path.join(baseline_output_path, "confound_analysis_normative_p.csv"), index=False)
wilcoxon_normative_p.to_csv(os.path.join(baseline_output_path, "wilcoxon_test_normative_p.csv"), index=False)

# Create a z-score composite across ROIs for youth and parent
# Flip the sign of the z-scores for ROIs with negative effect sizes to ensure that higher z-scores indicate greater deviation from the normative model in the direction of depression
for roi in roi_cols_y:
    if roi in mri_dep_y_sig["feature"].values:
        effect_size = mri_dep_y_sig.loc[mri_dep_y_sig["feature"] == roi, "effect_size"].values[0]
        if effect_size < 0:
            z_scores_y[roi] = -z_scores_y[roi]

for roi in roi_cols_p:
    if roi in mri_dep_p_sig["feature"].values:
        effect_size = mri_dep_p_sig.loc[mri_dep_p_sig["feature"] == roi, "effect_size"].values[0]
        if effect_size < 0:
            z_scores_p[roi] = -z_scores_p[roi]
        
# Apply effect size weights
def _apply_effect_size_weights(whitened_df, sig_df, composite_dict=None):
    weighted_df = whitened_df.copy()
    for col in weighted_df.columns:
        if col in sig_df["feature"].values:
            effect_size = sig_df.loc[sig_df["feature"] == col, "effect_size"].values[0]
            weight = 1 + abs(effect_size)
        # If the column is part of a composite, apply the average of the absolute effect sizes of the ROIs in the composite as the weight
        elif composite_dict is not None and col in composite_dict.keys():
            composite_rois = composite_dict[col]
            effect_sizes = sig_df.loc[sig_df["feature"].isin(composite_rois), "effect_size"].values
            weight = 1 + np.mean(np.abs(effect_sizes))
        else:
            print(f"Warning: Column {col} not found in features list. Skipping weighing for this column.")
            continue

        weighted_df[col] *= weight

    return weighted_df

# Apply weighing to the normalised mri data based on the absolute effect sizes of the significant ROIs by multiplying each ROI with its absolute effect size
z_scores_y_weighted = _apply_effect_size_weights(z_scores_y, mri_dep_y_sig)
# Reattach subject labels to wighted mri data
z_scores_y_weighted["subject_ids"] = z_scores_y["subject_ids"]

z_scores_p_weighted = _apply_effect_size_weights(z_scores_p, mri_dep_p_sig, mri_dep_p_sig_composite_dict)
# Reattach subject labels to wighted mri data
z_scores_p_weighted["subject_ids"] = z_scores_p["subject_ids"]

# Create composites based on covariance to control for correlation
# Create composites of significant ROIs for youth and parent
z_scores_y_composites, z_scores_y_composite_dict = create_composites(z_scores_y_weighted.drop(columns = ["subject_ids", "observations"]), overwrite = True, composite_output = os.path.join(baseline_output_path, "mri_dep_y_sig_composites.csv"))
z_scores_y_composites["subject_ids"] = z_scores_y_weighted["subject_ids"]

z_scores_p_composites, z_scores_p_composite_dict = create_composites(z_scores_p_weighted.drop(columns = ["subject_ids", "observations"]), overwrite = True, composite_output = os.path.join(baseline_output_path, "mri_dep_p_sig_composites.csv"))
z_scores_p_composites["subject_ids"] = z_scores_p_weighted["subject_ids"]

# Create a z-score composite across ROIs for youth and parent
z_scores_y_composites["z_score_composite"] = z_scores_y_composites.drop(columns=["subject_ids"]).mean(axis=1)
z_scores_p_composites["z_score_composite"] = z_scores_p_composites.drop(columns=["subject_ids"]).mean(axis=1)

# Plot z-score composite distribution for youth and parent and mark depressed subjects according to youth and parent KSADS questionnaire
# TODO: REDO PLOTTING AS VIOLIN PLOT DOESNT MAKE SENSE WITH Z-SCORES AS THE HEALTHY POPULATION IS NORMALZED
# INSTEAD DO LINE PLOT WITH REFERENCE LINE FOR HEALTHY POPULATION SORTED ASCENDING

# Check association between z-score composite and depression symptom load for youth and parent
# TODO: Implement

# EXTRACT FITBIT FEATURES
# Filter subjects based on inclusion criteria and extract metadata
fit_meta_df = filter_subjects(dta_path, test=False, overwrite=False)

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df, overwrite=False)

# Get fitbit data for subject timepoint pairs in the filtered mri dataset
# Get subject timepoint pairs for filtered mri dataset
subject_timepoint_pairs = mri_data_filtered[["participant_id", "session_id"]]
subject_timepoint_pairs.rename(columns={"participant_id": "subject", "session_id": "timepoint"}, inplace=True)

# Check which subject timepoint pairs exist in fit_meta_df
existing_pairs = pd.merge(subject_timepoint_pairs, fit_meta_df[["subject", "timepoint"]], on=["subject", "timepoint"], how="inner")
existing_pairs = existing_pairs.drop_duplicates(subset=["subject", "timepoint"])
print(f"Number of subject timepoint pairs in filtered mri dataset: {subject_timepoint_pairs.shape[0]}")
print(f"Number of existing subject timepoint pairs in fit_meta_df: {existing_pairs.shape[0]}")

# Check which subject timepoint pairs that exist in fit_meta_df have a depression diagnosis according to the youth KSADS questionnaire
existing_pairs_with_dep_y = pd.merge(existing_pairs, mri_data_filtered[["participant_id", "mh_y_ksads__dep__mdd__pres_dx"]], left_on="subject", right_on="participant_id", how="inner")
existing_pairs_with_dep_y = existing_pairs_with_dep_y[existing_pairs_with_dep_y["mh_y_ksads__dep__mdd__pres_dx"] == 1]
existing_pairs_with_dep_p = pd.merge(existing_pairs, mri_data_filtered[["participant_id", "mh_p_ksads__dep__mdd__pres_dx"]], left_on="subject", right_on="participant_id", how="inner")
existing_pairs_with_dep_p = existing_pairs_with_dep_p[existing_pairs_with_dep_p["mh_p_ksads__dep__mdd__pres_dx"] == 1]
print(f"Number of existing subject timepoint pairs in fit_meta_df with depression diagnosis according to youth KSADS questionnaire: {existing_pairs_with_dep_y.shape[0]}")
print(f"Number of existing subject timepoint pairs in fit_meta_df with depression diagnosis according to parent KSADS questionnaire: {existing_pairs_with_dep_p.shape[0]}")

# Extract fitbit features for subject timepoint pairs in the filtered mri dataset
fitbit_features = extract_fitbit_features_2(con, existing_pairs, output_path=os.path.join(baseline_output_path, "fitbit_features.csv"), overwrite=True)
fitbit_features.to_csv(os.path.join(baseline_output_path, "fitbit_features.csv"), index=False)

# Make sure that the fitbit features are aligned with the filtered mri dataset by merging on subject and timepoint
fitbit_features_filtered = fitbit_features.merge(existing_pairs, left_on=["subject", "timepoint"], right_on=["subject", "timepoint"], how="inner")

# Create composites of fitbit features
fitbit_features_composites, fitbit_features_composite_dict = create_composites(fitbit_features_filtered.drop(columns=["subject", "timepoint"]), overwrite=True, composite_output=os.path.join(baseline_output_path, "fitbit_features_composites.csv"))
fitbit_features_composites["subject"] = fitbit_features_filtered["subject"]
fitbit_features_composites["timepoint"] = fitbit_features_filtered["timepoint"]

fitbit_features_composites.to_csv(os.path.join(baseline_output_path, "fitbit_features_with_composites.csv"), index=False)

# Optional: Reimport fitbit features with composites
fitbit_features_composites = pd.read_csv(os.path.join(baseline_output_path, "fitbit_features_with_composites.csv"))

# Add depression diagnosis labels based on youth and parent KSADS questionnaires to the fitbit data
fitbit_features_filtered = (
    fitbit_features_composites
    .merge(
        mri_data_filtered[
            [
                "participant_id",
                "session_id",
                "mh_y_ksads__dep__mdd__pres_dx",
                "mh_p_ksads__dep__mdd__pres_dx",
            ]
        ],
        left_on=["subject", "timepoint"],
        right_on=["participant_id", "session_id"],
        how="inner", 
    )
)

# Get number of subjects in fitbit data with depression diagnosis according to youth and parent KSADS questionnaires
labels = (
    fitbit_features_filtered[
        ["subject", "timepoint",
         "mh_y_ksads__dep__mdd__pres_dx",
         "mh_p_ksads__dep__mdd__pres_dx"]
    ]
    .drop_duplicates()
)

num_depressed_youth = labels["mh_y_ksads__dep__mdd__pres_dx"].sum()
num_depressed_parent = labels["mh_p_ksads__dep__mdd__pres_dx"].sum()
print(f"Number of subjects with depression diagnosis in fitbit data according to youth KSADS questionnaire: {num_depressed_youth}")
print(f"Number of subjects with depression diagnosis in fitbit data according to parent KSADS questionnaire: {num_depressed_parent}")

# Impute missing values in fitbit features using median imputation
fitbit_features_filtered_imputed = fitbit_features_filtered.drop(columns=["subject", "timepoint", "session_id", "participant_id"], errors="ignore").copy()
imputer = SimpleImputer(strategy="median")
fitbit_features_filtered_imputed = pd.DataFrame(imputer.fit_transform(fitbit_features_filtered_imputed), columns=fitbit_features_filtered_imputed.columns)
fitbit_features_filtered_imputed["participant_id"] = fitbit_features_filtered["participant_id"].values

# Add age and sex to the fitbit features for classification
features = fitbit_features_filtered_imputed.merge(
    mri_data_filtered[["participant_id", "visit_age", "sex", "scan_site"]],
    left_on="participant_id",
    right_on="participant_id",
    how="left"
)

# EXPLORATORY GROUP DIFFERENCE ANALYSIS ON FITBIT FEATURES
# Conduct exploratory group difference analysis on fitbit features for youth and parent depression diagnosis
fitbit_dep_y_sig, fitbit_dep_y_all = exploratory_group_difference_analysis_fitbit(features, "mh_y_ksads__dep__mdd__pres_dx", 
                                                                     os.path.join(baseline_output_path, "fitbit_dep_y_results.csv"), 
                                                                     output_path_sig=os.path.join(baseline_output_path, "fitbit_dep_y_results_sig.csv"),
                                                                     overwrite=True)
fitbit_dep_p_sig, fitbit_dep_p_all = exploratory_group_difference_analysis_fitbit(features, "mh_p_ksads__dep__mdd__pres_dx", 
                                                                     os.path.join(baseline_output_path, "fitbit_dep_p_results.csv"), 
                                                                     output_path_sig=os.path.join(baseline_output_path, "fitbit_dep_p_results_sig.csv"),
                                                                     overwrite=True)

# Filter significant fitbit features to only include those with an absolute effect size >0.2
fitbit_dep_y_sig_filtered = fitbit_dep_y_sig[abs(fitbit_dep_y_sig["effect_size"]) > 0.2]
fitbit_dep_p_sig_filtered = fitbit_dep_p_sig[abs(fitbit_dep_p_sig["effect_size"]) > 0.2]
print(f"Number of significant fitbit features after filtering by effect size (youth): {fitbit_dep_y_sig_filtered.shape[0]}")
print(f"Number of significant fitbit features after filtering by effect size (parent): {fitbit_dep_p_sig_filtered.shape[0]}")

# NOTE: This analysis did not result in any significant fitbit features at all. 

# BASELINE CLASSIFICATION
# Prepare data for classification
# Define features and labels for classification
X = features.drop(columns=["mh_y_ksads__dep__mdd__pres_dx", "mh_p_ksads__dep__mdd__pres_dx"])
y_y = features["mh_y_ksads__dep__mdd__pres_dx"]
y_p = features["mh_p_ksads__dep__mdd__pres_dx"]

# Check for missing values in features and labels
missing_X = X.isnull().sum().sum()
missing_y_y = y_y.isnull().sum()
missing_y_p = y_p.isnull().sum()
print(f"Missing values in features: {missing_X}")
print(f"Missing values in youth labels: {missing_y_y}")
print(f"Missing values in parent labels: {missing_y_p}")

# Print class distribution for youth and parent labels
print("Class distribution for youth labels:")
print(y_y.value_counts())
print("Class distribution for parent labels:")
print(y_p.value_counts())

# Train-test split for classification
X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X, y_y, test_size=0.2, random_state=42, stratify=y_y)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_p, test_size=0.2, random_state=42, stratify=y_p)

# Print class distribution for training and testing sets for youth and parent labels
print("Class distribution for youth labels (training set):")
print(y_train_y.value_counts())
print("Class distribution for youth labels (testing set):")
print(y_test_y.value_counts())
print("Class distribution for parent labels (training set):")
print(y_train_p.value_counts())
print("Class distribution for parent labels (testing set):")
print(y_test_p.value_counts())

# Resample the training set to address class imbalance using random oversampler
resampler = RandomOverSampler(random_state=42)
X_train_y_resampled, y_train_y_resampled = resampler.fit_resample(X_train_y.drop(columns=["participant_id"], errors="ignore"), y_train_y)
X_train_p_resampled, y_train_p_resampled = resampler.fit_resample(X_train_p.drop(columns=["participant_id"], errors="ignore"), y_train_p)

# Print class distribution for resampled training sets for youth and parent labels
print("Class distribution for youth labels (resampled training set):")
print(pd.Series(y_train_y_resampled).value_counts())
print("Class distribution for parent labels (resampled training set):")
print(pd.Series(y_train_p_resampled).value_counts())

# YOUTH MODEL
# Train and evaluate baseline classification models using nested cross-validation
final_model_y, best_hyperparams_y, train_predictions_y = train_final_model(
    X_train_y.drop(columns=["participant_id"], errors="ignore"),
    y_train_y,
    model="SVM"
)

# Save final model, hyperparameters and predictions
joblib.dump(final_model_y, os.path.join(baseline_output_path, "final_model_y.joblib"))
joblib.dump(best_hyperparams_y, os.path.join(baseline_output_path, "best_hyperparams_y.json"))
train_predictions_df = pd.DataFrame(train_predictions_y, columns=["predicted_label"])
train_predictions_df["true_label"] = y_train_y.values
train_predictions_df.to_csv(os.path.join(baseline_output_path, "train_predictions_y.csv"), index=False)

# Optional: reimport final model for evaluation
final_model_y = joblib.load(os.path.join(baseline_output_path, "final_model_y.joblib"))

# Calculate performance metrics for youth labels on training set
auc_train_y = roc_auc_score(y_train_y, final_model_y.decision_function(X_train_y.drop(columns=["participant_id"], errors="ignore")))
print(f"Performance metrics for youth labels (training set):")
print(f"AUC: {auc_train_y:.4f}")

# Get predictions on the test set for youth labels
test_predictions_y = final_model_y.predict(X_test_y.drop(columns=["participant_id"], errors="ignore"))
test_predictions_df_y = pd.DataFrame(test_predictions_y, columns=["predicted_label"])
test_predictions_df_y["true_label"] = y_test_y.values
test_predictions_df_y.to_csv(os.path.join(baseline_output_path, "test_predictions_y.csv"), index=False)

# Calculate performance metrics for youth labels
auc_y = roc_auc_score(y_test_y, final_model_y.decision_function(X_test_y.drop(columns=["participant_id"], errors="ignore")))
f1_y = f1_score(y_test_y, test_predictions_y)
accuracy_y = accuracy_score(y_test_y, test_predictions_y)
precision_y = precision_score(y_test_y, test_predictions_y)
print(f"Performance metrics for youth labels:")
print(f"AUC: {auc_y:.4f}")
print(f"F1 Score: {f1_y:.4f}")
print(f"Accuracy: {accuracy_y:.4f}")
print(f"Precision: {precision_y:.4f}")

# Get SHAP values for youth model
explainer_y = shap.Explainer(final_model_y.decision_function, X_train_y.drop(columns=["participant_id"], errors="ignore"))
shap_values_y = explainer_y(X_test_y.drop(columns=["participant_id"], errors="ignore"))
shap_values_y_df = pd.DataFrame(shap_values_y.values, columns=X_test_y.drop(columns=["participant_id"], errors="ignore").columns)
shap_values_y_df["participant_id"] = X_test_y["participant_id"].values
shap_values_y_df.to_csv(os.path.join(baseline_output_path, "shap_values_y.csv"), index=False)

shap.summary_plot(shap_values_y, X_test_y.drop(columns=["participant_id"], errors="ignore"), show=False)
plt.savefig(os.path.join(baseline_output_path, "shap_summary_y.png"), dpi=300, bbox_inches="tight")
plt.close()

# Train youth model using resampled data to address class imbalance
final_model_y_resampled, best_hyperparams_y_resampled, train_predictions_y_resampled = train_final_model(
    X_train_y_resampled,
    y_train_y_resampled,
    model="SVM"
)

# Save final model, hyperparameters and predictions for resampled youth model
joblib.dump(final_model_y_resampled, os.path.join(baseline_output_path, "final_model_y_resampled.joblib"))
joblib.dump(best_hyperparams_y_resampled, os.path.join(baseline_output_path, "best_hyperparams_y_resampled.json"))
train_predictions_df_y_resampled = pd.DataFrame(train_predictions_y_resampled, columns=["predicted_label"])
train_predictions_df_y_resampled["true_label"] = y_train_y_resampled.values
train_predictions_df_y_resampled.to_csv(os.path.join(baseline_output_path, "train_predictions_y_resampled.csv"), index=False)

# Optional: reimport final model for evaluation
final_model_y_resampled = joblib.load(os.path.join(baseline_output_path, "final_model_y_resampled.joblib"))

# Calculate performance metrics for resampled youth model on training set
auc_train_y_resampled = roc_auc_score(y_train_y_resampled, final_model_y_resampled.decision_function(X_train_y_resampled))
print(f"Performance metrics for resampled youth model (training set):")
print(f"AUC: {auc_train_y_resampled:.4f}")

# Get predictions on the test set for resampled youth model
test_predictions_y_resampled = final_model_y_resampled.predict(X_test_y.drop(columns=["participant_id"], errors="ignore"))
test_predictions_df_y_resampled = pd.DataFrame(test_predictions_y_resampled, columns=["predicted_label"])
test_predictions_df_y_resampled["true_label"] = y_test_y.values
test_predictions_df_y_resampled.to_csv(os.path.join(baseline_output_path, "test_predictions_y_resampled.csv"), index=False)

# Calculate performance metrics for resampled youth model
auc_y_resampled = roc_auc_score(y_test_y, final_model_y_resampled.decision_function(X_test_y.drop(columns=["participant_id"], errors="ignore")))
f1_y_resampled = f1_score(y_test_y, test_predictions_y_resampled)
accuracy_y_resampled = accuracy_score(y_test_y, test_predictions_y_resampled)
precision_y_resampled = precision_score(y_test_y, test_predictions_y_resampled)
print(f"Performance metrics for resampled youth model:")
print(f"AUC: {auc_y_resampled:.4f}")
print(f"F1 Score: {f1_y_resampled:.4f}")
print(f"Accuracy: {accuracy_y_resampled:.4f}")
print(f"Precision: {precision_y_resampled:.4f}")

# Get SHAP values for resampled youth model
explainer_y_resampled = shap.Explainer(final_model_y_resampled.decision_function, X_train_y_resampled)
shap_values_y_resampled = explainer_y_resampled(X_test_y.drop(columns=["participant_id"], errors="ignore"))
shap_values_y_resampled_df = pd.DataFrame(shap_values_y_resampled.values, columns=X_test_y.drop(columns=["participant_id"], errors="ignore").columns)
shap_values_y_resampled_df["participant_id"] = X_test_y["participant_id"].values
shap_values_y_resampled_df.to_csv(os.path.join(baseline_output_path, "shap_values_y_resampled.csv"), index=False)

shap.summary_plot(shap_values_y_resampled, X_test_y.drop(columns=["participant_id"], errors="ignore"), show=False)
plt.savefig(os.path.join(baseline_output_path, "shap_summary_y_resampled.png"), dpi=300, bbox_inches="tight")
plt.close()

# PARENT MODEL
# Train and evaluate baseline classification models using nested cross-validation
final_model_p, best_hyperparams_p, train_predictions_p = train_final_model(
    X_train_p.drop(columns=["participant_id"], errors="ignore"),
    y_train_p,
    model="SVM"
)

# Save final model, hyperparameters and predictions
joblib.dump(final_model_p, os.path.join(baseline_output_path, "final_model_p.joblib"))
joblib.dump(best_hyperparams_p, os.path.join(baseline_output_path, "best_hyperparams_p.json"))
train_predictions_df_p = pd.DataFrame(train_predictions_p, columns=["predicted_label"])
train_predictions_df_p["true_label"] = y_train_p.values
train_predictions_df_p.to_csv(os.path.join(baseline_output_path, "train_predictions_p.csv"), index=False)

# Calculate performance metrics for parent labels on training set
auc_train_p = roc_auc_score(y_train_p, final_model_p.decision_function(X_train_p.drop(columns=["participant_id"], errors="ignore")))
print(f"Performance metrics for parent labels (training set):")
print(f"AUC: {auc_train_p:.4f}")

# Get predictions on the test set for parent labels
test_predictions_p = final_model_p.predict(X_test_p.drop(columns=["participant_id"], errors="ignore"))
test_predictions_df_p = pd.DataFrame(test_predictions_p, columns=["predicted_label"])
test_predictions_df_p["true_label"] = y_test_p.values
test_predictions_df_p.to_csv(os.path.join(baseline_output_path, "test_predictions_p.csv"), index=False)

# Calculate performance metrics for parent labels
auc_p = roc_auc_score(y_test_p, final_model_p.decision_function(X_test_p.drop(columns=["participant_id"], errors="ignore")))
f1_p = f1_score(y_test_p, test_predictions_p)
accuracy_p = accuracy_score(y_test_p, test_predictions_p)
precision_p = precision_score(y_test_p, test_predictions_p)
print(f"Performance metrics for parent labels:")
print(f"AUC: {auc_p:.4f}")
print(f"F1 Score: {f1_p:.4f}")
print(f"Accuracy: {accuracy_p:.4f}")
print(f"Precision: {precision_p:.4f}")

# Get SHAP values for parent model
explainer_p = shap.Explainer(final_model_p.decision_function, X_train_p.drop(columns=["participant_id"], errors="ignore"))
shap_values_p = explainer_p(X_test_p.drop(columns=["participant_id"], errors="ignore"))
shap_values_p_df = pd.DataFrame(shap_values_p.values, columns=X_test_p.drop(columns=["participant_id"], errors="ignore").columns)
shap_values_p_df["participant_id"] = X_test_p["participant_id"].values
shap_values_p_df.to_csv(os.path.join(baseline_output_path, "shap_values_p.csv"), index=False)

shap.summary_plot(shap_values_p, X_test_p.drop(columns=["participant_id"], errors="ignore"), show=False)
plt.savefig(os.path.join(baseline_output_path, "shap_summary_p.png"), dpi=300, bbox_inches="tight")
plt.close()

# Train parent model using resampled data to address class imbalance
final_model_p_resampled, best_hyperparams_p_resampled, train_predictions_p_resampled = train_final_model(
    X_train_p_resampled,
    y_train_p_resampled,
    model="SVM"
)

# Save final model, hyperparameters and predictions for resampled parent model
joblib.dump(final_model_p_resampled, os.path.join(baseline_output_path, "final_model_p_resampled.joblib"))
joblib.dump(best_hyperparams_p_resampled, os.path.join(baseline_output_path, "best_hyperparams_p_resampled.json"))
train_predictions_df_p_resampled = pd.DataFrame(train_predictions_p_resampled, columns=["predicted_label"])

# Optional: reimport final model for evaluation
final_model_p_resampled = joblib.load(os.path.join(baseline_output_path, "final_model_p_resampled.joblib"))

# Calculate performance metrics for resampled parent model on training set
auc_train_p_resampled = roc_auc_score(y_train_p_resampled, final_model_p_resampled.decision_function(X_train_p_resampled))
print(f"Performance metrics for resampled parent model (training set):")
print(f"AUC: {auc_train_p_resampled:.4f}")

# Get predictions on the test set for resampled parent model
test_predictions_p_resampled = final_model_p_resampled.predict(X_test_p.drop(columns=["participant_id"], errors="ignore"))
test_predictions_df_p_resampled = pd.DataFrame(test_predictions_p_resampled, columns=["predicted_label"])
test_predictions_df_p_resampled["true_label"] = y_test_p.values
test_predictions_df_p_resampled.to_csv(os.path.join(baseline_output_path, "test_predictions_p_resampled.csv"), index=False)

# Calculate performance metrics for resampled parent model
auc_p_resampled = roc_auc_score(y_test_p, final_model_p_resampled.decision_function(X_test_p.drop(columns=["participant_id"], errors="ignore")))
f1_p_resampled = f1_score(y_test_p, test_predictions_p_resampled)
accuracy_p_resampled = accuracy_score(y_test_p, test_predictions_p_resampled)
precision_p_resampled = precision_score(y_test_p, test_predictions_p_resampled)
print(f"Performance metrics for resampled parent model:")
print(f"AUC: {auc_p_resampled:.4f}")
print(f"F1 Score: {f1_p_resampled:.4f}")
print(f"Accuracy: {accuracy_p_resampled:.4f}")
print(f"Precision: {precision_p_resampled:.4f}")

# Get SHAP values for resampled parent model
explainer_p_resampled = shap.Explainer(final_model_p_resampled.decision_function, X_train_p_resampled)
shap_values_p_resampled = explainer_p_resampled(X_test_p.drop(columns=["participant_id"], errors="ignore"))
shap_values_p_resampled_df = pd.DataFrame(shap_values_p_resampled.values, columns=X_test_p.drop(columns=["participant_id"], errors="ignore").columns)
shap_values_p_resampled_df["participant_id"] = X_test_p["participant_id"].values
shap_values_p_resampled_df.to_csv(os.path.join(baseline_output_path, "shap_values_p_resampled.csv"), index=False)

shap.summary_plot(shap_values_p_resampled, X_test_p.drop(columns=["participant_id"], errors="ignore"), show=False)
plt.savefig(os.path.join(baseline_output_path, "shap_summary_p_resampled.png"), dpi=300, bbox_inches="tight")
plt.close()

# REGRESSION ANALYSIS
# Create regression target from z-score composite for youth and parent
# Get subjects indicator from train and test features
train_subjects_y = X_train_y["participant_id"]
test_subjects_y = X_test_y["participant_id"]
train_subjects_p = X_train_p["participant_id"]
test_subjects_p = X_test_p["participant_id"]

# Create regression target from z-score composite for youth and parent
reg_y_train_y = z_scores_y_composites[z_scores_y_composites["subject_ids"].isin(train_subjects_y)][["subject_ids", "z_score_composite"]].copy()
reg_y_test_y = z_scores_y_composites[z_scores_y_composites["subject_ids"].isin(test_subjects_y)][["subject_ids", "z_score_composite"]].copy()
reg_y_train_p = z_scores_p_composites[z_scores_p_composites["subject_ids"].isin(train_subjects_p)][["subject_ids", "z_score_composite"]].copy()
reg_y_test_p = z_scores_p_composites[z_scores_p_composites["subject_ids"].isin(test_subjects_p)][["subject_ids", "z_score_composite"]].copy()

# NOTE: Are confounding variables included in the regression model? If not TODO: Include confounding variables in the regression model to control for their effects on the z-score composite.

# Add classification labels to regression targets for stratification during training
y_train_y_df = pd.DataFrame({"subject_ids": train_subjects_y.values, "label": y_train_y.values})
reg_y_train_y = reg_y_train_y.merge(y_train_y_df,on="subject_ids",how="left")
y_test_y_df = pd.DataFrame({"subject_ids": test_subjects_y.values,"label": y_test_y.values})
reg_y_test_y = reg_y_test_y.merge(y_test_y_df,on="subject_ids",how="left")
y_train_p_df = pd.DataFrame({"subject_ids": train_subjects_p.values,"label": y_train_p.values})
reg_y_train_p = reg_y_train_p.merge(y_train_p_df,on="subject_ids",how="left")
y_test_p_df = pd.DataFrame({"subject_ids": test_subjects_p.values,"label": y_test_p.values})
reg_y_test_p = reg_y_test_p.merge(y_test_p_df, on="subject_ids", how="left")

# Check that the regression target is aligned with the training and testing features for youth
assert set(reg_y_train_y["subject_ids"]) == set(train_subjects_y), "Mismatch between training features and regression target for youth"
assert set(reg_y_test_y["subject_ids"]) == set(test_subjects_y), "Mismatch between testing features and regression target for youth"
assert set(reg_y_train_p["subject_ids"]) == set(train_subjects_p), "Mismatch between training features and regression target for parent"
assert set(reg_y_test_p["subject_ids"]) == set(test_subjects_p), "Mismatch between testing features and regression target for parent"

# Check for missing values in regression targets for youth and parent
missing_reg_y_train_y = reg_y_train_y.isnull().sum().sum()
missing_reg_y_test_y = reg_y_test_y.isnull().sum().sum()
missing_reg_y_train_p = reg_y_train_p.isnull().sum().sum()
missing_reg_y_test_p = reg_y_test_p.isnull().sum().sum()
print(f"Missing values in regression target for youth (training set): {missing_reg_y_train_y}")
print(f"Missing values in regression target for youth (testing set): {missing_reg_y_test_y}")
print(f"Missing values in regression target for parent (training set): {missing_reg_y_train_p}")
print(f"Missing values in regression target for parent (testing set): {missing_reg_y_test_p}")

# Print class distribution for regression targets in train and test sets for youth and parent
print("Class distribution for regression targets (youth) in training set:")
print(reg_y_train_y["label"].value_counts())
print("Class distribution for regression targets (youth) in testing set:")
print(reg_y_test_y["label"].value_counts())
print("Class distribution for regression targets (parent) in training set:")
print(reg_y_train_p["label"].value_counts())
print("Class distribution for regression targets (parent) in testing set:")
print(reg_y_test_p["label"].value_counts())

# Resample the training set to address class imbalance using random oversampler for regression targets
oversampler = RandomOverSampler(random_state=42)
X_res, labels_res = oversampler.fit_resample(X_train, reg_y_train["label"])
indices = oversampler.sample_indices_
y_res = reg_y_train["z_score_composite"].iloc[indices]

# YOUTH MODEL 
# Train final regression model for youth z-score composite
final_reg_model_y, best_hyperparams_reg_y, train_predictions_reg_y = train_final_regression_model(
    X_train_y.drop(columns=["participant_id"], errors="ignore"),
    reg_y_train_y[["z_score_composite", "label"]].squeeze(),
    model="SVR"
)

# Save final regression model, hyperparameters and predictions for youth
joblib.dump(final_reg_model_y, os.path.join(baseline_output_path, "final_reg_model_y.joblib"))
joblib.dump(best_hyperparams_reg_y, os.path.join(baseline_output_path, "best_hyperparams_reg_y.json"))
train_predictions_reg_df_y = pd.DataFrame(train_predictions_reg_y, columns=["predicted_z_score_composite"])
train_predictions_reg_df_y["true_z_score_composite"] = reg_y_train_y["z_score_composite"].values
train_predictions_reg_df_y.to_csv(os.path.join(baseline_output_path, "train_predictions_reg_y.csv"), index=False)

# Get predictions on the test set for youth z-score composite
test_predictions_reg_y = final_reg_model_y.predict(X_test_y.drop(columns=["participant_id"], errors="ignore"))
test_predictions_reg_df_y = pd.DataFrame(test_predictions_reg_y, columns=["predicted_z_score_composite"])
test_predictions_reg_df_y["true_z_score_composite"] = reg_y_test_y["z_score_composite"].values
test_predictions_reg_df_y.to_csv(os.path.join(baseline_output_path, "test_predictions_reg_y.csv"), index=False)

# Calculate performance metrics for youth z-score composite regression
mse_reg_y = mean_squared_error(reg_y_test_y["z_score_composite"], test_predictions_reg_y)
r2_reg_y = r2_score(reg_y_test_y["z_score_composite"], test_predictions_reg_y)
print(f"Performance metrics for youth z-score composite regression:")
print(f"MSE: {mse_reg_y:.4f}")
print(f"R^2: {r2_reg_y:.4f}")

# Train regression model using resampled data to address class imbalance for youth z-score composite
final_reg_model_y_resampled, best_hyperparams_reg_y_resampled, train_predictions_reg_y_resampled = train_final_regression_model(
    X_train_y_resampled_reg,
    reg_y_train_y_resampled[["z_score_composite", "label"]].squeeze(),
    model="SVR"
)

# Save final regression model, hyperparameters and predictions for resampled youth
joblib.dump(final_reg_model_y_resampled, os.path.join(baseline_output_path, "final_reg_model_y_resampled.joblib"))
joblib.dump(best_hyperparams_reg_y_resampled, os.path.join(baseline_output_path, "best_hyperparams_reg_y_resampled.json"))
train_predictions_reg_df_y_resampled = pd.DataFrame(train_predictions_reg_y_resampled, columns=["predicted_z_score_composite"])
train_predictions_reg_df_y_resampled["true_z_score_composite"] = reg_y_train_y_resampled["z_score_composite"].values
train_predictions_reg_df_y_resampled.to_csv(os.path.join(baseline_output_path, "train_predictions_reg_y_resampled.csv"), index=False)

# Get predictions on the test set for resampled youth z-score composite
test_predictions_reg_y_resampled = final_reg_model_y_resampled.predict(X_test_y.drop(columns=["participant_id"], errors="ignore"))
test_predictions_reg_df_y_resampled = pd.DataFrame(test_predictions_reg_y_resampled, columns=["predicted_z_score_composite"])
test_predictions_reg_df_y_resampled["true_z_score_composite"] = reg_y_test_y["z_score_composite"].values
test_predictions_reg_df_y_resampled.to_csv(os.path.join(baseline_output_path, "test_predictions_reg_y_resampled.csv"), index=False)

# Calculate performance metrics for resampled youth z-score composite regression
mse_reg_y_resampled = mean_squared_error(reg_y_test_y["z_score_composite"], test_predictions_reg_y_resampled)
r2_reg_y_resampled = r2_score(reg_y_test_y["z_score_composite"], test_predictions_reg_y_resampled)
print(f"Performance metrics for resampled youth z-score composite regression:")
print(f"MSE: {mse_reg_y_resampled:.4f}")
print(f"R^2: {r2_reg_y_resampled:.4f}")

# PARENT MODEL 
# Train final regression model for parent z-score composite
final_reg_model_p, best_hyperparams_reg_p, train_predictions_reg_p = train_final_regression_model(
    X_train_p.drop(columns=["participant_id"], errors="ignore"),
    reg_y_train_p[["z_score_composite", "label"]].squeeze(),
    model="SVR"
)

# Save final regression model, hyperparameters and predictions for parent
joblib.dump(final_reg_model_p, os.path.join(baseline_output_path, "final_reg_model_p.joblib"))
joblib.dump(best_hyperparams_reg_p, os.path.join(baseline_output_path, "best_hyperparams_reg_p.json"))
train_predictions_reg_df_p = pd.DataFrame(train_predictions_reg_p, columns=["predicted_z_score_composite"])
train_predictions_reg_df_p["true_z_score_composite"] = reg_y_train_p["z_score_composite"].values
train_predictions_reg_df_p.to_csv(os.path.join(baseline_output_path, "train_predictions_reg_p.csv"), index=False)

# Get predictions on the test set for parent z-score composite
test_predictions_reg_p = final_reg_model_p.predict(X_test_p.drop(columns=["participant_id"], errors="ignore"))
test_predictions_reg_df_p = pd.DataFrame(test_predictions_reg_p, columns=["predicted_z_score_composite"])
test_predictions_reg_df_p["true_z_score_composite"] = reg_y_test_p["z_score_composite"].values
test_predictions_reg_df_p.to_csv(os.path.join(baseline_output_path, "test_predictions_reg_p.csv"), index=False)

# Calculate performance metrics for parent z-score composite regression
mse_reg_p = mean_squared_error(reg_y_test_p["z_score_composite"], test_predictions_reg_p)
r2_reg_p = r2_score(reg_y_test_p["z_score_composite"], test_predictions_reg_p)
print(f"Performance metrics for parent z-score composite regression:")
print(f"MSE: {mse_reg_p:.4f}")
print(f"R^2: {r2_reg_p:.4f}")

# Train regression model using resampled data to address class imbalance for parent z-score composite
final_reg_model_p_resampled, best_hyperparams_reg_p_resampled, train_predictions_reg_p_resampled = train_final_regression_model(
    X_train_p_resampled_reg,
    reg_y_train_p_resampled[["z_score_composite", "label"]].squeeze(),
    model="SVR"
)

# Save final regression model, hyperparameters and predictions for resampled parent
joblib.dump(final_reg_model_p_resampled, os.path.join(baseline_output_path, "final_reg_model_p_resampled.joblib"))
joblib.dump(best_hyperparams_reg_p_resampled, os.path.join(baseline_output_path, "best_hyperparams_reg_p_resampled.json"))
train_predictions_reg_df_p_resampled = pd.DataFrame(train_predictions_reg_p_resampled, columns=["predicted_z_score_composite"])
train_predictions_reg_df_p_resampled["true_z_score_composite"] = reg_y_train_p_resampled["z_score_composite"].values
train_predictions_reg_df_p_resampled.to_csv(os.path.join(baseline_output_path, "train_predictions_reg_p_resampled.csv"), index=False)

# Get predictions on the test set for resampled parent z-score composite
test_predictions_reg_p_resampled = final_reg_model_p_resampled.predict(X_test_p.drop(columns=["participant_id"], errors="ignore"))
test_predictions_reg_df_p_resampled = pd.DataFrame(test_predictions_reg_p_resampled, columns=["predicted_z_score_composite"])
test_predictions_reg_df_p_resampled["true_z_score_composite"] = reg_y_test_p["z_score_composite"].values
test_predictions_reg_df_p_resampled.to_csv(os.path.join(baseline_output_path, "test_predictions_reg_p_resampled.csv"), index=False)

# Calculate performance metrics for resampled parent z-score composite regression
mse_reg_p_resampled = mean_squared_error(reg_y_test_p["z_score_composite"], test_predictions_reg_p_resampled)
r2_reg_p_resampled = r2_score(reg_y_test_p["z_score_composite"], test_predictions_reg_p_resampled)
print(f"Performance metrics for resampled parent z-score composite regression:")
print(f"MSE: {mse_reg_p_resampled:.4f}")
print(f"R^2: {r2_reg_p_resampled:.4f}")

# EXPLORATORY SYMPTOM ANALYSIS
# Create symptom profiles based on fitbit features and investigate their influence on depression diagnosis according to youth and parent
# Conduct unsupervised clustering on fitbit features to identify symptom profiles
# Get fitbit features for only depressed subjects according to youth and parent KSADS questionnaires
fitbit_features_depressed_y = features[features["mh_y_ksads__dep__mdd__pres_dx"] == 1].drop(columns=["participant_id", "mh_y_ksads__dep__mdd__pres_dx", "mh_p_ksads__dep__mdd__pres_dx"], errors="ignore")
fitbit_features_depressed_p = features[features["mh_p_ksads__dep__mdd__pres_dx"] == 1].drop(columns=["participant_id", "mh_y_ksads__dep__mdd__pres_dx", "mh_p_ksads__dep__mdd__pres_dx"], errors="ignore")

# Residualize fitbit features by regressing out age and sex for youth and parent separately
def residualize_features(df, feature_cols, covariates):
    """
    Residualize `feature_cols` against `covariates` (e.g., age, sex).
    Returns a new dataframe with residualized features; other columns
    (IDs, labels, etc.) are left untouched.
    """
    residualized_df = df.copy()

    # Ensure categorical covariates (e.g. sex as 'M'/'F') are numeric dummies
    X_cov = pd.get_dummies(df[covariates], drop_first=True)
    X_cov = sm.add_constant(X_cov)

    for col in feature_cols:
        y = df[col]

        # Align and drop rows with missing values in either y or X_cov
        valid = y.notna() & X_cov.notna().all(axis=1)

        model = sm.OLS(y[valid], X_cov[valid]).fit()
        resid = model.predict(X_cov[valid])  # fitted values, for residual calc below
        residualized_df.loc[valid, col] = y[valid] - model.fittedvalues
        residualized_df.loc[~valid, col] = pd.NA  # or keep as NaN

    return residualized_df

feature_cols_y = fitbit_features_depressed_y.drop(columns=["participant_id", "visit_age", "sex"], errors="ignore").columns.tolist()
covariates_y = ["visit_age", "sex"]
feature_cols_p = fitbit_features_depressed_p.drop(columns=["participant_id", "visit_age", "sex"], errors="ignore").columns.tolist()
covariates_p = ["visit_age", "sex"]

fitbit_features_depressed_y_resid = residualize_features(fitbit_features_depressed_y, feature_cols_y, covariates_y)
fitbit_features_depressed_p_resid = residualize_features(fitbit_features_depressed_p, feature_cols_p, covariates_p)

# Check residualization by fitting a linear model and checking the correlation between residuals and covariates
summary, stats = confound_analysis(
    data_pre=fitbit_features_depressed_y,
    data_post=fitbit_features_depressed_y_resid,
    feature_cols=feature_cols_y,
    base_terms=[
        "bs(visit_age, df = 4)",
        "C(sex)",
    ],
    confounds={
        "Age": "visit_age",
        "Sex": "C(sex)",
    },
)

# Standardize residualized features for clustering
scaler_y = StandardScaler()
fitbit_features_depressed_y_resid_scaled = fitbit_features_depressed_y_resid.copy()
fitbit_features_depressed_y_resid_scaled[feature_cols_y] = scaler_y.fit_transform(fitbit_features_depressed_y_resid[feature_cols_y])

scaler_p = StandardScaler()
fitbit_features_depressed_p_resid_scaled = fitbit_features_depressed_p_resid.copy()
fitbit_features_depressed_p_resid_scaled[feature_cols_p] = scaler_p.fit_transform(fitbit_features_depressed_p_resid[feature_cols_p])

# Plot data of depressed subjects in 2D PaCMAP space for youth and parent
reducer = PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
fitbit_features_depressed_y_pacmap = reducer.fit_transform(fitbit_features_depressed_y_resid_scaled[feature_cols_y])
fitbit_features_depressed_p_pacmap = reducer.fit_transform(fitbit_features_depressed_p_resid_scaled[feature_cols_p])
plot_y_df = pd.DataFrame(fitbit_features_depressed_y_pacmap, columns=["PaCMAP_1", "PaCMAP_2"])
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=plot_y_df,
    x="PaCMAP_1",
    y="PaCMAP_2",
    alpha=0.7,
    s=50,
)
plt.title("PaCMAP projection of residualized and standardized fitbit features (Youth)")
plt.xlabel("PaCMAP 1")
plt.ylabel("PaCMAP 2")
plt.tight_layout()
plt.savefig(os.path.join(baseline_output_path, "fitbit_features_depressed_y_pacmap.png"), dpi=300, bbox_inches="tight")
plt.close()

# Perform clustering on residualized and standardized fitbit features for youth
class_labels_y = clustering(fitbit_features_depressed_y_resid_scaled[feature_cols_y], 
                                clustering_output = "class_labels_y", 
                                cl=["HDBSCAN", "BayesianGMM"],
                                max_clusters=10,
                                bootstrapping=False,
                                overwrite=True)

# Print size of each cluster for youth
print("Cluster sizes for youth:")
print(class_labels_y["label"].value_counts())
# NOTE: No cluster solutions met the filtering criteria for youth, so no clusters were identified.

# Perform clustering on residualized and standardized fitbit features for parent
class_labels_p = clustering(fitbit_features_depressed_p_resid_scaled[feature_cols_p],
                                clustering_output = "class_labels_p", 
                                cl=["HDBSCAN", "BayesianGMM"],
                                max_clusters=10,
                                bootstrapping=False,
                                overwrite=True)

# Print size of each cluster for parent
print("Cluster sizes for parent:")
print(class_labels_p["label"].value_counts())
