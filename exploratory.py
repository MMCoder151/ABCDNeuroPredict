from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pacmap import PaCMAP
from sklearn.ensemble import IsolationForest
from sklearn.impute import IterativeImputer
from neuroCombat import neuroCombat
import seaborn as sns
from src.mri_rois import *
from functools import reduce
from pygam import LinearGAM, s, l, f
from statsmodels.formula.api import ols
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import shap
import statsmodels.api as sm
from pcntoolkit import NormativeModel, BLR
from pcntoolkit.dataio.norm_data import NormData
import numpy as np
import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pathlib
import duckdb
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.seasonal import STL
from src.modelling import *
import joblib

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
def partial_r2(full_formula, reduced_formula, data):
    full = ols(full_formula, data=data).fit()
    reduced = ols(reduced_formula, data=data).fit()

    return (reduced.ssr - full.ssr) / reduced.ssr

def confound_analysis(data_pre, data_post, feature_cols, base_terms, confounds):
    """
    Parameters
    ----------
    data_pre, data_post : DataFrame
        Data before and after harmonization/residualization.

    feature_cols : list[str]
        Features to evaluate.

    base_terms : list[str]
        Terms in the full model.

        Example MRI:
        [
            "bs(visit_age, df=4)",
            "C(sex)",
            "C(scan_site)",
            "TIV_z"
        ]

        Example Fitbit:
        [
            "visit_age",
            "C(sex)"
        ]

    confounds : dict
        Mapping from confound name to the exact model term.

        Example MRI:
        {
            "Age": "bs(visit_age, df=4)",
            "Sex": "C(sex)",
            "Site": "C(scan_site)",
            "TIV": "TIV_z"
        }

        Example Fitbit:
        {
            "Age": "visit_age",
            "Sex": "C(sex)"
        }
    """

    summary = []

    full_r2_pre = []
    full_r2_post = []

    partial_pre = {k: [] for k in confounds}
    partial_post = {k: [] for k in confounds}

    for feature in tqdm(feature_cols):

        response = f'Q("{feature}")'

        full_formula = response + " ~ " + " + ".join(base_terms)

        model_pre = ols(full_formula, data=data_pre).fit()
        model_post = ols(full_formula, data=data_post).fit()

        row = {
            "Feature": feature,
            "R2_pre": model_pre.rsquared,
            "R2_post": model_post.rsquared,
        }

        full_r2_pre.append(model_pre.rsquared)
        full_r2_post.append(model_post.rsquared)

        for name, term in confounds.items():

            reduced_terms = [t for t in base_terms if t != term]

            reduced_formula = response + " ~ " + " + ".join(reduced_terms)

            r2_pre = partial_r2(full_formula, reduced_formula, data_pre)
            r2_post = partial_r2(full_formula, reduced_formula, data_post)

            row[f"{name}_partial_R2_pre"] = r2_pre
            row[f"{name}_partial_R2_post"] = r2_post

            partial_pre[name].append(r2_pre)
            partial_post[name].append(r2_post)

        summary.append(row)

    summary_df = pd.DataFrame(summary)

    # Mean row
    mean_row = {"Feature": "Mean"}

    for col in summary_df.columns[1:]:
        mean_row[col] = summary_df[col].mean()

    summary_df = pd.concat(
        [summary_df, pd.DataFrame([mean_row])],
        ignore_index=True,
    )

    # Wilcoxon tests
    wilcoxon_row = {
        "Feature": "Wilcoxon",
        "R2_stat": wilcoxon(full_r2_pre, full_r2_post).statistic,
        "R2_p": wilcoxon(full_r2_pre, full_r2_post).pvalue,
    }

    for name in confounds:

        stat, p = wilcoxon(partial_pre[name], partial_post[name])

        wilcoxon_row[f"{name}_stat"] = stat
        wilcoxon_row[f"{name}_p"] = p

    wilcoxon_df = pd.DataFrame([wilcoxon_row])

    return summary_df, wilcoxon_df

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
def _cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std
 
def _cohens_d_std(x, y, d=None):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    if d is None:
        d = _cohens_d(x, y)
    var_d = ((nx + ny) / (nx * ny)) + ((d ** 2) / (2 * (nx + ny - 2)))
    return np.sqrt(var_d)
 
def _stratified_bootstrap_indices(group_labels, n_bootstraps, rng):
    """Resample WITHIN each group separately, so every replicate preserves
    the original group sizes and always contains both groups."""
    idx_dep = np.where(group_labels == 1)[0]
    idx_nondep = np.where(group_labels == 0)[0]
    boots = []
    for _ in range(n_bootstraps):
        boot_dep = rng.choice(idx_dep, size=len(idx_dep), replace=True)
        boot_nondep = rng.choice(idx_nondep, size=len(idx_nondep), replace=True)
        boots.append(np.concatenate([boot_dep, boot_nondep]))
    return boots
 
def exploratory_group_difference_analysis(
    mri_data_filtered,
    group_col,
    output_path_all,
    output_path_sig,
    overwrite=False,
    n_bootstraps=100,
    seed=0,
):
    if not overwrite and os.path.exists(output_path_all) and os.path.exists(output_path_sig):
        print("Results already exist. Skipping analysis.")
        return pd.read_csv(output_path_sig), pd.read_csv(output_path_all)
 
    rng = np.random.default_rng(seed)
 
    mri_data_filtered = mri_data_filtered.copy()
    mri_data_filtered["scan_site_code"] = mri_data_filtered["scan_site"].astype("category").cat.codes
 
    cols_to_exclude = [
        "participant_id", "session_id",
        "mh_y_ksads__dep__mdd__pres_dx", "mh_p_ksads__dep__mdd__pres_dx",
        "sex", "scan_site", "scan_site_code", "visit_age", "age_squared", "visit_age_squared",
        "mr_y_smri__vol__aseg__icv_sum", group_col,
    ]
 
    nuisance_cols = ["visit_age", "scan_site_code", "sex", "mr_y_smri__vol__aseg__icv_sum"]
    full_cols = nuisance_cols + [group_col]
    group_term_index = full_cols.index(group_col)
 
    group_labels = mri_data_filtered[group_col].to_numpy()
    boot_indices = _stratified_bootstrap_indices(group_labels, n_bootstraps, rng)
 
    gam_results = []
    for col in tqdm(mri_data_filtered.columns, desc="GAM group-difference analysis"):
        if col in cols_to_exclude:
            continue
        try:
            y = mri_data_filtered[col]
 
            # --- Point estimate on the real (non-resampled) data ---
            X_full = mri_data_filtered[full_cols]
            gam_full = LinearGAM(s(0) + f(1) + l(2) + l(3) + l(4)).fit(X_full, y)
            p_value = gam_full.statistics_["p_values"][group_term_index]
 
            X_nuisance = mri_data_filtered[nuisance_cols]
            gam_nuisance = LinearGAM(s(0) + f(1) + l(2) + l(3)).fit(X_nuisance, y)
            adjusted_residuals = (y - gam_nuisance.predict(X_nuisance)).to_numpy()
 
            group_dep = adjusted_residuals[group_labels == 1]
            group_nondep = adjusted_residuals[group_labels == 0]
            effect_size = _cohens_d(group_dep, group_nondep)
            effect_size_std_analytic = _cohens_d_std(group_dep, group_nondep, d=effect_size)
 
            # --- Bootstrap distribution (resample residuals, don't refit GAMs) ---
            # --- Bootstrap distribution (refit nuisance GAM each replicate) ---
            boot_effect_sizes = np.empty(n_bootstraps)

            for b, indices in enumerate(boot_indices):

                boot_df = mri_data_filtered.iloc[indices].copy()

                X_boot = boot_df[nuisance_cols]
                y_boot = boot_df[col]
                group_boot = boot_df[group_col].to_numpy()

                try:
                    gam_boot = LinearGAM(
                        s(0) + f(1) + l(2) + l(3)
                    ).fit(X_boot, y_boot)

                    adjusted_boot = (
                        y_boot - gam_boot.predict(X_boot)
                    ).to_numpy()

                    boot_effect_sizes[b] = _cohens_d(
                        adjusted_boot[group_boot == 1],
                        adjusted_boot[group_boot == 0]
                    )

                except Exception:
                    boot_effect_sizes[b] = np.nan
 
            boot_effect_sizes = boot_effect_sizes[np.isfinite(boot_effect_sizes)]

            effect_size_std_boot = np.std(boot_effect_sizes, ddof=1)

            ci_lower, ci_upper = np.percentile(
                boot_effect_sizes,
                [2.5, 97.5]
            )

            sign_stability = np.mean(
                np.sign(boot_effect_sizes) == np.sign(effect_size)
            )
 
            gam_results.append((
                col, p_value, effect_size, effect_size_std_analytic,
                effect_size_std_boot, ci_lower, ci_upper, sign_stability,
            ))
        except Exception as e:
            print(f"Error occurred while fitting GAM model for column {col}: {e}")
 
    results_df = pd.DataFrame(gam_results, columns=[
        "feature", "p_value", "effect_size", "effect_size_std",
        "effect_size_std_boot", "ci_lower_boot", "ci_upper_boot", "sign_stability",
    ])
    results_df = results_df.sort_values("p_value")
 
    print("Performing FDR correction for multiple comparisons...")
    rejected, corrected_p_values, _, _ = multipletests(results_df["p_value"], alpha=0.05, method="fdr_bh")
    results_df["corrected_p_value"] = corrected_p_values
    results_df["significant_fdr"] = rejected
 
    results_df.to_csv(output_path_all, index=False)
    print(f"All results saved to: {output_path_all}")
 
    sig_results_df = results_df[results_df["significant_fdr"]]
    sig_results_df.to_csv(output_path_sig, index=False)
    print(f"Significant ROIs saved to: {output_path_sig}")
 
    num_significant_rois = results_df["significant_fdr"].sum()
    print(f"Number of significant ROIs after FDR correction: {num_significant_rois}")
 
    return sig_results_df, results_df

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
# Check overlap between subjects with fibtit features and subjects with mri features

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

def filter_subjects(dta_path, dta_path_tabular, test=False, overwrite=True, output_path=pathlib.Path("output")):
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
        dta_path_tabular (Path): Path to the tabular data directory
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

    # add depression marker to mri_meta_df
    # Read in clinical data for depression marker
    youth_directory = dta_path_tabular / "mh_y_ksads__dep.tsv"
    parent_directory = dta_path_tabular / "mh_p_ksads__dep.tsv"

    ksads_youth = pd.read_csv(youth_directory, sep="\t")
    ksads_parent = pd.read_csv(parent_directory, sep="\t")
    
    # Filter to only include subjects and timepoints that are present in mri_meta_df
    ksads_youth = ksads_youth.merge(
        mri_meta_df[["subject", "timepoint"]],
        left_on=["participant_id", "session_id"],
        right_on=["subject", "timepoint"],
        how="inner"
    )
    ksads_parent = ksads_parent.merge(
        mri_meta_df[["subject", "timepoint"]],
        left_on=["participant_id", "session_id"],
        right_on=["subject", "timepoint"],
        how="inner"
    )

    # Filter to only include the first timepoint for each subject
    ksads_youth = ksads_youth.sort_values(by=["participant_id", "session_id"]).groupby("participant_id").first().reset_index()
    ksads_parent = ksads_parent.sort_values(by=["participant_id", "session_id"]).groupby("participant_id").first().reset_index()

    # Get list of depressed subjects based on KSADS depression diagnosis (youth and parent report)
    diagnosis_cols_youth = {#"mh_y_ksads__dep__mdd__partrem_dx"  :"Diagnosis: Major depressive disorder (F32.4) - Partial remission [Youth]",
                            "mh_y_ksads__dep__mdd__pres_dx"     :"Diagnosis: Major depressive disorder - Present [Youth]",
                            #"mh_y_ksads__dep__pdd__oth__pres_dx":"Diagnosis: Other specified depressive disorder, persistent depressive disorder (impairment does not meet full criteria) (F32.8) - Present [Youth]",
                            #"mh_y_ksads__dep__pdd__partrem_dx"  :"Diagnosis: Persistent depressive disorder (Dysthymia) (F34.1) - Partial remission [Youth]",
                            "mh_y_ksads__dep__pdd__pres_dx"     :"Diagnosis: Persistent depressive disorder (Dysthymia) (F34.1) - Present [Youth]"}
    diagnosis_cols_parent = {#"mh_p_ksads__dep__mdd__partrem_dx"  :"Diagnosis: Major depressive disorder (F32.4) - Partial remission [Parent]",
                            "mh_p_ksads__dep__mdd__pres_dx"     :"Diagnosis: Major depressive disorder - Present [Parent]",
                            #"mh_p_ksads__dep__pdd__oth__pres_dx":"Diagnosis: Other specified depressive disorder, persistent depressive disorder (impairment does not meet full criteria) (F32.8) - Present [Parent]",
                            #"mh_p_ksads__dep__pdd__partrem_dx"  :"Diagnosis: Persistent depressive disorder (Dysthymia) (F34.1) - Partial remission [Parent]",
                            "mh_p_ksads__dep__pdd__pres_dx"     :"Diagnosis: Persistent depressive disorder (Dysthymia) (F34.1) - Present [Parent]"}
    
    # Create a binary depression marker for each subject based on youth and parent report
    diagnosis_youth_cols = list(diagnosis_cols_youth.keys())
    y_depr = (ksads_youth[diagnosis_youth_cols] == 1).any(axis=1)

    diagnosis_parent_cols = list(diagnosis_cols_parent.keys())
    p_depr = (ksads_parent[diagnosis_parent_cols] == 1).any(axis=1)

    depr = y_depr | p_depr

    # Create a binary depression marker for each subject
    subjects_depr = set(ksads_youth.loc[depr, "participant_id"]) | set(ksads_parent.loc[depr, "participant_id"])

    # Add depression marker to mri_meta_df
    mri_meta_df["dep_dx"] = mri_meta_df["subject"].apply(lambda x: 1 if x in subjects_depr else 0)

    # Add parent and youth depression markers to mri_meta_df
    mri_meta_df["dep_dx_y"] = mri_meta_df["subject"].apply(lambda x: 1 if x in set(ksads_youth.loc[y_depr, "participant_id"]) else 0)
    mri_meta_df["dep_dx_p"] = mri_meta_df["subject"].apply(lambda x: 1 if x in set(ksads_parent.loc[p_depr, "participant_id"]) else 0)

    # Add raw diagnosis columns to mri_meta_df for reference
    mri_meta_df = mri_meta_df.merge(
        ksads_youth[["participant_id", *diagnosis_youth_cols]],
        left_on="subject",
        right_on="participant_id",
        how="left"
    ).drop(columns=["participant_id"])

    mri_meta_df = mri_meta_df.merge(
        ksads_parent[["participant_id", *diagnosis_parent_cols]],
        left_on="subject",
        right_on="participant_id",
        how="left"
    ).drop(columns=["participant_id"])

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

# Filter subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = filter_subjects(dta_path, dta_path_tabular, test=False, overwrite=False)

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
                        try:
                            imputer = IterativeImputer(random_state=0, max_iter=20)
                            daily_stats = pd.DataFrame(
                                imputer.fit_transform(daily_stats),
                                index=daily_stats.index,
                                columns=daily_stats.columns,
                            )
                        except Exception as e:
                            print(f"Iterative imputation failed for subject {subject}, metric {metric}: {e}")
                            daily_stats = daily_stats.ffill().bfill()
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
                            print(f"STL decomposition failed for subject {subject}, metric {metric}: {e}")
        features_list.append(feature_dict)
    fitbit_features_df = pd.DataFrame(features_list)
    
    fitbit_features_df.to_csv(Path("output")/ "fitbit_features.csv", index=False)

    return fitbit_features_df

fitbit_features = extract_fitbit_features_2(con, existing_pairs, output_path=os.path.join(baseline_output_path, "fitbit_features.csv"), overwrite=False)
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
def exploratory_group_difference_analysis_fitbit(
    mri_data_filtered,
    group_col,
    output_path_all,
    output_path_sig,
    overwrite=False,
    n_bootstraps=100,
    seed=0,
):
    if not overwrite and os.path.exists(output_path_all) and os.path.exists(output_path_sig):
        print("Results already exist. Skipping analysis.")
        return pd.read_csv(output_path_sig), pd.read_csv(output_path_all)
 
    rng = np.random.default_rng(seed)
 
    mri_data_filtered = mri_data_filtered.copy()
    mri_data_filtered["scan_site_code"] = mri_data_filtered["scan_site"].astype("category").cat.codes
 
    cols_to_exclude = [
        "participant_id", "session_id",
        "mh_y_ksads__dep__mdd__pres_dx", "mh_p_ksads__dep__mdd__pres_dx",
        "sex", "scan_site", "scan_site_code", "visit_age", "age_squared", "visit_age_squared",
        "mr_y_smri__vol__aseg__icv_sum", group_col,
    ]
 
    nuisance_cols = ["visit_age", "sex"]
    full_cols = nuisance_cols + [group_col]
    group_term_index = full_cols.index(group_col)
 
    group_labels = mri_data_filtered[group_col].to_numpy()
    boot_indices = _stratified_bootstrap_indices(group_labels, n_bootstraps, rng)
 
    gam_results = []
    for col in tqdm(mri_data_filtered.columns, desc="GAM group-difference analysis"):
        if col in cols_to_exclude:
            continue
        try:
            y = mri_data_filtered[col]
 
            # --- Point estimate on the real (non-resampled) data ---
            X_full = mri_data_filtered[full_cols]
            gam_full = LinearGAM(s(0) + f(1) + l(2)).fit(X_full, y)
            p_value = gam_full.statistics_["p_values"][group_term_index]
 
            X_nuisance = mri_data_filtered[nuisance_cols]
            gam_nuisance = LinearGAM(s(0) + f(1)).fit(X_nuisance, y)
            adjusted_residuals = (y - gam_nuisance.predict(X_nuisance)).to_numpy()
 
            group_dep = adjusted_residuals[group_labels == 1]
            group_nondep = adjusted_residuals[group_labels == 0]
            effect_size = _cohens_d(group_dep, group_nondep)
            effect_size_std_analytic = _cohens_d_std(group_dep, group_nondep, d=effect_size)
 
            # --- Bootstrap distribution (resample residuals, don't refit GAMs) ---
            # --- Bootstrap distribution (refit nuisance GAM each replicate) ---
            boot_effect_sizes = np.empty(n_bootstraps)

            for b, indices in enumerate(boot_indices):

                boot_df = mri_data_filtered.iloc[indices].copy()

                X_boot = boot_df[nuisance_cols]
                y_boot = boot_df[col]
                group_boot = boot_df[group_col].to_numpy()

                try:
                    gam_boot = LinearGAM(
                        s(0) + f(1) + l(2) + l(3)
                    ).fit(X_boot, y_boot)

                    adjusted_boot = (
                        y_boot - gam_boot.predict(X_boot)
                    ).to_numpy()

                    boot_effect_sizes[b] = _cohens_d(
                        adjusted_boot[group_boot == 1],
                        adjusted_boot[group_boot == 0]
                    )

                except Exception:
                    boot_effect_sizes[b] = np.nan
 
            boot_effect_sizes = boot_effect_sizes[np.isfinite(boot_effect_sizes)]

            if boot_effect_sizes.size == 0:
                print(f"Warning: all {n_bootstraps} bootstrap replicates failed for column '{col}'. Skipping stability stats.")
                effect_size_std_boot = np.nan
                ci_lower, ci_upper = np.nan, np.nan
                sign_stability = np.nan
            else:
                effect_size_std_boot = np.std(boot_effect_sizes, ddof=1) if boot_effect_sizes.size > 1 else np.nan
                ci_lower, ci_upper = np.percentile(boot_effect_sizes, [2.5, 97.5])
                sign_stability = np.mean(np.sign(boot_effect_sizes) == np.sign(effect_size))
 
            gam_results.append((
                col, p_value, effect_size, effect_size_std_analytic,
                effect_size_std_boot, ci_lower, ci_upper, sign_stability,
            ))
        except Exception as e:
            print(f"Error occurred while fitting GAM model for column {col}: {e}")
 
    results_df = pd.DataFrame(gam_results, columns=[
        "feature", "p_value", "effect_size", "effect_size_std",
        "effect_size_std_boot", "ci_lower_boot", "ci_upper_boot", "sign_stability",
    ])
    results_df = results_df.sort_values("p_value")
 
    print("Performing FDR correction for multiple comparisons...")
    rejected, corrected_p_values, _, _ = multipletests(results_df["p_value"], alpha=0.05, method="fdr_bh")
    results_df["corrected_p_value"] = corrected_p_values
    results_df["significant_fdr"] = rejected
 
    results_df.to_csv(output_path_all, index=False)
    print(f"All results saved to: {output_path_all}")
 
    sig_results_df = results_df[results_df["significant_fdr"]]
    sig_results_df.to_csv(output_path_sig, index=False)
    print(f"Significant ROIs saved to: {output_path_sig}")
 
    num_significant_rois = results_df["significant_fdr"].sum()
    print(f"Number of significant ROIs after FDR correction: {num_significant_rois}")
 
    return sig_results_df, results_df

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
