from pathlib import Path
import os
from src.preprocessing import *
from src.feature_extraction import *
from src.data_analysis import *
from src.modelling import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
from pacmap import PaCMAP
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.impute import IterativeImputer
from neuroCombat import neuroCombat
import seaborn as sns
from src.mri_rois import *
from functools import reduce
from pygam import LinearGAM, s, l, f
from sklearn.metrics import jaccard_score

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
feature_cols = mri_data.columns
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
mri_data_filtered[numeric_cols] = mri_data_combat["data"].transpose()

# Conduct confound analysis pre and post harmonization
# TODO: Implement

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

mri_dep_y_sig.columns

# Filter significant ROIs to only include those with an absolute effect size >0.2
mri_dep_y_sig_filtered = mri_dep_y_sig[abs(mri_dep_y_sig["effect_size"]) > 0.2]
mri_dep_p_sig_filtered = mri_dep_p_sig[abs(mri_dep_p_sig["effect_size"]) > 0.2]

# Print the number of significant ROIs after filtering by effect size
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

# Create composites of significant ROIs for youth and parent
mri_dep_y_sig_composites, mri_dep_y_sig_composite_dict = create_composites(mri_data_filtered[mri_dep_y_sig_filtered_rois], overwrite = True, composite_output = os.path.join(baseline_output_path, "mri_dep_y_sig_composites.csv"))
mri_dep_y_sig_composites["participant_id"] = mri_data_filtered["participant_id"]

mri_dep_p_sig_composites, mri_dep_p_sig_composite_dict = create_composites(mri_data_filtered[mri_dep_p_sig_filtered_rois], overwrite = True, composite_output = os.path.join(baseline_output_path, "mri_dep_p_sig_composites.csv"))
mri_dep_p_sig_composites["participant_id"] = mri_data_filtered["participant_id"]

# Create scatter plot of data in 2D PCA space for parent significant ROIs and mark depressed subjects
pca_p = PCA(n_components=2, random_state=42)
mri_dep_p_sig_pca = pca_p.fit_transform(mri_data_filtered[mri_dep_p_sig_filtered_rois])
mri_dep_p_sig_pca_df = pd.DataFrame(mri_dep_p_sig_pca, columns=["PCA_1", "PCA_2"])
mri_dep_p_sig_pca_df["depressed_parent"] = mri_data_filtered["mh_p_ksads__dep__mdd__pres_dx"]
plt.figure(figsize=(10, 8))
sns.scatterplot(data=mri_dep_p_sig_pca_df, x="PCA_1", y="PCA_2", hue="depressed_parent", palette={0: "blue", 1: "red"}, alpha=0.7)
plt.title("PCA of Parent Significant ROIs (Depressed vs Non-Depressed)")
plt.savefig(os.path.join(baseline_output_path, "mri_dep_p_sig_pca.png"))
plt.close()

# Site harmonization using ComBat for all significant ROIs for youth and parent
# TODO: Implement

# z-score normalization of significant ROIs for youth and parent
scaler_y = StandardScaler()
mri_dep_y_sig_scaled = pd.DataFrame(
    scaler_y.fit_transform(mri_dep_y_sig_composites.drop(columns=["participant_id"])),
    columns=mri_dep_y_sig_composites.drop(columns=["participant_id"]).columns,
    index=mri_dep_y_sig_composites.index,
)
mri_dep_y_sig_scaled["participant_id"] = mri_dep_y_sig_composites["participant_id"]

scaler_p = StandardScaler()
mri_dep_p_sig_scaled = pd.DataFrame(
    scaler_p.fit_transform(mri_dep_p_sig_composites.drop(columns=["participant_id"])),
    columns=mri_dep_p_sig_composites.drop(columns=["participant_id"]).columns,
    index=mri_dep_p_sig_composites.index,
)
mri_dep_p_sig_scaled["participant_id"] = mri_dep_p_sig_composites["participant_id"]

# WHITEN AND WEIGH MRI FEATURES
# Calculate covariance matrix for whitening using Mahalanobis distance
mri_identifier_cols_y = [col for col in mri_dep_y_sig_scaled.columns if col in {"subject", "subject_ids", "participant_id"}]
mri_identifier_cols_p = [col for col in mri_dep_p_sig_scaled.columns if col in {"subject", "subject_ids", "participant_id"}]

mri_feature_cols_y = [col for col in mri_dep_y_sig_scaled.columns if col not in mri_identifier_cols_y]
mri_feature_cols_p = [col for col in mri_dep_p_sig_scaled.columns if col not in mri_identifier_cols_p]

mri_feature_matrix_y = mri_dep_y_sig_scaled[mri_feature_cols_y].to_numpy(dtype=float)
mri_feature_matrix_p = mri_dep_p_sig_scaled[mri_feature_cols_p].to_numpy(dtype=float)

Sigma_y = np.cov(mri_feature_matrix_y, rowvar=False)
Sigma_p = np.cov(mri_feature_matrix_p, rowvar=False)
lw_y = LedoitWolf().fit(mri_dep_y_sig_scaled[mri_feature_cols_y].to_numpy(dtype=float))
lw_p = LedoitWolf().fit(mri_dep_p_sig_scaled[mri_feature_cols_p].to_numpy(dtype=float))
Sigma_y = lw_y.covariance_
Sigma_p = lw_p.covariance_

# Compute whitening matrix
eigvals_y, eigvecs_y = eigh(Sigma_y)
eigvals_p, eigvecs_p = eigh(Sigma_p)
eigvals_y = np.clip(eigvals_y, a_min=1e-10, a_max=None)
eigvals_p = np.clip(eigvals_p, a_min=1e-10, a_max=None)
W_pca_y = eigvecs_y @ np.diag(1.0 / np.sqrt(eigvals_y)) @ eigvecs_y.T
W_pca_p = eigvecs_p @ np.diag(1.0 / np.sqrt(eigvals_p)) @ eigvecs_p.T

# Get subject labels for resampled mri data to ensure alignment with whitened data
subject_labels_y = mri_dep_y_sig_scaled["participant_id"].reset_index(drop=True)
subject_labels_p = mri_dep_p_sig_scaled["participant_id"].reset_index(drop=True)

# Apply whitening to the normalised mri data
mri_dep_y_whitened = pd.DataFrame(
    mri_dep_y_sig_scaled[mri_feature_cols_y].to_numpy(dtype=float) @ W_pca_y,
    columns=mri_feature_cols_y,
    index=mri_dep_y_sig_scaled.index,
)
mri_dep_p_whitened = pd.DataFrame(
    mri_dep_p_sig_scaled[mri_feature_cols_p].to_numpy(dtype=float) @ W_pca_p,
    columns=mri_feature_cols_p,
    index=mri_dep_p_sig_scaled.index,
)

# Sanity check
print(np.cov(mri_dep_y_whitened.to_numpy(dtype=float), rowvar=False))
print(np.cov(mri_dep_p_whitened.to_numpy(dtype=float), rowvar=False))

def _apply_effect_size_weights(whitened_df, sig_df, composite_dict, label):
    weighted_df = whitened_df.copy()
    for col in weighted_df.columns:
        if col in sig_df["feature"].values:
            effect_size = sig_df.loc[sig_df["feature"] == col, "effect_size"].values[0]
            weight = abs(effect_size)
        elif col in composite_dict:
            individual_rois = composite_dict[col]
            effect_sizes = sig_df[sig_df["feature"].isin(individual_rois)]["effect_size"].values
            if effect_sizes.size == 0:
                print(f"Warning: No significant ROIs found for composite {col} in {label}. Skipping weighing for this column.")
                continue
            weight = np.mean(np.abs(effect_sizes))
        else:
            print(f"Warning: Column {col} not found in significant features or composite dictionary for {label}. Skipping weighing for this column.")
            continue

        weighted_df[col] *= weight

    return weighted_df

# Apply weighing to the normalised mri data based on the absolute effect sizes of the significant ROIs by multiplying each ROI with its absolute effect size
mri_dep_y_weighted = _apply_effect_size_weights(mri_dep_y_whitened, mri_dep_y_sig, mri_dep_y_sig_composite_dict, "youth")
# Reattach subject labels to wighted mri data
mri_dep_y_weighted["subject"] = subject_labels_y

mri_dep_p_weighted = _apply_effect_size_weights(mri_dep_p_whitened, mri_dep_p_sig, mri_dep_p_sig_composite_dict, "parent")
# Reattach subject labels to wighted mri data
mri_dep_p_weighted["subject"] = subject_labels_p

# Create scatter plot for weighted and whitened mri data in 2D PCA space for parent significant ROIs and mark depressed subjects
pca_p = PCA(n_components=2, random_state=42)
mri_dep_p_weighted_pca = pca_p.fit_transform(mri_dep_p_weighted.drop(columns=["subject"]))
mri_dep_p_weighted_pca_df = pd.DataFrame(mri_dep_p_weighted_pca, columns=["PCA_1", "PCA_2"])
mri_dep_p_weighted_pca_df["depressed_parent"] = mri_data_filtered["mh_p_ksads__dep__mdd__pres_dx"].reset_index(drop=True)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=mri_dep_p_weighted_pca_df, x="PCA_1", y="PCA_2", hue="depressed_parent", palette={0: "blue", 1: "red"}, alpha=0.7)
plt.title("PCA of Weighted and Whitened Parent Significant ROIs (Depressed vs Non-Depressed)")
plt.savefig(os.path.join(baseline_output_path, "mri_dep_p_weighted_pca.png"))
plt.close()


# NORMATIVE MODELING
# Prepare data for normative modeling
# Filter to only include healthy subjects (no depression diagnosis according to youth KSADS questionnaire) for normative modeling
healthy_subjects_y = mri_data_filtered[mri_data_filtered["mh_y_ksads__dep__mdd__pres_dx"] == 0]
healthy_subjects_p = mri_data_filtered[mri_data_filtered["mh_p_ksads__dep__mdd__pres_dx"] == 0]

# Define response variables (significant ROIs) for youth and parent
roi_cols_y = mri_dep_y_sig_filtered_rois
roi_cols_p = mri_dep_p_sig_filtered_rois

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
        dataframe=mri_data_filtered,
        covariates=["visit_age", "age_squared", "mr_y_smri__vol__aseg__icv_sum"],
        batch_effects=["scan_site", "sex"],
        response_vars=roi_cols_y,
        subject_ids="participant_id",
        remove_Nan=True,
    )

data_full_p = NormData.from_dataframe(
        name="mri_norm_full_p",
        dataframe=mri_data_filtered,
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

# Create z-score composites for youth and parent
z_scores_y_composites, z_scores_y_composite_dict = create_composites(z_scores_y[roi_cols_y], overwrite = True, composite_output = os.path.join(normative_output_dir_str_y, "z_scores_y_composites.csv"))
z_scores_y_composites["participant_id"] = z_scores_y["subject_ids"]
z_scores_p_composites, z_scores_p_composite_dict = create_composites(z_scores_p[roi_cols_p], overwrite = True, composite_output = os.path.join(normative_output_dir_str_p, "z_scores_p_composites.csv"))
z_scores_p_composites["participant_id"] = z_scores_p["subject_ids"]

# Plot z-score distributions in 2D PaCMAP space for youth significant ROIs and mark depressed subjects according to youth KSADS questionnaire
pacmap_y = PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
z_scores_y_pacmap = pacmap_y.fit_transform(z_scores_y[roi_cols_y])
z_scores_y_pacmap_df = pd.DataFrame(z_scores_y_pacmap, columns=["PaCMAP_1", "PaCMAP_2"])
z_scores_y_pacmap_df["depressed_youth"] = mri_data_filtered["mh_y_ksads__dep__mdd__pres_dx"].reset_index(drop=True)
plt.figure(figsize=(10, 8))
sns.scatterplot(data=z_scores_y_pacmap_df, x="PaCMAP_1", y="PaCMAP_2", hue="depressed_youth", palette={0: "blue", 1: "red"}, alpha=0.7)
plt.title("PaCMAP of Z-scores for Youth Significant ROIs (Depressed vs Non-Depressed)")
plt.savefig(os.path.join(normative_output_dir_str_y, "z_scores_y_pacmap.png"))
plt.close()

# For each ROI, plot z-score distribution for youth and mark depressed subjects according to youth KSADS questionnaire
z_scores_y = z_scores_y.merge(
    mri_data_filtered[["participant_id", "mh_y_ksads__dep__mdd__pres_dx"]],
    left_on="subject_ids",
    right_on="participant_id",
    how="left",
)
diagnosis_col = "mh_y_ksads__dep__mdd__pres_dx"

for roi in roi_cols_y:
    # Sort z-score and diagnosis together so they stay aligned
    plot_df = z_scores_y[[roi, diagnosis_col]].sort_values(roi).reset_index(drop=True)
    healthy = plot_df[plot_df[diagnosis_col] == 0]
    depressed = plot_df[plot_df[diagnosis_col] == 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(healthy.index, healthy[roi], c="lightgray", alpha=0.5, s=15, label="Healthy")
    plt.scatter(depressed.index, depressed[roi], c="crimson", alpha=0.9, s=30,
                edgecolor="black", linewidth=0.3, label="Depressed", zorder=5)

    plt.axhline(0, color="black", lw=1, linestyle="--", alpha=0.6)
    plt.axhline(1.96, color="gray", lw=0.8, linestyle=":")
    plt.axhline(-1.96, color="gray", lw=0.8, linestyle=":")

    plt.title(f"Z-score distribution for {roi} (Youth KSADS)")
    plt.xlabel("Subjects (sorted by Z-score)")
    plt.ylabel("Z-score")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(normative_output_dir_str_y, f"z_score_distribution_{roi}_youth.png"))
    plt.close()

# For each ROI, plot z-score distribution for parent and mark depressed subjects according to parent KSADS questionnaire
z_scores_p = z_scores_p.merge(
    mri_data_filtered[["participant_id", "mh_p_ksads__dep__mdd__pres_dx"]],
    left_on="subject_ids",
    right_on="participant_id",
    how="left",
)
diagnosis_col = "mh_y_ksads__dep__mdd__pres_dx"

for roi in roi_cols_p:
    # Sort z-score and diagnosis together so they stay aligned
    plot_df = z_scores_p[[roi, diagnosis_col]].sort_values(roi).reset_index(drop=True)
    healthy = plot_df[plot_df[diagnosis_col] == 0]
    depressed = plot_df[plot_df[diagnosis_col] == 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(healthy.index, healthy[roi], c="lightgray", alpha=0.5, s=15, label="Healthy")
    plt.scatter(depressed.index, depressed[roi], c="crimson", alpha=0.9, s=30,
                edgecolor="black", linewidth=0.3, label="Depressed", zorder=5)

    plt.axhline(0, color="black", lw=1, linestyle="--", alpha=0.6)
    plt.axhline(1.96, color="gray", lw=0.8, linestyle=":")
    plt.axhline(-1.96, color="gray", lw=0.8, linestyle=":")

    plt.title(f"Z-score distribution for {roi} (Parent KSADS)")
    plt.xlabel("Subjects (sorted by Z-score)")
    plt.ylabel("Z-score")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(normative_output_dir_str_p, f"z_score_distribution_{roi}_parent.png"))
    plt.close()

# BASELINE CLASSIFICATION
# Check overlap between subjects with fibtit features and subjects with mri features
# Filter subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = filter_subjects(dta_path, dta_path_tabular, test=False, overwrite=False)

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df, overwrite=False)

# Get fitbit data for subject timepoint pairs in the filtered mri dataset
# Get subject timepoint pairs for filtered mri dataset
subject_timepoint_pairs = mri_data_filtered[["participant_id", "session_id"]]
subject_timepoint_pairs.rename(columns={"participant_id": "subject", "session_id": "timepoint"}, inplace=True)

mri_data_filtered = None

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
        subject_fitbit_df = con.execute(query).fetchdf()
        fitbit_metric_cols = [col for col in subject_fitbit_df.columns if col not in ["subject", "timepoint", "Wear_Time"]]
        for col in fitbit_metric_cols:
            subject_fitbit_df[col] = pd.to_numeric(subject_fitbit_df[col], errors="coerce")
        feature_dict = {"subject": row.subject}
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

fitbit_features = extract_fitbit_features_2(con, subject_timepoint_pairs, output_path=os.path.join(baseline_output_path, "fitbit_features.csv"), overwrite=True)
fitbit_features.to_csv(os.path.join(baseline_output_path, "fitbit_features.csv"), index=False)

# Make sure that the fitbit features are aligned with the filtered mri dataset by merging on subject and timepoint
fitbit_features_filtered = fitbit_features.merge(subject_timepoint_pairs, left_on=["subject", "timepoint"], right_on=["subject", "timepoint"], how="inner")

# Create composites of fitbit features
fitbit_features_composites, fitbit_features_composite_dict = create_composites(fitbit_features_filtered.drop(columns=["subject", "timepoint"]), overwrite=True, composite_output=os.path.join(baseline_output_path, "fitbit_features_composites.csv"))
fitbit_features_composites["subject"] = fitbit_features_filtered["subject"]
fitbit_features_composites["timepoint"] = fitbit_features_filtered["timepoint"]

fitbit_features_composites.to_csv(os.path.join(baseline_output_path, "fitbit_features_composites.csv"), index=False)

# Add depression diagnosis labels based on youth and parent KSADS questionnaires to the fitbit data
fitbit_features_filtered = fitbit_features_filtered.merge(mri_data_filtered[["participant_id", "session_id", "mh_y_ksads__dep__mdd__pres_dx", "mh_p_ksads__dep__mdd__pres_dx"]], left_on=["subject", "timepoint"], right_on=["participant_id", "session_id"], how="left")

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
print(f"Number of subjects with depression diagnosis according to youth KSADS questionnaire: {num_depressed_youth}")
print(f"Number of subjects with depression diagnosis according to parent KSADS questionnaire: {num_depressed_parent}")












# UNSUPERVISED CLUSTERING
# Conduct unsupervised clustering on the normalized mri data of significant ROIs for youth KSADS questionnaire
labels_y = mri_clustering(mri_dep_y_sig_scaled,
                          dr=["PaCMAP", "PCA"],
                          cl=["HDBSCAN"],
                          n_clusters=2,
                          output_path=os.path.join(baseline_output_path),
                          clustering_output="labels_y.csv",
                          bootstrapping=False,
                          overwrite=True
                          )

# For each discovered cluster, get size and overlap with depressed subjects according to youth KSADS questionnaire
for cluster in np.unique(labels_y["label"]):
    cluster_size = np.sum(labels_y["label"] == cluster)
    depressed_overlap = np.sum((labels_y["label"] == cluster) & (mri_data_filtered["mh_y_ksads__dep__mdd__pres_dx"].reset_index(drop=True) == 1))
    print(f"Cluster {cluster}: Size = {cluster_size}, Overlap with depressed subjects = {depressed_overlap}")

# Conduct unsupervised clustering on the normalized mri data of significant ROIs for parent KSADS questionnaire
labels_p = mri_clustering(mri_dep_p_sig_scaled,
                          dr=["PaCMAP", "PCA"],
                          cl=["AgglomerativeClustering", "HDBSCAN"],
                          n_clusters=2,
                          output_path=os.path.join(baseline_output_path),
                          clustering_output="labels_p.csv",
                          bootstrapping=False,
                          overwrite=True
                          )

# For each discovered cluster, get size and overlap with depressed subjects according to parent KSADS questionnaire
for cluster in np.unique(labels_p["label"]):
    cluster_size = np.sum(labels_p["label"] == cluster)
    depressed_overlap = np.sum((labels_p["label"] == cluster) & (mri_data_filtered["mh_p_ksads__dep__mdd__pres_dx"].reset_index(drop=True) == 1))
    print(f"Cluster {cluster}: Size = {cluster_size}, Overlap with depressed subjects = {depressed_overlap}")

# Conduct unsupervised clustering on the weighted and whitened mri data of significant ROIs for youth KSADS questionnaire
labels_y_weighted = mri_clustering(mri_dep_y_weighted.drop(columns=["subject"]),
                          dr=["PaCMAP", "PCA"],
                          cl=["Agglomerative", "KMeans"],
                          n_clusters=2,
                          output_path=os.path.join(baseline_output_path),
                          bootstrapping=False,
                          overwrite=True
                          )

# For each discovered cluster, get size and overlap with depressed subjects according to youth KSADS questionnaire
for cluster in np.unique(labels_y_weighted):
    cluster_size = np.sum(labels_y_weighted == cluster)
    depressed_overlap = np.sum((labels_y_weighted == cluster) & (mri_data_filtered["mh_y_ksads__dep__mdd__pres_dx"].reset_index(drop=True) == 1))
    print(f"Weighted Cluster {cluster}: Size = {cluster_size}, Overlap with depressed subjects = {depressed_overlap}")

# Conduct unsupervised clustering on the weighted and whitened mri data of significant ROIs for parent KSADS questionnaire
labels_p_weighted = mri_clustering(mri_dep_p_weighted.drop(columns=["subject"]),
                          dr=["PaCMAP", "PCA"],
                          cl=["Agglomerative", "KMeans"],
                          n_clusters=2,
                          output_path=os.path.join(baseline_output_path),
                          bootstrapping=False,
                          overwrite=True
                          )