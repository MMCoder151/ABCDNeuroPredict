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
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.impute import IterativeImputer
from neuroHarmonize import harmonizationLearn

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

# ---- DATA WRANGLING & PREPROCESSING----

# Filter subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = filter_subjects(dta_path, dta_path_tabular, test=False, overwrite=False)

# Print descriptive statistics of filtered subjects
describe_subjects(fit_meta_df, mri_meta_df)

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df, overwrite=False)

# Use a subject-level MRI label table for all subject counts.
mri_subject_labels = mri_meta_df[["subject", "dep_dx"]].drop_duplicates(subset=["subject"])

# MISSINGNESS ANALYSIS
# Calculate association of missingness in fitbit data with diagnosis group using logistic regression 
missingness_df = fit_meta_df[["subject", "missing_days_percentage"]].merge(mri_subject_labels, on="subject")
missingness_df = missingness_df.dropna(subset=["missing_days_percentage", "dep_dx"])
X_missingness = missingness_df[["missing_days_percentage"]].astype(np.float64)
y_missingness = missingness_df["dep_dx"]
model_missingness = LogisticRegression(max_iter=1000)
model_missingness.fit(X_missingness, y_missingness)
print("Missingness association with diagnosis group:", pd.Series(model_missingness.coef_.ravel(), index=X_missingness.columns))

##############################
# ---- FEATURE EXTRACTION ----
##############################

###########################
# ACTIVITY FEATURE EXTRACTION
###########################

# Extract features from fitbit data
fitbit_features_df = extr_fitbit_features(con, dem_df, overwrite=False)

# Analyse feature colinearity using Variance Inflation Factor (VIF) and create composite scores to account for multicollinearity
fitbit_features_with_composites, composite_dict = create_composites(fitbit_features_df, overwrite=False, composite_output = "fitbit_composites")

# Reappend subject column to fitbit_features_with_composites 
fitbit_features_with_composites = fitbit_features_with_composites.merge(fitbit_features_df[["subject"]], left_index=True, right_index=True, how="left")

# ADD SMARTPHONE FEATURES
# Read in precomputed smartphone features and merge with fitbit features
earsapp_features = pd.read_csv(os.path.join(dta_path_tabular, "nt_y_earsapp.tsv"), sep="\t")

earsapp_features.shape

# Filter to only include subjects and timepoints that are present in fit_meta_df and mri_meta_df
earsapp_features_filtered = earsapp_features[
    (earsapp_features["participant_id"].isin(fit_meta_df["subject"])) &
    (earsapp_features["session_id"].isin(fit_meta_df["timepoint"]))
]

earsapp_features_filtered.shape

# Get number of depressed subjects in earsapp_features_filtered
num_depressed_subjects = earsapp_features_filtered["participant_id"].isin(mri_subject_labels[mri_subject_labels["dep_dx"] == 1]["subject"]).sum()
print(f"Number of depressed subjects in earsapp_features_filtered: {num_depressed_subjects}")

# NOTE: No sufficient overlap with fitbit data nor with depressed subjects to include

######################
# EXTRACT MRI FEATURES
######################

# Get mri rois that show significant differences between depressed and non-depressed subjects
mri_rois_sig, mri_rois_results = extract_mri_rois(dta_path_tabular, dta_path, mri_meta_df, overwrite=False)

# Filter mri_rois_sig to only inclue ROIs with effect size smaller than -0.2 or greater than 0.2
mri_rois_sig_filtered = mri_rois_results.loc[(mri_rois_results["effect_size"] < -0.2) | (mri_rois_results["effect_size"] > 0.2), "mri_feature"].tolist()
print(f"\nNumber of significant MRI ROIs with effect size < -0.2 or > 0.2: {len(mri_rois_sig_filtered)}")
print(f"Significant MRI ROIs with effect size < -0.2 or > 0.2:")
for roi in mri_rois_sig_filtered:
    effect_size = mri_rois_results.loc[mri_rois_results["mri_feature"] == roi, "effect_size"].values[0]
    std = mri_rois_results.loc[mri_rois_results["mri_feature"] == roi, "effect_size_std"].values[0]
    print(f"  {roi}: {effect_size:.4f} (std: {std:.4f})")

# Get raw mri data for subjects in mri_meta_df at the first timepoint
subjects_in_meta = mri_meta_df["subject"].unique().tolist()
query = f"""
        SELECT "subject", {', '.join(f'"{col}"' for col in mri_rois_sig_filtered)}
        FROM mri_data
        WHERE subject IN ({', '.join(f"'{sub}'" for sub in subjects_in_meta)})
        AND timepoint IN (
            SELECT MIN(timepoint)
            FROM mri_data AS sub_mri
            WHERE sub_mri.subject = mri_data.subject
        )
    """
raw_mri_data = con.execute(query).df()

# Calculate correlation matrix of raw mri data and save to CSV
correlation_matrix = raw_mri_data.drop(columns=["subject"]).corr()
correlation_matrix.to_csv(os.path.join(output_path, "raw_mri_correlation_matrix.csv"))

# Print features with a correlation over 0.8
highly_correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            highly_correlated_features.add(correlation_matrix.columns[i])
            highly_correlated_features.add(correlation_matrix.columns[j])

print(f"Highly correlated features (correlation > 0.8): {highly_correlated_features}")

# Drop highly correlated MRI feature 
raw_mri_data = raw_mri_data.drop(columns=["mr_y_rsfmri__corr__gpnet__smh__smm_mean"])

#########################
# ---- PREPROCESSING ----
#########################

# Check if there are duplicate subjects in fitbit_features_with_composites
duplicate_subjects_fitbit = fitbit_features_with_composites[fitbit_features_with_composites.duplicated(subset=["subject"], keep=False)]
print(f"Number of duplicate subjects in fitbit features: {len(duplicate_subjects_fitbit)}")

# Check subject overlap between fitbit features and raw mri data
fitbit_subjects = set(fitbit_features_with_composites["subject"])
mri_subjects = set(raw_mri_data["subject"])
overlap_subjects = fitbit_subjects.intersection(mri_subjects)
print(f"Number of subjects in fitbit features: {len(fitbit_subjects)}")
print(f"Number of subjects in raw mri data: {len(mri_subjects)}")
print(f"Number of overlapping subjects between fitbit features and raw mri data: {len(overlap_subjects)}")

# Drop subjects that aren't in both fitbit features and raw mri data
fitbit_features_with_composites = fitbit_features_with_composites[fitbit_features_with_composites["subject"].isin(overlap_subjects)]
raw_mri_data = raw_mri_data[raw_mri_data["subject"].isin(overlap_subjects)]

# NOTE: I am not sure why the number of subjects isn't even despite the rigurous filtering during data wrangling.

# OUTLIER DETECTION AND REMOVAL
# Conduct outlier detection using IsolationForest on fitbit features of subjects without depression diagnosis and remove outliers
iso_forest_fit = IsolationForest(contamination=0.005, random_state=42)
fitbit_features_with_composites_no_dep = fitbit_features_with_composites[fitbit_features_with_composites["subject"].isin(mri_subject_labels[mri_subject_labels["dep_dx"] == 0]["subject"])]
outlier_mask_fit = iso_forest_fit.fit_predict(fitbit_features_with_composites_no_dep.drop(columns=["subject"]))
fitbit_outliers = set(fitbit_features_with_composites_no_dep.loc[outlier_mask_fit == -1, "subject"])
print(f"Number of outliers detected in fitbit features: {np.sum(outlier_mask_fit == -1)}")

# Conduct outlier detection using IsolationForest on raw mri data of subjects without depression diagnosis and remove outliers
iso_forest_mri = IsolationForest(contamination=0.005, random_state=42)
raw_mri_data_no_dep = raw_mri_data[raw_mri_data["subject"].isin(mri_subject_labels[mri_subject_labels["dep_dx"] == 0]["subject"])]
outlier_mask_mri = iso_forest_mri.fit_predict(raw_mri_data_no_dep.drop(columns=["subject"]))
mri_outliers = set(raw_mri_data_no_dep.loc[outlier_mask_mri == -1, "subject"])
print(f"Number of outliers detected in raw mri data: {np.sum(outlier_mask_mri == -1)}")

# Check overlap between outliers detected in fitbit features and raw mri data
outlier_subjects_fit = fitbit_features_with_composites_no_dep.loc[outlier_mask_fit == -1, "subject"]
outlier_subjects_mri = raw_mri_data_no_dep.loc[outlier_mask_mri == -1, "subject"]
overlap_outliers = set(outlier_subjects_fit).intersection(set(outlier_subjects_mri))
print(f"Number of overlapping outliers detected in both fitbit features and raw mri data: {len(overlap_outliers)}")

# Create scatter plot in 2D PCA space of outliers detected in fitbit features and raw mri data
pca = PCA(n_components=2)
fitbit_pca = pca.fit_transform(fitbit_features_with_composites_no_dep.drop(columns=["subject"]))
mri_pca = pca.transform(raw_mri_data_no_dep.drop(columns=["subject"]))

plt.figure(figsize=(10, 8))
plt.scatter(fitbit_pca[:, 0], fitbit_pca[:, 1], c='blue', label='Fitbit Outliers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Fitbit Outliers in 2D PCA Space')
plt.legend()
plt.savefig(os.path.join(output_path, "fitbit_outliers_pca_scatter.png"))
plt.close()

plt.figure(figsize=(10, 8))
plt.scatter(mri_pca[:, 0], mri_pca[:, 1], c='red', label='MRI Outliers')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('MRI Outliers in 2D PCA Space')
plt.legend()
plt.savefig(os.path.join(output_path, "mri_outliers_pca_scatter.png"))
plt.close()

# Remove outliers in either fitbit or raw mri data
outlier_subjects = fitbit_outliers | mri_outliers
print(f"Total unique outlier subjects: {len(outlier_subjects)}")
fitbit_features_with_composites = fitbit_features_with_composites[~fitbit_features_with_composites["subject"].isin(outlier_subjects)]
raw_mri_data = raw_mri_data[~raw_mri_data["subject"].isin(outlier_subjects)]

# IMPUTATION OF MISSING VALUES
# Impute missing values in fitbit features using IterativeImputer
imputer = IterativeImputer(random_state=42)
fitbit_features_with_composites_imputed = imputer.fit_transform(fitbit_features_with_composites.drop(columns=["subject"]))
# Get amount of imputed data in fitbit features
imputed_count_fitbit = np.sum(np.isnan(fitbit_features_with_composites.drop(columns=["subject"]).values), axis=0)
print(f"Number of imputed values in fitbit features: {imputed_count_fitbit}")

fitbit_feature_cols = fitbit_features_with_composites.drop(columns=["subject"]).columns

fitbit_features_with_composites_imputed = pd.DataFrame(
    fitbit_features_with_composites_imputed,
    columns=fitbit_feature_cols,
    index=fitbit_features_with_composites.index  # preserve original row alignment
)

fitbit_features_with_composites_imputed["subject"] = fitbit_features_with_composites["subject"]

# Impute missing values in raw mri data using IterativeImputer
imputer = IterativeImputer(random_state=42)
raw_mri_data_imputed = imputer.fit_transform(raw_mri_data.drop(columns=["subject"]))
# Get amount of imputed data in raw mri data
imputed_count_mri = np.sum(np.isnan(raw_mri_data.drop(columns=["subject"]).values), axis=0)
print(f"Number of imputed values in raw mri data: {imputed_count_mri}")

mri_feature_cols = raw_mri_data.drop(columns=["subject"]).columns

raw_mri_data_imputed = pd.DataFrame(
    raw_mri_data_imputed,
    columns=mri_feature_cols,
    index=raw_mri_data.index  # preserve original row alignment
)

raw_mri_data_imputed["subject"] = raw_mri_data["subject"]

# SITE HARMONIZATION OF MRI DATA
# Regress out site effects from the raw mri data using neuroHarmonize
covars = pd.DataFrame({
    "SITE": dem_df.loc[raw_mri_data_imputed.index, "scan_site"].values,
    "AGE": dem_df.loc[raw_mri_data_imputed.index, "age_at_first_mri"].values,
    "SEX": dem_df.loc[raw_mri_data_imputed.index, "sex"].values,
    "TIV": mri_meta_df.loc[raw_mri_data_imputed.index, "mr_y_smri__vol__aseg__icv_sum"].values,
})

covars["SEX"] = covars["SEX"].map({"M": 0, "F": 1})

imputer = IterativeImputer(random_state=42)
covars = pd.DataFrame(
    imputer.fit_transform(covars),
    columns=covars.columns,
    index=covars.index,
)

covars["SITE"] = covars["SITE"].astype(str)

model, raw_mri_data_res = harmonizationLearn(raw_mri_data_imputed.drop(columns=["subject"]).to_numpy(dtype=float), covars, smooth_terms=["AGE"])
raw_mri_data_res = pd.DataFrame(raw_mri_data_res, columns=raw_mri_data_imputed.drop(columns=["subject"]).columns, index=raw_mri_data_imputed.index)
raw_mri_data_res["subject"] = raw_mri_data_imputed["subject"]

# Conduct confound analysis of raw mri data pre and post residualization
confound_effects_mri_df = analyse_confounds(dem_df, mri_meta_df, transformed_data=raw_mri_data_res, raw_data=raw_mri_data_imputed)
confound_effects_mri_df.to_csv(os.path.join(output_path, "confound_effects_mri.csv"), index=False)

print(model['gamma_hat'].shape)
print(pd.Series(model['gamma_hat'].flatten()).describe())
print(model['delta_hat'].shape)
print(pd.Series(model['delta_hat'].flatten()).describe())

# PREPARE FITBIT FEATURES FOR MODELING
# Add sex and age to selected_subjects_with_composites for modeling
features = fitbit_features_with_composites.merge(dem_df[["subject", "sex", "age_at_first_mri"]], left_on="subject", right_on="subject", how="left")
features["sex"] = features["sex"].map({"M": 0, "F": 1})
features["sex"] = features["sex"].astype(np.float64)
features["age_at_first_mri"] = features["age_at_first_mri"].astype(np.float64)

# Attach subject-level labels before splitting so features, and diagnosis stay aligned.
subject_labels = (
    dem_df[["subject"]]
    .drop_duplicates(subset=["subject"])
    .merge(
        mri_meta_df[["subject", "dep_dx"]].drop_duplicates(subset=["subject"]),
        on="subject",
        how="left",
    )
)

# Train-Test Split
train_X, test_X, train_labels, test_labels = train_test_split(
    features,
    subject_labels,
    test_size=0.2,
    stratify=subject_labels["dep_dx"],
    random_state=42,
)

# Create labels for train and test sets based on depression diagnosis in mri_meta_df
train_y_dx = train_labels[["subject", "dep_dx"]].reset_index(drop=True)
test_y_dx = test_labels[["subject", "dep_dx"]].reset_index(drop=True)

# Save features to CSV
train_X.to_csv(os.path.join(output_path, "train_features.csv"), index=False)
test_X.to_csv(os.path.join(output_path, "test_features.csv"), index=False)
train_y_dx.to_csv(os.path.join(output_path, "train_labels_dx.csv"), index=False)
test_y_dx.to_csv(os.path.join(output_path, "test_labels_dx.csv"), index=False)

# OPTIONAL: Reimport features and labels from CSV for modeling
train_X = pd.read_csv(os.path.join(output_path, "train_features.csv"))
test_X = pd.read_csv(os.path.join(output_path, "test_features.csv"))
train_y_dx = pd.read_csv(os.path.join(output_path, "train_labels_dx.csv"))
test_y_dx = pd.read_csv(os.path.join(output_path, "test_labels_dx.csv"))

# Z-SCORE NORMALIZE DATA
# z-score the mri data
scaler_mri = StandardScaler()
raw_mri_data_norm = pd.DataFrame(
    scaler_mri.fit_transform(raw_mri_data_res.drop(columns=["subject"])),
    columns=raw_mri_data_res.drop(columns=["subject"]).columns,
    index=raw_mri_data_res.index,
)
raw_mri_data_norm["subject"] = raw_mri_data["subject"]

# z-score the fitbit features
scaler_fitbit = StandardScaler()
fitbit_features_with_composites_norm = pd.DataFrame(
    scaler_fitbit.fit_transform(train_X.drop(columns=["subject"])),
    columns=train_X.drop(columns=["subject"]).columns,
    index=train_X.index,
)
fitbit_features_with_composites_norm["subject"] = train_X["subject"]

# RESAMPLE FOR BALANCED CLASS DISTRIBUTION
# Resample raw mri data for balanced class distribution
raw_mri_subject_labels = raw_mri_data[["subject"]].drop_duplicates().merge(mri_subject_labels, on="subject", how="left")
raw_mri_data_resampled, _ = resample(raw_mri_data_norm, raw_mri_subject_labels["dep_dx"])

# Resample fitbit features for balanced class distribution
fitbit_features_labels = fitbit_features_with_composites_norm[["subject"]].drop_duplicates().merge(mri_subject_labels, on="subject", how="left")
fitbit_features_with_composites_resampled, _ = resample(fitbit_features_with_composites_norm, train_y_dx["dep_dx"])

# Resample fitbit training data
train_X_resampled, train_y_dx_resampled = resample(train_X, train_y_dx["dep_dx"])

# Save resampled training data to CSV
train_X_resampled.to_csv(os.path.join(output_path, "train_features_resampled.csv"), index=False)
train_y_dx_resampled.to_csv(os.path.join(output_path, "train_labels_dx_resampled.csv"), index=False)

# OPTIONAL: Reimport resampled training data from CSV for modeling
train_X_resampled = pd.read_csv(os.path.join(output_path, "train_features_resampled.csv"))
train_y_dx_resampled = pd.read_csv(os.path.join(output_path, "train_labels_dx_resampled.csv"))

# WHITEN AND WEIGH MRI FEATURES
# Calculate covariance matrix for whitening using Mahalanobis distance
mri_identifier_cols = [col for col in raw_mri_data_resampled.columns if col in {"subject", "subject_ids"}]
mri_feature_cols = [col for col in raw_mri_data_resampled.columns if col not in mri_identifier_cols]
mri_feature_matrix = raw_mri_data_resampled[mri_feature_cols].to_numpy(dtype=float)
Sigma = np.cov(mri_feature_matrix, rowvar=False)
lw = LedoitWolf().fit(raw_mri_data_resampled[mri_feature_cols].to_numpy(dtype=float))
Sigma = lw.covariance_

# Compute whitening matrix
eigvals, eigvecs = eigh(Sigma)
eigvals = np.clip(eigvals, a_min=1e-10, a_max=None)
W_pca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

# Get subject labels for resampled mri data to ensure alignment with whitened data
subject_labels = raw_mri_data_resampled["subject_ids"].reset_index(drop=True)

# Apply whitening to the normalised mri data
mri_data_whitened = pd.DataFrame(
    raw_mri_data_resampled[mri_feature_cols].to_numpy(dtype=float) @ W_pca,
    columns=mri_feature_cols,
    index=raw_mri_data_resampled.index,
)

# Sanity check
print(np.cov(mri_data_whitened.to_numpy(dtype=float), rowvar=False))

# Apply weighing to the normalised mri data based on the absolute effect sizes of the significant ROIs by multiplying each ROI with its absolute effect size
mri_data_weighted = mri_data_whitened.copy()  
for col in mri_data_weighted.columns:
    effect_size = mri_rois_results.loc[mri_rois_results["mri_feature"] == col, "effect_size"].values[0]
    mri_data_weighted[col] *= abs(effect_size)

# Reattach subject labels to wighted mri data
mri_data_weighted["subject"] = subject_labels

# ---- BASELINE CLASSIFICATION MODEL ----

# BASELINE CLASSIFICATION MODELING
# Train and evaluate baseline classification models using nested cross-validation
final_model, best_hyperparams, train_predictions = train_final_model(
    train_X.drop(columns=["subject"]),
    (train_y_dx.drop(columns=["subject"])).squeeze(),
    model="SVM"
)

# Save final model, hyperparameters, and predictions to CSV
with open(os.path.join(baseline_output_path, "final_model.pkl"), "wb") as f:
    pickle.dump(final_model, f)
with open(os.path.join(baseline_output_path, "final_model_hyperparams.json"), "w") as f:
    json.dump(best_hyperparams, f, indent=4)
train_predictions_df = pd.DataFrame({
    "subject": train_X["subject"],
    "predicted_dep_dx": train_predictions
})
train_predictions_df.to_csv(os.path.join(baseline_output_path, "final_model_predictions.csv"), index=False)

# Get predictions on test set using the final model
test_predictions = final_model.predict(test_X.drop(columns=["subject"]))
test_predictions_df = pd.DataFrame({
    "subject": test_X["subject"],
    "predicted_dep_dx": test_predictions
})
test_predictions_df.to_csv(os.path.join(baseline_output_path, "final_model_test_predictions.csv"), index=False)

# Get confusion matrix of final model on the test set and save to CSV
confusion_matrix_df = pd.DataFrame(confusion_matrix(test_y_dx["dep_dx"], test_predictions), 
    index=["Actual_Negative", "Actual_Positive"], 
    columns=["Predicted_Negative", "Predicted_Positive"])
confusion_matrix_df.to_csv(os.path.join(baseline_output_path, "final_model_confusion_matrix.csv"))

# Calculate F1 score on test set
f1 = f1_score(test_y_dx["dep_dx"], test_predictions)
print(f"Final model F1 score on test set: {f1:.4f}")

# RESAMPLED CLASSIFICATION MODEL
# Train final model using the best model based on resampled cross-validation scores
final_resampled_model, best_resampled_hyperparams, train_resampled_predictions = train_final_model(
    train_X_resampled.drop(columns=["subject"]),
    (train_y_dx_resampled.drop(columns=["subject"])).squeeze(),
    model="SVM"
)

# Save final resampled model, hyperparameters, and predictions to CSV
with open(os.path.join(baseline_output_path, "final_resampled_model.pkl"), "wb") as f:
    pickle.dump(final_resampled_model, f)
with open(os.path.join(baseline_output_path, "final_resampled_model_hyperparams.json"), "w") as f:
    json.dump(best_resampled_hyperparams, f, indent=4)
train_resampled_predictions_df = pd.DataFrame({
    "subject": train_X_resampled["subject"],
    "predicted_dep_dx": train_resampled_predictions
})
train_resampled_predictions_df.to_csv(os.path.join(baseline_output_path, "final_resampled_model_predictions.csv"), index=False)

# Get predictions on test set using the final resampled model
test_resampled_predictions = final_resampled_model.predict(test_X.drop(columns=["subject"]))
test_resampled_predictions_df = pd.DataFrame({
    "subject": test_X["subject"],
    "predicted_dep_dx": test_resampled_predictions
})
test_resampled_predictions_df.to_csv(os.path.join(baseline_output_path, "final_resampled_model_test_predictions.csv"), index=False)

# Get confusion matrix of final resampled model on the test set and save to CSV
confusion_matrix_resampled_df = pd.DataFrame(confusion_matrix(test_y_dx["dep_dx"], test_resampled_predictions), 
    index=["Actual_Negative", "Actual_Positive"], 
    columns=["Predicted_Negative", "Predicted_Positive"])
confusion_matrix_resampled_df.to_csv(os.path.join(baseline_output_path, "final_resampled_model_confusion_matrix.csv"))

# Calculate F1 score on test set for resampled model
f1_resampled = f1_score(test_y_dx["dep_dx"], test_resampled_predictions)
print(f"Final resampled model F1 score on test set: {f1_resampled:.4f}")

# ---- UNSUPERVISED LABEL ASSIGNMENT ----

# NORMATIVE SELECTION OF MRI DATA
raw_mri_data_resampled.rename(columns={"subject_ids": "subject"}, inplace=True)
mri_rois_sig_filtered = [roi for roi in mri_rois_sig_filtered if roi in raw_mri_data_resampled.columns]
selected_subjects = normative_selection(data=raw_mri_data_resampled, roi_cols=mri_rois_sig_filtered, mri_meta_df=mri_meta_df, overwrite=True)

# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_subjects = normative_selection(data=raw_mri_data_resampled, roi_cols=mri_rois_sig_filtered, mri_meta_df=mri_meta_df, overwrite=True)

# Check overlap between normative selected subjects and subjects with depression diagnosis
depression_diagnosis_df = mri_meta_df[mri_meta_df["dep_dx"] == 1]
depression_diagnosis_df = (
    mri_meta_df[mri_meta_df["dep_dx"] == 1]
    .drop_duplicates(subset=["subject"])
)
overlap_subjects = set(selected_subjects["subject_ids"]).intersection(set(depression_diagnosis_df["subject"]))
print(f"Number of subjects selected by normative modeling with depression diagnosis: {len(overlap_subjects)}")
print(f"Overlap percentage: {len(overlap_subjects) / len(selected_subjects) * 100:.2f}%")

# Conduct confound analysis pre and post normative modeling
z_scores = pd.read_csv(os.path.join(output_path, "normative_modelling", "results","Z_mri_norm.csv"))
confound_effects_df = analyse_confounds(dem_df, z_scores, con = con, view = "mri_data")

# Print descriptive statistics of normative selected subjects
selected_fit_meta_df = fit_meta_df[fit_meta_df["subject"].isin(selected_subjects["subject_ids"])]
selected_mri_meta_df = mri_meta_df[mri_meta_df["subject"].isin(selected_subjects["subject_ids"])]
describe_subjects(selected_fit_meta_df, selected_mri_meta_df)

# Print descriptive statistics of non-selected subjects
non_selected_fit_meta_df = fit_meta_df[~fit_meta_df["subject"].isin(selected_subjects["subject_ids"])]
non_selected_mri_meta_df = mri_meta_df[~mri_meta_df["subject"].isin(selected_subjects["subject_ids"])]
describe_subjects(non_selected_fit_meta_df, non_selected_mri_meta_df)

# UNSUPERVISED CLUSTERING
# Conduct unsupervised clustering of resampled mri data for label assignment
subject_labels_resampled = mri_clustering(raw_mri_data_resampled, 
                                          n_clusters=2, 
                                          cl=["HDBSCAN", "AgglomerativeClustering"],
                                          dr=["PCA", "PaCMAP"],
                                          mri_meta_df=mri_meta_df,
                                          clustering_output="label_assignment_resampled", 
                                          bootstrapping=False, 
                                          overwrite=True)

# Per discovered subtype, get overlap with subjects with depression diagnosis
for subtype in subject_labels_resampled["label"].unique():
    subtype_subjects = subject_labels_resampled[subject_labels_resampled["label"] == subtype]["subject_ids"].tolist()
    depression_diagnosis_df = mri_meta_df[mri_meta_df["dep_dx"] == 1].drop_duplicates(subset=["subject"])
    overlap_subjects = set(subtype_subjects).intersection(set(depression_diagnosis_df["subject"]))
    print(f"\nLabel {subtype}:")
    print(f"Number of subjects in subtype: {len(subtype_subjects)}")
    print(f"Number of subjects in subtype with depression diagnosis: {len(overlap_subjects)}")
    print(f"Percentage of all depressed subjects in subtype: {len(overlap_subjects) / len(depression_diagnosis_df) * 100:.2f}%")

# Per discovered subtype, get overlap with subjects that exist in mri_meta_df
for subtype in subject_labels_resampled["label"].unique():
    subtype_subjects = subject_labels_resampled[subject_labels_resampled["label"] == subtype]["subject_ids"].tolist()
    overlap_subjects = set(subtype_subjects).intersection(set(mri_meta_df["subject"]))
    print(f"\nLabel {subtype}:")
    print(f"Number of subjects in subtype: {len(subtype_subjects)}")
    print(f"Number of subjects in subtype that exist in mri_meta_df: {len(overlap_subjects)}")
    print(f"Percentage of all subjects in mri_meta_df in subtype: {len(overlap_subjects) / len(mri_meta_df) * 100:.2f}%")

cluster_confound_df = (
    subject_labels_resampled[["subject_ids", "label"]]
    .drop_duplicates(subset=["subject_ids"])
    .merge(
        mri_meta_df[["subject", "sex", "scan_site", "age_at_mri", "mr_y_smri__vol__aseg__icv_sum"]].drop_duplicates(subset=["subject"]),
        left_on="subject_ids",
        right_on="subject",
        how="inner",
    )
)

from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, pointbiserialr

results = {}

# Categorical confounds: sex, scan_site
for confound in ["sex", "scan_site"]:
    sub = cluster_confound_df[["label", confound]].dropna()
    contingency = pd.crosstab(sub["label"], sub[confound])
    chi2, p, dof, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan
    results[confound] = {"chi2": chi2, "p": p, "cramers_v": cramers_v}

# Continuous confounds: age, ICV
for confound in ["age_at_mri", "mr_y_smri__vol__aseg__icv_sum"]:
    sub = cluster_confound_df[["label", confound]].dropna()
    g0 = sub.loc[sub["label"] == 0, confound]
    g1 = sub.loc[sub["label"] == 1, confound]
    t_stat, p = ttest_ind(g0, g1, equal_var=False)  # Welch's, safer default
    r_pb, p_pb = pointbiserialr(sub["label"], sub[confound])
    results[confound] = {"t_stat": t_stat, "p": p, "point_biserial_r": r_pb}

print(results)

# Conduct unsupervised clustering of WHITENED & WEIGHTED mri data for label assignment
subject_labels_weighted = mri_clustering(mri_data_whitened, 
                                         n_clusters=2, 
                                         dr=["PCA", "PaCMAP"],
                                         cl=["HDBSCAN", "AgglomerativeClustering"],
                                         mri_meta_df=mri_meta_df,
                                         clustering_output="label_assignment_weighted", 
                                         bootstrapping=False, 
                                         overwrite=True)

# Per discovered subtype, get overlap with subjects with depression diagnosis
for subtype in subject_labels_weighted["label"].unique():
    subtype_subjects = subject_labels_weighted[subject_labels_weighted["label"] == subtype]["subject_ids"].tolist()
    overlap_subjects = set(subtype_subjects).intersection(set(depression_diagnosis_df["subject"]))
    print(f"\nWeighted Label {subtype}:")
    print(f"Number of subjects in weighted label: {len(subtype_subjects)}")
    print(f"Number of subjects in weighted label with depression diagnosis: {len(overlap_subjects)}")
    print(f"Overlap percentage: {len(overlap_subjects) / len(subtype_subjects) * 100:.2f}%")









# UNSUPERVISED SUBTPYING
# Filter raw_mri_data_norm to only include subjects with depression diagnosis
depression_diagnosis_df = mri_meta_df[mri_meta_df["dep_dx"] == 1].drop_duplicates(subset=["subject"])
raw_mri_data_norm_dep = raw_mri_data_norm[raw_mri_data_norm["subject"].isin(depression_diagnosis_df["subject"])]

# visualise the data in 2D PaCMAP space for the normalized depressed subjects
pacmap_model = pacmap.PaCMAP(n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
pacmap_embedding = pacmap_model.fit_transform(raw_mri_data_norm_dep.drop(columns=["subject"]))
plt.figure(figsize=(8, 6))
plt.scatter(pacmap_embedding[:, 0], pacmap_embedding[:, 1], alpha=0.7)
plt.title('PaCMAP Embedding of Normalized Depressed Subjects')
plt.xlabel('PaCMAP1')
plt.ylabel('PaCMAP2')
plt.savefig(os.path.join(output_path, "pacmap_embedding_norm_dep_subjects.png"))
plt.close()

# Conduct unsupervised clustering of normalized depressed subjects for subtyping
subject_labels_norm_dep = mri_clustering(raw_mri_data_norm_dep,
                                         clustering_output="label_assignment_norm_dep",
                                         max_clusters=10,
                                         bootstrapping=False,
                                         overwrite=True)

# per discovered subtype, get size
for subtype in subject_labels_norm_dep["label"].unique():
    subtype_subjects = subject_labels_norm_dep[subject_labels_norm_dep["label"] == subtype]["subject_ids"].tolist()
    print(f"\nNormalized Label {subtype}:")
    print(f"Number of subjects in normalized label: {len(subtype_subjects)}")

# Visualize the discovered subtypes in a radar chart for each significant MRI ROI
radar_data = subject_labels_norm_dep
mri_cols = [col for col in raw_mri_data_norm_dep.columns if col not in ["subject", "subject_ids", "label", "labels"]]
radar_data = radar_data.groupby("label", as_index=False)[mri_cols].mean()

categories = mri_cols
value_matrix = radar_data[categories].to_numpy(dtype=float)
min_value = np.nanmin(value_matrix)
max_value = np.nanmax(value_matrix)
radius_offset = min(0.0, min_value)
if radius_offset < 0:
    radius_offset -= 0.1 * (max_value - min_value if max_value > min_value else 1.0)

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"polar": True})

for subtype in sorted(radar_data["label"].unique()):
    subtype_values = radar_data.loc[radar_data["label"] == subtype, categories].iloc[0].tolist()
    subtype_values = [value - radius_offset for value in subtype_values]
    subtype_values += subtype_values[:1]
    ax.plot(angles, subtype_values, linewidth=2, label=f"Subtype {subtype}")
    ax.fill(angles, subtype_values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8)
ax.set_title("Mean MRI ROI Z-scores by subtype", pad=20)
tick_values = np.linspace(min_value, max_value, num=5)
ax.set_yticks((tick_values - radius_offset).tolist())
ax.set_yticklabels([f"{tick_value:.2f}" for tick_value in tick_values])
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(output_path, "radar_chart_mean_z_scores_by_subtype_norm.png"))
plt.close()







# Filter raw_mri_data_resampled to only include subjects with depression diagnosis
depression_diagnosis_df = mri_meta_df[mri_meta_df["dep_dx"] == 1].drop_duplicates(subset=["subject"])
raw_mri_data_resampled_dep = raw_mri_data_resampled[raw_mri_data_resampled["subject_ids"].isin(depression_diagnosis_df["subject"])]

# Add over-sampled subjects from raw_mri_data_resampled that don't have subject_ids
over_sampled_subjects = raw_mri_data_resampled[~raw_mri_data_resampled["subject_ids"].isin(mri_meta_df["subject"])]["subject_ids"].tolist()
raw_mri_data_resampled_dep = pd.concat([raw_mri_data_resampled_dep, raw_mri_data_resampled[raw_mri_data_resampled["subject_ids"].isin(over_sampled_subjects)]])

# Visualise the data in 2D PaCMAP space for the resampled depressed subjects
pacmap_model = pacmap.PaCMAP(n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0, random_state=42)
pacmap_embedding = pacmap_model.fit_transform(raw_mri_data_resampled_dep.drop(columns=["subject_ids"]))
plt.figure(figsize=(8, 6))
plt.scatter(pacmap_embedding[:, 0], pacmap_embedding[:, 1], alpha=0.7)
plt.title('PaCMAP Embedding of Resampled Depressed Subjects')
plt.xlabel('PaCMAP1')
plt.ylabel('PaCMAP2')
plt.savefig(os.path.join(output_path, "pacmap_embedding_resampled_dep_subjects.png"))
plt.close()

# Conduct unsupervised clustering of resampled depressed subjects for subtyping
subject_labels_resampled_dep = mri_clustering(raw_mri_data_resampled_dep,
                                             clustering_output="label_assignment_resampled_dep",
                                             max_clusters=10,
                                             bootstrapping=False,
                                             overwrite=True)

# Per discovered subtype, get overlap with subjects that exist in mri_meta_df
for subtype in subject_labels_resampled_dep["label"].unique():
    subtype_subjects = subject_labels_resampled_dep[subject_labels_resampled_dep["label"] == subtype]["subject_ids"].tolist()
    overlap_subjects = set(subtype_subjects).intersection(set(mri_meta_df["subject"]))
    print(f"\nLabel {subtype}:")
    print(f"Number of subjects in subtype: {len(subtype_subjects)}")
    print(f"Number of subjects in subtype that exist in mri_meta_df: {len(overlap_subjects)}")
    print(f"Percentage of all subjects in mri_meta_df in subtype: {len(overlap_subjects) / len(mri_meta_df) * 100:.2f}%")

# Visualize the discovered subtypes in 2D UMAP space
raw_mri_data_resampled_dep = raw_mri_data_resampled_dep.reset_index(drop=True)
subject_labels_resampled_dep = subject_labels_resampled_dep.reset_index(drop=True)

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(raw_mri_data_resampled_dep.drop(columns="subject_ids"))

plt.figure(figsize=(8, 6))
for subtype in sorted(subject_labels_resampled_dep["label"].unique()):
    mask = subject_labels_resampled_dep["label"] == subtype
    plt.scatter(
        embedding[mask.to_numpy(), 0],
        embedding[mask.to_numpy(), 1],
        label=f"Label {subtype}",
        alpha=0.7,
    )
plt.title("Scatter Plot of Resampled Depressed Subjects in 2D UMAP Space")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_path, "scatter_plot_resampled_dep_subjects_umap.png"))
plt.close()

# Visualize the discovered subtypes in a radar chart for each significant MRI ROI
radar_data = subject_labels_resampled_dep
radar_data = radar_data.dropna(subset=["label"])
mri_cols = [col for col in raw_mri_data_resampled_dep.columns if col not in ["subject_ids", "label", "labels"]]
radar_data = radar_data.groupby("label", as_index=False)[mri_cols].mean()

categories = mri_cols
value_matrix = radar_data[categories].to_numpy(dtype=float)
min_value = np.nanmin(value_matrix)
max_value = np.nanmax(value_matrix)
radius_offset = min(0.0, min_value)
if radius_offset < 0:
    radius_offset -= 0.1 * (max_value - min_value if max_value > min_value else 1.0)

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"polar": True})

for subtype in sorted(radar_data["label"].unique()):
    subtype_values = radar_data.loc[radar_data["label"] == subtype, categories].iloc[0].tolist()
    subtype_values = [value - radius_offset for value in subtype_values]
    subtype_values += subtype_values[:1]
    ax.plot(angles, subtype_values, linewidth=2, label=f"Subtype {subtype}")
    ax.fill(angles, subtype_values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8)
ax.set_title("Mean MRI ROI Z-scores by subtype", pad=20)
tick_values = np.linspace(min_value, max_value, num=5)
ax.set_yticks((tick_values - radius_offset).tolist())
ax.set_yticklabels([f"{tick_value:.2f}" for tick_value in tick_values])
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(output_path, "radar_chart_mean_z_scores_by_subtype.png"))
plt.close()





# ---- MODELING ----

# CLASSIFICATION MODELING
final_cv_scores = train_and_evaluate_models(
    train_X.drop(columns=["subject"]),
    (train_y_dx.drop(columns=["subject"])).squeeze(),
    search="random",
    outer_splits=10,
    inner_splits=10,
    models_to_train=["Logistic Regression", "Random Forest", "LightGBM", "SVM"]
)
print(final_cv_scores)
final_cv_scores.to_csv(os.path.join(output_path, "final_cv_scores_dx.csv"), index=False)

# Train final model using the best model
best_model_name = max(final_cv_scores, key=lambda m: final_cv_scores[m]["mean"])
final_model_final, best_hyperparams_dx = train_final_model(
    train_X.drop(columns=["subject"]),
    (train_y_dx.drop(columns=["subject"])).squeeze(),
    best_model_name=best_model_name
)

# Get final model predictions on the test set
final_model_predictions_dx = final_model_final.predict(test_X.drop(columns=["subject"]))

# Save final model using pickle
with open(os.path.join(output_path, "final_model_dx", "final_model_dx.pkl"), "wb") as f:
    pickle.dump(final_model_final, f)

# Save final model hyperparameters to JSON
with open(os.path.join(output_path, "final_model_dx", "final_model_hyperparams.json"), "w") as f:
    json.dump(best_hyperparams_dx, f, indent=4)

# Save final model predictions
final_model_predictions_dx_df = pd.DataFrame({
    "subject": test_X["subject"],
    "predicted_dep_dx": final_model_predictions_dx
})
final_model_predictions_dx_df.to_csv(os.path.join(output_path, "final_model_dx", "final_model_predictions_dx.csv"), index=False)

# MULTI-TARGET REGRESSION MODELING

# Get raw mri data for depressed subjects in mri_meta_df at the first timepoint
depressed_subjects = mri_meta_df[mri_meta_df["dep_dx"] == 1]["subject"].unique().tolist()
query_depressed = f"""
        SELECT "subject", "timepoint", {', '.join(f'"{col}"' for col in mri_rois_sig_filtered)}
        FROM mri_data
        WHERE subject IN ({', '.join(f"'{sub}'" for sub in depressed_subjects)})
        AND timepoint IN (
            SELECT MIN(timepoint)
            FROM mri_data AS sub_mri
            WHERE sub_mri.subject = mri_data.subject
        )
    """
raw_mri_data_depressed = con.execute(query_depressed).df()

# Filter to only include subjects in the training set
raw_mri_data_depressed_train = raw_mri_data_depressed[raw_mri_data_depressed["subject"].isin(train_X["subject"])]
raw_mri_data_depressed_test = raw_mri_data_depressed[raw_mri_data_depressed["subject"].isin(test_X["subject"])]

# Train and evaluate regression models for each z-score target column
regression_results, regression_failures = train_multi_target_regression(
    train_X.drop(columns=["subject"]),
    raw_mri_data_depressed_train.drop(columns=["subject", "timepoint"])
)

# Save regression results and failures to JSON files
with open(os.path.join(output_path, "multi_target_regression", "multi_target_regression_results_raw_dx.json"), "w") as f:
    json.dump(regression_results, f, indent=4)

with open(os.path.join(output_path, "multi_target_regression", "multi_target_regression_failures_raw_dx.json"), "w") as f:
    json.dump(regression_failures, f, indent=4)

# Train final regression models for each target using the best model identified in the previous step
best_model_per_target = {target: max(models, key=lambda m: regression_results[target][m]["mean"]) for target, models in regression_results.items()}
best_models, best_hyperparams = train_final_models_multi_target_regression(
    train_X.drop(columns=["subject"]),
    raw_mri_data_depressed_train.drop(columns=["subject", "timepoint"]),
    best_model_per_target=best_model_per_target,
    results_dir="final_regression_models"
)

# Save best model per target to JSON file
with open(os.path.join(output_path, "multi_target_regression", "best_model_per_target_raw_dx.json"), "w") as f:
    json.dump(best_model_per_target, f, indent=4)

# Save best model hyperparameters to JSON file
with open(os.path.join(output_path, "multi_target_regression", "best_model_hyperparams_raw_dx.json"), "w") as f:
    json.dump(best_hyperparams, f, indent=4)

# Get predictions from the best models on the test set
test_predictions = {}
for target, model in best_models.items():
    test_predictions[target] = model.predict(test_X.drop(columns=["subject"]))

# Save test predictions to CSV
test_predictions_df = pd.DataFrame(test_predictions)
test_predictions_df.to_csv(os.path.join(output_path, "multi_target_regression", "test_predictions_raw_dx.csv"), index=False)

# Save test performance metrics to CSV
test_performance_df = pd.DataFrame(test_performance).T
test_performance_df.to_csv(os.path.join(output_path, "multi_target_regression", "test_performance_metrics_raw_dx.csv"), index=True)