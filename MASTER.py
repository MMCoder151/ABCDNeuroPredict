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

# Set raw data directory 
dta_path = Path.home() / "dairc" / "rawdata"

# Set tabular data directory 
dta_path_tabular = Path.home() / "dairc" / "abcd" / "rawdata" / "phenotype"

# Set output path
output_path = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ---- DATA WRANGLING & PREPROCESSING----

# Filter subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = filter_subjects(dta_path, dta_path_tabular, test=False, overwrite=False)

# Print descriptive statistics of filtered subjects
describe_subjects(fit_meta_df, mri_meta_df)

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df, overwrite=False)

# MISSINGNESS ANALYSIS
# Calculate association of missingness in fitbit data with diagnosis group using logistic regression 
missingness_df = fit_meta_df[["subject", "missing_days_percentage"]].merge(mri_meta_df[["subject", "dep_dx"]], on="subject")
missingness_df[["dep_dx"]].nunique()  # Check unique values in group column
X_missingness = missingness_df[["dep_dx"]]
y_missingness = missingness_df["missing_days_percentage"]
model_missingness = LogisticRegression()
model_missingness.fit(X_missingness, y_missingness)
print("Missingness association with diagnosis group:", pd.Series(model_missingness.coef_, index=X_missingness.columns))

# ---- FEATURE EXTRACTION ----

# FITBIT FEATURE EXTRACTION
# Extract features from fitbit data
fitbit_features_df = extr_fitbit_features(con, dem_df, overwrite=False)

# Analyse feature colinearity using Variance Inflation Factor (VIF) and create composite scores to account for multicollinearity
fitbit_features_with_composites, composite_dict = create_composites(fitbit_features_df, overwrite=False)

# PREPARATION FOR MODELING
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

# RESIDUALIZATION OF FITBIT FEATURES
# Fit residualization models on training data to remove confounding effects of age and sex
#models = fit_residualiser(train_X, dem_df.loc[train_X.index], overwrite=True)

# Apply residualization to training and test data
#train_X_residualized = apply_residualiser(models, train_X, dem_df.loc[train_X.index])
#test_X_residualized = apply_residualiser(models, test_X, dem_df.loc[test_X.index])

# Save to csv
#train_X_residualized.to_csv(os.path.join(output_path, "train_features_residualized.csv"), index=False)
#test_X_residualized.to_csv(os.path.join(output_path, "test_features_residualized.csv"), index=False)

# OPTIONAL: Reimport resudialised fitbit features
#train_X_residualized = pd.read_csv(os.path.join(output_path, "train_features_residualized.csv"))
#test_X_residualized = pd.read_csv(os.path.join(output_path, "test_features_residualized.csv"))

# Conduct confound analysis of fitbit features pre and post residualization
#confound_effects_residualized_df = analyse_confounds(dem_df, train_X_residualized, raw_data = train_X)

# ---- BASELINE ASSUMPTION CHECKS ----

# Create baseline output directory
baseline_output_path = os.path.join(output_path, "baseline_checks")
os.makedirs(baseline_output_path, exist_ok=True)

# MRI GROUP DIFFERENCES
# Get mri rois that show significant differences between depressed and non-depressed subjects
mri_rois_sig, mri_rois_results = extract_mri_rois(dta_path_tabular, dta_path, mri_meta_df, overwrite=False)

# Print mean, min, max and std of effect sizes for significant MRI ROIs
effect_sizes = mri_rois_results.loc[mri_rois_results["mri_feature"].isin(mri_rois_sig), "effect_size"]
print("\nDescriptive statistics of effect sizes for significant MRI ROIs:")
print(f"Mean: {effect_sizes.mean():.4f}")
print(f"Min: {effect_sizes.min():.4f}")
print(f"Max: {effect_sizes.max():.4f}")
print(f"Std: {effect_sizes.std():.4f}")

# Filter mri_rois_sig to only inclue ROIs with effect size smaller than -0.2 or greater than 0.2
mri_rois_sig_filtered = mri_rois_results.loc[(mri_rois_results["effect_size"] < -0.2) | (mri_rois_results["effect_size"] > 0.2), "mri_feature"].tolist()
print(f"\nNumber of significant MRI ROIs with effect size < -0.2 or > 0.2: {len(mri_rois_sig_filtered)}")
print(f"Significant MRI ROIs with effect size < -0.2 or > 0.2: {mri_rois_sig_filtered}")

# BASELINE CLASSIFICATION MODELING
# Train and evaluate baseline classification models using nested cross-validation
cv_scores = train_and_evaluate_models(
    train_X.drop(columns=["subject"]),
    (train_y_dx.drop(columns=["subject"])).squeeze(),
    search="random",
    outer_splits=10,
    inner_splits=10,
    models_to_train=["Logistic Regression", "Random Forest", "LightGBM", "SVM"]
)

print("Baseline model cross-validation scores:")
for model_name, scores in cv_scores.items():
    print(f"  {model_name}: {scores}")

# Save baseline model cross-validation scores to CSV
cv_scores_df = pd.DataFrame.from_dict(cv_scores, orient="index")
cv_scores_df.to_csv(os.path.join(baseline_output_path, "baseline_model_cv_scores.csv"))

# Train final model using the best model based on cross-validation scores
best_model_name = max(cv_scores, key=lambda m: cv_scores[m]["mean"])
final_model, best_hyperparams, train_predictions = train_final_model(
    train_X.drop(columns=["subject"]),
    (train_y_dx.drop(columns=["subject"])).squeeze(),
    model=best_model_name
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

# Get confusion matrix of final model on the test set and save to CSV
confusion_matrix_df = pd.DataFrame(confusion_matrix(test_y_dx["dep_dx"], train_predictions), 
    index=["Actual_Negative", "Actual_Positive"], 
    columns=["Predicted_Negative", "Predicted_Positive"])
confusion_matrix_df.to_csv(os.path.join(baseline_output_path, "final_model_confusion_matrix.csv"))

# BASELINE REGRESSION MODELING
# Get raw mri data of the ROIs with significant group difference for subjects in mri_meta_df at the first timepoint
subjects_in_meta = mri_meta_df["subject"].unique().tolist()
query = f"""
        SELECT "subject", "timepoint", {', '.join(f'"{col}"' for col in mri_rois_sig_filtered)}
        FROM mri_data
        WHERE subject IN ({', '.join(f"'{sub}'" for sub in subjects_in_meta)})
        AND timepoint IN (
            SELECT MIN(timepoint)
            FROM mri_data AS sub_mri
            WHERE sub_mri.subject = mri_data.subject
        )
    """
raw_mri_data = con.execute(query).df()

# Train and evaluate regression models for each MRI ROI target column
regression_results, regression_failures = train_multi_target_regression(
    train_X.drop(columns=["subject"]),
    raw_mri_data[raw_mri_data["subject"].isin(train_X["subject"])].drop(columns=["subject", "timepoint"])
)

# Print regression results
print("Baseline regression model results:")
for target, models in regression_results.items():
    print(f"  Target: {target}")
    for model_name, metrics in models.items():
        print(f"    Model: {model_name}, Mean Score: {metrics['mean']:.4f}, Std Score: {metrics['std']:.4f}")

# Save baseline regression results to CSV
regression_results_df = pd.DataFrame.from_dict({(target, model): metrics for target, models in regression_results.items() for model, metrics in models.items()}, orient='index')
regression_results_df.to_csv(os.path.join(baseline_output_path, "baseline_regression_results.csv"))

# Train final regression models for each target using the best model identified in the previous step
best_model_per_target = {target: max(models, key=lambda m: regression_results[target][m]["mean"]) for target, models in regression_results.items()}
best_models, best_hyperparams = train_final_models_multi_target_regression(
    train_X.drop(columns=["subject"]),
    raw_mri_data[raw_mri_data["subject"].isin(train_X["subject"])].drop(columns=["subject", "timepoint"]),
    best_model_per_target=best_model_per_target
)

# Get final regression model predictions on the test set and save to CSV
test_predictions = {}
for target, model in best_models.items():
    test_predictions[target] = model.predict(test_X.drop(columns=["subject"]))
test_predictions_df = pd.DataFrame({
    "subject": test_X["subject"]
})
for target, predictions in test_predictions.items():
    test_predictions_df[f"predicted_{target}"] = predictions
test_predictions_df.to_csv(os.path.join(baseline_output_path, "final_regression_predictions.csv"), index=False)

# Evaluate the performance of the best models on the test set by calculating RMSE and R-squared for each target
test_performance = {}
for target, model in best_models.items():
    y_true = raw_mri_data[raw_mri_data["subject"].isin(test_X["subject"])][target]
    y_pred = test_predictions[target]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    test_performance[target] = {"RMSE": rmse, "R-squared": r2}
test_performance_df = pd.DataFrame(test_performance).T
test_performance_df.to_csv(os.path.join(baseline_output_path, "final_regression_performance_metrics.csv"), index=True)
print("Final regression model performance on the test set:")
print(test_performance_df)

# Save best model per target using pickle 
for target, model in best_models.items():
    with open(os.path.join(baseline_output_path, f"final_model_{target}.pkl"), "wb") as f:
        pickle.dump(model, f)

# ---- UNSUPERVISED LABEL ASSIGNMENT ----

# NORMATIVE SELECTION OF MRI DATA
# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_subjects = normative_selection(con, mri_meta_df, mri_rois_sig_filtered, overwrite=True)

# Check overlap between normative selected subjects and subjects with depression diagnosis
depression_diagnosis_df = mri_meta_df[mri_meta_df["dep_dx"] == 1]
depression_diagnosis_df = (
    mri_meta_df[mri_meta_df["dep_dx"] == 1]
    .drop_duplicates(subset=["subject"])
)
overlap_subjects = set(selected_subjects["subject_ids"]).intersection(set(depression_diagnosis_df["subject"]))
print(f"Number of subjects selected by normative modeling and with depression diagnosis: {len(overlap_subjects)}")
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

# Add group labels to dem_df based on selected_subjects
dem_df["group"] = dem_df["subject"].apply(lambda x: 1 if x in selected_subjects["subject_ids"].values else 0)

# UNSUPERVISED CLUSTERING
# Get raw mri data for subjects in mri_meta_df at the first timepoint
subjects_in_meta = mri_meta_df["subject"].unique().tolist()
query = f"""
        SELECT "subject", "timepoint", {', '.join(f'"{col}"' for col in mri_rois_sig_filtered)}
        FROM mri_data
        WHERE subject IN ({', '.join(f"'{sub}'" for sub in subjects_in_meta)})
        AND timepoint IN (
            SELECT MIN(timepoint)
            FROM mri_data AS sub_mri
            WHERE sub_mri.subject = mri_data.subject
        )
    """
raw_mri_data = con.execute(query).df()

# Conduct unsupervised clustering of raw mri data for label assignment
subject_labels_raw = mri_clustering(raw_mri_data, n_clusters=2, clustering_output="label_assignment_raw", bootstrapping=True, overwrite=True)

# Per discovered subtype, get overlap with subjects with depression diagnosis
for subtype in subject_labels_raw["subtype"].unique():
    subtype_subjects = subject_labels_raw[subject_labels_raw["subtype"] == subtype]["subject_ids"].tolist()
    overlap_subjects = set(subtype_subjects).intersection(set(depression_diagnosis_df["subject"]))
    print(f"\nSubtype {subtype}:")
    print(f"Number of subjects in subtype: {len(subtype_subjects)}")
    print(f"Number of subjects in subtype with depression diagnosis: {len(overlap_subjects)}")
    print(f"Overlap percentage: {len(overlap_subjects) / len(subtype_subjects) * 100:.2f}%")

# Conduct unsupervised clustering of selected subjects' z-scores for subtype discovery
subject_subtypes = mri_clustering(selected_subjects, clustering_output="mri_clustering", bootstrapping=True, overwrite=False)

# Add cluster labels to dem_df based on subject_subtypes
dem_df = dem_df.merge(subject_subtypes[["subject_ids", "subtype"]], left_on="subject", right_on="subject_ids", how="left")
# Add -99 for subjects without cluster labels (non-selected subjects)
dem_df["subtype"] = dem_df["subtype"].fillna(-99)

# EM-REGRESSION MODEL LABEL ASSIGNMENT
# Fit EM-regression model to raw mri data for label assignment
em_regression_labels = em_regression_label_assignment(raw_mri_data, n_components=2, em_output="label_assignment_em", overwrite=True)

# Add EM-regression labels to dem_df based on em_regression_labels
dem_df = dem_df.merge(em_regression_labels[["subject_ids", "em_label"]], left_on="subject", right_on="subject_ids", how="left")

# Calculate overlap of EM-regression labels with subjects with depression diagnosis












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