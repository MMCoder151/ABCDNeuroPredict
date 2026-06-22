from pathlib import Path
import os
from src.preprocessing import *
from src.feature_extraction import *
from src.data_analysis import *
from src.modelling import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

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

# ---- DATA ANALYSIS MRI ----

# NORMATIVE SELECTION OF MRI DATA
# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_subjects = normative_selection(con, mri_meta_df, overwrite=False)

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
# Conduct unsupervised clustering of selected subjects' z-scores for subtype discovery
subject_subtypes = mri_clustering(selected_subjects, bootstrapping=True, overwrite=False)

# Add cluster labels to dem_df based on subject_subtypes
dem_df = dem_df.merge(subject_subtypes[["subject_ids", "subtype"]], left_on="subject", right_on="subject_ids", how="left")
# Add -99 for subjects without cluster labels (non-selected subjects)
dem_df["subtype"] = dem_df["subtype"].fillna(-99)

# Calculate cluster association with age using linear regression
age_cluster_df = dem_df[["subject", "subtype"]].dropna().merge(mri_meta_df[["subject", "age_at_mri"]], on="subject")
X_cluster = pd.get_dummies(age_cluster_df["subtype"], drop_first=False)
y_cluster = age_cluster_df["age_at_mri"]
model_cluster = LinearRegression()
model_cluster.fit(X_cluster, y_cluster)
print("Cluster association with age (coefficients):", pd.Series(model_cluster.coef_, index=X_cluster.columns))

# MISSINGNESS ANALYSIS
# Calculate association of missingness in fitbit data with group using logistic regression 
missingness_df = fit_meta_df[["subject", "missing_days_percentage"]].merge(dem_df[["subject", "group"]], on="subject")
missingness_df[["group"]].nunique()  # Check unique values in group column
X_missingness = missingness_df[["group"]]
y_missingness = missingness_df["missing_days_percentage"]
model_missingness = LogisticRegression()
model_missingness.fit(X_missingness, y_missingness)
print("Missingness association with group (coefficients):", pd.Series(model_missingness.coef_, index=X_missingness.columns))

# ---- FITBIT FEATURE EXTRACTION ----

# Extract features from fitbit data
fitbit_features_df = extr_fitbit_features(con, dem_df)

# save extracted features to CSV
fitbit_features_df.to_csv(os.path.join(output_path, "fitbit_features.csv"), index=False)

# OPTIONAL: Reimport extracted features from CSV for analysis and modeling
fitbit_features_df = pd.read_csv(os.path.join(output_path, "fitbit_features.csv"))

# Analyse feature colinearity using Variance Inflation Factor (VIF) and create composite scores to account for multicollinearity
fitbit_features_with_composites, composite_dict = create_composites(fitbit_features_df)

# Save to CSV
fitbit_features_with_composites.to_csv(os.path.join(output_path, "fitbit_features_with_composites.csv"), index=False)
composite_df = pd.DataFrame({
    "composite_name": list(composite_dict.keys()),
    "features_included": [", ".join(features) for features in composite_dict.values()]
})
composite_df.to_csv(os.path.join(output_path, "composite_dictionary.csv"), index=False)

# OPTIONAL: Reimport features with composites from CSV for analysis and modeling
fitbit_features_with_composites = pd.read_csv(os.path.join(output_path, "fitbit_features_with_composites.csv"))
composite_df = pd.read_csv(os.path.join(output_path, "composite_dictionary.csv"))

# Add sex and age to selected_subjects_with_composites for modeling
features = fitbit_features_with_composites.merge(dem_df[["subject", "sex", "age_at_first_mri"]], left_on="subject", right_on="subject", how="left")
features["sex"] = features["sex"].map({"M": 0, "F": 1})
features["sex"] = features["sex"].astype(np.float64)
features["age_at_first_mri"] = features["age_at_first_mri"].astype(np.float64)

if "subtype" not in features.columns:
    features = features.merge(dem_df[["subject", "subtype"]], left_on="subject", right_on="subject", how="left")

# TRAIN-TEST SPLIT
train_X, test_X = train_test_split(features, test_size=0.2, stratify=dem_df["group"], random_state=42)

# Create labels for train and test sets
train_y = dem_df[dem_df["subject"].isin(train_X["subject"])][["subject", "group"]]
test_y = dem_df[dem_df["subject"].isin(test_X["subject"])][["subject", "group"]]

# Save features to CSV
train_X.to_csv(os.path.join(output_path, "train_features.csv"), index=False)
test_X.to_csv(os.path.join(output_path, "test_features.csv"), index=False)
train_y.to_csv(os.path.join(output_path, "train_labels.csv"), index=False)
test_y.to_csv(os.path.join(output_path, "test_labels.csv"), index=False)

# OPTIONAL: Reimport features and labels from CSV for modeling
train_X = pd.read_csv(os.path.join(output_path, "train_features.csv"))
test_X = pd.read_csv(os.path.join(output_path, "test_features.csv"))
train_y = pd.read_csv(os.path.join(output_path, "train_labels.csv"))
test_y = pd.read_csv(os.path.join(output_path, "test_labels.csv"))

# ---- DATA ANALYSIS FITBIT ----

# NORMATIVE SELECTION OF FITBIT DATA
# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_fitbit_subjects = normative_selection_fitbit(dem_df, fitbit_features_with_composites, overwrite=True)

# Conduct confound analysis pre and post normative modeling
z_scores_fitbit = pd.read_csv(os.path.join(output_path, "normative_modelling_fitbit", "results","Z_fitbit_norm.csv"))
confound_effects_fitbit_df = analyse_confounds(dem_df, z_scores_fitbit, raw_data = fitbit_features_with_composites)

# Calculate overlap of normative selected fitbit subjects with selected subjects from MRI normative modeling
overlap_subjects = set(selected_subjects["subject_ids"]).intersection(set(selected_fitbit_subjects["subject_ids"]))
print(f"Number of subjects selected by both MRI and fitbit normative modeling: {len(overlap_subjects)}")
print(f"Overlap percentage: {len(overlap_subjects) / len(selected_subjects) * 100:.2f}%")

# RESIDUALIZATION OF FITBIT FEATURES
# Fit residualization models on training data to remove confounding effects of age and sex
models = fit_residualiser(train_X, dem_df.loc[train_X.index])

train_X.columns.nunique()

# Apply residualization to training and test data
train_X_residualized = apply_residualiser(train_X, models)
test_X_residualized = apply_residualiser(test_X, models)

# Save to csv
train_X_residualized.to_csv(os.path.join(output_path, "train_features_residualized.csv"), index=False)
test_X_residualized.to_csv(os.path.join(output_path, "test_features_residualized.csv"), index=False)

# Conduct confound analysis of fitbit features pre and post residualization
confound_effects_residualized_df = analyse_confounds(dem_df, train_X_residualized, raw_data = train_X)

# ---- MODELING ----

cv_logreg = train_and_evaluate_models(
    train_X.drop(columns=["subject"]), 
    (train_y.drop(columns=["subject"])).squeeze(), 
    search="random", 
    outer_splits=10, 
    inner_splits=10, 
    models_to_train=["Logistic Regression"]
    )

cv_svm = train_and_evaluate_models(
    train_X.drop(columns=["subject"]), 
    (train_y.drop(columns=["subject"])).squeeze(), 
    search="random", 
    outer_splits=10, 
    inner_splits=10, 
    models_to_train=["SVM"]
    )

cv_rf = train_and_evaluate_models(
    train_X.drop(columns=["subject"]), 
    (train_y.drop(columns=["subject"])).squeeze(), 
    search="random", 
    outer_splits=10, 
    inner_splits=10, 
    models_to_train=["Random Forest"]
    )

cv_lightgbm = train_and_evaluate_models(
    train_X.drop(columns=["subject"]), 
    (train_y.drop(columns=["subject"])).squeeze(), 
    search="random", 
    outer_splits=10, 
    inner_splits=10, 
    models_to_train=["LightGBM"]
    )

nested_cv_scores = [
    cv_logreg["Logistic Regression"]["mean"],
    cv_rf["Random Forest"]["mean"],
    cv_lightgbm["LightGBM"]["mean"],
    cv_svm["SVM"]["mean"],
]

nested_cv_scores_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "LightGBM", "SVM"],
    "Nested CV Score": nested_cv_scores
})
print(nested_cv_scores_df)
nested_cv_scores_df.to_csv(os.path.join(output_path, "nested_cv_scores.csv"), index=False)