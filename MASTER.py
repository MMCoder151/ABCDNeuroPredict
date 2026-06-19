from pathlib import Path
import os
from src.preprocessing import *
from src.feature_extraction import *
from src.data_analysis import *
from src.modelling import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
con = setup_duckdb(dta_path, fit_meta_df, overwrite=True)

# ---- DATA ANALYSIS ----

# NORMATIVE SELECTION OF MRI DATA
# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_subjects = normative_selection(con, mri_meta_df, overwrite=False)

# Conduct confound analysis pre and post normative modeling
confound_effects_df = analyse_confounds(con, dem_df, mri_meta_df)

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
# Calculate association of missingness in fitbit data with group using linear regression
missingness_df = fit_meta_df[["subject", "missing_days_percentage"]].merge(dem_df[["subject", "group"]], on="subject")
missingness_df[["group"]].nunique()  # Check unique values in group column
X_missingness = missingness_df[["group"]]
y_missingness = missingness_df["missing_days_percentage"]
model_missingness = LinearRegression()
model_missingness.fit(X_missingness, y_missingness)
print("Missingness association with group (coefficients):", pd.Series(model_missingness.coef_, index=X_missingness.columns))

# ---- FEATURE EXTRACTION ----

dem_df["subject"].nunique()

# Filter dem_df to include only non-short subjects
#dem_df_features = dem_df[~dem_df["subject"].isin(fit_meta_df[fit_meta_df["short"] == True]["subject"])]
#dem_df_features["subject"].nunique()

# Train-test split of subjects in dem_df
train_df, test_df = train_test_split(dem_df[["subject", "subtype"]], test_size=0.2, stratify=dem_df["group"], random_state=42)

# Extract features from selected subjects fitbit data for train set
train_features = extr_fitbit_features(con, train_df)
train_y = dem_df[dem_df["subject"].isin(train_features["subject"])][["subject", "group"]]

# Extract features from selected subjects fitbit data for test set
test_features = extr_fitbit_features(con, test_df)
test_y = dem_df[dem_df["subject"].isin(test_features["subject"])][["subject", "group"]]

# Save features to CSV
train_features.to_csv(os.path.join(output_path, "train_features.csv"), index=False)
test_features.to_csv(os.path.join(output_path, "test_features.csv"), index=False)
train_y.to_csv(os.path.join(output_path, "train_labels.csv"), index=False)
test_y.to_csv(os.path.join(output_path, "test_labels.csv"), index=False)

# OPTIONAL: Reimport features and labels from CSV for modeling
train_features = pd.read_csv(os.path.join(output_path, "train_features.csv"))
test_features = pd.read_csv(os.path.join(output_path, "test_features.csv"))
train_y = pd.read_csv(os.path.join(output_path, "train_labels.csv"))
test_y = pd.read_csv(os.path.join(output_path, "test_labels.csv"))

# Analyse feature colinearity using Variance Inflation Factor (VIF)
# TODO: Implement

# Create feature composites
# TODO: Implement

# ---- DATA ANALYSIS #2 ----

# NORMATIVE SELECTION OF FITBIT DATA
# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
# TODO: Implement

# Conduct confound analysis pre and post normative modeling
# TODO: Implement

# Calculate group overlap with selected subjects from MRI normative modeling
# TODO: Implement

# ---- MODELING ----

cv_logreg = train_and_evaluate_models(
    train_features.drop(columns=["subject"]), 
    (train_y.drop(columns=["subject"])).squeeze(), 
    search="random", 
    outer_splits=10, 
    inner_splits=10, 
    models_to_train=["Logistic Regression"]
    )

cv_svm = train_and_evaluate_models(
    train_features.drop(columns=["subject"]), 
    (train_y.drop(columns=["subject"])).squeeze(), 
    search="random", 
    outer_splits=10, 
    inner_splits=10, 
    models_to_train=["SVM"]
    )

cv_rf = train_and_evaluate_models(
    train_features.drop(columns=["subject"]), 
    (train_y.drop(columns=["subject"])).squeeze(), 
    search="random", 
    outer_splits=10, 
    inner_splits=10, 
    models_to_train=["Random Forest"]
    )

cv_lightgbm = train_and_evaluate_models(
    train_features.drop(columns=["subject"]), 
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