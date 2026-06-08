from pathlib import Path
import os
from src.preprocessing import *
from src.feature_extraction import *
from sklearn.model_selection import train_test_split

# Set data directory paths
dta_path = Path.home() / "dairc" / "rawdata"

# Set output path
output_path = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ---- DATA WRANGLING ----

# Select subjects based on inclusion criteria and extract metadata
dem_df, mri_meta_df, fit_meta_df = filter_subjects(dta_path, test=False, overwrite=False)

# Print demographics of selected subjects with both fitbit and mri data
print("Demographics of filtered subjects with both fitbit and mri files:")
print(f"N: {mri_meta_df['subject'].nunique()}")
print("\nMRI Age Statistics:")
print(f"Mean: {mri_meta_df['age_at_mri'].mean()}, Std: {mri_meta_df['age_at_mri'].std()}, Min: {mri_meta_df['age_at_mri'].min()}, Max: {mri_meta_df['age_at_mri'].max()}")
print("\nSex Distribution:")
print(dem_df["sex"].value_counts())

# Print Fitbit completeness at the first available timepoint, split by file domain
first_timepoint_fit_df = fit_meta_df[["subject", "timepoint", "filename", "short"]].copy()
first_timepoint_fit_df["domain"] = "other"
first_timepoint_fit_df.loc[
    first_timepoint_fit_df["filename"].str.contains(r"fitbInt1m", case=False, regex=True),
    "domain",
] = "actigraphy"
first_timepoint_fit_df.loc[
    first_timepoint_fit_df["filename"].str.contains(r"fitbHR1m", case=False, regex=True),
    "domain",
] = "heart_rate"
first_timepoint_fit_df.loc[
    first_timepoint_fit_df["filename"].str.contains(r"fitbSlp1m", case=False, regex=True),
    "domain",
] = "sleep"
first_timepoint_fit_df = first_timepoint_fit_df[first_timepoint_fit_df["domain"] != "other"]
first_timepoint_fit_df["timepoint_order"] = first_timepoint_fit_df["timepoint"].str.extract(r"(\d+)")[0].astype(int)

first_timepoint_subject_df = (
    first_timepoint_fit_df.sort_values(["subject", "timepoint_order", "timepoint"])
    .drop_duplicates(subset=["subject"], keep="first")[["subject", "timepoint"]]
)

first_timepoint_fit_df = first_timepoint_fit_df.merge(first_timepoint_subject_df, on=["subject", "timepoint"], how="inner")

domain_subject_df = (
    first_timepoint_fit_df.groupby(["domain", "subject"], as_index=False)
    .agg(
        n_files=("filename", "size"),
        has_short=("short", lambda s: bool((s == 1).any())),
        has_non_short=("short", lambda s: bool((s == 0).any())),
    )
)

print("\nFitbit completeness at the first available timepoint, by domain:")
print("mixed = subjects with both short and non-short files within the same domain")

for domain in ["actigraphy", "heart_rate", "sleep"]:
    domain_df = domain_subject_df[domain_subject_df["domain"] == domain]
    if domain_df.empty:
        continue

    short_subjects = set(domain_df.loc[domain_df["has_short"], "subject"])
    non_short_subjects = set(domain_df.loc[domain_df["has_non_short"], "subject"])

    print(f"\n{domain.replace('_', ' ').title()}:")
    print(f"Subjects at first timepoint: {domain_df['subject'].nunique()}")
    print(f"Subjects with any short files: {len(short_subjects)}")
    print(f"Subjects with any non-short files: {len(non_short_subjects)}")

    for label, subject_set in [("short", short_subjects), ("non-short", non_short_subjects)]:
        subject_list = list(subject_set)
        if not subject_list:
            print(f"\n{domain.replace('_', ' ').title()} - {label.title()}:")
            print("N: 0")
            continue

        subject_dem_df = dem_df[dem_df["subject"].isin(subject_list)]
        subject_mri_df = mri_meta_df[mri_meta_df["subject"].isin(subject_list)]

        print(f"\n{domain.replace('_', ' ').title()} - {label.title()}:")
        print(f"N: {len(subject_set)}")
        print("MRI Age Statistics:")
        print(
            f"Mean: {subject_mri_df['age_at_mri'].mean()}, "
            f"Std: {subject_mri_df['age_at_mri'].std()}, "
            f"Min: {subject_mri_df['age_at_mri'].min()}, "
            f"Max: {subject_mri_df['age_at_mri'].max()}"
        )
        print("Sex Distribution:")
        print(subject_dem_df["sex"].value_counts())

# Transform data to make it easier to query with DuckDB
con = setup_duckdb(dta_path, fit_meta_df, overwrite=True)

# ---- FEATURE EXTRACTION ----

# Select subjects based on normative modeling of FIRST TIMEPOINT and composite z-scores
selected_subjects = normative_selection(con, mri_meta_df, dem_df, overwrite=False)

# Print demographics of normative selected subjects
print("Selected Subjects MRI Age Statistics:")
print(f"Mean: {mri_meta_df[mri_meta_df['subject'].isin(selected_subjects['subject_ids'])]['age_at_mri'].mean()}, Std: {mri_meta_df[mri_meta_df['subject'].isin(selected_subjects['subject_ids'])]['age_at_mri'].std()}, Min: {mri_meta_df[mri_meta_df['subject'].isin(selected_subjects['subject_ids'])]['age_at_mri'].min()}, Max: {mri_meta_df[mri_meta_df['subject'].isin(selected_subjects['subject_ids'])]['age_at_mri'].max()}")
print("\nSelected Subjects Sex Distribution:")
print(dem_df[dem_df["subject"].isin(selected_subjects["subject_ids"])]["sex"].value_counts())

# Print missing statistics of fitbit data for selected subjects
selected_fit_meta_df = fit_meta_df[fit_meta_df["subject"].isin(selected_subjects["subject_ids"])]
print("\nFitbit Data Missingness Statistics for Normative Selected Subjects:")
print(f"Mean missingness: {selected_fit_meta_df['missingness'].mean()}, Std: {selected_fit_meta_df['missingness'].std()}, Min: {selected_fit_meta_df['missingness'].min()}, Max: {selected_fit_meta_df['missingness'].max()}")
print(f"Number of subjects with short recordings (short == 1): {selected_fit_meta_df[selected_fit_meta_df['short'] == 1]['subject'].nunique()} ({selected_fit_meta_df[selected_fit_meta_df['short'] == 1]['subject'].nunique() / len(selected_fit_meta_df) * 100:.2f}% )")

# Print missingness statistics of fitbit data for non-selected subjects
non_selected_fit_meta_df = fit_meta_df[~fit_meta_df["subject"].isin(selected_subjects["subject_ids"])]
print("\nFitbit Data Missingness Statistics for Normative Non-Selected Subjects:")
print(f"Mean missingness: {non_selected_fit_meta_df['missingness'].mean()}, Std: {non_selected_fit_meta_df['missingness'].std()}, Min: {non_selected_fit_meta_df['missingness'].min()}, Max: {non_selected_fit_meta_df['missingness'].max()}")
print(f"Number of subjects with short recordings (short == 1): {non_selected_fit_meta_df[non_selected_fit_meta_df['short'] == 1]['subject'].nunique()} ({non_selected_fit_meta_df[non_selected_fit_meta_df['short'] == 1]['subject'].nunique() / len(non_selected_fit_meta_df) * 100:.2f}% )")

# Conduct confound analysis pre and post normative modeling
confound_effects_df = analyse_confounds(con, dem_df, mri_meta_df)

# Create composite mri z-scores based on VIF
selected_subjects_composites, composite_dict = create_mri_composites(con, selected_subjects)

# Save selected subjects to CSV
selected_subjects_composites.to_csv(os.path.join(output_path, "selected_subjects_composites.csv"), index=False)

# Add group labels to dem_df based on selected_subjects
dem_df["group"] = dem_df["subject"].apply(lambda x: 1 if x in selected_subjects["subject_ids"].values else 0)

# Train-test split of subjects in dem_df
train_df, test_df = train_test_split(dem_df["subject"], test_size=0.2, stratify=dem_df["group"], random_state=42)

# Extract features from selected subjects fitbit data for train set
train_features = extr_fitbit_features(con, train_df)
train_y = dem_df[dem_df["subject"].isin(train_df)]["group"]

# Extract features from selected subjects fitbit data for test set
test_features = extr_fitbit_features(con, test_df)
test_y = dem_df[dem_df["subject"].isin(test_df)]["group"]

# Save features to CSV
train_features.to_csv(os.path.join(output_path, "train_features.csv"), index=False)
test_features.to_csv(os.path.join(output_path, "test_features.csv"), index=False)
train_y.to_csv(os.path.join(output_path, "train_labels.csv"), index=False)
test_y.to_csv(os.path.join(output_path, "test_labels.csv"), index=False)

# ---- DATA ANALYSIS ----

# Conduct unsupervised clustering of selected subjects' z-scores for subtype discovery
#subject_subtypes = mri_clustering(dem_df, selected_subjects)

# Print number of unique subtypes discovered
#print(f"Number of unique subtypes discovered: {len(set(subject_subtypes.values())) - (1 if -1 in subject_subtypes.values() else 0))}")  # Exclude -1 if it exists, which represents subjects not assigned to any subtype

# Add subtype labels to dem_df based on subject_subtypes
#dem_df["subtype"] = dem_df["subject"].apply(lambda x: subject_subtypes.get(x, -1))  # Assign -1 for subjects not in subject_subtypes

# Conduct missingness analysis of fitbit data
#missingness_df = missingness_analysis(con, fit_meta_df)

# ---- MODELING ----

