import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pyampute.exploration.mcar_statistical_tests import MCARTest
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import hdbscan
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score
from pyampute.exploration.mcar_statistical_tests import MCARTest
import pacmap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

def mri_clustering(selected_subjects, output_path = Path("output")):
    '''
    This function performs clustering to identify subtypes of depression based on the selected subjects' MRI ROI data.
    It uses several different clustering algorithms (HBDSCAN and Bayesian Gaussian Mixture Models).
    Clustering stability is assessed using bootstrapping and assessing stability using the Jaccard index.
    Algorithms are compared based on their Silhouette coefficient, Density-Based Clustering Validation (DBCV) score, and Davies-Bouldin Index (DBI).

    Parameters:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their MRI ROI z-scores
    Returns:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their assigned cluster labels based on their MRI ROI z-scores 

    TODO: Add descriptive analysis of clusters and graphs

    '''

    # Read in the z-scores for the selected subjects and their MRI ROI data
    z_scores_path = os.path.join(output_path, "normative_modelling", "results", "Z_mri_norm.csv")
    subject_scores = pd.read_csv(z_scores_path)

    # Filter to selected subjects
    selected_subjects = selected_subjects.merge(subject_scores, left_on="subject_ids", right_on="subject_ids", how="left")

    selected_subjects.drop(columns=["observations", "composite_z", "rank"], inplace=True)

    def _align_labels(reference, target):
        '''Aligns cluster labels of the target clustering to the reference clustering using the Hungarian algorithm.'''
        # Compute confusion matrix between reference and target labels
        labels = np.union1d(np.asarray(reference), np.asarray(target))
        conf_matrix = confusion_matrix(reference, target, labels=labels)
        # Use Hungarian algorithm to find optimal label alignment
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        # Create a mapping from target labels to reference labels
        label_mapping = {labels[target_label]: labels[reference_label] for target_label, reference_label in zip(col_ind, row_ind)}
        # Apply the mapping to the target labels
        aligned_target = target.map(label_mapping).where(lambda s: s.notna(), target)
        return aligned_target 
    
    # Dimensionality reduction using PaCMAP
    reducer = pacmap.PaCMAP(n_components=2, random_state=42)
    X_pacmap = reducer.fit_transform(selected_subjects.drop(columns=["subject_ids"]))

    # Dimensionality reduction using PCA for comparison
    pca09 = PCA(n_components=0.9, random_state=42)
    X_pca09 = pca09.fit_transform(selected_subjects.drop(columns=["subject_ids"]))
    X_pca09.shape
    pca2 = PCA(n_components=2, random_state=42)
    X_pca2 = pca2.fit_transform(selected_subjects.drop(columns=["subject_ids"]))

    # Dimensionality reduction using t-SNE for comparison
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(selected_subjects.drop(columns=["subject_ids"]))

    # Dimensionality reduction using UMAP for comparison
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(selected_subjects.drop(columns=["subject_ids"]))

    # Evaluate local dimensionality reduction validity using knn overlap and trustworthiness
    def knn_overlap(X_orig, X_emb, k=10):
        nn_orig = NearestNeighbors(n_neighbors=k+1).fit(X_orig)
        nn_emb  = NearestNeighbors(n_neighbors=k+1).fit(X_emb)
        idx_orig = nn_orig.kneighbors(return_distance=False)[:,1:]  # exclude self
        idx_emb  = nn_emb.kneighbors(return_distance=False)[:,1:]
        overlaps = [(len(set(a).intersection(b))/k) for a,b in zip(idx_orig, idx_emb)]
        return np.mean(overlaps)
    
    knn_overlap_score = knn_overlap(selected_subjects.drop(columns=["subject_ids"]), X_pacmap)
    trustworthiness_score = trustworthiness(selected_subjects.drop(columns=["subject_ids"]), X_pacmap, n_neighbors=10)
    print(f"PaCMAP dimensionality reduction validity: KNN overlap={knn_overlap_score:.4f}, Trustworthiness={trustworthiness_score:.4f}")
    knn_overlap_score_pca09 = knn_overlap(selected_subjects.drop(columns=["subject_ids"]), X_pca09)
    trustworthiness_score_pca09 = trustworthiness(selected_subjects.drop(columns=["subject_ids"]), X_pca09, n_neighbors=10)
    print(f"PCA (0.9 components) dimensionality reduction validity: KNN overlap={knn_overlap_score_pca09:.4f}, Trustworthiness={trustworthiness_score_pca09:.4f}")
    knn_overlap_score_pca2 = knn_overlap(selected_subjects.drop(columns=["subject_ids"]), X_pca2)
    trustworthiness_score_pca2 = trustworthiness(selected_subjects.drop(columns=["subject_ids"]), X_pca2, n_neighbors=10)
    print(f"PCA (2 components) dimensionality reduction validity: KNN overlap={knn_overlap_score_pca2:.4f}, Trustworthiness={trustworthiness_score_pca2:.4f}")
    knn_overlap_score_tsne = knn_overlap(selected_subjects.drop(columns=["subject_ids"]), X_tsne)
    trustworthiness_score_tsne = trustworthiness(selected_subjects.drop(columns=["subject_ids"]), X_tsne, n_neighbors=10)
    print(f"t-SNE dimensionality reduction validity: KNN overlap={knn_overlap_score_tsne:.4f}, Trustworthiness={trustworthiness_score_tsne:.4f}")
    knn_overlap_score_umap = knn_overlap(selected_subjects.drop(columns=["subject_ids"]), X_umap)
    trustworthiness_score_umap = trustworthiness(selected_subjects.drop(columns=["subject_ids"]), X_umap, n_neighbors=10)
    print(f"UMAP dimensionality reduction validity: KNN overlap={knn_overlap_score_umap:.4f}, Trustworthiness={trustworthiness_score_umap:.4f}")

    # Evaluate global dimensionality reduction validity using pairwise distance correlation
    def pairwise_distance_correlation(X_orig, X_emb):
        from scipy.spatial.distance import pdist, squareform
        dist_orig = squareform(pdist(X_orig))
        dist_emb = squareform(pdist(X_emb))
        corr = np.corrcoef(dist_orig.flatten(), dist_emb.flatten())[0, 1]
        return corr
    
    distance_correlation = pairwise_distance_correlation(selected_subjects.drop(columns=["subject_ids"]), X_pacmap)
    print(f"\nPaCMAP global distance preservation: Pairwise distance correlation={distance_correlation:.4f}")
    distance_correlation_pca09 = pairwise_distance_correlation(selected_subjects.drop(columns=["subject_ids"]), X_pca09)
    print(f"PCA (0.9 components) global distance preservation: Pairwise distance correlation={distance_correlation_pca09:.4f}")
    distance_correlation_pca2 = pairwise_distance_correlation(selected_subjects.drop(columns=["subject_ids"]), X_pca2)
    print(f"PCA (2 components) global distance preservation: Pairwise distance correlation={distance_correlation_pca2:.4f}")
    distance_correlation_tsne = pairwise_distance_correlation(selected_subjects.drop(columns=["subject_ids"]), X_tsne)
    print(f"t-SNE global distance preservation: Pairwise distance correlation={distance_correlation_tsne:.4f}")
    distance_correlation_umap = pairwise_distance_correlation(selected_subjects.drop(columns=["subject_ids"]), X_umap)
    print(f"UMAP global distance preservation: Pairwise distance correlation={distance_correlation_umap:.4f}")

    # Create reference clustering on the lower dimensional data
    hdbscan_clusterer_umap = hdbscan.HDBSCAN()
    reference_labels_hdbscan_umap = pd.Series(
        hdbscan_clusterer_umap.fit_predict(X_umap),
        index=selected_subjects.index,
        name="reference_labels_hdbscan_umap",
    )
    hdbscan_clusterer_pca09 = hdbscan.HDBSCAN()
    reference_labels_hdbscan_pca09 = pd.Series(
        hdbscan_clusterer_pca09.fit_predict(X_pca09),
        index=selected_subjects.index,
        name="reference_labels_hdbscan_pca09",
    )
    print(f"\nHDBSCAN clustering on UMAP: Number of unique subtypes discovered: {len(set(reference_labels_hdbscan_umap)) - (1 if -1 in reference_labels_hdbscan_umap else 0)}")  # Exclude -1 if it exists, which represents subjects not assigned to any subtype
    print(f"HDBSCAN clustering on PCA (0.9 components): Number of unique subtypes discovered: {len(set(reference_labels_hdbscan_pca09)) - (1 if -1 in reference_labels_hdbscan_pca09 else 0)}")  # Exclude -1 if it exists, which represents subjects not assigned to any subtype

    # Create mask for non-noise points in HDBSCAN and GMM to use for alignment and scoring with label alignment
    mask_hdbscan_umap = (reference_labels_hdbscan_umap != -1)
    mask_hdbscan_pca09 = (reference_labels_hdbscan_pca09 != -1)

    tuning_results_umap = []
    tuning_results_pca09 = []

    for mcs in tqdm([5, 10, 15, 20], desc="Tuning HDBSCAN parameters"):
        for ms in [5, 10, 20]:  # Varying min_samples for HDBSCAN
            for i in range(100):  # Bootstrapping for clustering stability
                # Resample subjects with replacement
                bootstrap_sample = selected_subjects.sample(frac=1, replace=True, random_state=i) 
                bootstrap_indices = bootstrap_sample.index
                bootstrap_sample_umap = X_umap[bootstrap_indices]
                bootstrap_sample_pca09 = X_pca09[bootstrap_indices]

                # Get the reference labels for the bootstrap sample and align comparisons on the sampled rows
                ref_clean_hdbscan_umap = reference_labels_hdbscan_umap.loc[bootstrap_indices].reset_index(drop=True)
                ref_clean_hdbscan_pca09 = reference_labels_hdbscan_pca09.loc[bootstrap_indices].reset_index(drop=True)

                # HDBSCAN clustering
                hdbscan_clusterer_umap = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
                boot_labels_umap = pd.Series(hdbscan_clusterer_umap.fit_predict(bootstrap_sample_umap), index=bootstrap_indices).reset_index(drop=True)
                non_noise_mask_umap = (ref_clean_hdbscan_umap != -1) & (boot_labels_umap != -1)
                if non_noise_mask_umap.sum() == 0:
                    tuning_results_umap.append({"mcs": np.nan, "ms": np.nan, "jaccard": np.nan, "silhouette": np.nan})
                    continue
                aligned_boot_labels = _align_labels(ref_clean_hdbscan_umap[non_noise_mask_umap], boot_labels_umap[non_noise_mask_umap])
                jaccard_hdbscan = jaccard_score(ref_clean_hdbscan_umap[non_noise_mask_umap], aligned_boot_labels, average="macro")
                tuning_results_umap.append({"mcs": mcs, "ms": ms, "jaccard": jaccard_hdbscan, "silhouette": silhouette_score(bootstrap_sample_umap, boot_labels_umap)})
    
                hdbscan_clusterer_pca09 = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
                boot_labels_pca09 = pd.Series(hdbscan_clusterer_pca09.fit_predict(bootstrap_sample_pca09), index=bootstrap_indices).reset_index(drop=True)
                non_noise_mask_pca09 = (ref_clean_hdbscan_pca09 != -1) & (boot_labels_pca09 != -1)
                if non_noise_mask_pca09.sum() == 0:
                    tuning_results_pca09.append({"mcs": np.nan, "ms": np.nan, "jaccard": np.nan, "silhouette": np.nan})
                    continue
                aligned_boot_labels = _align_labels(ref_clean_hdbscan_pca09[non_noise_mask_pca09], boot_labels_pca09[non_noise_mask_pca09])
                jaccard_hdbscan = jaccard_score(ref_clean_hdbscan_pca09[non_noise_mask_pca09], aligned_boot_labels, average="macro")
                tuning_results_pca09.append({"mcs": mcs, "ms": ms, "jaccard": jaccard_hdbscan, "silhouette": silhouette_score(bootstrap_sample_pca09, boot_labels_pca09)})

                # Get best performing parameters based on mean Jaccard index across bootstraps
                best_params_umap = max(tuning_results_umap, key=lambda x: x["silhouette"])
                best_params_pca09 = max(tuning_results_pca09, key=lambda x: x["silhouette"])   

    print(f"\nBest HDBSCAN parameters for UMAP: min_cluster_size={best_params_umap['mcs']}, min_samples={best_params_umap['ms']}, silhouette={best_params_umap['silhouette']:.4f}")
    print(f"Cluster stability across bootstraps for UMAP: mean Jaccard index={pd.Series([r['jaccard'] for r in tuning_results_umap if not np.isnan(r['jaccard'])]).mean():.4f} with std={pd.Series([r['jaccard'] for r in tuning_results_umap if not np.isnan(r['jaccard'])]).std():.4f}")
    print(f"Variance in Silhouette score across bootstraps for UMAP: mean={pd.Series([r['silhouette'] for r in tuning_results_umap if not np.isnan(r['silhouette'])]).mean():.4f}, std={pd.Series([r['silhouette'] for r in tuning_results_umap if not np.isnan(r['silhouette'])]).std():.4f}, min={pd.Series([r['silhouette'] for r in tuning_results_umap if not np.isnan(r['silhouette'])]).min():.4f}, max={pd.Series([r['silhouette'] for r in tuning_results_umap if not np.isnan(r['silhouette'])]).max():.4f}")
    print(f"Number of unique subtypes discovered with best HDBSCAN parameters on UMAP: {len(set(reference_labels_hdbscan_umap)) - (1 if -1 in reference_labels_hdbscan_umap else 0)} with an average of {pd.Series(reference_labels_hdbscan_umap).value_counts().mean():.2f} subjects per subtype")  # Exclude -1 if it exists, which represents subjects not assigned to any subtype
    print(f"\nBest HDBSCAN parameters for PCA (0.9 components): min_cluster_size={best_params_pca09['mcs']}, min_samples={best_params_pca09['ms']}, silhouette={best_params_pca09['silhouette']:.4f}")
    print(f"Cluster stability across bootstraps for PCA (0.9 components): mean Jaccard index={pd.Series([r['jaccard'] for r in tuning_results_pca09 if not np.isnan(r['jaccard'])]).mean():.4f} with std={pd.Series([r['jaccard'] for r in tuning_results_pca09 if not np.isnan(r['jaccard'])]).std():.4f}")
    print(f"Variance in Silhouette score across bootstraps for PCA (0.9 components): mean={pd.Series([r['silhouette'] for r in tuning_results_pca09 if not np.isnan(r['silhouette'])]).mean():.4f}, std={pd.Series([r['silhouette'] for r in tuning_results_pca09 if not np.isnan(r['silhouette'])]).std():.4f}, min={pd.Series([r['silhouette'] for r in tuning_results_pca09 if not np.isnan(r['silhouette'])]).min():.4f}, max={pd.Series([r['silhouette'] for r in tuning_results_pca09 if not np.isnan(r['silhouette'])]).max():.4f}")
    print(f"Number of unique subtypes discovered with best HDBSCAN parameters on PCA (0.9 components): {len(set(reference_labels_hdbscan_pca09)) - (1 if -1 in reference_labels_hdbscan_pca09 else 0)} with an average of {pd.Series(reference_labels_hdbscan_pca09).value_counts().mean():.2f} subjects per subtype")  # Exclude -1 if it exists, which represents subjects not assigned to any subtype
    
    # Rerun with best parameters and assign cluster labels to subjects
    hdbscan_clusterer_umap = hdbscan.HDBSCAN(min_cluster_size=best_params_umap["mcs"], min_samples=best_params_umap["ms"])
    selected_subjects["hdbscan_umap_labels"] = hdbscan_clusterer_umap.fit_predict(X_umap)
    hdbscan_clusterer_pca09 = hdbscan.HDBSCAN(min_cluster_size=best_params_pca09["mcs"], min_samples=best_params_pca09["ms"])
    selected_subjects["hdbscan_pca09_labels"] = hdbscan_clusterer_pca09.fit_predict(X_pca09) 

    return selected_subjects

def missingness_analysis(con, fit_meta_df):
    '''
    This function analyzes the missingness patterns in the fitbit data for the selected subjects for MCAR, MAR, or MNAR missingness
    using Littles test from pyampute.
        1. Queries different fitbit domain data separately (i.e. actigraphy-based: Stps1m, METs1m, Int1m, heart-rate: HR1m, Sleep: Slp1m)
        2. For each, creates datetime index (days) with proper missing days based on min and max of the Wear_Time column for each subject and timepoint, 
        and merges with the original data to get missingness patterns
        3. Conducts Little's test for each fitbit domain to assess whether the missingness is MCAR, MAR, or MNAR

    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        fit_meta_df (DataFrame): DataFrame containing the fitbit metadata for the selected subjects
    Returns:
        missingness_df (DataFrame): DataFrame containing the missingness patterns in the fitbit data for the selected subjects.

    NOTE: This function is currently broken. Little's test is not computing properly and I have no fucking clue why. WIP
    '''

    # For each subject and timepoint, create datetime index with proper missing days based on min and max of the Wear_Time column, and merge with original data to get missingness patterns
    def create_missingness_df(df, domain_name):
        missingness_list = []
        grouped = df.groupby(["subject", "timepoint"])
        for (subject, timepoint), group in tqdm(grouped, total=grouped.ngroups, desc=f"Creating missingness patterns for {domain_name}"):
            min_date = group["Wear_Time"].min().floor("D")
            max_date = group["Wear_Time"].max().ceil("D")
            value_cols = [c for c in group.columns if c not in ["subject", "timepoint", "Wear_Time"]]
            daily_means = group.set_index("Wear_Time")[value_cols].resample("D").mean().reset_index()
            date_range = pd.date_range(start=min_date, end=max_date, freq="D")
            date_df = pd.DataFrame({"Wear_Time": date_range})
            merged_df = date_df.merge(daily_means, on="Wear_Time", how="left")
            merged_df["subject"] = subject
            merged_df["timepoint"] = timepoint
            missingness_list.append(merged_df)
        return pd.concat(missingness_list, ignore_index=True)
    
    # Conduct Little's test for each fitbit domain to assess if the missingness is MCAR
    mcar_test = MCARTest(method="little")

    def safe_little_test(df, min_obs_per_col=5, min_cols=2):
        df = df.copy()
        # coerce numeric and drop empty cols/rows
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.loc[:, df.notna().sum() >= min_obs_per_col]
        df = df.dropna(how='all')
        if df.shape[1] < min_cols or df.shape[0] < 2:
            return np.nan, "insufficient data after filtering"
        # compute pj/df quickly (same logic as pyampute)
        vars_ = df.dtypes.index.values
        n_var = df.shape[1]
        r = 1 * df.isnull()
        mdp = np.dot(r, [2**i for i in range(n_var)])
        pj = 0
        for i in np.unique(mdp):
            dataset_temp = df.loc[mdp == i, vars_]
            select_vars = ~dataset_temp.isnull().any()
            pj += np.sum(select_vars)
        df_val = pj - n_var
        if df_val <= 0:
            return np.nan, f"df <= 0 (pj={pj}, n_var={n_var})"
        # try running the test and catch numerical errors
        try:
            pval = MCARTest(method="little").little_mcar_test(df)
        except Exception as e:
            return np.nan, f"test error: {e}"
        return pval, None

    p, err = safe_little_test(actigraphy_missingness_df.drop(columns=["subject","timepoint","Wear_Time"]))
    print("p:", p, "err:", err)

    # Query all fitbit data
    fitbit_df = con.execute(f"SELECT subject, timepoint, Wear_Time, * FROM fitbit_data").df()
    print(f"Creating missingness pattern...")
    fitbit_missingness_df = create_missingness_df(fitbit_df, "fitbit")
    fitbit_df = None  # free memory
    fitbit_mcar = mcar_test.little_mcar_test(fitbit_missingness_df.drop(columns=["subject", "timepoint", "Wear_Time"]))
    fitbit_missingness_df = None  # free memory

    # Query actigraphy columns
    actigraphy_cols = ["Calories_Cal1m", "Steps_Stps1m", "METs_METs1m"]
    actigraphy_df = con.execute(f"SELECT subject, timepoint, Wear_Time, {', '.join(actigraphy_cols)} FROM fitbit_data").df()
    # Create missingness patterns for actigraphy data
    actigraphy_missingness_df = create_missingness_df(actigraphy_df, "actigraphy")
    actigraphy_df = None  # free memory
    # Conduct Little's test for actigraphy data
    actigraphy_mcar = mcar_test.little_mcar_test(actigraphy_missingness_df.drop(columns=["subject", "timepoint", "Wear_Time"]))
    actigraphy_missingness_df = None  # free memory

    actigraphy_df.head()
    actigraphy_df.shape
    actigraphy_missingness_df.head()
    actigraphy_missingness_df.shape
    actigraphy_missingness_df["Wear_Time"].min(), actigraphy_missingness_df["Wear_Time"].max()

    daily_means = actigraphy_df.set_index("Wear_Time")[actigraphy_cols].resample("D").mean().reset_index()
    daily_means.head(10)

    # Query heart rate columns
    heart_rate_df = con.execute(f"SELECT subject, timepoint, Wear_Time, Value_HR1m FROM fitbit_data").df()
    # Create missingness patterns for heart rate data
    heart_rate_missingness_df = create_missingness_df(heart_rate_df, "heart_rate")
    heart_rate_df = None  # free memory
    # Conduct Little's test for heart rate data
    heart_rate_mcar = mcar_test.little_mcar_test(heart_rate_missingness_df.drop(columns=["subject", "timepoint", "Wear_Time"]))
    heart_rate_missingness_df = None  # free memory

    # Query sleep columns
    sleep_df = con.execute(f"SELECT subject, timepoint, Wear_Time, value_Slp1m FROM fitbit_data").df()
    # Create missingness patterns for sleep data
    sleep_missingness_df = create_missingness_df(sleep_df, "sleep")
    sleep_df = None  # free memory
    # Conduct Little's test for sleep data
    sleep_mcar = mcar_test.little_mcar_test(sleep_missingness_df.drop(columns=["subject", "timepoint", "Wear_Time"]))
    sleep_missingness_df = None  # free memory

    print(f"Actigraphy missingness MCAR test: p-value={actigraphy_mcar:.4f}")
    print(f"Heart rate missingness MCAR test: p-value={heart_rate_mcar:.4f}")
    print(f"Sleep missingness MCAR test: p-value={sleep_mcar:.4f}")
