import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pyampute.exploration.mcar_statistical_tests import MCARTest
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, silhouette_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score, davies_bouldin_score, calinski_harabasz_score
from pyampute.exploration.mcar_statistical_tests import MCARTest
import pacmap
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from itertools import product
from sklearn.model_selection import ParameterGrid

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
        label_mapping = {labels[col]: labels[row] for row, col in zip(row_ind, col_ind)}
        # Apply the mapping to the target labels
        aligned_target = np.array([label_mapping.get(label, label) for label in target])
        return aligned_target 
    
    # Evaluate local dimensionality reduction validity using knn overlap and trustworthiness
    def knn_overlap(X_orig, X_emb, k=10):
        nn_orig = NearestNeighbors(n_neighbors=k+1).fit(X_orig)
        nn_emb  = NearestNeighbors(n_neighbors=k+1).fit(X_emb)
        idx_orig = nn_orig.kneighbors(return_distance=False)[:,1:]  # exclude self
        idx_emb  = nn_emb.kneighbors(return_distance=False)[:,1:]
        overlaps = [(len(set(a).intersection(b))/k) for a,b in zip(idx_orig, idx_emb)]
        return np.mean(overlaps)

    # Evaluate global dimensionality reduction validity using pairwise distance correlation
    def pairwise_distance_correlation(X_orig, X_emb):
        from scipy.spatial.distance import pdist, squareform
        dist_orig = squareform(pdist(X_orig))
        dist_emb = squareform(pdist(X_emb))
        corr = np.corrcoef(dist_orig.flatten(), dist_emb.flatten())[0, 1]
        return corr

    pacmac_grid = list(ParameterGrid({
        "n_components": [2],
        "random_state": [42],
        "n_neighbors": [5, 10, 15, 20],
        "MN_ratio": [0.5, 1.0, 2.0],
        "FP_ratio": [0.5, 1.0, 2.0]
        }))
    
    pca_grid = list(ParameterGrid({
        "n_components": [0.1, 0.25, 0.5, 0.75, 0.9], 
        "random_state": [42], 
        "whiten": [True, False]
        }))
    
    umap_grid = list(ParameterGrid({
        "n_components": [2], 
        "random_state": [42],
        "n_neighbors": [5, 10, 15, 20, 30, 40, 50],
        "min_dist": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "metric": ["euclidean", "manhattan", "cosine"]
        }))

    dr_models = (
        [("PaCMAP", pacmap.PaCMAP, p) for p in pacmac_grid] +
        [("PCA", PCA, p) for p in pca_grid] +
        [("UMAP", umap.UMAP, p) for p in umap_grid]
    )

    hdbscan_params = list(ParameterGrid({
        "min_cluster_size": [5, 10, 15, 20],
        "min_samples": [5, 10, 20]
    }))

    bayesian_gmm_params = list(ParameterGrid({
        "n_components": [5, 10, 15, 20],
        "random_state": [42]
    }))

    kmeans_params = list(ParameterGrid({
        "n_clusters": [5, 10, 15, 20],
        "random_state": [42]
    }))

    agglomerative_params = list(ParameterGrid({
        "n_clusters": [5, 10, 15, 20],
        "linkage": ["ward", "complete", "average", "single"]
    }))

    cl_models = (
        [("HDBSCAN", hdbscan.HDBSCAN, p) for p in hdbscan_params] +
        [("BayesianGMM", BayesianGaussianMixture, p) for p in bayesian_gmm_params] +
        [("KMeans", KMeans, p) for p in kmeans_params] +
        [("AgglomerativeClustering", AgglomerativeClustering, p) for p in agglomerative_params]
    )
    
    results = []

    for (dr_name, DR, dr_params), (cl_name, CL, cl_params) in tqdm(product(dr_models, cl_models), desc="Tuning parameters"):
        dr_model = DR(**dr_params)
        cl_model = CL(**cl_params)

        X_dr   = dr_model.fit_transform(selected_subjects.drop(columns=["subject_ids"]))
        labels = cl_model.fit_predict(X_dr)

        # Skip degenerate solutions
        n_clusters = len(np.unique(labels[labels != -1]))
        noise_pct  = (labels == -1).sum() / len(labels)
        if n_clusters < 2 or noise_pct > 0.20:
            continue

        # Evaluate dimensionality reduction
        knn_overlap_score = knn_overlap(selected_subjects.drop(columns=["subject_ids"]), X_dr)
        trustworthiness_score = trustworthiness(selected_subjects.drop(columns=["subject_ids"]), X_dr, n_neighbors=10)
        pairwise_distance = pairwise_distance_correlation(selected_subjects.drop(columns=["subject_ids"]), X_dr)

        # Evaluate clusters in original space without noise points
        mask = labels != -1
        sil = silhouette_score(selected_subjects.drop(columns=["subject_ids"])[mask], labels[mask])  
        db  = davies_bouldin_score(selected_subjects.drop(columns=["subject_ids"])[mask], labels[mask])
        ch  = calinski_harabasz_score(selected_subjects.drop(columns=["subject_ids"])[mask], labels[mask])

        # Evaluate bootstrap stability with Jaccard index
        jaccard_scores = []
        for i in range(100):  # Bootstrapping for clustering stability
            bootstrap_sample = selected_subjects.sample(frac=1, replace=True, random_state=i)
            X_bootstrap = dr_model.fit_transform(bootstrap_sample.drop(columns=["subject_ids"]))
            labels_bootstrap = cl_model.fit_predict(X_bootstrap)
            aligned_labels = _align_labels(labels[bootstrap_sample.index], labels_bootstrap)
            jaccard = jaccard_score(labels[bootstrap_sample.index], aligned_labels, average="macro")
            jaccard_scores.append(jaccard)
        m_jaccard = np.mean(jaccard_scores)
        sd_jaccard = np.std(jaccard_scores)

        results.append({
            "dr_model": dr_name, "dr_params": dr_params,
            "cl_model": cl_name, "cl_params": cl_params,
            "n_clusters": n_clusters, "noise_pct": noise_pct,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "knn_overlap": knn_overlap_score,
            "trustworthiness": trustworthiness_score,
            "pairwise_distance_correlation": pairwise_distance,
            "mean_jaccard": m_jaccard,
            "std_jaccard": sd_jaccard
        })

    # Create clustering output path
    clustering_output_path = os.path.join(output_path, "mri_clustering")
    os.makedirs(clustering_output_path, exist_ok=True)

    results_df = pd.DataFrame(results).sort_values(by="silhouette", ascending=False)
    results_df.to_csv(os.path.join(clustering_output_path, "clustering_results.csv"), index=False)

    print(f"\nBest clustering result: {results_df.iloc[0].to_dict()}")

    # Rerun with best parameters to get cluster labels for each subject
    best_dr = results_df.iloc[0]['dr_model']
    best_cl = results_df.iloc[0]['cl_model']
    best_dr_params = results_df.iloc[0]['dr_params']
    best_cl_params = results_df.iloc[0]['cl_params']
    dr_model = next(DR(**params) for name, DR, params in dr_models if name == best_dr)
    cl_model = next(CL(**params) for name, CL, params in cl_models if name == best_cl)
    X_dr = dr_model.fit_transform(selected_subjects.drop(columns=["subject_ids"]))
    selected_subjects["subtype"] = cl_model.fit_predict(X_dr) 

    # Save cluster labels to CSV
    selected_subjects[["subject_ids", "subtype"]].to_csv(os.path.join(clustering_output_path, "subject_subtypes.csv"), index=False)

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
