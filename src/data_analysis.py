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
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

def mri_clustering(dem_df, selected_subjects, output_path = Path("output")):
    '''
    This function performs clustering to identify subtypes of depression based on the selected subjects' MRI ROI data.
    It uses several different clustering algorithms (HBDSCAN and Bayesian Gaussian Mixture Models).
    Clustering stability is assessed using bootstrapping and assessing stability using the Jaccard index.
    Algorithms are compared based on their Silhouette coefficient, Density-Based Clustering Validation (DBCV) score, and Davies-Bouldin Index (DBI).

    Parameters:
        dem_df (DataFrame): DataFrame containing demographic information for the selected subjects
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their MRI ROI z-scores
    Returns:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their assigned cluster labels based on their MRI ROI data

    NOTE: This function is a WIP
    TODO: Finish implementing tuning of HDBSCAN parameters

    '''

    # Read in the z-scores for the selected subjects and their MRI ROI data
    z_scores_path = os.path.join(output_path, "normative_modelling", "results", "Z_mri_norm.csv")
    subject_scores = pd.read_csv(z_scores_path)

    # Filter to selected subjects
    selected_subjects = selected_subjects.merge(subject_scores, left_on="subject_ids", right_on="subject_ids", how="left")

    selected_subjects.drop(columns=["observations", "composite_z", "rank"], inplace=True)
  
    selected_subjects.head()
    selected_subjects.shape

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
    embedding = reducer.fit_transform(selected_subjects.drop(columns=["subject_ids"]))
    selected_subjects["embedding_1"] = embedding[:, 0]
    selected_subjects["embedding_2"] = embedding[:, 1]

    # Dimensionality reduction using PCA for comparison
    pca = PCA(n_components=0.9, random_state=42)
    X_pca = pca.fit_transform(selected_subjects.drop(columns=["subject_ids", "embedding_1", "embedding_2"]))
    X_pca.shape
    selected_subjects.shape
    # Evaluate local dimensionality reduction validity using knn overlap and trustworthiness
    def knn_overlap(X_orig, X_emb, k=10):
        nn_orig = NearestNeighbors(n_neighbors=k+1).fit(X_orig)
        nn_emb  = NearestNeighbors(n_neighbors=k+1).fit(X_emb)
        idx_orig = nn_orig.kneighbors(return_distance=False)[:,1:]  # exclude self
        idx_emb  = nn_emb.kneighbors(return_distance=False)[:,1:]
        overlaps = [(len(set(a).intersection(b))/k) for a,b in zip(idx_orig, idx_emb)]
        return np.mean(overlaps)
    
    knn_overlap_score = knn_overlap(selected_subjects.drop(columns=["subject_ids", "embedding_1", "embedding_2"]), embedding)
    trustworthiness_score = trustworthiness(selected_subjects.drop(columns=["subject_ids", "embedding_1", "embedding_2"]), embedding, n_neighbors=10)
    print(f"PaCMAP dimensionality reduction validity: KNN overlap={knn_overlap_score:.4f}, Trustworthiness={trustworthiness_score:.4f}")

    # Evaluate global dimensionality reduction validity using pairwise distance correlation
    def pairwise_distance_correlation(X_orig, X_emb):
        from scipy.spatial.distance import pdist, squareform
        dist_orig = squareform(pdist(X_orig))
        dist_emb = squareform(pdist(X_emb))
        corr = np.corrcoef(dist_orig.flatten(), dist_emb.flatten())[0, 1]
        return corr
    distance_correlation = pairwise_distance_correlation(selected_subjects.drop(columns=["subject_ids", "embedding_1", "embedding_2"]), embedding)
    print(f"PaCMAP global distance preservation: Pairwise distance correlation={distance_correlation:.4f}")
    distance_correlation_pca = pairwise_distance_correlation(selected_subjects.drop(columns=["subject_ids", "embedding_1", "embedding_2"]), X_pca)
    print(f"PCA global distance preservation: Pairwise distance correlation={distance_correlation_pca:.4f}")

    # Create reference clustering on the lower dimensional data
    hdbscan_clusterer_pacmap = hdbscan.HDBSCAN(min_cluster_size=10)
    reference_labels_hdbscan = pd.Series(
        hdbscan_clusterer_pacmap.fit_predict(selected_subjects[["embedding_1", "embedding_2"]]),
        index=selected_subjects.index,
        name="reference_labels_hdbscan",
    )
    hdbscan_clusterer_pca = hdbscan.HDBSCAN(min_cluster_size=10)
    reference_labels_hdbscan_pca = pd.Series(
        hdbscan_clusterer_pca.fit_predict(X_pca),
        index=selected_subjects.index,
        name="reference_labels_hdbscan_pca",
    )
    print(f"HDBSCAN clustering on PaCMAP embedding: Number of unique subtypes discovered: {len(set(reference_labels_hdbscan)) - (1 if -1 in reference_labels_hdbscan else 0)}")  # Exclude -1 if it exists, which represents subjects not assigned to any subtype
    print(f"HDBSCAN clustering on PCA embedding: Number of unique subtypes discovered: {len(set(reference_labels_hdbscan_pca)) - (1 if -1 in reference_labels_hdbscan_pca else 0)}")  # Exclude -1 if it exists, which represents subjects not assigned to any subtype
    #gmm_clusterer = BayesianGaussianMixture(n_components=3, max_iter=100, random_state=42)
    #reference_labels_gmm = gmm_clusterer.fit_predict(selected_subjects[["embedding_1", "embedding_2"]])

    # Create mask for non-noise points in HDBSCAN and GMM to use for alignment and scoring with label alignment
    mask_hdbscan = (reference_labels_hdbscan != -1)
    #mask_gmm = (reference_labels_gmm != -1)

    jaccard_indices_hdbscan = []
    #jaccard_indices_gmm = []

    for mcs in [5, 10, 15, 20]:  # Varying min_cluster_size for HDBSCAN
        for ms in [5, 10, 20]:  # Varying min_samples for HDBSCAN
            for i in tqdm(range(100), desc="Bootstrapping for clustering stability"):
                # Resample subjects with replacement
                bootstrap_sample = selected_subjects[["embedding_1", "embedding_2"]].sample(frac=1, replace=True, random_state=i)
                bootstrap_indices = bootstrap_sample.index

                # Get the reference labels for the bootstrap sample and align comparisons on the sampled rows
                ref_clean_hdbscan = reference_labels_hdbscan.loc[bootstrap_indices].reset_index(drop=True)
                #ref_clean_gmm = reference_labels_gmm[bootstrap_sample.index][mask_gmm]

                # HDBSCAN clustering
                hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
                boot_labels = pd.Series(hdbscan_clusterer.fit_predict(bootstrap_sample), index=bootstrap_indices).reset_index(drop=True)
                non_noise_mask = (ref_clean_hdbscan != -1) & (boot_labels != -1)
                if non_noise_mask.sum() == 0:
                    jaccard_indices_hdbscan.append(np.nan)
                    continue
                aligned_boot_labels = _align_labels(ref_clean_hdbscan[non_noise_mask], boot_labels[non_noise_mask])
                jaccard_hdbscan = jaccard_score(ref_clean_hdbscan[non_noise_mask], aligned_boot_labels, average="macro")
                jaccard_indices_hdbscan.append(jaccard_hdbscan)

                # Gaussian Mixture Models clustering
                #gmm_clusterer = BayesianGaussianMixture(n_components=3, max_iter=100, random_state=i)
                #boot_labels = gmm_clusterer.fit_predict(bootstrap_sample)
                #boot_clean = boot_labels[mask_gmm]
                #aligned_boot_labels = _align_labels(ref_clean_gmm[bootstrap_sample.index], boot_clean)
                #jaccard_gmm = jaccard_score(reference_labels_gmm[bootstrap_sample.index], aligned_boot_labels, average="macro")
                #jaccard_indices["gmm"].append(jaccard_gmm)
            print(f"HDBSCAN clustering stability (Jaccard index): mean={np.mean(jaccard_indices_hdbscan):.4f}, std={np.std(jaccard_indices_hdbscan):.4f}")
            print(f"Number of unique subtypes discovered by HDBSCAN: {len(set(reference_labels_hdbscan)) - (1 if -1 in reference_labels_hdbscan else 0)}")  # Exclude -1 if it exists, which represents subjects not assigned to any subtype
            print(f"Average cluster size for HDBSCAN: {pd.Series(reference_labels_hdbscan).value_counts().mean():.2f}")
            #print(f"GMM clustering stability (Jaccard index): mean={np.mean(jaccard_indices['gmm']):.4f}, std={np.std(jaccard_indices['gmm']):.4f}")

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
