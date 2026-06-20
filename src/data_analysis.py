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
from sklearn.metrics import jaccard_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from pyampute.exploration.mcar_statistical_tests import MCARTest
import pacmap
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from itertools import product
from sklearn.model_selection import ParameterGrid

def mri_clustering(selected_subjects, output_path = Path("output"), bootstrapping = True, overwrite = True):
    '''
    This function performs clustering to identify subtypes of depression based on the selected subjects' MRI ROI data.
    It uses several different clustering algorithms (HBDSCAN and Bayesian Gaussian Mixture Models).
    Clustering stability is assessed using bootstrapping and assessing stability using the Jaccard index.
    Algorithms are compared based on their Silhouette coefficient, Density-Based Clustering Validation (DBCV) score, and Davies-Bouldin Index (DBI).

    Parameters:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their MRI ROI z-scores
        output_path (str or Path): Path to save clustering results and visualizations
        bootstrapping (bool): Whether to perform bootstrapping for cluster stability assessment
        overwrite (bool): Whether to overwrite existing clustering results if they exist
    Returns:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their assigned cluster labels based on their MRI ROI z-scores 
    '''

    # Create clustering output path
    clustering_output_path = os.path.join(output_path, "mri_clustering")
    os.makedirs(clustering_output_path, exist_ok=True)

    if overwrite == False:
        existing_results_path = os.path.join(clustering_output_path, "subject_subtypes.csv")
        if os.path.exists(existing_results_path):
            print(f"Overwrite set to False. Loading existing results.")
            results_df = pd.read_csv(existing_results_path)
            return selected_subjects.merge(results_df[["subject_ids", "subtype"]], left_on="subject_ids", right_on="subject_ids", how="left")
        else:
            print(f"No existing clustering results found at {existing_results_path}. Running clustering analysis.")

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
        "n_jobs": [1],
        "n_neighbors": [5, 10, 15, 20],
        "min_dist": [0.0, 0.05, 0.1, 0.2],
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
        "linkage": ["ward", "complete", "average"]
    }))

    cl_models = (
        [("HDBSCAN", hdbscan.HDBSCAN, p) for p in hdbscan_params] +
        [("BayesianGMM", BayesianGaussianMixture, p) for p in bayesian_gmm_params] +
        [("KMeans", KMeans, p) for p in kmeans_params] +
        [("AgglomerativeClustering", AgglomerativeClustering, p) for p in agglomerative_params]
    )
    
    results = []

    for (dr_name, DR, dr_params), (cl_name, CL, cl_params) in tqdm(product(dr_models, cl_models), total=len(dr_models)*len(cl_models), desc="Tuning parameters"):
        dr_model = DR(**dr_params)
        cl_model = CL(**cl_params)

        X_dr   = dr_model.fit_transform(selected_subjects.drop(columns=["subject_ids"]))
        labels = cl_model.fit_predict(X_dr)
        n_dimensions = X_dr.shape[1]

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

        # Evaluate bootstrap stability with Jaccard index and ari score
        if bootstrapping:
            jaccard_scores = []
            ari_scores = []
            for i in range(10):  # Bootstrapping for clustering stability
                bootstrap_sample = selected_subjects.sample(frac=1, replace=True, random_state=i)
                X_bootstrap = dr_model.fit_transform(bootstrap_sample.drop(columns=["subject_ids"]))
                labels_bootstrap = cl_model.fit_predict(X_bootstrap)
                # skip degenerate solutions in bootstrap samples
                if len(np.unique(labels_bootstrap[labels_bootstrap != -1])) < 2 or (labels_bootstrap == -1).sum() / len(labels_bootstrap) > 0.20:
                    continue
                aligned_labels = _align_labels(labels[bootstrap_sample.index], labels_bootstrap)
                jaccard = jaccard_score(labels[bootstrap_sample.index], aligned_labels, average="macro")
                jaccard_scores.append(jaccard)
                ari = adjusted_rand_score(labels[bootstrap_sample.index], aligned_labels)
                ari_scores.append(ari)
            m_jaccard = np.mean(jaccard_scores)
            sd_jaccard = np.std(jaccard_scores)
            m_ari = np.mean(ari_scores)
            sd_ari = np.std(ari_scores)

        results.append({
            "dr_model": dr_name, "dr_params": dr_params, "n_dimensions": n_dimensions,
            "cl_model": cl_name, "cl_params": cl_params,
            "n_clusters": n_clusters, "noise_pct": noise_pct,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "knn_overlap": knn_overlap_score,
            "trustworthiness": trustworthiness_score,
            "pairwise_distance_correlation": pairwise_distance,
        })
        if bootstrapping:
            results[-1].update({
                "mean_jaccard": m_jaccard,
                "std_jaccard": sd_jaccard,
                "mean_ari": m_ari,
                "std_ari": sd_ari
            })

    results_df = pd.DataFrame(results).sort_values(by="silhouette", ascending=False)
    results_df.to_csv(os.path.join(clustering_output_path, "clustering_results.csv"), index=False)

    results_df_filtered = results_df[
        (results_df["silhouette"] > 0.25) &
        (results_df["davies_bouldin"] < 1.0) &
        (results_df["trustworthiness"] > 0.8) &
        (results_df["pairwise_distance_correlation"] > 0.75) &
        (results_df["knn_overlap"] > 0.5)
    ].sort_values(by="silhouette", ascending=False)
    if bootstrapping:
        results_df_filtered = results_df_filtered[(results_df_filtered["mean_jaccard"] > 0.5)]
    results_df_filtered.to_csv(os.path.join(clustering_output_path, "filtered_clustering_results.csv"), index=False)

    if results_df_filtered.empty:
        print("No clustering solutions met the filtering criteria.")
        return selected_subjects
    
    print(f"Best filtered clustering result: {results_df_filtered.iloc[0].to_dict()}")

    # Rerun with best parameters to get cluster labels for each subject
    best_dr = results_df_filtered.iloc[0]['dr_model']
    best_cl = results_df_filtered.iloc[0]['cl_model']
    best_dr_params = results_df_filtered.iloc[0]['dr_params']
    best_cl_params = results_df_filtered.iloc[0]['cl_params']
    dr_model = next(DR(**params) for name, DR, params in dr_models if name == best_dr)
    cl_model = next(CL(**params) for name, CL, params in cl_models if name == best_cl)
    X_dr = dr_model.fit_transform(selected_subjects.drop(columns=["subject_ids"]))
    selected_subjects["subtype"] = cl_model.fit_predict(X_dr) 

    # Save cluster labels to CSV
    selected_subjects[["subject_ids", "subtype"]].to_csv(os.path.join(clustering_output_path, "subject_subtypes.csv"), index=False)

    # TODO: Fix visualization

    # Reduce dimensionality for visualization if not already 2D
    #if X_dr.shape[1] > 2:
    #    dr_vis = umap.UMAP(n_components=2, random_state=42)
    #    X_dr_vis = dr_vis.fit_transform(X_dr)
    #else:        X_dr_vis = X_dr

    # Create colored scatter plot of clusters in dimensionality reduction space
    #plt.figure(figsize=(10, 6))
    #for subtype in np.unique(selected_subjects["subtype"]):
    #    subtype_data = X_dr_vis[selected_subjects["subtype"] == subtype]
    #    plt.scatter(subtype_data[:, 0], subtype_data[:, 1], label=f"Subtype {subtype}", alpha=0.6)
    #plt.title("Clusters in Dimensionality Reduction Space")
    #plt.xlabel("Component 1")
    #plt.ylabel("Component 2")
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig(os.path.join(clustering_output_path, f"clusters_in_dr_space.png"))
    #plt.close()

    # Create cluster profile plots for top 5 clusters
    #top_clusters = selected_subjects["subtype"].value_counts().index[:5]
    #for cluster in top_clusters:
    #    cluster_profile = selected_subjects[selected_subjects["subtype"] == cluster].drop(columns=["subject_ids", "subtype"]).mean()
    #    plt.figure(figsize=(10, 6))
    #    cluster_profile.plot(kind="bar")
    #    plt.title(f"Cluster {cluster} Profile")
    #    plt.ylabel("Mean Z-score")
    #    plt.xticks(rotation=45, ha="right")
    #    plt.tight_layout()
    #    plt.savefig(os.path.join(clustering_output_path, f"cluster_{cluster}_profile.png"))
    #    plt.close()

    # Create cluster centoroid plots in dimensionality reduction space
    #plt.figure(figsize=(10, 6))
    #centroids = (
    #    pd.DataFrame(X_dr_vis)
    #    .groupby(selected_subjects["subtype"])
    #    .mean()
    #)   
    #plt.scatter(centroids[0], centroids[1], s=200, marker="X")
    #plt.title("Cluster Centroids in Dimensionality Reduction Space")
    #plt.xlabel("Component 1")
    #plt.ylabel("Component 2")
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig(os.path.join(clustering_output_path, f"cluster_centroids_in_dr_space.png"))
    #plt.close()

    return selected_subjects

def missingness_analysis(con, fit_meta_df, output_path = Path("output")):
    '''
    This function analyzes the missingness patterns in the fitbit data for the selected subjects for MCAR, MAR, or MNAR missingness
    using Littles test from pyampute.
        1. Queries each fitbit file for the first timepoint of the filtered subjects
        2. For each, creates datetime index (days) with proper missing days based on min and max of the Wear_Time column for each subject and timepoint, 
        and merges with the original data to get missingness patterns
        3. Conducts Little's test for each fitbit domain to assess whether the missingness is MCAR, MAR, or MNAR

    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        fit_meta_df (DataFrame): DataFrame containing the fitbit metadata for the selected subjects
    Returns:
        mcar_results_df (DataFrame): DataFrame containing the results of Little's test for each fitbit domain, including the p-value and any errors encountered during testing

    NOTE: This function is currently broken. Little's test is not computing properly and I have no fucking clue why. WIP
    '''

    # For each subject create datetime index with proper missing days based on min and max of the Wear_Time column, and merge with original data to get missingness patterns
    def create_missingness_df(df, domain_name):
        missingness_list = []
        grouped = df.groupby("subject")
        for subject, group in grouped:
            min_date = group["Wear_Time"].min().floor("D")
            max_date = group["Wear_Time"].max().ceil("D")
            value_cols = [c for c in group.columns if c not in ["subject", "Wear_Time"]]
            daily_means = group.set_index("Wear_Time")[value_cols].resample("D").mean().reset_index()
            date_range = pd.date_range(start=min_date, end=max_date, freq="D")
            date_df = pd.DataFrame({"Wear_Time": date_range})
            merged_df = date_df.merge(daily_means, on="Wear_Time", how="left")
            merged_df["subject"] = subject
            missingness_list.append(merged_df)
        return pd.concat(missingness_list, ignore_index=True)

    # Define safe littles test for debugging
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
    
    mcar_test = MCARTest(method="little")

    # Setup bootstrapping for random selection of subjects to run Little's test with reduced memory load
    bootstrap_results = []
    for i in tqdm(range(10), desc="Bootstrapping Little's test for MCAR"):
        
        bootstrap_subjects = fit_meta_df["subject"].sample(frac=0.25, random_state=i).values
        query = f"""
            WITH earliest_timepoint AS (
            SELECT subject, MIN(timepoint) AS first_tp
            FROM fitbit_data
            WHERE subject IN (SELECT unnest($subjects))
            GROUP BY subject
            ),
            daily_agg AS (
                SELECT 
                    f.subject,
                    CAST(f.Wear_Time AS DATE) AS date,
                    AVG(f.Value_HR1m)       AS Value_HR1m,
                    AVG(f.Steps_Stps1m)     AS Steps_Stps1m,
                    AVG(f.Calories_Cal1m)   AS Calories_Cal1m,
                    AVG(f.METs_METs1m)      AS METs_METs1m,
                    AVG(f.Intensity_Int1m)  AS Intensity_Int1m,
                    AVG(f.value_Slp1m)      AS value_Slp1m
                FROM fitbit_data f
                INNER JOIN earliest_timepoint e
                    ON f.subject = e.subject 
                    AND f.timepoint = e.first_tp
                WHERE f.subject IN (SELECT unnest($subjects))
                GROUP BY f.subject, CAST(f.Wear_Time AS DATE)
            )
            SELECT * FROM daily_agg
            """
        df = con.execute(query, {"subjects": list(bootstrap_subjects)}).df()

        # Drop columns with all missing values and zero variance to avoid errors in Little's test
        #df = df.drop(columns=["subject","timepoint", "Wear_Time", "Level_Slp1m"])  
        df = df.drop(columns=["subject", "date"])
        df = df.dropna(axis=1, how="all")
        numeric_cols = df.select_dtypes(include="number").columns
        zero_var_cols = numeric_cols[df[numeric_cols].std() == 0]
        df = df.drop(columns=zero_var_cols)
        df = df.apply(lambda col: col.astype(float) 
                            if hasattr(col, 'dtype') and pd.api.types.is_extension_array_dtype(col) 
                            else col)

        # Convert columns to numeric and Wear_Time to datetime
        cols_to_convert = [c for c in df.columns if c not in ["subject", "Wear_Time"]]
        df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        #df["Wear_Time"] = pd.to_datetime(df["Wear_Time"])

        #missingness_df = create_missingness_df(df, domain_name=f"bootstrap_{i}")
        #df = None  # free memory

        # --- Diagnostics ---
        #test_df = missingness_df.drop(columns=["subject", "Wear_Time"])
        #print(f"Shape: {test_df.shape}")
        #print(f"Missingness per column (%):\n{test_df.isna().mean().sort_values(ascending=False).head(20)}")
        #print(f"Columns with all NaN: {test_df.isna().all().sum()}")
        #print(f"Columns with zero variance: {(test_df.std() == 0).sum()}")
        #print(f"Sample of data:\n{test_df.head()}")

        #print(f"Shape: {df.shape}")
        #print(f"Missingness per column (%):\n{df.isna().mean().sort_values(ascending=False)}")
        #print(f"Dtype of each column:\n{df.dtypes}")
        #print(f"Any <NA> (pandas NA, not numpy NaN):\n{df.isin([pd.NA]).any()}")
        #print(f"Sample:\n{df.head(10)}")
        # ------------------

        #mcar = mcar_test.little_mcar_test(missingness_df.drop(columns=["subject", "Wear_Time"]))
        mcar = mcar_test.little_mcar_test(df)

        # --- Diagnostics ---
        #print(f"Raw result object: {mcar}")
        #print(f"Type of result: {type(mcar)}")
        # -------------------

        print(f"Bootstrap iteration {i}: Little's test p-value = {mcar}")
        bootstrap_results.append((i, mcar))

    bootstrap_results_df = pd.DataFrame(bootstrap_results, columns=["bootstrap_iteration", "little_mcar_pval", "little_mcar_error"])
    bootstrap_results_df.to_csv(os.path.join(output_path, "fitbit_missingness_mcar_bootstrap_results.csv"), index=False)
    print("Completed bootstrapping for Little's test. Results saved to CSV.")
    print("Average p-value across bootstraps:", bootstrap_results_df["little_mcar_pval"].mean())

    # Get unique column names from fitbit tables
    query = """
    SELECT column_name
    FROM (DESCRIBE fitbit_data)
    """
    fitbit_columns = con.execute(query).df()["column_name"].tolist()
    fitbit_columns.remove("subject")
    fitbit_columns.remove("timepoint")
    fitbit_columns.remove("Wear_Time")

    

    query = f"""
            SELECT subject, Wear_Time, {fitbit_columns[0]}
            FROM fitbit_data   
            WHERE timepoint = (
                SELECT MIN(timepoint)
                FROM fitbit_data f2
                WHERE f2.subject = fitbit_data.subject
            )
            """
    df = con.execute(query).df()

    results = []
    for column in tqdm(fitbit_columns, desc="Testing MCAR for each column"):
        print(column)
        query = f"""
            SELECT subject, Wear_Time, {column}
            FROM fitbit_data   
            WHERE timepoint = (
                SELECT MIN(timepoint)
                FROM fitbit_data f2
                WHERE f2.subject = fitbit_data.subject
            )
            """
        df = con.execute(query).df()
        missingness_df = create_missingness_df(df, column)
        df = None  # free memory
        mcar = mcar_test.little_mcar_test(missingness_df.drop(columns=["subject", "Wear_Time"]))
        print(f"Little's test for {column}: p-value = {mcar}")
        print("Safe test:")
        pval, error = safe_little_test(missingness_df.drop(columns=["subject", "Wear_Time"]))
        print(f"Column: {column}, p-value: {pval}, error: {error}")
        missingness_df = None  # free memory
        results.append((column, mcar, pval, error))
    mcar_results_df = pd.DataFrame(results, columns=["filename", "little_mcar", "safe_little_pval", "safe_little_error"])
    mcar_results_df.to_csv(os.path.join(output_path, "fitbit_missingness_mcar_results.csv"), index=False)

    return mcar_results_df