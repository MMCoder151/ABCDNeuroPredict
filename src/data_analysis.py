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
from sklearn.metrics import jaccard_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, matthews_corrcoef
from pyampute.exploration.mcar_statistical_tests import MCARTest
import pacmap
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ParameterGrid
from statsmodels.stats.multitest import multipletests
from pygam import LinearGAM, s, l
from src.mri_rois import mri_rois
from scipy.stats import fisher_exact

def _cohens_d(x, y):
    """
    Calculate Cohen's d effect size between two groups.
    Parameters:
        x (array-like): Values for group 1
        y (array-like): Values for group 2
    Returns:
        d (float): Cohen's d effect size
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def _cohens_d_std(x, y, d=None):
    """
    Approximate standard deviation (standard error) of Cohen's d.
    Parameters:
        x (array-like): Values for group 1
        y (array-like): Values for group 2
        d (float, optional): Precomputed Cohen's d
    Returns:
        float: Standard deviation (SE) estimate for Cohen's d
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    if d is None:
        d = _cohens_d(x, y)
    var_d = ((nx + ny) / (nx * ny)) + ((d ** 2) / (2 * (nx + ny - 2)))
    return np.sqrt(var_d)

def extract_mri_rois(dta_path_tabular, dta_path, mri_meta_df, overwrite = True, output_path=Path("output")):
    '''
    This function analyses group differences between depressed an non-depressed subjects for each MRI feature in the specified mri_files to 
    extract the MRI ROIs that show significant differences between the two groups.
    1. Extract binary depression marker
    2. For each MRI feature, perform regression analysis using a generalized additive model (GAM) to account for sex and non-linear age influence and extract p-values for group differences
    3. Return a list of MRI ROIs that show significant differences between the two groups (p < 0.05)
    Parameters:
        dta_path_tabular (Path): Path to the tabular data directory
        dta_path (Path): Path to the raw data directory
    Returns:
        mri_rois (list): List of MRI ROIs that show significant differences between depressed and non-depressed subjects
        mri_rois_results (DataFrame): DataFrame containing the regression results for each MRI feature
        mri_rois_results (CSV): CSV file containing the regression results for each MRI feature
    NOTE: This function currently doesn't use DuckDB 
    '''
    if overwrite == False:
        existing_results_path = output_path / "mri_rois_results.csv"
        if existing_results_path.exists():
            print(f"Overwrite set to False. Loading existing results.")
            gam_results_df = pd.read_csv(existing_results_path)
            mri_rois_sig = gam_results_df.loc[gam_results_df["significant_fdr"], "mri_feature"].tolist()
            return mri_rois_sig, gam_results_df
        else:
            print(f"No existing results found at {existing_results_path}. Running analysis.")

    # Read in clinical data for depression marker
    youth_directory = dta_path_tabular / "mh_y_ksads__dep.tsv"
    parent_directory = dta_path_tabular / "mh_p_ksads__dep.tsv"

    ksads_youth = pd.read_csv(youth_directory, sep="\t")
    ksads_parent = pd.read_csv(parent_directory, sep="\t")
    
    # Filter to only include subjects and timepoints that are present in mri_meta_df
    ksads_youth = ksads_youth.merge(
        mri_meta_df[["subject", "timepoint"]],
        left_on=["participant_id", "session_id"],
        right_on=["subject", "timepoint"],
        how="inner"
    )
    ksads_parent = ksads_parent.merge(
        mri_meta_df[["subject", "timepoint"]],
        left_on=["participant_id", "session_id"],
        right_on=["subject", "timepoint"],
        how="inner"
    )

    # Filter to only include the first timepoint for each subject
    ksads_youth = ksads_youth.sort_values(by=["participant_id", "session_id"]).groupby("participant_id").first().reset_index()
    ksads_parent = ksads_parent.sort_values(by=["participant_id", "session_id"]).groupby("participant_id").first().reset_index()

    # Get list of depressed subjects based on KSADS depression diagnosis (youth and parent report)
    diagnosis_cols_youth = {#"mh_y_ksads__dep__mdd__partrem_dx"  :"Diagnosis: Major depressive disorder (F32.4) - Partial remission [Youth]",
                            "mh_y_ksads__dep__mdd__pres_dx"     :"Diagnosis: Major depressive disorder - Present [Youth]",
                            #"mh_y_ksads__dep__pdd__oth__pres_dx":"Diagnosis: Other specified depressive disorder, persistent depressive disorder (impairment does not meet full criteria) (F32.8) - Present [Youth]",
                            #"mh_y_ksads__dep__pdd__partrem_dx"  :"Diagnosis: Persistent depressive disorder (Dysthymia) (F34.1) - Partial remission [Youth]",
                            "mh_y_ksads__dep__pdd__pres_dx"     :"Diagnosis: Persistent depressive disorder (Dysthymia) (F34.1) - Present [Youth]"}
    diagnosis_cols_parent = {#"mh_p_ksads__dep__mdd__partrem_dx"  :"Diagnosis: Major depressive disorder (F32.4) - Partial remission [Parent]",
                            "mh_p_ksads__dep__mdd__pres_dx"     :"Diagnosis: Major depressive disorder - Present [Parent]",
                            #"mh_p_ksads__dep__pdd__oth__pres_dx":"Diagnosis: Other specified depressive disorder, persistent depressive disorder (impairment does not meet full criteria) (F32.8) - Present [Parent]",
                            #"mh_p_ksads__dep__pdd__partrem_dx"  :"Diagnosis: Persistent depressive disorder (Dysthymia) (F34.1) - Partial remission [Parent]",
                            "mh_p_ksads__dep__pdd__pres_dx"     :"Diagnosis: Persistent depressive disorder (Dysthymia) (F34.1) - Present [Parent]"}
    
    # Create a binary depression marker for each subject based on youth and parent report
    diagnosis_youth_cols = list(diagnosis_cols_youth.keys())
    y_depr = (ksads_youth[diagnosis_youth_cols] == 1).any(axis=1)

    diagnosis_parent_cols = list(diagnosis_cols_parent.keys())
    p_depr = (ksads_parent[diagnosis_parent_cols] == 1).any(axis=1)

    depr = y_depr | p_depr

    # Create a binary depression marker for each subject
    subjects_depr = set(ksads_youth.loc[depr, "participant_id"]) | set(ksads_parent.loc[depr, "participant_id"])

    # Read in MRI data for each subject and timepoint from mri_files
    mri_files, roi_names = mri_rois()
    # Filter to only include the last three mri_files (i.e., the MRI features of interest)
    mri_files = mri_files[-3:]
    mri_df = []
    for mri_file in mri_files:
        mri_df.append(pd.read_csv(dta_path / "phenotype" / mri_file, sep="\t"))

    # Filter to only include subjects and timepoints that are present in mri_meta_df
    mri_df = [df.merge(mri_meta_df[["subject", "timepoint"]], left_on=["participant_id", "session_id"], right_on=["subject", "timepoint"], how="inner") for df in mri_df]

    # Filter to only include the first timepoint for each subject
    mri_df = [df.sort_values(by=["participant_id", "session_id"]).groupby("participant_id").first().reset_index() for df in mri_df]

    # TODO: Add a check to ensure subjects and timepoints match between mri and clinical data

    # Add depression marker to each MRI dataframe
    for df in mri_df:
        df["depression_marker"] = df["participant_id"].apply(lambda x: 1 if x in subjects_depr else 0)

    # Get demographic information
    stc_df = pd.read_csv(dta_path / "participants.tsv", sep="\t")
    
    # Append age from clinical file and sex from dem_df to the mri dataframe
    mri_df = [df.merge(ksads_youth[["participant_id", "mh_y_ksads__dep_age"]], on="participant_id", how="left") for df in mri_df]
    mri_df = [df.merge(stc_df[["participant_id", "sex"]], on="participant_id", how="left") for df in mri_df]

    # Add scan_site from mri_meta_df to the mri dataframe
    mri_meta_first = mri_meta_df.sort_values(["subject", "timepoint"]).groupby("subject").first().reset_index()
    mri_df = [
        df.merge(mri_meta_first[["subject", "scan_site"]], left_on="participant_id", right_on="subject", how="left")
        .drop(columns=["subject"], errors="ignore")  # drop the redundant key column from this merge
        for df in mri_df
    ]

    # Encode scan_site to numerical values
    scan_site_mapping = {site: idx for idx, site in enumerate(mri_meta_df["scan_site"].unique())}
    for df in mri_df:
        df["scan_site"] = df["scan_site"].map(scan_site_mapping)

    # Print number and percentage of depressed subjects in the MRI dataframe
    depressed_count = mri_df[0]["depression_marker"].sum()
    total_count = len(mri_df[0])
    print(f"Number of depressed subjects in MRI dataframe: {depressed_count} ({(depressed_count/total_count)*100:.2f}%)")

    mri_df[0]["depression_marker"].value_counts()

    print(len(subjects_depr))
    print(ksads_youth["participant_id"].nunique())

    print(ksads_youth.shape)
    print(ksads_youth["participant_id"].nunique())
    print(ksads_youth.shape[0] - ksads_youth["participant_id"].nunique())  # excess rows if duplicated

    # Recode sex from M/F to 0/1
    for df in mri_df:
        df["sex"] = df["sex"].map({"M": 0, "F": 1})

    # Get total intracranial volume (TIV)
    subcortical_vol = pd.read_csv(dta_path / "phenotype" / "mr_y_smri__vol__aseg.tsv", sep="\t")
    mri_df = [df.merge(subcortical_vol[["participant_id", "mr_y_smri__vol__aseg__icv_sum"]], on="participant_id", how="left") for df in mri_df]

    # Drop subjects with sex, age or TIV missing
    mri_df_filtered = [df.dropna(subset=["sex", "mh_y_ksads__dep_age", "mr_y_smri__vol__aseg__icv_sum"]) for df in mri_df]

    # Perform regression analysis per MRI feature while accounting for sex and non-linear age influence using a generalized additive model (GAM) and extract p-values for group differences
    # Also calculate the effect size (Cohen's d) for each MRI feature to quantify the magnitude of group differences
    gam_results = []
    for df in tqdm(mri_df_filtered, desc="Performing GAM regression analysis for MRI features"):
        for col in df.columns:
            if col not in ["participant_id", "session_id", "subject", "subject_x", "subject_y", "acq_time", "depression_marker", "mh_y_ksads__dep_age", "sex", 
                           "mr_y_smri__vol__aseg__icv_sum", "timepoint", "timepoint_x", "timepoint_y", "scan_site"]:
                model_cols = ["mh_y_ksads__dep_age", "sex", "depression_marker", "mr_y_smri__vol__aseg__icv_sum", "scan_site", col]
                df_fit = df[model_cols].replace([np.inf, -np.inf], np.nan).dropna()
                n_dropped = len(df) - len(df_fit)
                if n_dropped > 0:
                    print(f"{col}: dropped {n_dropped} rows ({(n_dropped/len(df)*100):.2f}%) with missing/invalid data")
                try:
                    gam = LinearGAM(s(0) + l(1) + l(2) + l(3) + l(4)).fit(df_fit[["mh_y_ksads__dep_age", "scan_site", "sex","mr_y_smri__vol__aseg__icv_sum", "depression_marker"]], df_fit[col])
                    group_dep = df_fit.loc[df_fit["depression_marker"] == 1, col]
                    group_nondep = df_fit.loc[df_fit["depression_marker"] == 0, col]
                    effect_size = _cohens_d(group_dep, group_nondep)
                    effect_size_std = _cohens_d_std(group_dep, group_nondep, d=effect_size)
                    gam_results.append((col, gam.statistics_['p_values'][4], effect_size, effect_size_std))  # p-value for depression_marker
                except Exception as e:
                    print(f"Error occurred while fitting GAM model for column {col}: {e}")
    # Combine GAM p-values, effect sizes, and effect size standard deviations into a single DataFrame
    results_df = pd.DataFrame(gam_results, columns=["mri_feature", "p_value", "effect_size", "effect_size_std"])
    results_df = results_df.sort_values("p_value")

    # Perform FDR correction for multiple comparisons
    print("Performing FDR correction for multiple comparisons...")
    rejected, corrected_p_values, _, _ = multipletests(results_df["p_value"], alpha=0.05, method='fdr_bh')
    results_df["corrected_p_value"] = corrected_p_values
    results_df["significant_fdr"] = rejected

    # Print the number of significant MRI ROIs after FDR correction
    num_significant_rois = results_df["significant_fdr"].sum()
    print(f"Number of significant MRI ROIs after FDR correction: {num_significant_rois}")

    # Save to CSV
    results_df.to_csv(output_path / "mri_rois_results.csv", index=False)

    # Return list of MRI ROIs that show significant differences between depressed and non-depressed subjects (p < 0.05)
    mri_rois_sig = results_df.loc[results_df["significant_fdr"], "mri_feature"].tolist()

    # Return the list of MRI ROIs and the results dataframe
    return mri_rois_sig, results_df

def mri_clustering(data, n_clusters=None, max_clusters=None,dr=None, dr_params=None, cl=None, cl_params=None, mri_meta_df=None, output_path = Path("output"), clustering_output = None, bootstrapping = True, overwrite = True):
    '''
    This function performs clustering to identify subtypes of depression based on the selected subjects' MRI ROI data.
    It uses several different clustering algorithms (HBDSCAN and Bayesian Gaussian Mixture Models).
    Clustering stability is assessed using bootstrapping and assessing stability using the Jaccard index.
    Algorithms are compared based on their Silhouette coefficient, Density-Based Clustering Validation (DBCV) score, and Davies-Bouldin Index (DBI).

    Parameters:
        data (DataFrame): DataFrame containing the data to cluster
        n_clusters (int): Number of clusters to form
        output_path (str or Path): Path to save clustering results and visualizations
        clustering_output (str or Path): Path to save clustering output
        bootstrapping (bool): Whether to perform bootstrapping for cluster stability assessment
        overwrite (bool): Whether to overwrite existing clustering results if they exist
    Returns:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their assigned cluster labels based on their MRI ROI z-scores 
    '''

    # Create clustering output path
    if clustering_output is None:
        clustering_output_path = os.path.join(output_path, "clustering")
        os.makedirs(clustering_output_path, exist_ok=True)
    else: 
        clustering_output_path = os.path.join(output_path, str(clustering_output))
        os.makedirs(clustering_output_path, exist_ok=True)

    # Load existing clustering results if overwrite is set to False
    if overwrite == False:
        existing_results_path = os.path.join(clustering_output_path, "subject_labels.csv")
        if os.path.exists(existing_results_path):
            print(f"Overwrite set to False. Loading existing results.")
            results_df = pd.read_csv(existing_results_path)
            return results_df
        else:
            print(f"No existing clustering results found at {existing_results_path}. Running clustering analysis.")

    subject_id_col = None
    for candidate in ["subject_ids", "subject", "participant_id", "subject_id"]:
        if candidate in data.columns:
            subject_id_col = candidate
            break

    if subject_id_col is not None:
        subject_ids = data[subject_id_col].reset_index(drop=True).copy()
    else:
        subject_ids = pd.Series(data.index.astype(str), name="subject_ids")

    data = data.reset_index(drop=True)

    # Drop columns that are not needed for clustering
    columns_to_drop = ["subject_ids", "observations", "composite_z", "rank", "subject", "timepoint", "scan_site", "sex", "age", "mr_y_smri__vol__aseg__icv_sum", 
                       "participant_id", "session_id", "subject_x", "subject_y", "timepoint_x", "timepoint_y", "acq_time"]
    for col in columns_to_drop:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Drop rows with missing values and keep subject identifiers aligned with the remaining rows
    valid_rows = data.notna().all(axis=1)
    data = data.loc[valid_rows].reset_index(drop=True)
    subject_ids = subject_ids.loc[valid_rows].reset_index(drop=True)

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
    
    # Create a list of percentage values of max dimensions with the minimum of 2 dimensions for dimensionality reduction
    max_dims = data.shape[1] - 1  # Maximum number of dimensions for dimensionality reduction
    dim_percentages = [0.1, 0.25, 0.5, 0.75, 0.9]
    dr_components = [max(2, int(round(max_dims * pct, 0))) for pct in dim_percentages]

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
        "n_components": dr_components, 
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

    # Drop dr_models from analysis that are not in dr if dr is not None
    if dr is not None:
        invalid_dr = [model for model in dr if model not in ["PaCMAP", "PCA", "UMAP"]]
        if invalid_dr:
            raise ValueError(f"Invalid dimensionality reduction model(s): {invalid_dr}. Must be one of ['PaCMAP', 'PCA', 'UMAP'].")
        dr_models = [model for model in dr_models if model[0] in dr]

    if dr_params is not None:
        dr_models = [(name, model, {**params, **dr_params}) for name, model, params in dr_models]

    if n_clusters is not None:
        kmeans_params = list(ParameterGrid({
            "n_clusters": [n_clusters],
            #"random_state": [42]
        }))

        agglomerative_params = list(ParameterGrid({
            "n_clusters": [n_clusters],
            "linkage": ["ward", "complete", "average"]
        }))

        bayesian_gmm_params = list(ParameterGrid({
            "n_components": [n_clusters],
            #"random_state": [42]
        }))

        hdbscan_params = list(ParameterGrid({
        "min_cluster_size": [10, 15, 20],
        "min_samples": [10, 15, 20],
        "cluster_selection_epsilon": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        }))

        cl_models = (
            [("KMeans", KMeans, p) for p in kmeans_params] +
            [("AgglomerativeClustering", AgglomerativeClustering, p) for p in agglomerative_params] +
            [("BayesianGMM", BayesianGaussianMixture, p) for p in bayesian_gmm_params] +
            [("HDBSCAN", hdbscan.HDBSCAN, p) for p in hdbscan_params]
        )
    else:

        hdbscan_params = list(ParameterGrid({
            "min_cluster_size": [10, 15, 20],
            "min_samples": [10, 15, 20],
            "cluster_selection_epsilon": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        }))

        bayesian_gmm_params = list(ParameterGrid({
            "n_components": [5, 10, 15, 20],
            #"random_state": [42]
        }))

        kmeans_params = list(ParameterGrid({
            "n_clusters": [5, 10, 15, 20],
            #"random_state": [42]
        }))

        agglomerative_params = list(ParameterGrid({
            "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "linkage": ["ward", "complete", "average"]
        }))

        cl_models = (
            [("HDBSCAN", hdbscan.HDBSCAN, p) for p in hdbscan_params] +
            [("BayesianGMM", BayesianGaussianMixture, p) for p in bayesian_gmm_params] +
            [("KMeans", KMeans, p) for p in kmeans_params] +
            [("AgglomerativeClustering", AgglomerativeClustering, p) for p in agglomerative_params]
        )
    
    # Drop cl_models from analysis that are not in cl if cl is not None
    if cl is not None:
        invalid_cl = [model for model in cl if model not in ["HDBSCAN", "BayesianGMM", "KMeans", "AgglomerativeClustering"]]
        if invalid_cl:
            raise ValueError(f"Invalid clustering model(s): {invalid_cl}. Must be one of ['HDBSCAN', 'BayesianGMM', 'KMeans', 'AgglomerativeClustering'].")
        cl_models = [model for model in cl_models if model[0] in cl]
    
    if cl_params is not None:
        cl_models = [(name, model, {**params, **cl_params}) for name, model, params in cl_models]

    results = []

    for dr_name, DR, dr_params in tqdm(dr_models, desc="Analysing cluster solutions", position=0):
        dr_model = DR(**dr_params)
        X_dr   = dr_model.fit_transform(data)

        # Evaluate dimensionality reduction
        knn_overlap_score = knn_overlap(data, X_dr)
        trustworthiness_score = trustworthiness(data, X_dr, n_neighbors=10)
        pairwise_distance = pairwise_distance_correlation(data, X_dr)

        for cl_name, CL, cl_params in tqdm(cl_models, desc=f"Fitting CL models with {dr_name}", position=1, leave=False):
            cl_model = CL(**cl_params)
            labels = cl_model.fit_predict(X_dr)
            n_dimensions = X_dr.shape[1]

            # Skip degenerate solutions
            n_clusters = len(np.unique(labels[labels != -1]))
            noise_pct  = (labels == -1).sum() / len(labels)
            if n_clusters < 2 or noise_pct > 0.20:
                continue

            # Evaluate clusters in original space without noise points
            mask = labels != -1
            sil = silhouette_score(data[mask], labels[mask])  
            db  = davies_bouldin_score(data[mask], labels[mask])
            ch  = calinski_harabasz_score(data[mask], labels[mask])

            # If mri_meta_df is provided, evaluate clustering against depression marker using adjusted rand index and Matthews correlation coefficient
            if mri_meta_df is not None:
                merged_df = pd.DataFrame({
                    "subject_ids": subject_ids,
                    "labels": labels
                }).merge(mri_meta_df[["subject", "dep_dx"]], left_on="subject_ids", right_on="subject", how="left")

                # Exclude noise points AND subjects with no matched depression_marker
                merged_df = merged_df[(merged_df["labels"] != -1) & (merged_df["dep_dx"].notna())]

                if merged_df["dep_dx"].nunique() > 1 and merged_df["labels"].nunique() > 1:
                    ari_dpx = adjusted_rand_score(merged_df["dep_dx"], merged_df["labels"])
                    mcc = matthews_corrcoef(merged_df["dep_dx"], merged_df["labels"])
                    contingency = pd.crosstab(merged_df["labels"], merged_df["dep_dx"])
                    _, fisher_p = fisher_exact(contingency) if contingency.shape == (2, 2) else (None, np.nan)
                else:
                    ari_dpx, mcc, fisher_p = np.nan, np.nan, np.nan
            else:
                ari_dpx, mcc, fisher_p = np.nan, np.nan, np.nan

            # Evaluate bootstrap stability with Jaccard index and ari score
            if bootstrapping:
                jaccard_scores = []
                ari_scores = []
                for i in range(10):  # Bootstrapping for clustering stability
                    bootstrap_sample = data.sample(frac=1, replace=True, random_state=i)
                    X_bootstrap = dr_model.fit_transform(bootstrap_sample)
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
                "ari_depression_marker": ari_dpx,
                "mcc_depression_marker": mcc,
                "fisher_p_depression_marker": fisher_p
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
        (results_df["trustworthiness"] > 0.8)
        #(results_df["pairwise_distance_correlation"] > 0.75) &
        #(results_df["knn_overlap"] > 0.5)
    ].sort_values(by="silhouette", ascending=False)
    if max_clusters is not None:
        results_df_filtered = results_df_filtered[(results_df_filtered["n_clusters"] <= max_clusters)]
    elif bootstrapping:
        results_df_filtered = results_df_filtered[(results_df_filtered["mean_jaccard"] > 0.5)]
    results_df_filtered.to_csv(os.path.join(clustering_output_path, "filtered_clustering_results.csv"), index=False)

    if results_df_filtered.empty:
        print("No clustering solutions met the filtering criteria.")
        return data
    
    print(f"Best filtered clustering result: {results_df_filtered.iloc[0].to_dict()}")

    # Rerun with best parameters to get cluster labels for each subject
    best_dr = results_df_filtered.iloc[0]['dr_model']
    best_cl = results_df_filtered.iloc[0]['cl_model']
    best_dr_params = results_df_filtered.iloc[0]['dr_params']
    best_cl_params = results_df_filtered.iloc[0]['cl_params']
    dr_model = next(DR(**params) for name, DR, params in dr_models if name == best_dr and params == best_dr_params)
    cl_model = next(CL(**params) for name, CL, params in cl_models if name == best_cl and params == best_cl_params)
    X_dr = dr_model.fit_transform(data)
    data["label"] = cl_model.fit_predict(X_dr)
    data["subject_ids"] = subject_ids

    # Save cluster labels to CSV
    data[["subject_ids", "label"]].to_csv(os.path.join(clustering_output_path, "subject_labels.csv"), index=False)

    # TODO: Fix visualization

    return data

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

def group_difference_analysis(data, group_col, mri_meta_df, analysis_output = None, output_path = Path("output")):
    '''
    This function performs group difference analysis for the selected subjects' data based on the specified group column.
    It uses a generalized additive model (GAM) to account for non-linear effects of covariates and extracts p-values for group differences.
    It also calculates the effect size (Cohen's d) for each feature to quantify the magnitude of group differences.

    Parameters:
        data (DataFrame): DataFrame containing the data to analyze
        group_col (str): Column name in the DataFrame that defines the groups for comparison
    Returns:
        results_df (DataFrame): DataFrame containing the results of the group difference analysis, including p-values, effect sizes, and effect size standard deviations for each feature
    '''

    def _cohens_d(x, y):
        """
        Calculate Cohen's d effect size between two groups.
        Parameters:
            x (array-like): Values for group 1
            y (array-like): Values for group 2
        Returns:
            d (float): Cohen's d effect size
        """
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
        return (np.mean(x) - np.mean(y)) / pooled_std

    def _cohens_d_std(x, y, d=None):
        """
        Approximate standard deviation (standard error) of Cohen's d.
        Parameters:
            x (array-like): Values for group 1
            y (array-like): Values for group 2
            d (float, optional): Precomputed Cohen's d
        Returns:
            float: Standard deviation (SE) estimate for Cohen's d
        """
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        if d is None:
            d = _cohens_d(x, y)
        var_d = ((nx + ny) / (nx * ny)) + ((d ** 2) / (2 * (nx + ny - 2)))
        return np.sqrt(var_d)
    
    # Add covariates to data from mri_meta_df
    covariate_cols = ["age_at_mri", "sex", "mr_y_smri__vol__aseg__icv_sum", "scan_site"]
    for col in covariate_cols:
        if col in mri_meta_df.columns:
            data[col] = mri_meta_df[col]
        else:
            raise ValueError(f"Covariate column '{col}' not found in mri_meta_df.")
    
    # Recode sex to numeric
    data["sex"] = data["sex"].map({"M": 0, "F": 1})

    # Group difference analysis for each fitbit feature between discovered subtypes using GAM with effect size calculation
    gam_results = []
    for col in tqdm(data.columns, desc="Performing GAM regression analysis for MRI features"):
        if col not in ["participant_id", "session_id", "subject", "subject_x", "subject_y", "acq_time", "depression_marker", "mh_y_ksads__dep_age", "sex", 
                        "mr_y_smri__vol__aseg__icv_sum", "timepoint", "timepoint_x", "timepoint_y", "scan_site", "subject_ids", "label", "labels", "age_at_first_mri", group_col]:
            model_cols = ["age_at_mri", "sex", group_col, "mr_y_smri__vol__aseg__icv_sum", "scan_site", col]
            df_fit = data[model_cols].replace([np.inf, -np.inf], np.nan).dropna()
            n_dropped = len(data) - len(df_fit)
            if n_dropped > 0:
                print(f"{col}: dropped {n_dropped} rows ({(n_dropped/len(data)*100):.2f}%) with missing/invalid data")
            try:
                gam = LinearGAM(s(0) + l(1) + l(2) + l(3) + l(4)).fit(df_fit[["age_at_mri", "scan_site", "sex","mr_y_smri__vol__aseg__icv_sum", group_col]], df_fit[col])
                group_dep = df_fit.loc[df_fit[group_col] == 1, col]
                group_nondep = df_fit.loc[df_fit[group_col] == 0, col]
                effect_size = _cohens_d(group_dep, group_nondep)
                effect_size_std = _cohens_d_std(group_dep, group_nondep, d=effect_size)
                gam_results.append((col, gam.statistics_['p_values'][4], effect_size, effect_size_std))  # p-value for label
            except Exception as e:
                print(f"Error occurred while fitting GAM model for column {col}: {e}")
    # Combine GAM p-values, effect sizes, and effect size standard deviations into a single DataFrame
    results_df = pd.DataFrame(gam_results, columns=["feature", "p_value", "effect_size", "effect_size_std"])
    results_df = results_df.sort_values("p_value")

    # Perform FDR correction for multiple comparisons
    print("Performing FDR correction for multiple comparisons...")
    rejected, corrected_p_values, _, _ = multipletests(results_df["p_value"], alpha=0.05, method='fdr_bh')
    results_df["corrected_p_value"] = corrected_p_values
    results_df["significant_fdr"] = rejected

    # If analysis_output is provided, save the results to a CSV file
    if analysis_output is not None:
        results_df.to_csv(os.path.join(output_path, str(analysis_output)), index=False)
        print(f"Results saved to: {os.path.join(output_path, str(analysis_output))}")

    # Print the number of significant ROIs after FDR correction
    num_significant_rois = results_df["significant_fdr"].sum()
    print(f"Number of significant ROIs after FDR correction: {num_significant_rois}")

    sig_results_df = results_df[results_df["significant_fdr"]]
    # If analysis_output is provided, save the significant results to a CSV file
    if analysis_output is not None:
        sig_results_df.to_csv(os.path.join(output_path, str(analysis_output)), index=False)
        print(f"Significant ROIs saved to: {os.path.join(output_path, str(analysis_output))}")

    return sig_results_df, results_df