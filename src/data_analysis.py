import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture
import hdbscan
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, silhouette_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, matthews_corrcoef
import pacmap
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ParameterGrid
from statsmodels.stats.multitest import multipletests
from pygam import LinearGAM, s, l, f
from scipy.stats import fisher_exact
from scipy.stats import wilcoxon
from statsmodels.formula.api import ols

def partial_r2(full_formula, reduced_formula, data):
    full = ols(full_formula, data=data).fit()
    reduced = ols(reduced_formula, data=data).fit()

    return (reduced.ssr - full.ssr) / reduced.ssr

def confound_analysis(data_pre, data_post, feature_cols, base_terms, confounds):
    """
    Parameters
    ----------
    data_pre, data_post : DataFrame
        Data before and after harmonization/residualization.

    feature_cols : list[str]
        Features to evaluate.

    base_terms : list[str]
        Terms in the full model.

        Example MRI:
        [
            "bs(visit_age, df=4)",
            "C(sex)",
            "C(scan_site)",
            "TIV_z"
        ]

        Example Fitbit:
        [
            "visit_age",
            "C(sex)"
        ]

    confounds : dict
        Mapping from confound name to the exact model term.

        Example MRI:
        {
            "Age": "bs(visit_age, df=4)",
            "Sex": "C(sex)",
            "Site": "C(scan_site)",
            "TIV": "TIV_z"
        }

        Example Fitbit:
        {
            "Age": "visit_age",
            "Sex": "C(sex)"
        }
    """

    summary = []

    full_r2_pre = []
    full_r2_post = []

    partial_pre = {k: [] for k in confounds}
    partial_post = {k: [] for k in confounds}

    for feature in tqdm(feature_cols, desc="Confound analysis", position=0):

        response = f'Q("{feature}")'

        full_formula = response + " ~ " + " + ".join(base_terms)

        model_pre = ols(full_formula, data=data_pre).fit()
        model_post = ols(full_formula, data=data_post).fit()

        row = {
            "Feature": feature,
            "R2_pre": model_pre.rsquared,
            "R2_post": model_post.rsquared,
        }

        full_r2_pre.append(model_pre.rsquared)
        full_r2_post.append(model_post.rsquared)

        for name, term in confounds.items():

            reduced_terms = [t for t in base_terms if t != term]

            reduced_formula = response + " ~ " + " + ".join(reduced_terms)

            r2_pre = partial_r2(full_formula, reduced_formula, data_pre)
            r2_post = partial_r2(full_formula, reduced_formula, data_post)

            row[f"{name}_partial_R2_pre"] = r2_pre
            row[f"{name}_partial_R2_post"] = r2_post

            partial_pre[name].append(r2_pre)
            partial_post[name].append(r2_post)

        summary.append(row)

    summary_df = pd.DataFrame(summary)

    # Mean row
    mean_row = {"Feature": "Mean"}

    for col in summary_df.columns[1:]:
        mean_row[col] = summary_df[col].mean()

    summary_df = pd.concat(
        [summary_df, pd.DataFrame([mean_row])],
        ignore_index=True,
    )

    # Wilcoxon tests
    wilcoxon_row = {
        "Feature": "Wilcoxon",
        "R2_stat": wilcoxon(full_r2_pre, full_r2_post).statistic,
        "R2_p": wilcoxon(full_r2_pre, full_r2_post).pvalue,
    }

    for name in confounds:

        stat, p = wilcoxon(partial_pre[name], partial_post[name])

        wilcoxon_row[f"{name}_stat"] = stat
        wilcoxon_row[f"{name}_p"] = p

    wilcoxon_df = pd.DataFrame([wilcoxon_row])

    return summary_df, wilcoxon_df

def _cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std
 
def _cohens_d_std(x, y, d=None):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    if d is None:
        d = _cohens_d(x, y)
    var_d = ((nx + ny) / (nx * ny)) + ((d ** 2) / (2 * (nx + ny - 2)))
    return np.sqrt(var_d)
 
def _stratified_bootstrap_indices(group_labels, n_bootstraps, rng):
    """Resample WITHIN each group separately, so every replicate preserves
    the original group sizes and always contains both groups."""
    idx_dep = np.where(group_labels == 1)[0]
    idx_nondep = np.where(group_labels == 0)[0]
    boots = []
    for _ in range(n_bootstraps):
        boot_dep = rng.choice(idx_dep, size=len(idx_dep), replace=True)
        boot_nondep = rng.choice(idx_nondep, size=len(idx_nondep), replace=True)
        boots.append(np.concatenate([boot_dep, boot_nondep]))
    return boots
 
def exploratory_group_difference_analysis(
    mri_data_filtered,
    group_col,
    output_path_all,
    output_path_sig,
    overwrite=False,
    n_bootstraps=100,
    seed=0,
):
    if not overwrite and os.path.exists(output_path_all) and os.path.exists(output_path_sig):
        print("Results already exist. Skipping analysis.")
        return pd.read_csv(output_path_sig), pd.read_csv(output_path_all)
 
    rng = np.random.default_rng(seed)
 
    mri_data_filtered = mri_data_filtered.copy()
    mri_data_filtered["scan_site_code"] = mri_data_filtered["scan_site"].astype("category").cat.codes
 
    cols_to_exclude = [
        "participant_id", "session_id",
        "mh_y_ksads__dep__mdd__pres_dx", "mh_p_ksads__dep__mdd__pres_dx",
        "sex", "scan_site", "scan_site_code", "visit_age", "age_squared", "visit_age_squared",
        "mr_y_smri__vol__aseg__icv_sum", group_col,
    ]
 
    nuisance_cols = ["visit_age", "scan_site_code", "sex", "mr_y_smri__vol__aseg__icv_sum"]
    full_cols = nuisance_cols + [group_col]
    group_term_index = full_cols.index(group_col)
 
    group_labels = mri_data_filtered[group_col].to_numpy()
    boot_indices = _stratified_bootstrap_indices(group_labels, n_bootstraps, rng)
 
    gam_results = []
    for col in tqdm(mri_data_filtered.columns, desc="GAM group-difference analysis"):
        if col in cols_to_exclude:
            continue
        try:
            y = mri_data_filtered[col]
 
            # --- Point estimate on the real (non-resampled) data ---
            X_full = mri_data_filtered[full_cols]
            gam_full = LinearGAM(s(0) + f(1) + f(2) + l(3) + f(4)).fit(X_full, y)
            p_value = gam_full.statistics_["p_values"][group_term_index]
 
            X_nuisance = mri_data_filtered[nuisance_cols]
            gam_nuisance = LinearGAM(s(0) + f(1) + f(2) + l(3)).fit(X_nuisance, y)
            adjusted_residuals = (y - gam_nuisance.predict(X_nuisance)).to_numpy()
 
            group_dep = adjusted_residuals[group_labels == 1]
            group_nondep = adjusted_residuals[group_labels == 0]
            effect_size = _cohens_d(group_dep, group_nondep)
            effect_size_std_analytic = _cohens_d_std(group_dep, group_nondep, d=effect_size)
 
            # --- Bootstrap distribution (resample residuals, don't refit GAMs) ---
            # --- Bootstrap distribution (refit nuisance GAM each replicate) ---
            boot_effect_sizes = np.empty(n_bootstraps)

            for b, indices in enumerate(boot_indices):

                boot_df = mri_data_filtered.iloc[indices].copy()

                X_boot = boot_df[nuisance_cols]
                y_boot = boot_df[col]
                group_boot = boot_df[group_col].to_numpy()

                try:
                    gam_boot = LinearGAM(
                        s(0) + f(1) + f(2) + l(3)
                    ).fit(X_boot, y_boot)

                    adjusted_boot = (
                        y_boot - gam_boot.predict(X_boot)
                    ).to_numpy()

                    boot_effect_sizes[b] = _cohens_d(
                        adjusted_boot[group_boot == 1],
                        adjusted_boot[group_boot == 0]
                    )

                except Exception:
                    boot_effect_sizes[b] = np.nan
 
            boot_effect_sizes = boot_effect_sizes[np.isfinite(boot_effect_sizes)]

            effect_size_std_boot = np.std(boot_effect_sizes, ddof=1)

            ci_lower, ci_upper = np.percentile(
                boot_effect_sizes,
                [2.5, 97.5]
            )

            sign_stability = np.mean(
                np.sign(boot_effect_sizes) == np.sign(effect_size)
            )
 
            gam_results.append((
                col, p_value, effect_size, effect_size_std_analytic,
                effect_size_std_boot, ci_lower, ci_upper, sign_stability,
            ))
        except Exception as e:
            print(f"Error occurred while fitting GAM model for column {col}: {e}")
 
    results_df = pd.DataFrame(gam_results, columns=[
        "feature", "p_value", "effect_size", "effect_size_std",
        "effect_size_std_boot", "ci_lower_boot", "ci_upper_boot", "sign_stability",
    ])
    results_df = results_df.sort_values("p_value")
 
    print("Performing FDR correction for multiple comparisons...")
    rejected, corrected_p_values, _, _ = multipletests(results_df["p_value"], alpha=0.05, method="fdr_bh")
    results_df["corrected_p_value"] = corrected_p_values
    results_df["significant_fdr"] = rejected
 
    results_df.to_csv(output_path_all, index=False)
    print(f"All results saved to: {output_path_all}")
 
    sig_results_df = results_df[results_df["significant_fdr"]]
    sig_results_df.to_csv(output_path_sig, index=False)
    print(f"Significant ROIs saved to: {output_path_sig}")
 
    num_significant_rois = results_df["significant_fdr"].sum()
    print(f"Number of significant ROIs after FDR correction: {num_significant_rois}")
 
    return sig_results_df, results_df

def exploratory_group_difference_analysis_fitbit(
    mri_data_filtered,
    group_col,
    output_path_all,
    output_path_sig,
    overwrite=False,
    n_bootstraps=100,
    seed=0,
):
    if not overwrite and os.path.exists(output_path_all) and os.path.exists(output_path_sig):
        print("Results already exist. Skipping analysis.")
        return pd.read_csv(output_path_sig), pd.read_csv(output_path_all)
 
    rng = np.random.default_rng(seed)
 
    mri_data_filtered = mri_data_filtered.copy()
    mri_data_filtered["scan_site_code"] = mri_data_filtered["scan_site"].astype("category").cat.codes
 
    cols_to_exclude = [
        "participant_id", "session_id",
        "mh_y_ksads__dep__mdd__pres_dx", "mh_p_ksads__dep__mdd__pres_dx",
        "sex", "scan_site", "scan_site_code", "visit_age", "age_squared", "visit_age_squared",
        "mr_y_smri__vol__aseg__icv_sum", group_col,
    ]
 
    nuisance_cols = ["visit_age", "sex"]
    full_cols = nuisance_cols + [group_col]
    group_term_index = full_cols.index(group_col)
 
    group_labels = mri_data_filtered[group_col].to_numpy()
    boot_indices = _stratified_bootstrap_indices(group_labels, n_bootstraps, rng)
 
    gam_results = []
    for col in tqdm(mri_data_filtered.columns, desc="GAM group-difference analysis"):
        if col in cols_to_exclude:
            continue
        try:
            y = mri_data_filtered[col]
 
            # --- Point estimate on the real (non-resampled) data ---
            X_full = mri_data_filtered[full_cols]
            gam_full = LinearGAM(s(0) + f(1) + l(2)).fit(X_full, y)
            p_value = gam_full.statistics_["p_values"][group_term_index]
 
            X_nuisance = mri_data_filtered[nuisance_cols]
            gam_nuisance = LinearGAM(s(0) + f(1)).fit(X_nuisance, y)
            adjusted_residuals = (y - gam_nuisance.predict(X_nuisance)).to_numpy()
 
            group_dep = adjusted_residuals[group_labels == 1]
            group_nondep = adjusted_residuals[group_labels == 0]
            effect_size = _cohens_d(group_dep, group_nondep)
            effect_size_std_analytic = _cohens_d_std(group_dep, group_nondep, d=effect_size)
 
            # --- Bootstrap distribution (resample residuals, don't refit GAMs) ---
            # --- Bootstrap distribution (refit nuisance GAM each replicate) ---
            boot_effect_sizes = np.empty(n_bootstraps)

            for b, indices in enumerate(boot_indices):

                boot_df = mri_data_filtered.iloc[indices].copy()

                X_boot = boot_df[nuisance_cols]
                y_boot = boot_df[col]
                group_boot = boot_df[group_col].to_numpy()

                try:
                    gam_boot = LinearGAM(
                        s(0) + f(1) + l(2) + l(3)
                    ).fit(X_boot, y_boot)

                    adjusted_boot = (
                        y_boot - gam_boot.predict(X_boot)
                    ).to_numpy()

                    boot_effect_sizes[b] = _cohens_d(
                        adjusted_boot[group_boot == 1],
                        adjusted_boot[group_boot == 0]
                    )

                except Exception:
                    boot_effect_sizes[b] = np.nan
 
            boot_effect_sizes = boot_effect_sizes[np.isfinite(boot_effect_sizes)]

            if boot_effect_sizes.size == 0:
                print(f"Warning: all {n_bootstraps} bootstrap replicates failed for column '{col}'. Skipping stability stats.")
                effect_size_std_boot = np.nan
                ci_lower, ci_upper = np.nan, np.nan
                sign_stability = np.nan
            else:
                effect_size_std_boot = np.std(boot_effect_sizes, ddof=1) if boot_effect_sizes.size > 1 else np.nan
                ci_lower, ci_upper = np.percentile(boot_effect_sizes, [2.5, 97.5])
                sign_stability = np.mean(np.sign(boot_effect_sizes) == np.sign(effect_size))
 
            gam_results.append((
                col, p_value, effect_size, effect_size_std_analytic,
                effect_size_std_boot, ci_lower, ci_upper, sign_stability,
            ))
        except Exception as e:
            print(f"Error occurred while fitting GAM model for column {col}: {e}")
 
    results_df = pd.DataFrame(gam_results, columns=[
        "feature", "p_value", "effect_size", "effect_size_std",
        "effect_size_std_boot", "ci_lower_boot", "ci_upper_boot", "sign_stability",
    ])
    results_df = results_df.sort_values("p_value")
 
    print("Performing FDR correction for multiple comparisons...")
    rejected, corrected_p_values, _, _ = multipletests(results_df["p_value"], alpha=0.05, method="fdr_bh")
    results_df["corrected_p_value"] = corrected_p_values
    results_df["significant_fdr"] = rejected
 
    results_df.to_csv(output_path_all, index=False)
    print(f"All results saved to: {output_path_all}")
 
    sig_results_df = results_df[results_df["significant_fdr"]]
    sig_results_df.to_csv(output_path_sig, index=False)
    print(f"Significant ROIs saved to: {output_path_sig}")
 
    num_significant_rois = results_df["significant_fdr"].sum()
    print(f"Number of significant ROIs after FDR correction: {num_significant_rois}")
 
    return sig_results_df, results_df

def clustering(data, n_clusters=None, max_clusters=None,dr=None, dr_params=None, cl=None, cl_params=None, mri_meta_df=None, output_path = Path("output"), clustering_output = None, bootstrapping = True, overwrite = True):
    '''
    This function performs clustering to identify subtypes of depression based on the selected subjects' data.
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
            "random_state": [42]
        }))

        agglomerative_params = list(ParameterGrid({
            "n_clusters": [n_clusters],
            "linkage": ["ward", "complete", "average"]
        }))

        bayesian_gmm_params = list(ParameterGrid({
            "n_components": [n_clusters],
            "max_iter": [300],
            "n_init": [5],
            "random_state": [42]
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
            "max_iter": [300],
            "n_init": [5],
            "random_state": [42]
        }))

        kmeans_params = list(ParameterGrid({
            "n_clusters": [5, 10, 15, 20],
            "random_state": [42]
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
