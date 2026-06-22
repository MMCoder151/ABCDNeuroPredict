import pandas as pd
import numpy as np
from pathlib import Path
from src.mri_rois import mri_rois
from pcntoolkit import NormativeModel, BLR, Runner
from pcntoolkit.dataio.norm_data import NormData
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import STL
import pathlib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
from scipy.stats import wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


def _one_hot_encode(df, col="scan_site", prefix="scan_site", categories=None):
    """One-hot encode a scan-site column while keeping a stable category order."""
    encoded = df.copy()
    site_values = encoded[col].astype("string")
    if categories is None:
        categories = sorted(site_values.dropna().unique().tolist())
    else:
        categories = [str(category) for category in categories]
    site_cat = pd.Categorical(site_values, categories=categories)
    dummies = pd.get_dummies(site_cat, prefix=prefix, drop_first=True, dtype=float)
    encoded = encoded.drop(columns=[col])
    encoded = pd.concat([encoded, dummies], axis=1)
    return encoded, list(dummies.columns), categories

def analyse_confounds(dem_df, transformed_data, output_path=pathlib.Path("output"), raw_data = None, con = None, view = None):
    '''
    This function runs linear regression and extracts the total and unique variance explained (R squared and adjusted R squared) 
    for each confound before and after normative modeling of MRI data
    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        dem_df (DataFrame): DataFrame containing demographic information for the subjects
        mri_meta_df (DataFrame): DataFrame containing MRI metadata for the subjects
        output_path (pathlib.Path): Path to the output directory where the results will be saved
    Returns:
        confound_effects_analysis.csv (file): CSV file containing the results of the confound effect analyses for each MRI ROI
        confound_effects_analysis.json (file): JSON file containing the results of the confound effect analyses for each MRI ROI
        confound_effects_df (DataFrame): DataFrame containing the results of the confound effect analyses for each MRI ROI
    '''

    # Import raw data 
    if con is None and raw_data is None:
        print("Error: No raw data or DuckDB connection provided. Please provide either raw data or a DuckDB connection with the appropriate view.")
        return None
    
    if con is not None and raw_data is not None:
        print("Warning: Both raw data and DuckDB connection provided. Using raw data and ignoring DuckDB connection.")
    
    if con is not None and raw_data is None:
        if view is not None:
            query = f"""
                SELECT *
                FROM {view}
                WHERE subject IN ({', '.join(f"'{sub}'" for sub in dem_df['subject'].unique())})
                AND timepoint = (
                    SELECT MIN(timepoint)
                    FROM {view} AS sub_mri
                    WHERE sub_mri.subject = {view}.subject
                )
            """
            raw_data = con.execute(query).df()
        else:
            print("Warning: DuckDB connection provided but view is None. Please provide view or raw data.")
            return None
    
    # Get columns to include in analysis
    exclude_cols = ["subject", "subject_ids", "timepoint", "Wear_Time", "subtype", "group",
                "age_at_first_mri", "age_at_first_mri_c", "age_at_first_mri_c_sq",
                "sex", "scan_site"]
    raw_analysis_cols = [col for col in raw_data.columns if col not in exclude_cols]
    transformed_analysis_cols = [col for col in transformed_data.columns if col not in exclude_cols]
    # Check that analysis_cols match in both raw and transformed data
    if not set(raw_analysis_cols).issubset(set(transformed_analysis_cols)):
        print(f"Error: Columns do not match between raw and transformed data.")
        print(f"Dropping columns from analysis that are not present in both datasets: {set(raw_analysis_cols) - set(transformed_analysis_cols)}")
        analysis_cols = [col for col in raw_analysis_cols if col in transformed_analysis_cols]
    else:
        analysis_cols = raw_analysis_cols

    # Merge z_scores_df with demographic data to get age, sex and scan_site for each subject
    transformed_data = transformed_data.merge(
        dem_df[["subject", "age_at_first_mri", "sex", "scan_site"]].drop_duplicates(),
        left_on="subject_ids",
        right_on="subject",
        how="inner"
    )
    transformed_data.drop(columns=["subject_ids"], inplace=True)

    # Merge raw data with demographic data to get age, sex and scan_site for each subject
    raw_data = raw_data.merge(
        dem_df[["subject", "age_at_first_mri", "sex", "scan_site"]].drop_duplicates(),
        left_on="subject",
        right_on="subject",
        how="inner"
    )

    # Add age squared centered around the mean to both raw and transformed data for confound analysis
    transformed_data["age_at_first_mri_c"] = transformed_data["age_at_first_mri"] - transformed_data["age_at_first_mri"].mean()
    transformed_data["age_at_first_mri_c_sq"] = transformed_data["age_at_first_mri_c"] ** 2
    raw_data["age_at_first_mri_c"] = raw_data["age_at_first_mri"] - raw_data["age_at_first_mri"].mean()
    raw_data["age_at_first_mri_c_sq"] = raw_data["age_at_first_mri_c"] ** 2

    # One-hot encode scan site in both dataframes and ensure the same categories and column names
    site_categories = sorted(
        pd.concat([raw_data["scan_site"], transformed_data["scan_site"]], ignore_index=True)
        .astype("string")
        .dropna()
        .unique()
        .tolist()
    )
    raw_data, site_dummy_cols, site_categories = _one_hot_encode(
        raw_data,
        col="scan_site",
        prefix="scan_site",
        categories=site_categories,
    )
    transformed_data, transformed_site_dummy_cols, _ = _one_hot_encode(
        transformed_data,
        col="scan_site",
        prefix="scan_site",
        categories=site_categories,
    )
    if site_dummy_cols != transformed_site_dummy_cols:
        raise ValueError("Site dummy columns do not align between raw and post-normative data.")
    
    # One-hot encode sex in both dataframes and ensure the same categories and column names
    sex_categories = sorted(
        pd.concat([raw_data["sex"], transformed_data["sex"]], ignore_index=True)
        .astype("string")
        .dropna()
        .unique()
        .tolist()
    )
    raw_data, sex_dummy_cols, sex_categories = _one_hot_encode(
        raw_data,
        col="sex",
        prefix="sex",
        categories=sex_categories,
    )
    transformed_data, transformed_sex_dummy_cols, _ = _one_hot_encode(
        transformed_data,
        col="sex",
        prefix="sex",
        categories=sex_categories,
    )
    if sex_dummy_cols != transformed_sex_dummy_cols:
        raise ValueError("Sex dummy columns do not align between raw and post-normative data.")

    # Fit hierarchical linear regression models for each MRI ROI with age, sex, and site as predictors before and after noromative modeling
    # and extract both total and unique variance explained (R squared and adjusted R squared) for each confound
    confound_effects = []

    # Define model hierarchy
    # Order reflects theoretical priority: site first (nuisance),
    # then age (primary biological), then sex
    model_hierarchy = {
        'site only':          site_dummy_cols,
        'site + age':         site_dummy_cols + ['age_at_first_mri'],
        'site + age + age^2': site_dummy_cols + ['age_at_first_mri', 'age_at_first_mri_c_sq'],
        'site + age + sex':   site_dummy_cols + ['age_at_first_mri'] + sex_dummy_cols,
        'site + age + age^2 + sex': site_dummy_cols + ['age_at_first_mri', 'age_at_first_mri_c_sq'] + sex_dummy_cols 
    }

    for roi in tqdm(analysis_cols, desc="Analyzing confound effects"):
        # Prepare data for regression
        pre_df = raw_data[["subject", "age_at_first_mri", "age_at_first_mri_c_sq", roi] + site_dummy_cols + sex_dummy_cols].dropna(subset=[roi])
        pre_df = pre_df.apply(pd.to_numeric, errors='coerce').astype('float64')
        post_df = transformed_data[["subject", "age_at_first_mri", "age_at_first_mri_c_sq", roi] + site_dummy_cols + sex_dummy_cols].dropna(subset=[roi])
        post_df = post_df.apply(pd.to_numeric, errors='coerce').astype('float64')
        X_pre = pre_df[site_dummy_cols + ["age_at_first_mri", "age_at_first_mri_c_sq"] + sex_dummy_cols]
        X_post = post_df[site_dummy_cols + ["age_at_first_mri", "age_at_first_mri_c_sq"] + sex_dummy_cols]
        y_pre = pre_df[roi]
        y_post = post_df[roi]
        X_pre_const = sm.add_constant(X_pre)
        X_post_const = sm.add_constant(X_post)
        # Fit models according to hierarchy and extract R squared and adjusted R squared 
        model_results = {}
        for model_name, predictors in model_hierarchy.items():
            model_pre = sm.OLS(y_pre, X_pre_const[["const"] + predictors]).fit()
            model_post = sm.OLS(y_post, X_post_const[["const"] + predictors]).fit()
            model_results[model_name] = {
                "R_squared_pre": model_pre.rsquared,
                "Adj_R_squared_pre": model_pre.rsquared_adj,
                "p_values_pre": model_pre.pvalues.to_dict(),
                "coefficients_pre": model_pre.params.to_dict(),
                "R_squared_post": model_post.rsquared,
                "Adj_R_squared_post": model_post.rsquared_adj,
                "p_values_post": model_post.pvalues.to_dict(),
                "coefficients_post": model_post.params.to_dict()
            }
        confound_effects.append({
            "variable": roi,
            "model_results": model_results
        })

    rows = []
    for item in confound_effects:
        roi = item['variable']
        for mname, res in item['model_results'].items():
            rows.append({
                'variable': roi,
                'model': mname,
                'R2_pre': res['R_squared_pre'],
                'R2_post': res['R_squared_post'],
                'AdjR2_pre': res['Adj_R_squared_pre'],
                'AdjR2_post': res['Adj_R_squared_post'],
                'pvals_pre': res['p_values_pre'],
                'pvals_post': res['p_values_post'],
                'coef_pre': res['coefficients_pre'],
                'coef_post': res['coefficients_post']
            })
    df = pd.DataFrame(rows)

    pivot = df.pivot(index='variable', columns='model')

    # Age effect = (site+age) - (site only)
    age_R2_pre  = pivot['R2_pre']['site + age'] - pivot['R2_pre']['site only']
    age_R2_post = pivot['R2_post']['site + age'] - pivot['R2_post']['site only']
    age_reduction = (age_R2_pre - age_R2_post)
    print(f"Age effect: mean R2 pre={age_R2_pre.mean():.4f}, mean R2 post={age_R2_post.mean():.4f}")
    print(f"Age effect: mean R2 reduction={age_reduction.mean():.4f}")

    # Age effect with age^2 = (site+age+age^2) - (site only)
    age2_R2_pre  = pivot['R2_pre']['site + age + age^2'] - pivot['R2_pre']['site only']
    age2_R2_post = pivot['R2_post']['site + age + age^2'] - pivot['R2_post']['site only']
    age2_reduction = (age2_R2_pre - age2_R2_post)
    print(f"Age^2 effect: mean R2 pre={age2_R2_pre.mean():.4f}, mean R2 post={age2_R2_post.mean():.4f}")
    print(f"Age^2 effect: mean R2 reduction={age2_reduction.mean():.4f}")

    # Sex effect = (site+age+sex) - (site+age)
    sex_R2_pre  = pivot['R2_pre']['site + age + sex'] - pivot['R2_pre']['site + age']
    sex_R2_post = pivot['R2_post']['site + age + sex'] - pivot['R2_post']['site + age']
    sex_reduction = (sex_R2_pre - sex_R2_post)
    print(f"Sex effect: mean R2 pre={sex_R2_pre.mean():.4f}, mean R2 post={sex_R2_post.mean():.4f}")
    print(f"Sex effect: mean R2 reduction={sex_reduction.mean():.4f}")

    # Sex effect with age^2 = (site+age+age^2+sex) - (site+age+age^2)
    sex2_R2_pre  = pivot['R2_pre']['site + age + age^2 + sex'] - pivot['R2_pre']['site + age + age^2']
    sex2_R2_post = pivot['R2_post']['site + age + age^2 + sex'] - pivot['R2_post']['site + age + age^2']
    sex2_reduction = (sex2_R2_pre - sex2_R2_post)
    print(f"Sex with age^2 effect: mean R2 pre={sex2_R2_pre.mean():.4f}, mean R2 post={sex2_R2_post.mean():.4f}")
    print(f"Sex with age^2 effect: mean R2 reduction={sex2_reduction.mean():.4f}")

    # Site effect is just the R2 of 'site only'
    site_R2_pre  = pivot['R2_pre']['site only']
    site_R2_post = pivot['R2_post']['site only']
    site_reduction = site_R2_pre - site_R2_post
    print(f"Site effect: mean R2 pre={site_R2_pre.mean():.4f}, mean R2 post={site_R2_post.mean():.4f}")
    print(f"Site effect: mean R2 reduction={site_reduction.mean():.4f}")

    residual_association_df = df[df["model"].isin(model_hierarchy)].copy()
    residual_association_df = residual_association_df[[
        "variable",
        "model",
        "R2_post",
        "AdjR2_post",
        "pvals_post",
        "coef_post",
    ]]
    residual_association_df.to_csv(
        output_path / "post_normative_residual_association.csv",
        index=False,
    )
    print(
        "Post-normative residual association (mean R2): "
        f"site-only={residual_association_df.loc[residual_association_df['model'] == 'site only', 'R2_post'].mean():.4f}, "
        f"site+age={residual_association_df.loc[residual_association_df['model'] == 'site + age', 'R2_post'].mean():.4f}, "
        f"site+age+age^2={residual_association_df.loc[residual_association_df['model'] == 'site + age + age^2', 'R2_post'].mean():.4f}, "
        f"site+age+age^2+sex={residual_association_df.loc[residual_association_df['model'] == 'site + age + age^2 + sex', 'R2_post'].mean():.4f}"
    )

    # Wilcoxon signed-rank test to compare the R2 values for each confound pre and post normative modeling across all MRI ROIs
    valid = (~age_R2_pre.isna()) & (~age_R2_post.isna())
    stat, p_age = wilcoxon(age_R2_pre[valid], age_R2_post[valid])
    print('Age R2 Wilcoxon p=', p_age)

    valid = (~age2_R2_pre.isna()) & (~age2_R2_post.isna())
    stat, p_age2 = wilcoxon(age2_R2_pre[valid], age2_R2_post[valid])
    print('Age^2 R2 Wilcoxon p=', p_age2)

    valid = (~sex_R2_pre.isna()) & (~sex_R2_post.isna())
    stat, p_sex = wilcoxon(sex_R2_pre[valid], sex_R2_post[valid])
    print('Sex R2 Wilcoxon p=', p_sex)
    
    valid = (~site_R2_pre.isna()) & (~site_R2_post.isna())
    stat, p_site = wilcoxon(site_R2_pre[valid], site_R2_post[valid])
    print('Site R2 Wilcoxon p=', p_site)

    def extract_coef(series, key):
        return series.map(lambda d: d.get(key, np.nan) if isinstance(d, dict) else np.nan)

    pivot_coefs = df.copy()
    pivot_coefs['age_coef_pre']  = extract_coef(pivot_coefs['coef_pre'], 'age_at_mri')
    pivot_coefs['age_coef_post'] = extract_coef(pivot_coefs['coef_post'], 'age_at_mri')

    plt.figure(figsize=(6,4))
    plt.hist(age_R2_pre - age_R2_post, bins=40)
    plt.axvline(0, color='k', linestyle='--')
    plt.title('Age: reduction in R2 (pre - post)')
    plt.xlabel('Δ R2')
    plt.savefig(output_path / "confound_effects_analysis_age_R2_reduction.png")

    plt.figure(figsize=(6,6))
    plt.scatter(age_R2_pre, age_R2_post, alpha=0.7)
    plt.plot([0, max(age_R2_pre.max(), age_R2_post.max())],[0, max(age_R2_pre.max(), age_R2_post.max())], 'k--')
    plt.xlabel('R2 pre')
    plt.ylabel('R2 post')
    plt.title('R2 pre vs post (age effect)')
    plt.savefig(output_path / "confound_effects_analysis_age_R2_pre_vs_post.png")

    confound_effects_df = pd.DataFrame(confound_effects)
    confound_effects_df.to_csv(output_path / "confound_effects_analysis.csv", index=False)
    confound_effects_df.to_json(output_path / "confound_effects_analysis.json", orient="records", lines=False, indent=2)
    return confound_effects_df

def normative_selection(con, mri_meta_df, output_path=pathlib.Path("output"), overwrite=True):
    '''
    This function performs normative modeling and selects subjects based on their composite absolute z-score. 
    It selects the top 10% (based on prevalence) of subjects with the highest cumulative z-score.
    These subjects are considered to have abnormal development in the selected MRI ROIs associated with depression.

    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        mri_meta_df (DataFrame): DataFrame containing MRI metadata
        dem_df (DataFrame): DataFrame containing demographic data
        output_path (pathlib.Path): Path to the output directory where the normative model results will be saved
    Returns:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their MRI ROI data and respective z-scores
        normative_modelling (Folder): Folder containing the normative model, results, and plots created in the output directory
    '''

    if overwrite == False:
        print("Normative modeling and subject selection skipped (overwrite=False). To re-run normative modeling and subject selection, set overwrite=True.")
        try:
            selected_subjects = pd.read_csv(Path(output_path) / "normative_modelling" / "results" / "selected_subjects.csv")
            return selected_subjects
        except Exception as e:
            print(f"Error loading selected subjects: {e}")
            print("Please check that the selected_subjects.csv file exists in the normative_modelling results directory and is correctly formatted.")
            raise e

    # Get MRI ROIs to include in the normative model
    _, mri_rois_dict = mri_rois()
    mri_roi_cols = list(mri_rois_dict.keys())

    # Get subjects to include from mri_meta_df
    included_subjects = mri_meta_df["subject"].unique()
    first_mri_meta_df = (
        mri_meta_df.sort_values(["subject", "timepoint"])
        .drop_duplicates(subset=["subject"], keep="first")
        [["subject", "sex", "age_at_mri", "scan_site"]]
    )
    
    # Query MRI data for the first timepoint for each included subject
    query = f"""
        SELECT *
        FROM mri_data
        WHERE subject IN ({', '.join(f"'{sub}'" for sub in included_subjects)})
        AND timepoint IN (
            SELECT MIN(timepoint)
            FROM mri_data AS sub_mri
            WHERE sub_mri.subject = mri_data.subject
        )
    """
    mri_df = con.execute(query).df()
    print(f"MRI data loaded: {len(mri_df)} subjects")

    # Merge MRI data 
    df = mri_df.merge(
        first_mri_meta_df,
        on="subject",
        how="inner"
    )

    missing_cols = set(mri_roi_cols) - set(df.columns)
    if missing_cols:
        print(f"Warning: The following MRI ROI columns are missing from the dataframe: {missing_cols}")

    # Encode sex explicitly so F/M map to 1/2 before passing data to NormData.
    sex_map = {'F': 1, 'M': 2}
    if not pd.api.types.is_numeric_dtype(df['sex']):
        sex_values = df['sex'].astype('string').str.strip().str.upper()
        mapped_sex = sex_values.map(sex_map)
        if mapped_sex.isna().any():
            unexpected_values = sorted(sex_values[mapped_sex.isna()].dropna().unique().tolist())
            raise ValueError(
                f"Unmapped sex values found: {unexpected_values}. Expected F/M or numeric input."
            )
        df['sex'] = mapped_sex.astype('Int64')

    df['sex'] = pd.to_numeric(df['sex'], errors='raise').astype(float)
    df['age_at_mri'] = pd.to_numeric(df['age_at_mri'], errors='raise').astype(float)
    df['subject'] = df['subject'].astype(str)

    df, site_dummy_cols, _ = _one_hot_encode(df, col="scan_site", prefix="scan_site")

    # Prepare data for normative modeling
    data = NormData.from_dataframe(
        name="mri_norm",
        dataframe=df,
        covariates=["sex", "age_at_mri"] + site_dummy_cols,
        response_vars=mri_roi_cols,
        subject_ids="subject",
        remove_Nan=True,
    )

    normative_output_dir = Path(output_path) / "normative_modelling"
    if not normative_output_dir.exists():
        normative_output_dir.mkdir(parents=True)
    normative_output_dir_str = str(normative_output_dir)

    # setup normative model
    model = NormativeModel(
        BLR(),
        # Whether to save the model after fitting.
        savemodel=True,
        # Whether to evaluate the model after fitting.
        evaluate_model=True,
        # Whether to save the results after evaluation.
        saveresults=True,
        # Whether to save the plots after fitting.
        saveplots=False,
        # The directory to save the model, results, and plots.
        save_dir=normative_output_dir_str,
        # The scaler to use for the input data. Can be either one of "standardize", "minmax", "robminmax", "none"
        inscaler="standardize",
        # The scaler to use for the output data. Can be either one of "standardize", "minmax", "robminmax", "none"
        outscaler="standardize",
        )

    model.fit(data)

    # create a runner
    #runner = Runner(cross_validate = True)

    # fit the model 
    #runner.fit(model, data)

    # Read in z-score file from normative modeling 
    centiles_df = pd.read_csv(normative_output_dir / "results" / "Z_mri_norm.csv")
    # Calculate composite absolute z-score across all MRI ROIs for each subject
    centiles_df["composite_z"] = centiles_df[mri_roi_cols].abs().sum(axis=1)
    subject_scores = (
        centiles_df[["subject_ids", "composite_z"]]
        .dropna()
        .drop_duplicates(subset=["subject_ids"])
        .sort_values("composite_z")
        .reset_index(drop=True)
    )
    subject_scores["subject_ids"].nunique()
    subject_scores["rank"] = np.arange(len(subject_scores))
    # Select top 10% of subjects with the highest composite z-score based on ranked prevalence
    n_select = int(np.ceil(0.10 * len(subject_scores)))
    selected_subject_ids = subject_scores.nlargest(n_select, "composite_z")["subject_ids"]
    selected_subjects = subject_scores[subject_scores["subject_ids"].isin(selected_subject_ids)]
    
    print(f"Selected {len(selected_subject_ids)} subjects with the highest composite z-scores based on a prevalence threshold of 10%.")
    selected_subjects.to_csv(normative_output_dir / "results" / "selected_subjects.csv", index=False)
    
    # create scatter plot of composite z-scores for all subjects, highlighting selected subjects in a different color
    plot_df = centiles_df[["subject_ids", "composite_z"]].dropna().sort_values("composite_z").reset_index(drop=True)
    plot_df["rank"] = np.arange(len(plot_df))

    plt.figure(figsize=(10, 6))
    plt.scatter(subject_scores["rank"], subject_scores["composite_z"], label="All subjects", alpha=0.5, s=12)
    selected_plot = subject_scores[subject_scores["subject_ids"].isin(selected_subject_ids)]
    plt.scatter(selected_plot["rank"], selected_plot["composite_z"], label="Selected subjects", color="red", s=18)
    plt.xlabel("Subject rank by composite z-score")
    plt.ylabel("Composite Absolute Z-Score")
    plt.title("Composite Absolute Z-Scores for MRI ROIs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(normative_output_dir / "results" / "composite_z_scores.png")
    plt.close()

    # Create summary table with mean, std, min, and max per mri_roi for the selected subjects
    stats_selected = (
        centiles_df[mri_roi_cols]
        .agg(['mean', 'std', 'min', 'max'])
        .transpose()
        .reset_index()
        .rename(columns={"index": "mri_roi"})
    )
    stats_selected.to_csv(normative_output_dir / "results" / "mri_roi_statistics.csv", index=False)

    # Create results summary table with mean, std, min, and max per metric
    stats_df = pd.read_csv(normative_output_dir / "results" / "statistics_mri_norm.csv")
    summary = stats_df.assign(
        mean = stats_df[mri_roi_cols].mean(axis=1),
        std  = stats_df[mri_roi_cols].std(axis=1),
        min  = stats_df[mri_roi_cols].min(axis=1),
        max  = stats_df[mri_roi_cols].max(axis=1),
    )[["statistic", "mean", "std", "min", "max"]]
    summary.to_csv(normative_output_dir / "results" / "statistics_summary.csv", index=False)

    return selected_subjects

def create_composites(selected_subjects, vif_threshold=10):
    """
    Creates composite scores out of given variables based on variance inflation factors (VIF).
 
    Parameters:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their
            MRI ROI data and respective z-scores
        vif_threshold (float): VIF value above which a variable triggers compositing
        max_iterations (int): safety cap so a degenerate case can't loop forever
 
    Returns:
        selected_subjects (DataFrame): DataFrame with high-VIF columns replaced by composites
        composite_dict (dict): Maps composite score name -> list of *original* base columns
                                that were averaged together to build it (flattened, not nested)
    """
    vif_cols = [
        col for col in selected_subjects.columns
        if col not in ["subject", "composite_z", "Wear_Time", "subtype", "group"]
    ]
 
    # Drop zero-variance columns (VIF undefined for these)
    variances = selected_subjects[vif_cols].var()
    zero_variance_cols = variances[variances == 0].index.tolist()
    if zero_variance_cols:
        print(f"Dropping columns with zero variance: {len(zero_variance_cols)}")
        print(zero_variance_cols)
        vif_cols = [c for c in vif_cols if c not in zero_variance_cols]
 
    vif_data = selected_subjects[vif_cols].dropna()
    selected_subjects = selected_subjects[vif_cols].copy()
 
    # Snapshot of original values, indexed the same as vif_data, used so we can always
    # average from true base columns even after they've been dropped from selected_subjects.
    original_values = vif_data.copy()
 
    # composite_dict maps a SHORT composite id -> flattened list of original base columns.
    # base_members maps current-column-name -> flattened list of original base columns,
    # for every column currently alive in vif_data (whether original or composite).
    composite_dict = {}
    base_members = {col: [col] for col in vif_data.columns}
    composite_counter = 0
 
    def vif_table(df):
        return pd.DataFrame({
            "variable": df.columns,
            "vif": [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        }).sort_values("vif", ascending=False)
 
    vif_df = vif_table(vif_data)
 
    while vif_df["vif"].max() > vif_threshold:
 
        high_vif_col = vif_df.iloc[0]["variable"]
 
        correlations = vif_data.corr()[high_vif_col].drop(high_vif_col).abs()
        if correlations.empty:
            # Nothing left to pair with — stop rather than crash
            break
        most_correlated_col = correlations.idxmax()
 
        # Short, stable, human-readable name — does NOT concatenate ancestry
        composite_counter += 1
        composite_name = f"composite_{composite_counter}"
 
        # Flattened provenance: original base variables in both parents, deduped,
        # order-preserved
        merged_members = []
        for c in base_members[high_vif_col] + base_members[most_correlated_col]:
            if c not in merged_members:
                merged_members.append(c)
 
        print(
            f"Created {composite_name} from '{high_vif_col}' + '{most_correlated_col}' "
            f"(VIF={vif_df.iloc[0]['vif']:.1f}, corr={correlations.max():.3f}) "
            f"-> {len(merged_members)} base vars: {merged_members}"
        )
 
        # Average from the ORIGINAL base columns (not from intermediate composites),
        # so a variable's influence on the final composite doesn't depend on which
        # merge order it happened to go through. We must read these from the
        # `original_values` snapshot taken before the loop started, because by this
        # point some of `merged_members` may already have been dropped from
        # `selected_subjects` in an earlier iteration (folded into a prior composite).
        # NOTE: `original_values` only has rows that survived the initial dropna() for
        # VIF purposes, so this assignment will introduce NaN in `selected_subjects`
        # for any row that had a NaN in ANY vif_col originally. If you need composite
        # scores for those rows too, compute composites on a per-pair basis from
        # selected_subjects directly instead (see alternative below).
        selected_subjects[composite_name] = original_values[merged_members].mean(axis=1)
 
        composite_dict[composite_name] = merged_members
 
        # Drop the two parent columns, register the new composite
        selected_subjects.drop(columns=[high_vif_col, most_correlated_col], inplace=True, errors="ignore")
        vif_data = vif_data.drop(columns=[high_vif_col, most_correlated_col])
        vif_data[composite_name] = selected_subjects[composite_name]
 
        base_members.pop(high_vif_col, None)
        base_members.pop(most_correlated_col, None)
        base_members[composite_name] = merged_members
 
        vif_df = vif_table(vif_data)
 
    selected_subjects = selected_subjects.copy()  # defragment
 
    # composite_dict can still contain entries for intermediate composites that were
    # absorbed into a later, larger composite (e.g. composite_1 -> [A, B] got folded
    # into composite_2 -> [A, B, C]). The composite_1 *column* is already gone from
    # selected_subjects at this point — it was dropped the moment it got merged — but
    # the dict entry lingers as a bookkeeping artifact. Prune any entry whose member
    # set is a strict subset of another entry's member set, since it no longer
    # corresponds to an actual column and is purely redundant provenance info.
    obsolete = set()
    for name_a, members_a in composite_dict.items():
        set_a = set(members_a)
        for name_b, members_b in composite_dict.items():
            if name_a == name_b:
                continue
            if set_a < set(members_b):  # strict subset
                obsolete.add(name_a)
                break
    for name in obsolete:
        composite_dict.pop(name)
 
    print(f"\nComposites created: {len(composite_dict)}")
    for name, members in composite_dict.items():
        print(f"  {name}: {members}")
 
    return selected_subjects, composite_dict

def extr_fitbit_features(con, selected_subjects, overwrite=True):
    '''
    This function extracts features from the fitbit data for the selected subjects.
        1. Creates daily mean, std, min, and max for each fitbit metric
        2. Imputes missing days 
        3. Conducts weekly Seasonal Trend Decomposition using Loess (STL) for each daily fitbit metric
        4. Creates mean, stdm, min, and max for the trend, seasonal, and residual components of the STL decomposition for each fitbit metric
    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        selected_subjects (DataFrame): DataFrame containing the subjects to extract fitbit features from 
    Returns:
        fitbit_features_df (DataFrame): DataFrame containing the extracted fitbit features for each subject
    '''

    # Get a de-duplicated list of selected subjects.
    if hasattr(selected_subjects, "columns") and "subject" in selected_subjects.columns:
        selected_subjects_list = selected_subjects["subject"].dropna().unique().tolist()
    else:
        selected_subjects_list = pd.Series(selected_subjects).dropna().unique().tolist()

    # Create a dataframe to hold the extracted features
    features_list = []

    # Loop through each subject to extract features
    #grouped = fitbit_df.groupby(["subject"])
    for subject in tqdm(selected_subjects_list, total=len(selected_subjects_list), desc="Extracting Fitbit features"):
        # query first timepoint for the subject
        query = f"""
        SELECT *
        FROM fitbit_data        
        WHERE subject = '{subject}'
        AND timepoint = (
            SELECT MIN(timepoint)
            FROM fitbit_data f2
            WHERE f2.subject = fitbit_data.subject
        )
        """
        subject_fitbit_df = con.execute(query).df()
        fitbit_metric_cols = [col for col in subject_fitbit_df.columns if col not in ["subject", "timepoint", "Wear_Time"]]
        for col in fitbit_metric_cols:
            subject_fitbit_df[col] = pd.to_numeric(subject_fitbit_df[col], errors="coerce")
        feature_dict = {"subject": subject}
        for metric in fitbit_metric_cols:
            # Check if the metric column exists in the group
            if metric in subject_fitbit_df.columns:
                daily_data = subject_fitbit_df[["Wear_Time", metric]].dropna()
                if not daily_data.empty:
                    # Create daily features (mean, std, min, max)
                    daily_data.set_index("Wear_Time", inplace=True)
                    daily_stats = daily_data.resample("D").agg(['mean', 'std', 'min', 'max'])
                    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
                    # Create datetime index with proper missing days based on the daily resampling range
                    # TODO: Check if using the first non-NaN date as the start date for reindexing is more appropriate than using the min date of the daily_stats index, which may be affected by outliers or missing data.
                    daily_stats = daily_stats.dropna(how="all")
                    min_date = daily_stats.index.min()
                    max_date = daily_stats.index.max()
                    date_range = pd.date_range(start=min_date, end=max_date, freq="D")
                    # Reindex to include missing days and impute missing values with multiple imputation
                    daily_stats = daily_stats.reindex(date_range)
                    if daily_stats.shape[0] > 1 and daily_stats.notna().sum().sum() > daily_stats.shape[1]:
                        try:
                            imputer = IterativeImputer(random_state=0, max_iter=20)
                            daily_stats = pd.DataFrame(
                                imputer.fit_transform(daily_stats),
                                index=daily_stats.index,
                                columns=daily_stats.columns,
                            )
                        except Exception as e:
                            print(f"Iterative imputation failed for subject {subject}, metric {metric}: {e}")
                            daily_stats = daily_stats.ffill().bfill()
                    else:
                        daily_stats = daily_stats.ffill().bfill()
                    feature_dict.update(daily_stats.mean().to_dict())
                    # STL decomposition on the imputed daily aggregate series.
                    for agg in ["mean", "std", "min", "max"]:
                        try:
                            stl_input = daily_stats[f"{metric}_{agg}"].copy()
                            stl_input.index = pd.to_datetime(stl_input.index)
                            stl_input = stl_input.sort_index().asfreq("D")
                            stl = STL(stl_input, period=7, robust=True)
                            result = stl.fit()
                            stl_features = {
                                f"{metric}_{agg}_trend_mean": result.trend.mean(),
                                f"{metric}_{agg}_trend_std": result.trend.std(),
                                f"{metric}_{agg}_trend_min": result.trend.min(),
                                f"{metric}_{agg}_trend_max": result.trend.max(),
                                f"{metric}_{agg}_seasonal_mean": result.seasonal.mean(),
                                f"{metric}_{agg}_seasonal_std": result.seasonal.std(),
                                f"{metric}_{agg}_seasonal_min": result.seasonal.min(),
                                f"{metric}_{agg}_seasonal_max": result.seasonal.max(),
                                f"{metric}_{agg}_resid_mean": result.resid.mean(),
                                f"{metric}_{agg}_resid_std": result.resid.std(),
                                f"{metric}_{agg}_resid_min": result.resid.min(),
                                f"{metric}_{agg}_resid_max": result.resid.max(),
                            }
                            feature_dict.update(stl_features)
                        except Exception as e:
                            print(f"STL decomposition failed for subject {subject}, metric {metric}: {e}")
        features_list.append(feature_dict)
    fitbit_features_df = pd.DataFrame(features_list)
    # add subtype labels
    fitbit_features_df = fitbit_features_df.merge(
        selected_subjects[["subject", "subtype"]],
        left_on="subject",
        right_on="subject",
        how="left"
    )

    return fitbit_features_df

def normative_selection_fitbit(dem_df, fitbit_features, output_path = Path("output"), overwrite=True):
    '''
    This function performs normative selection on the fitbit features to test overlap with the MRI normative modeling.
    Parameters:
        dem_df (DataFrame): DataFrame containing demographic information for each subject
        fitbit_features (DataFrame): DataFrame containing the extracted fitbit features for each subject
        output_path (Path): Path to the output directory where the normative model results will be saved
    Returns:
        selected_fitbit_subjects (DataFrame): DataFrame containing the selected subjects based on normative modeling of fitbit features
        normative_modelling_fitbit (Folder): Folder containing the normative model, results, and plots created in the output directory for fitbit features
    '''
    if overwrite == False:
        print("Normative modeling and subject selection for fitbit features skipped (overwrite=False). To re-run normative modeling and subject selection for fitbit features, set overwrite=True.")
        try:
            selected_fitbit_subjects = pd.read_csv(Path(output_path) / "normative_modelling_fitbit" / "results" / "selected_fitbit_subjects.csv")
            return selected_fitbit_subjects
        except Exception as e:
            print(f"Error loading selected subjects: {e}")
            print("Please check that the selected_fitbit_subjects.csv file exists in the normative_modelling_fitbit results directory and is correctly formatted.")
            raise e

    # get columns to model
    model_cols = [col for col in fitbit_features.columns if col not in ["subject", "subtype"]]

    # Drop colums with zero variance from analysis
    variances = fitbit_features[model_cols].var()
    zero_variance_cols = variances[variances == 0].index.tolist()
    if zero_variance_cols:
        print(f"Dropping columns with zero variance from analysis: {len(zero_variance_cols)}")
        print(zero_variance_cols)
        model_cols = [c for c in model_cols if c not in zero_variance_cols]

    # Merge fitbit features with demographic data to get age, sex and scan site
    df = fitbit_features.merge(
        dem_df[["subject", "age_at_first_mri", "sex", "scan_site"]].drop_duplicates(),
        left_on="subject",
        right_on="subject",
        how="inner"
    )

    # Encode sex explicitly so F/M map to 1/2 before passing data to NormData.
    sex_map = {'F': 1, 'M': 2}
    if not pd.api.types.is_numeric_dtype(df['sex']):
        sex_values = df['sex'].astype('string').str.strip().str.upper()
        mapped_sex = sex_values.map(sex_map)
        if mapped_sex.isna().any():
            unexpected_values = sorted(sex_values[mapped_sex.isna()].dropna().unique().tolist())
            raise ValueError(
                f"Unmapped sex values found: {unexpected_values}. Expected F/M or numeric input."
            )
        df['sex'] = mapped_sex.astype('Int64')

    df['sex'] = pd.to_numeric(df['sex'], errors='raise').astype(float)
    df['age_at_first_mri'] = pd.to_numeric(df['age_at_first_mri'], errors='raise').astype(float)
    df['subject'] = df['subject'].astype(str)

    # Encode scan site
    df, site_dummy_cols, _ = _one_hot_encode(df, col="scan_site", prefix="scan_site")

    # Prepare data for normative modeling
    data = NormData.from_dataframe(
        name="fitbit_norm",
        dataframe=df,
        covariates=["sex", "age_at_first_mri"] + site_dummy_cols,
        response_vars=model_cols,
        subject_ids="subject",
        remove_Nan=True,
    )

    # define normative modeling output path
    normative_output_dir = Path(output_path) / "normative_modelling_fitbit"
    if not normative_output_dir.exists():
        normative_output_dir.mkdir(parents=True)
    normative_output_dir_str = str(normative_output_dir)

    # setup normative model
    model = NormativeModel(
        BLR(),
        savemodel=True,
        evaluate_model=True,
        saveresults=True,
        saveplots=False,
        save_dir=normative_output_dir_str,
        inscaler="standardize",
        outscaler="standardize",
        )
    model.fit(data)

    # Read in z-score file from normative modeling 
    centiles_df = pd.read_csv(normative_output_dir / "results" / "Z_fitbit_norm.csv")
    # Calculate composite absolute z-score across all fitbit features for each subject
    centiles_df["composite_z"] = centiles_df[model_cols].abs().sum(axis=1)
    subject_scores = (
        centiles_df[["subject_ids", "composite_z"]]
        .dropna()
        .drop_duplicates(subset=["subject_ids"])
        .sort_values("composite_z")
        .reset_index(drop=True)
    )
    subject_scores["subject_ids"].nunique()
    subject_scores["rank"] = np.arange(len(subject_scores))
    # Select top 10% of subjects with the highest composite z-score based on ranked prevalence
    n_select = int(np.ceil(0.10 * len(subject_scores)))
    selected_subject_ids = subject_scores.nlargest(n_select, "composite_z")["subject_ids"]
    selected_fitbit_subjects = subject_scores[subject_scores["subject_ids"].isin(selected_subject_ids)]

    print(f"Selected {len(selected_subject_ids)} subjects with the highest composite z-scores based on a prevalence threshold of 10%.")
    selected_fitbit_subjects.to_csv(normative_output_dir / "results" / "selected_fitbit_subjects.csv", index=False)

    # Create summary table with mean, std, min, and max per mri_roi for the selected subjects
    stats_selected = (
        centiles_df[model_cols]
        .agg(['mean', 'std', 'min', 'max'])
        .transpose()
        .reset_index()
        .rename(columns={"index": "mri_roi"})
    )
    stats_selected.to_csv(normative_output_dir / "results" / "fitbit_features_statistics.csv", index=False)

    # Create results summary table with mean, std, min, and max per metric
    stats_df = pd.read_csv(normative_output_dir / "results" / "statistics_fitbit_norm.csv")
    summary = stats_df.assign(
        mean = stats_df[model_cols].mean(axis=1),
        std  = stats_df[model_cols].std(axis=1),
        min  = stats_df[model_cols].min(axis=1),
        max  = stats_df[model_cols].max(axis=1),
    )[["statistic", "mean", "std", "min", "max"]]
    summary.to_csv(normative_output_dir / "results" / "statistics_summary.csv", index=False)

    return selected_fitbit_subjects

def fit_residualiser(X_train, dem_df):
    '''
    Fit a GPR per feature on TRAINING data only.
    Covariates: age (continuous) + sex (dummy-coded).
    '''
    X_train.dropna(inplace=True)
    X_train.drop(columns=["subject", "subtype"], inplace=True, errors="ignore")

    # drop columns with zero variance
    variances = X_train.var()
    zero_variance_cols = variances[variances == 0].index.tolist()
    if zero_variance_cols:
        print(f"Dropping columns with zero variance from residualisation: {len(zero_variance_cols)}")
        print(zero_variance_cols)
        X_train = X_train.drop(columns=zero_variance_cols, errors="ignore")

    age_train = dem_df.loc[X_train.index, "age_at_first_mri"].values.reshape(-1, 1)
    sex_train = pd.get_dummies(dem_df.loc[X_train.index, "sex"], drop_first=True).values
    design_matrix_train = np.hstack([age_train, sex_train])

    n_features = X_train.shape[1]
    models = []
    for i in tqdm(range(n_features), desc="Fitting GPR residualiser"):
        y_train = X_train.iloc[:, i].values
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
        gpr.fit(design_matrix_train, y_train)
        models.append(gpr)
    return models

def apply_residualiser(models, X, dem_df):
    '''
    Apply the fitted GPR residualiser to new data (e.g. test set).
    '''
    age = dem_df.loc[X.index, "age_at_first_mri"].values.reshape(-1, 1)
    sex = pd.get_dummies(dem_df.loc[X.index, "sex"], drop_first=True).values
    design_matrix = np.hstack([age, sex])
    
    X_residualised = X.copy()
    for i, gpr in enumerate(models):
        X_residualised.iloc[:, i] = gpr.predict(design_matrix)
    
    return X_residualised