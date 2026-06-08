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
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
from scipy.stats import wilcoxon


def _one_hot_encode_scan_site(df, site_col="scan_site", prefix="scan_site", categories=None):
    """One-hot encode a scan-site column while keeping a stable category order."""
    encoded = df.copy()
    site_values = encoded[site_col].astype("string")
    if categories is None:
        categories = sorted(site_values.dropna().unique().tolist())
    else:
        categories = [str(category) for category in categories]
    site_cat = pd.Categorical(site_values, categories=categories)
    dummies = pd.get_dummies(site_cat, prefix=prefix, drop_first=True, dtype=float)
    encoded = encoded.drop(columns=[site_col])
    encoded = pd.concat([encoded, dummies], axis=1)
    return encoded, list(dummies.columns), categories

def analyse_confounds(con, dem_df, mri_meta_df, output_path=pathlib.Path("output")):
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
    # Import z-score file from normative modeling
    if not (output_path / "normative_modelling" / "results" / "Z_mri_norm.csv").exists():
        print("Z-score file from normative modeling not found. Please run normative modeling before running confound effect analysis.")
        return None
    z_scores_df = pd.read_csv(output_path / "normative_modelling" / "results" / "Z_mri_norm.csv")
    z_scores_df.drop(columns=["observations"], inplace=True, errors ="ignore")

    # Import raw MRI data for the FIRST timepoint for each INCLUDED subject in dem_df
    query = f"""
        SELECT *
        FROM mri_data
        WHERE subject IN ({', '.join(f"'{sub}'" for sub in dem_df['subject'].unique())})
        AND timepoint = (
            SELECT MIN(timepoint)
            FROM mri_data AS sub_mri
            WHERE sub_mri.subject = mri_data.subject
        )
    """
    mri_df = con.execute(query).df()

    # Get MRI ROIs to include in the analysis
    _, mri_rois_dict = mri_rois()
    mri_roi_cols = list(mri_rois_dict.keys())

    # Encode sex explicitly so F/M map to 1/2 before passing data to NormData.
    sex_map = {'F': 1, 'M': 2}
    if not pd.api.types.is_numeric_dtype(dem_df['sex']):
        sex_values = dem_df['sex'].astype('string').str.strip().str.upper()
        mapped_sex = sex_values.map(sex_map)
        if mapped_sex.isna().any():
            unexpected_values = sorted(sex_values[mapped_sex.isna()].dropna().unique().tolist())
            raise ValueError(
                f"Unmapped sex values found: {unexpected_values}. Expected F/M or numeric input."
            )
        dem_df['sex'] = mapped_sex.astype('Int64')

    # Add age_at_mri from mri_meta_df for the first timepoint to dem_df
    first_age = mri_meta_df.sort_values("timepoint").drop_duplicates(subset="subject", keep="first")[["subject","age_at_mri"]]
    dem_df = dem_df.merge(
        first_age,
        on="subject",
        how="left"
    )

    # Merge z_scores_df with demographic data to get age for each subject
    z_scores_df = z_scores_df.merge(
        dem_df[["subject", "age_at_mri", "sex", "scan_site"]].drop_duplicates(),
        left_on="subject_ids",
        right_on="subject",
        how="inner"
    )
    z_scores_df.drop(columns=["subject_ids"], inplace=True)

    # Merge raw MRI data with demographic data to get age for each subject
    mri_df = mri_df.merge(
        dem_df[["subject", "age_at_mri", "sex", "scan_site"]].drop_duplicates(),
        left_on="subject",
        right_on="subject",
        how="inner"
    )

    # One-hot encode scan site in both dataframes and ensure the same categories and column names
    site_categories = sorted(
        pd.concat([mri_df["scan_site"], z_scores_df["scan_site"]], ignore_index=True)
        .astype("string")
        .dropna()
        .unique()
        .tolist()
    )
    mri_df, site_dummy_cols, site_categories = _one_hot_encode_scan_site(
        mri_df,
        categories=site_categories,
    )
    z_scores_df, z_site_dummy_cols, _ = _one_hot_encode_scan_site(
        z_scores_df,
        categories=site_categories,
    )
    if site_dummy_cols != z_site_dummy_cols:
        raise ValueError("Site dummy columns do not align between raw and post-normative data.")

    # Fit hierarchical linear regression models for each MRI ROI with age, sex, and site as predictors before and after noromative modeling
    # and extract both total and unique variance explained (R squared and adjusted R squared) for each confound
    confound_effects = []

    # Define model hierarchy
    # Order reflects theoretical priority: site first (nuisance),
    # then age (primary biological), then sex
    model_hierarchy = {
        'site only':          site_dummy_cols,
        'site + age':         site_dummy_cols + ['age_at_mri'],
        'site + age + sex':   site_dummy_cols + ['age_at_mri', 'sex']
    }

    for roi in tqdm(mri_roi_cols, desc="Analyzing confound effects"):
        # Prepare data for regression
        pre_df = mri_df[["subject", "age_at_mri", "sex", roi] + site_dummy_cols].dropna(subset=[roi])
        pre_df = pre_df.apply(pd.to_numeric, errors='coerce').astype('float64')
        post_df = z_scores_df[["subject", "age_at_mri", "sex", roi] + site_dummy_cols].dropna(subset=[roi])
        post_df = post_df.apply(pd.to_numeric, errors='coerce').astype('float64')
        X_pre = pre_df[site_dummy_cols + ["age_at_mri", "sex"]]
        X_post = post_df[site_dummy_cols + ["age_at_mri", "sex"]]
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
            "mri_roi": roi,
            "model_results": model_results
        })

    rows = []
    for item in confound_effects:
        roi = item['mri_roi']
        for mname, res in item['model_results'].items():
            rows.append({
                'mri_roi': roi,
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

    pivot = df.pivot(index='mri_roi', columns='model')

    # Age effect = (site+age) - (site only)
    age_R2_pre  = pivot['R2_pre']['site + age'] - pivot['R2_pre']['site only']
    age_R2_post = pivot['R2_post']['site + age'] - pivot['R2_post']['site only']
    age_reduction = (age_R2_pre - age_R2_post)
    print(f"Age effect: mean R2 pre={age_R2_pre.mean():.4f}, mean R2 post={age_R2_post.mean():.4f}")
    print(f"Age effect: mean R2 reduction={age_reduction.mean():.4f}")

    # Sex effect = (site+age+sex) - (site+age)
    sex_R2_pre  = pivot['R2_pre']['site + age + sex'] - pivot['R2_pre']['site + age']
    sex_R2_post = pivot['R2_post']['site + age + sex'] - pivot['R2_post']['site + age']
    sex_reduction = (sex_R2_pre - sex_R2_post)
    print(f"Sex effect: mean R2 pre={sex_R2_pre.mean():.4f}, mean R2 post={sex_R2_post.mean():.4f}")
    print(f"Sex effect: mean R2 reduction={sex_reduction.mean():.4f}")

    # Site effect is just the R2 of 'site only'
    site_R2_pre  = pivot['R2_pre']['site only']
    site_R2_post = pivot['R2_post']['site only']
    site_reduction = site_R2_pre - site_R2_post
    print(f"Site effect: mean R2 pre={site_R2_pre.mean():.4f}, mean R2 post={site_R2_post.mean():.4f}")
    print(f"Site effect: mean R2 reduction={site_reduction.mean():.4f}")

    residual_association_df = df[df["model"].isin(model_hierarchy)].copy()
    residual_association_df = residual_association_df[[
        "mri_roi",
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
        f"site+age+sex={residual_association_df.loc[residual_association_df['model'] == 'site + age + sex', 'R2_post'].mean():.4f}"
    )

    # Wilcoxon signed-rank test to compare the R2 values for each confound pre and post normative modeling across all MRI ROIs
    valid = (~age_R2_pre.isna()) & (~age_R2_post.isna())
    stat, p_age = wilcoxon(age_R2_pre[valid], age_R2_post[valid])
    print('Age R2 Wilcoxon p=', p_age)
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
        mri_meta_df[["subject", "sex", "age_at_mri", "scan_site"]].drop_duplicates(),
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

    df, site_dummy_cols, _ = _one_hot_encode_scan_site(df)

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

def create_mri_composites(selected_subjects):
    '''
    This function creates composite scores out of selected subject's z-scores based on 
    variance inflation factors (VIF) to account for multicollinearity between MRI ROIs.

    Parameters:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their MRI ROI data and respective z-scores
    Returns:
        composite_scores_df (DataFrame): DataFrame containing the selected subjects and their composite scores based on the selected MRI ROIs
        composite_dict (dict): Dictionary mapping composite score names to the MRI ROIs included in each composite
    '''

    # Get MRI ROIs to include in the normative model
    _, mri_rois_dict = mri_rois()
    mri_roi_cols = list(mri_rois_dict.keys())

    # Calculate variance inflation factors (VIF) for the selected MRI ROIs
    vif_data = selected_subjects[mri_roi_cols].dropna()
    vif_df = pd.DataFrame({
        "mri_roi": vif_data.columns,
        "vif": [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
    }).sort_values("vif", ascending=False)
    vif_df.head()

    # Replace variables with a vif above 10 with their average until all variables have a vif below 10
    composite_dict = {}
    while vif_df["vif"].max() > 10:
        # Get the variable with the highest VIF
        high_vif_roi = vif_df.iloc[0]["mri_roi"]
        # Create a composite score by averaging this variable with the variable it is most correlated with
        correlations = vif_data.corr()[high_vif_roi].drop(high_vif_roi).abs()
        most_correlated_roi = correlations.idxmax()
        composite_name = f"composite_{high_vif_roi}_{most_correlated_roi}"
        selected_subjects[composite_name] = selected_subjects[[high_vif_roi, most_correlated_roi]].mean(axis=1)
        composite_dict[composite_name] = [high_vif_roi, most_correlated_roi]
        # Drop the original variable with high VIF from the data used to calculate VIFs in the next iteration
        vif_data = vif_data.drop(columns=[high_vif_roi])
        # Recalculate VIFs
        vif_df = pd.DataFrame({
            "mri_roi": vif_data.columns,
            "vif": [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
        }).sort_values("vif", ascending=False)
    print(f"Composites created: {composite_dict}")

    return selected_subjects, composite_dict

def extr_fitbit_features(con, selected_subjects):
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
    # Get list of selected subjects
    selected_subjects_list = selected_subjects["subject"].unique().tolist()

    # Query FIRST timepoint for the selected subjects
    query = f"""
    SELECT *
    FROM fitbit_data
    WHERE timepoint = (
        SELECT MIN(timepoint)
        FROM fitbit_data f2
        WHERE f2.subject = fitbit_data.subject
    )
    AND subject IN ({', '.join(map(str, selected_subjects_list))})
    """
    print("Querying fitbit data for feature extraction...")
    fitbit_df = con.execute(query).df()

    # Get fitbit metric columns (exclude subject, timepoint, and Wear_Time)
    fitbit_metric_cols = [col for col in fitbit_df.columns if col not in ["subject", "timepoint", "Wear_Time"]]

    # Coerce to numeric
    for col in fitbit_metric_cols:
        fitbit_df[col] = pd.to_numeric(fitbit_df[col], errors="coerce")

    # Create a dataframe to hold the extracted features
    features_list = []

    # Loop through each subject and timepoint to extract features
    grouped = fitbit_df.groupby(["subject", "timepoint"])
    for (subject, timepoint), group in tqdm(grouped, total=grouped.ngroups, desc="Extracting Fitbit features"):
        feature_dict = {"subject": subject, "timepoint": timepoint}
        for metric in fitbit_metric_cols:
            # Check if the metric column exists in the group
            #if metric in group.columns:
                daily_data = group[["Wear_Time", metric]]#.dropna()
                if not daily_data.empty:
                    # Create daily features (mean, std, min, max)
                    daily_data.set_index("Wear_Time", inplace=True)
                    daily_stats = daily_data.resample("D").agg(['mean', 'std', 'min', 'max'])
                    daily_stats.columns = ['_'.join(col) for col in daily_stats.columns]
                    # Create datetime index with proper missing days based on the daily resampling range
                    min_date = daily_stats.index.min().floor("D")
                    max_date = daily_stats.index.max().ceil("D")
                    date_range = pd.date_range(start=min_date, end=max_date, freq="D")
                    # Reindex to include missing days and impute missing values with multiple imputation
                    daily_stats = daily_stats.reindex(date_range)
                    #if daily_stats.shape[0] > 1 and daily_stats.notna().sum().sum() > daily_stats.shape[1]:
                    try:
                        imputer = IterativeImputer(random_state=0, max_iter=20)
                        daily_stats = pd.DataFrame(
                            imputer.fit_transform(daily_stats),
                            index=daily_stats.index,
                            columns=daily_stats.columns,
                        )
                    except Exception as e:
                        print(f"Iterative imputation failed for subject {subject}, timepoint {timepoint}, metric {metric}: {e}")
                        daily_stats = daily_stats.ffill().bfill()
                    #else:
                        #daily_stats = daily_stats.ffill().bfill()
                    feature_dict.update(daily_stats.mean().to_dict())
                    # STL decomposition
                    try:
                        stl = STL(daily_data[metric], period=7, robust=True)
                        result = stl.fit()
                        stl_features = {
                            f"{metric}_trend_mean": result.trend.mean(),
                            f"{metric}_trend_std": result.trend.std(),
                            f"{metric}_trend_min": result.trend.min(),
                            f"{metric}_trend_max": result.trend.max(),
                            f"{metric}_seasonal_mean": result.seasonal.mean(),
                            f"{metric}_seasonal_std": result.seasonal.std(),
                            f"{metric}_seasonal_min": result.seasonal.min(),
                            f"{metric}_seasonal_max": result.seasonal.max(),
                            f"{metric}_resid_mean": result.resid.mean(),
                            f"{metric}_resid_std": result.resid.std(),
                            f"{metric}_resid_min": result.resid.min(),
                            f"{metric}_resid_max": result.resid.max(),
                        }
                        feature_dict.update(stl_features)
                    except Exception as e:
                        print(f"STL decomposition failed for subject {subject}, timepoint {timepoint}, metric {metric}: {e}")
        features_list.append(feature_dict)
    fitbit_features_df = pd.DataFrame(features_list)

    return fitbit_features_df
