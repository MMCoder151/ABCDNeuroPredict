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
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

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

def analyse_confounds(dem_df, mri_meta_df, transformed_data, output_path=pathlib.Path("output"), raw_data = None, con = None, view = None):
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
                "sex", "scan_site", "age_at_mri", "mr_y_smri__vol__aseg__icv_sum"]
    raw_analysis_cols = [col for col in raw_data.columns if col not in exclude_cols]
    transformed_analysis_cols = [col for col in transformed_data.columns if col not in exclude_cols]

    if "subject_ids" in transformed_data.columns:
        transformed_data.rename(columns={"subject_ids": "subject"}, inplace=True)
    if "subject_ids" in raw_data.columns:
        raw_data.rename(columns={"subject_ids": "subject"}, inplace=True)

    # Check that analysis_cols match in both raw and transformed data
    if not set(raw_analysis_cols).issubset(set(transformed_analysis_cols)):
        print(f"Error: Columns do not match between raw and transformed data.")
        print(f"Dropping columns from analysis that are not present in both datasets: {set(raw_analysis_cols) - set(transformed_analysis_cols)}")
        analysis_cols = [col for col in raw_analysis_cols if col in transformed_analysis_cols]
    else:
        analysis_cols = raw_analysis_cols

    # Drop zero variance columns in raw data from analysis_cols
    zero_variance_cols = raw_data[analysis_cols].var()[raw_data[analysis_cols].var() == 0].index.tolist()
    if zero_variance_cols:
        print(f"Dropping zero variance columns from analysis: {len(zero_variance_cols)}")
        print(zero_variance_cols)
        analysis_cols = [c for c in analysis_cols if c not in zero_variance_cols]

    # Define columns to attach and merge them onto both raw and transformed data
    dem_cols_to_attach = ["age_at_first_mri", "sex", "scan_site"]
    mri_cols_to_attach = ["mr_y_smri__vol__aseg__icv_sum"]

    transformed_data = transformed_data.drop(
        columns=[c for c in dem_cols_to_attach if c in transformed_data.columns]
    )
    transformed_data = transformed_data.drop(
        columns=[c for c in mri_cols_to_attach if c in transformed_data.columns]
    )
    raw_data = raw_data.drop(
        columns=[c for c in dem_cols_to_attach if c in raw_data.columns]
    )
    raw_data = raw_data.drop(
        columns=[c for c in mri_cols_to_attach if c in raw_data.columns]
    )

    # Merge transformed data with demographic data to get age, sex, scan_site, and TIV for each subject
    transformed_data = transformed_data.merge(
        dem_df[["subject", "age_at_first_mri", "sex", "scan_site"]].drop_duplicates(),
        left_on="subject",
        right_on="subject",
        how="inner"
    )
    transformed_data = transformed_data.merge(
        mri_meta_df[["subject", "mr_y_smri__vol__aseg__icv_sum"]].drop_duplicates(),
        left_on="subject",
        right_on="subject",
        how="inner"
    )

    # Merge raw data with demographic data to get age, sex, scan_site, and TIV for each subject
    raw_data = raw_data.merge(
        dem_df[["subject", "age_at_first_mri", "sex", "scan_site"]].drop_duplicates(),
        left_on="subject",
        right_on="subject",
        how="inner"
    )
    raw_data = raw_data.merge(
        mri_meta_df[["subject", "mr_y_smri__vol__aseg__icv_sum"]].drop_duplicates(),
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

    confound_effects = []

    # Define model hierarchy
    model_hierarchy = {
        'site only':          site_dummy_cols,
        'site + TIV':         site_dummy_cols + ['mr_y_smri__vol__aseg__icv_sum'],
        'site + TIV + age':         site_dummy_cols + ['mr_y_smri__vol__aseg__icv_sum', 'age_at_first_mri'],
        'site + TIV + age^2': site_dummy_cols + ['mr_y_smri__vol__aseg__icv_sum', 'age_at_first_mri_c_sq'],
        'site + TIV + age + sex':   site_dummy_cols + ['mr_y_smri__vol__aseg__icv_sum', 'age_at_first_mri'] + sex_dummy_cols,
        'site + TIV + age^2 + sex': site_dummy_cols + ['mr_y_smri__vol__aseg__icv_sum', 'age_at_first_mri_c_sq'] + sex_dummy_cols,
        'TIV + age^2 + sex': ['mr_y_smri__vol__aseg__icv_sum', 'age_at_first_mri_c_sq'] + sex_dummy_cols
    }

    # Impute missing TIV values
    imputer = SimpleImputer(strategy='mean')
    raw_data['mr_y_smri__vol__aseg__icv_sum'] = imputer.fit_transform(raw_data[['mr_y_smri__vol__aseg__icv_sum']])
    transformed_data['mr_y_smri__vol__aseg__icv_sum'] = imputer.fit_transform(transformed_data[['mr_y_smri__vol__aseg__icv_sum']])

    for roi in tqdm(analysis_cols, desc="Analyzing confound effects"):
        # Prepare data for regression
        pre_df = raw_data[["subject", "age_at_first_mri", "age_at_first_mri_c_sq", "mr_y_smri__vol__aseg__icv_sum", roi] + site_dummy_cols + sex_dummy_cols].dropna(subset=[roi])
        pre_df = pre_df.apply(pd.to_numeric, errors='coerce').astype('float64')
        post_df = transformed_data[["subject", "age_at_first_mri", "age_at_first_mri_c_sq", "mr_y_smri__vol__aseg__icv_sum", roi] + site_dummy_cols + sex_dummy_cols].dropna(subset=[roi])
        post_df = post_df.apply(pd.to_numeric, errors='coerce').astype('float64')

        if pre_df[roi].nunique() <= 1 or len(pre_df) <= 1:
            print(f"WARNING: {roi} has no variance or too few rows in raw_data (n={len(pre_df)}, unique={pre_df[roi].nunique()})")
        if post_df[roi].nunique() <= 1 or len(post_df) <= 1:
            print(f"WARNING: {roi} has no variance or too few rows in transformed_data (n={len(post_df)}, unique={post_df[roi].nunique()})")

        X_pre = pre_df[site_dummy_cols + ["age_at_first_mri", "age_at_first_mri_c_sq", "mr_y_smri__vol__aseg__icv_sum"] + sex_dummy_cols]
        X_post = post_df[site_dummy_cols + ["age_at_first_mri", "age_at_first_mri_c_sq", "mr_y_smri__vol__aseg__icv_sum"] + sex_dummy_cols]
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

    # Print site effect R2 pre and post per ROI
    site_effects = df[df['model'] == 'site only']
    print("Site Effect R2 per ROI:")
    for _, row in site_effects.iterrows():
        print(f"  {row['variable']}: R2 pre={row['R2_pre']:.4f}, R2 post={row['R2_post']:.4f}")

    # Pivot the dataframe to have models as columns and variables as rows for easier comparison
    pivot = df.pivot(index='variable', columns='model')

    # TIV effect = (site+TIV) - (site only)
    tiv_R2_pre  = pivot['R2_pre']['site + TIV'] - pivot['R2_pre']['site only']
    tiv_R2_post = pivot['R2_post']['site + TIV'] - pivot['R2_post']['site only']
    tiv_reduction = (tiv_R2_pre - tiv_R2_post)
    print(f"TIV effect: mean R2 pre={tiv_R2_pre.mean():.4f}, mean R2 post={tiv_R2_post.mean():.4f}")
    print(f"TIV effect: mean R2 reduction={tiv_reduction.mean():.4f}")

    # Age effect = (site+TIV+age) - (site+TIV)
    age_R2_pre  = pivot['R2_pre']['site + TIV + age'] - pivot['R2_pre']['site + TIV']
    age_R2_post = pivot['R2_post']['site + TIV + age'] - pivot['R2_post']['site + TIV']
    age_reduction = (age_R2_pre - age_R2_post)
    print(f"Age effect: mean R2 pre={age_R2_pre.mean():.4f}, mean R2 post={age_R2_post.mean():.4f}")
    print(f"Age effect: mean R2 reduction={age_reduction.mean():.4f}")

    # Age^2 effect = (site+TIV+age^2) - (site+TIV)
    age2_R2_pre  = pivot['R2_pre']['site + TIV + age^2'] - pivot['R2_pre']['site + TIV']
    age2_R2_post = pivot['R2_post']['site + TIV + age^2'] - pivot['R2_post']['site + TIV']
    age2_reduction = (age2_R2_pre - age2_R2_post)
    print(f"Age^2 effect: mean R2 pre={age2_R2_pre.mean():.4f}, mean R2 post={age2_R2_post.mean():.4f}")
    print(f"Age^2 effect: mean R2 reduction={age2_reduction.mean():.4f}")

    # Sex effect = (site+TIV+age+sex) - (site+TIV+age)
    sex_R2_pre  = pivot['R2_pre']['site + TIV + age + sex'] - pivot['R2_pre']['site + TIV + age']
    sex_R2_post = pivot['R2_post']['site + TIV + age + sex'] - pivot['R2_post']['site + TIV + age']
    sex_reduction = (sex_R2_pre - sex_R2_post)
    print(f"Sex effect: mean R2 pre={sex_R2_pre.mean():.4f}, mean R2 post={sex_R2_post.mean():.4f}")
    print(f"Sex effect: mean R2 reduction={sex_reduction.mean():.4f}")

    # Site effect is just the R2 of 'site only'
    site_R2_pre  = pivot['R2_pre']['site only']
    site_R2_post = pivot['R2_post']['site only']
    site_reduction = site_R2_pre - site_R2_post
    print(f"Site effect: mean R2 pre={site_R2_pre.mean():.4f}, mean R2 post={site_R2_post.mean():.4f}")
    print(f"Site effect: mean R2 reduction={site_reduction.mean():.4f}")

    # Partial site efffect = (site + TIV + age^2 + sex) - (TIV + age^2 + sex)
    site_partial_R2_pre  = pivot['R2_pre']['site + TIV + age^2 + sex'] - pivot['R2_pre']['TIV + age^2 + sex']
    site_partial_R2_post = pivot['R2_post']['site + TIV + age^2 + sex'] - pivot['R2_post']['TIV + age^2 + sex']
    site_partial_reduction = (site_partial_R2_pre - site_partial_R2_post)
    print(f"Partial site effect: mean R2 pre={site_partial_R2_pre.mean():.4f}, mean R2 post={site_partial_R2_post.mean():.4f}")
    print(f"Partial site effect: mean R2 reduction={site_partial_reduction.mean():.4f}")

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
        f"site + TIV={residual_association_df.loc[residual_association_df['model'] == 'site + TIV', 'R2_post'].mean():.4f}, "
        f"site + TIV + age={residual_association_df.loc[residual_association_df['model'] == 'site + TIV + age', 'R2_post'].mean():.4f}, "
        f"site + TIV + age^2={residual_association_df.loc[residual_association_df['model'] == 'site + TIV + age^2', 'R2_post'].mean():.4f}, "
        f"site + TIV + age + sex={residual_association_df.loc[residual_association_df['model'] == 'site + TIV + age + sex', 'R2_post'].mean():.4f}, "
        f"site + TIV + age^2 + sex={residual_association_df.loc[residual_association_df['model'] == 'site + TIV + age^2 + sex', 'R2_post'].mean():.4f}, "
        f"TIV + age^2 + sex={residual_association_df.loc[residual_association_df['model'] == 'TIV + age^2 + sex', 'R2_post'].mean():.4f}"
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

    valid = (~tiv_R2_pre.isna()) & (~tiv_R2_post.isna())
    stat, p_tiv = wilcoxon(tiv_R2_pre[valid], tiv_R2_post[valid])
    print('TIV R2 Wilcoxon p=', p_tiv)

    valid = (~site_partial_R2_pre.isna()) & (~site_partial_R2_post.isna())
    stat, p_site_partial = wilcoxon(site_partial_R2_pre[valid], site_partial_R2_post[valid])
    print('Partial Site R2 Wilcoxon p=', p_site_partial)

    confound_effects_df = pd.DataFrame({
        "variable": analysis_cols,
        "site_R2_pre": site_R2_pre.values,
        "site_R2_post": site_R2_post.values,
        "age_R2_pre": age_R2_pre.values,
        "age_R2_post": age_R2_post.values,
        "age2_R2_pre": age2_R2_pre.values,
        "age2_R2_post": age2_R2_post.values,
        "tiv_R2_pre": tiv_R2_pre.values,
        "tiv_R2_post": tiv_R2_post.values
    })

    return confound_effects_df

def normative_selection(mri_meta_df, roi_cols, data = None, con = None, output_path=pathlib.Path("output"), overwrite=True):
    '''
    This function performs normative modeling and selects subjects based on their composite absolute z-score. 
    It selects the top 10% (based on prevalence) of subjects with the highest cumulative z-score.
    These subjects are considered to have abnormal development in the selected MRI ROIs associated with depression.

    Parameters:
        con (duckdb.Connection): DuckDB connection with views for fitbit and mri data
        mri_meta_df (DataFrame): DataFrame containing MRI metadata
        roi_cols (list): List of MRI ROI column names
        data (DataFrame, optional): Pre-loaded MRI data. If provided, this will be used instead of querying the database.
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

    # Get subjects to include from mri_meta_df
    included_subjects = mri_meta_df["subject"].unique()
    first_mri_meta_df = (
        mri_meta_df.sort_values(["subject", "timepoint"])
        .drop_duplicates(subset=["subject"], keep="first")
        [["subject", "sex", "age_at_mri", "scan_site", "mr_y_smri__vol__aseg__icv_sum"]]
    )
    
    if data is not None:
        print("Using provided data for normative modeling.")
        mri_df = data
    else:
        # Query MRI data for the first timepoint for each included subject
        query = f"""
            SELECT "subject", "timepoint", {', '.join(f'"{col}"' for col in roi_cols)}
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

    # Filter df to only include healthy subjects (subjects with dep_dx == 0 in mri_meta_df)
    healthy_subjects = mri_meta_df[mri_meta_df["dep_dx"] == 0]["subject"].unique()
    df_reference = df[df["subject"].isin(healthy_subjects)]

    # Prepare data for normative modeling
    data_reference = NormData.from_dataframe(
        name="mri_norm_reference",
        dataframe=df_reference,
        covariates=["sex", "age_at_mri", "mr_y_smri__vol__aseg__icv_sum"] + site_dummy_cols,
        response_vars=roi_cols,
        subject_ids="subject",
        remove_Nan=True,
    )

    data_full = NormData.from_dataframe(
        name="mri_norm",
        dataframe=df,
        covariates=["sex", "age_at_mri", "mr_y_smri__vol__aseg__icv_sum"] + site_dummy_cols,
        response_vars=roi_cols,
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

    model.fit(data_reference)
    model.predict(data_full)

    # create a runner
    #runner = Runner(cross_validate = True)

    # fit the model 
    #runner.fit(model, data)

    # Read in z-score file from normative modeling 
    centiles_df = pd.read_csv(normative_output_dir / "results" / "Z_mri_norm.csv")

    # Read in results file from roi selection
    mri_rois_results = pd.read_csv(output_path / "mri_rois_results.csv")

    def _build_sign_aligned_composite(centiles_df, roi_cols, mri_rois_results):
        """
        Build a sign-aligned composite deviation score, using the direction of each
        ROI's group-difference effect size to align z-scores before combining.
        """
        # Map each ROI to the sign of its effect size from the group-difference analysis
        effect_size_lookup = mri_rois_results.set_index("mri_feature")["effect_size"]
        sign_lookup = np.sign(effect_size_lookup.reindex(roi_cols))

        # Sanity check: make sure every ROI used here actually has a known direction
        missing = sign_lookup[sign_lookup.isna()]
        if len(missing) > 0:
            raise ValueError(f"Missing effect-size sign for ROIs: {missing.index.tolist()}")

        aligned = centiles_df[roi_cols].copy()
        for roi in roi_cols:
            aligned[roi] = centiles_df[roi] * sign_lookup[roi]

        # Now "more positive" consistently means "more depression-like" across all ROIs
        centiles_df["composite_z_aligned"] = aligned.sum(axis=1)
        return centiles_df

    # Calculate composite absolute z-score across all MRI ROIs for each subject
    centiles_df = _build_sign_aligned_composite(centiles_df, roi_cols, mri_rois_results)

    # Weigh z-scores by absolute effect size of each ROI from the group-difference analysis
    effect_size_lookup = mri_rois_results.set_index("mri_feature")["effect_size"]
    weighted = centiles_df[roi_cols].copy()
    for roi in roi_cols:
        weighted[roi] = centiles_df[roi] * abs(effect_size_lookup[roi])
    centiles_df["composite_z_weighted"] = weighted.sum(axis=1)

    from sklearn.metrics import roc_auc_score, roc_curve
    depressed_subject_set = set(mri_meta_df[mri_meta_df["dep_dx"] == 1]["subject"].unique())
    y_true = centiles_df["subject_ids"].isin(depressed_subject_set).astype(int)  # deduplicated!
    y_score = centiles_df["composite_z_aligned"]
    auc = roc_auc_score(y_true, y_score)
    print(f"AUC: {auc:.4f}")

    # Check AUC per ROI in roi_cols
    print("Calculating AUC per ROI...")
    y_true = centiles_df["subject_ids"].isin(depressed_subject_set).astype(int)

    roi_auc_results = []
    for roi in roi_cols:
        roi_scores = centiles_df[roi]
        valid = roi_scores.notna()
        if valid.sum() < 10:  # skip ROIs with too little data to compute a meaningful AUC
            continue
        auc = roc_auc_score(y_true[valid], roi_scores[valid])
        roi_auc_results.append({"roi": roi, "auc": auc, "n": valid.sum()})

    roi_auc_df = pd.DataFrame(roi_auc_results).sort_values("auc", ascending=False).reset_index(drop=True)
    print(roi_auc_df.to_string(index=False))

    roi_auc_df["auc_abs"] = roi_auc_df["auc"].apply(lambda a: max(a, 1 - a))
    roi_auc_df = roi_auc_df.sort_values("auc_abs", ascending=False).reset_index(drop=True)

    roi_auc_df = roi_auc_df.merge(
        mri_rois_results[["mri_feature", "effect_size"]],
        left_on="roi", right_on="mri_feature", how="left"
    )
    roi_auc_df["direction_agrees"] = (
        (roi_auc_df["effect_size"] > 0) & (roi_auc_df["auc"] > 0.5)
    ) | (
        (roi_auc_df["effect_size"] < 0) & (roi_auc_df["auc"] < 0.5)
    )
    print(roi_auc_df[["roi", "effect_size", "auc", "auc_abs", "direction_agrees"]].to_string(index=False))

    subject_scores = (
        centiles_df[["subject_ids", "composite_z_aligned"]]
        .dropna()
        .drop_duplicates(subset=["subject_ids"])
        .sort_values("composite_z_aligned")
        .reset_index(drop=True)
    )

    subject_scores["subject_ids"] = subject_scores["subject_ids"].astype(str)
    subject_scores["subject_ids"].nunique()
    subject_scores["rank"] = np.arange(len(subject_scores))

    # Select top 5% of subjects with the highest composite z-score based on ranked prevalence
    n_select = int(np.ceil(0.05 * len(subject_scores)))
    selected_subject_ids = subject_scores.nlargest(n_select, "composite_z_aligned")["subject_ids"]
    selected_subjects = subject_scores[subject_scores["subject_ids"].isin(selected_subject_ids)]
    
    print(f"Selected {len(selected_subject_ids)} subjects with the highest composite z-scores based on a prevalence threshold of 5%.")
    selected_subjects.to_csv(normative_output_dir / "results" / "selected_subjects.csv", index=False)
    
    # create scatter plot of composite z-scores for all subjects, highlighting selected subjects in a different color
    plot_df = centiles_df[["subject_ids", "composite_z_aligned"]].dropna().sort_values("composite_z_aligned").reset_index(drop=True)
    plot_df["rank"] = np.arange(len(plot_df))

    plt.figure(figsize=(10, 6))
    plt.scatter(subject_scores["rank"], subject_scores["composite_z_aligned"], label="All subjects", alpha=0.5, s=12)
    selected_plot = subject_scores[subject_scores["subject_ids"].isin(selected_subject_ids)]
    plt.scatter(selected_plot["rank"], selected_plot["composite_z_aligned"], label="Selected subjects", color="red", s=18)
    plt.xlabel("Subject rank by composite z-score")
    plt.ylabel("Composite Absolute Z-Score")
    plt.title("Composite Absolute Z-Scores for MRI ROIs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(normative_output_dir / "results" / "composite_z_scores.png")
    plt.close()

    # Create summary table with mean, std, min, and max per mri_roi for the selected subjects
    stats_selected = (
        centiles_df[roi_cols]
        .agg(['mean', 'std', 'min', 'max'])
        .transpose()
        .reset_index()
        .rename(columns={"index": "mri_roi"})
    )
    stats_selected.to_csv(normative_output_dir / "results" / "mri_roi_statistics.csv", index=False)

    # Create results summary table with mean, std, min, and max per metric
    stats_df = pd.read_csv(normative_output_dir / "results" / "statistics_mri_norm.csv")
    summary = stats_df.assign(
        mean = stats_df[roi_cols].mean(axis=1),
        std  = stats_df[roi_cols].std(axis=1),
        min  = stats_df[roi_cols].min(axis=1),
        max  = stats_df[roi_cols].max(axis=1),
    )[["statistic", "mean", "std", "min", "max"]]
    summary.to_csv(normative_output_dir / "results" / "statistics_summary.csv", index=False)

    return selected_subjects

def create_composites(selected_subjects, vif_threshold=10, overwrite=True, output_path=Path("output"), composite_output = None):
    """
    Creates composite scores out of given variables based on variance inflation factors (VIF) and drops zero variance columns.
 
    Parameters:
        selected_subjects (DataFrame): DataFrame containing the selected subjects and their data to be analysed
        vif_threshold (float): VIF value above which a variable triggers compositing
        max_iterations (int): safety cap so a degenerate case can't loop forever
 
    Returns:
        selected_subjects (DataFrame): DataFrame with high-VIF columns replaced by composites
        composite_dict (dict): Maps composite score name -> list of *original* base columns
                                that were averaged together to build it (flattened, not nested)
    """

    if overwrite == False:
        print("Overwrite set to False. Reimporting composites.")
        try:
            if composite_output is not None:
                composite_output_path = Path(output_path) / composite_output
                features_path = composite_output_path / "features_with_composites.csv"
                composite_dict_path = composite_output_path / "composite_dictionary.csv"
            else:
                features_path = Path(output_path) / "fitbit_features_with_composites.csv"
                composite_dict_path = Path(output_path) / "composite_dictionary.csv"

            features_with_composites = pd.read_csv(features_path)
            composite_df = pd.read_csv(composite_dict_path)
            return features_with_composites, composite_df
        except Exception as e:
            print(f"An error occured: {e}")
            return e

    vif_cols = [
        col for col in selected_subjects.columns
        if col not in ["subject", "participant_id", "composite_z", "Wear_Time", "subtype", "group"]
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
        if df.shape[1] == 0:
            return pd.DataFrame(columns=["variable", "vif"])
        if df.shape[1] == 1:
            return pd.DataFrame({"variable": df.columns, "vif": [np.inf]})

        df_with_const = sm.add_constant(df)
        vifs = [
            variance_inflation_factor(df_with_const.values, i)
            for i in range(1, df_with_const.shape[1])  # skip the constant column (index 0)
        ]
        return pd.DataFrame({"variable": df.columns, "vif": vifs}).sort_values("vif", ascending=False)
 
    vif_df = vif_table(vif_data)
 
    while not vif_df.empty and vif_df["vif"].max() > vif_threshold:
        if vif_data.shape[1] < 2:
            print("Stopping composite creation because fewer than two VIF columns remain.")
            break
 
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

    if zero_variance_cols:
        print(f"Dropping columns with zero variance: {len(zero_variance_cols)}")
        print(zero_variance_cols)
        selected_subjects.drop(columns=zero_variance_cols, inplace=True, errors="ignore")

    if composite_output is not None:
        composite_output_path = Path(output_path) / composite_output
        if not composite_output_path.exists():
            composite_output_path.mkdir(parents=True)
        selected_subjects.to_csv(Path(composite_output_path / "features_with_composites.csv"), index=False)
        composite_df = pd.DataFrame({
            "composite_name": list(composite_dict.keys()),
            "features_included": [", ".join(features) for features in composite_dict.values()]
        })
        composite_df.to_csv(Path(composite_output_path / "composite_dictionary.csv"), index=False)
    else:    
        selected_subjects.to_csv(Path(output_path / "features_with_composites.csv"), index=False)
        composite_df = pd.DataFrame({
            "composite_name": list(composite_dict.keys()),
            "features_included": [", ".join(features) for features in composite_dict.values()]
        })
        composite_df.to_csv(Path(output_path / "composite_dictionary.csv"), index=False)
    
    return selected_subjects, composite_dict

def extr_fitbit_features(con, selected_subjects, overwrite=True, output_path=Path("output")):
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

    if overwrite == False:
        print("Overwirte set to False. Reimporting features.")
        try:
            fitbit_features = pd.read_csv(Path(output_path) / "fitbit_features.csv")
            return fitbit_features
        except Exception as e:
            print(f"An error occured: {e}")
            return e

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
                    # Create afternoon features (mean, std, min, max) for the time range 15:00 to 20:00
                    afternoon_data = daily_data.between_time("15:00", "20:00")
                    if not afternoon_data.empty:
                        afternoon_stats = afternoon_data.resample("D").agg(['mean', 'std', 'min', 'max'])
                        afternoon_stats.columns = ['_an_'.join(col) for col in afternoon_stats.columns]
                        daily_stats = pd.concat([daily_stats, afternoon_stats], axis=1)

                    # Create datetime index with proper missing days based on the daily resampling range
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
    
    fitbit_features_df.to_csv(Path("output")/ "fitbit_features.csv", index=False)

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

def fit_residualiser(X_train, dem_df, mri_meta_df, overwrite=True, residualiser_output="residualisation"):
    '''
    Fit a GPR per feature on TRAINING data only.
    Covariates: age (continuous) + sex (dummy-coded) + scan site (dummy-coded) + TIV (continuous)
    '''

    if overwrite == False:
        print("Residualiser fitting skipped (overwrite=False). To re-run residualiser fitting, set overwrite=True.")
        try:
            models = []
            residualisation_dir = Path("output") / residualiser_output
            for i in range(len(X_train.columns)-1):
                gpr = joblib.load(residualisation_dir / f"gpr_model_{i}.joblib")
                models.append(gpr)
            return models
        except Exception as e:
            print(f"Error loading GPR models: {e}")
            print("Please check that the gpr_model_*.joblib files exist in the residualisation directory and are correctly formatted.")
            raise e
        
    X_train.dropna(inplace=True)
    columns_to_drop = ["subject", "subtype", "age_at_first_mri", "sex", "Wear_Time", "timepoint"]
    columns_to_drop = [c for c in columns_to_drop if c in X_train.columns]
    X_train = X_train.drop(columns=columns_to_drop, errors="ignore")

    # drop columns with zero variance
    variances = X_train.var()
    zero_variance_cols = variances[variances == 0].index.tolist()
    if zero_variance_cols:
        print(f"Dropping columns with zero variance from residualisation: {len(zero_variance_cols)}")
        print(zero_variance_cols)
        X_train = X_train.drop(columns=zero_variance_cols, errors="ignore")

    age_train = dem_df.loc[X_train.index, "age_at_first_mri"].values.reshape(-1, 1)
    sex_train = pd.get_dummies(dem_df.loc[X_train.index, "sex"], drop_first=True).values
    scan_site_train = pd.get_dummies(dem_df.loc[X_train.index, "scan_site"], drop_first=True).values
    tiv_train = mri_meta_df.loc[X_train.index, "mr_y_smri__vol__aseg__icv_sum"].values.reshape(-1, 1)
    tiv_train = np.nan_to_num(tiv_train, nan=np.nanmean(tiv_train))
    # z-score age and tiv
    scaler = StandardScaler()
    age_train = scaler.fit_transform(age_train)
    tiv_train = scaler.fit_transform(tiv_train)

    design_matrix_train = np.hstack([age_train, sex_train, scan_site_train, tiv_train])

    n_features = X_train.shape[1]
    models = []
    for i in tqdm(range(n_features), desc="Fitting GPR residualiser"):
        y_train = X_train.iloc[:, i].values
        n_dims = design_matrix_train.shape[1]
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_dims)) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=5)
        gpr.fit(design_matrix_train, y_train)
        models.append(gpr)
        
    #Save models to disk in residualisation folder
    residualisation_dir = Path("output") / residualiser_output
    if not residualisation_dir.exists():
        residualisation_dir.mkdir(parents=True)
    for i, gpr in enumerate(models):
        joblib.dump(gpr, residualisation_dir / f"gpr_model_{i}.joblib")
    return models

def apply_residualiser(models, X, dem_df, mri_meta_df):
    '''
    Apply the fitted GPR residualiser to new data (e.g. test set).
    '''

    X.dropna(inplace=True)
    columns_to_drop = ["subject", "subtype", "age_at_first_mri", "sex", "Wear_Time", "timepoint"]
    columns_to_drop = [c for c in columns_to_drop if c in X.columns]
    dropped_cols_df = X[columns_to_drop].copy()
    X = X.drop(columns=columns_to_drop, errors="ignore")

    # drop columns with zero variance
    variances = X.var()
    zero_variance_cols = variances[variances == 0].index.tolist()
    if zero_variance_cols:
        print(f"Dropping columns with zero variance from residualisation: {len(zero_variance_cols)}")
        print(zero_variance_cols)
        dropped_var_cols_df = X[zero_variance_cols].copy()
        dropped_cols_df = pd.concat([dropped_cols_df, dropped_var_cols_df], axis=1)
        X = X.drop(columns=zero_variance_cols, errors="ignore")

    age = dem_df.loc[X.index, "age_at_first_mri"].values.reshape(-1, 1)
    sex = pd.get_dummies(dem_df.loc[X.index, "sex"], drop_first=True).values
    scan_site = pd.get_dummies(dem_df.loc[X.index, "scan_site"], drop_first=True).values
    tiv = mri_meta_df.loc[X.index, "mr_y_smri__vol__aseg__icv_sum"].values.reshape(-1, 1)
    tiv = np.nan_to_num(tiv, nan=np.nanmean(tiv))
    design_matrix = np.hstack([age, sex, scan_site, tiv])

    X_residualised = X.copy()
    for i, gpr in tqdm(enumerate(models), desc="Applying GPR residualiser", total=len(models)):
        predicted = gpr.predict(design_matrix)
        X_residualised.iloc[:, i] = X.iloc[:, i].values - predicted

    X_residualised = pd.concat([dropped_cols_df.loc[X_residualised.index], X_residualised], axis=1)
    
    return X_residualised

def resample(X, y):
    '''
    Resamples data to balance classes in the target y using a hybrid under- and over-sampling approach, while minimising the amount of synthetic data generated.
        1. Under-sample the majority class using RandomUnderSampler to 75% of the original majority class size
        2. Over-sample the minority class using SMOTE to 25% of the original majority class size
    Parameters:
        X (DataFrame): DataFrame containing the features to be resampled
        y (Series): Series containing the target variable to be resampled
    Returns:
        X_resampled (DataFrame): DataFrame containing the resampled features
        y_resampled (Series): Series containing the resampled target variable
    NOTE: Works mainly with binary class labels
    '''

    # Get feature columns
    cols_to_exclude = ["subject", "subtype", "age_at_first_mri", "sex", "Wear_Time", "timepoint"]
    feature_cols = [col for col in X.columns if col not in cols_to_exclude]

    # Under-sample the majority class to 75% of the original majority class size
    original_class_distribution = y.value_counts()
    print(f"Original class distribution: {original_class_distribution.to_dict()}")

    majority_class = original_class_distribution.idxmax()
    minority_class = original_class_distribution.idxmin()

    target_majority_count = int(original_class_distribution[majority_class] * 0.75)

    undersample = RandomUnderSampler(
        sampling_strategy={
            majority_class: target_majority_count,
            minority_class: original_class_distribution[minority_class]
        },
        random_state=42
    )

    X_undersampled, y_undersampled = undersample.fit_resample(
        X, y
    )

    # Recover subject IDs for the original undersampled rows; synthetic SMOTE rows get missing IDs.
    undersampled_source_indices = getattr(undersample, "sample_indices_", None)
    if undersampled_source_indices is None:
        raise RuntimeError("RandomUnderSampler did not expose sample_indices_; cannot recover subject IDs safely.")
    undersampled_subject_ids = X.iloc[undersampled_source_indices]["subject"].reset_index(drop=True)

    
    # If there are missings in the undersampled data, impute them using IterativeImputer
    if X_undersampled[feature_cols].isnull().any().any():
        print("Missing values detected in undersampled data. Imputing missing values using IterativeImputer...")
        imputer = IterativeImputer(random_state=0, max_iter=20)
        X_undersampled_imputed = pd.DataFrame(
            imputer.fit_transform(X_undersampled[feature_cols]),
            columns=feature_cols,
            index=X_undersampled.index
        )

    # Over-sample the minority class to 25% of the original majority class size
    target_minority_count = int(original_class_distribution[majority_class] * 0.25)

    smote = SMOTE(
        sampling_strategy={
            majority_class: target_majority_count,
            minority_class: target_minority_count
        },
        random_state=42
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_undersampled_imputed, y_undersampled
    )

    X_resampled["subject_ids"] = pd.concat(
    [
        undersampled_subject_ids,
        pd.Series([pd.NA] * (len(X_resampled) - len(undersampled_subject_ids))),
    ],
    ignore_index=True,
    )   

    # Print class distribution of original and resampled data
    resampled_class_distribution = pd.Series(y_resampled).value_counts()
    print("Original class distribution:")
    print(original_class_distribution)
    print("Resampled class distribution:")
    print(resampled_class_distribution)

    return X_resampled, y_resampled
