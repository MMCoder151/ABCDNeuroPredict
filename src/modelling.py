from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier --- Issues with xgboost installation
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
    
class FractionalSelectKBest(BaseEstimator, TransformerMixin):
    """
    SelectKBest where k is defined as a fraction of incoming features,
    making it robust to variable upstream feature counts.
    """
    def __init__(self, fraction=0.5, score_func=mutual_info_classif):
        self.fraction = fraction
        self.score_func = score_func

    def fit(self, X, y=None):
        k = max(1, int(X.shape[1] * self.fraction))
        self.selector_ = SelectKBest(score_func=self.score_func, k=k)
        self.selector_.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.selector_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.selector_.get_feature_names_out(input_features)

class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely NaN in the training split."""

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self._is_dataframe = True
            self.kept_columns_ = X.columns[~X.isna().all(axis=0)].tolist()
        else:
            self._is_dataframe = False
            X_array = np.asarray(X)
            self.kept_indices_ = np.where(~np.isnan(X_array).all(axis=0))[0]
        return self

    def transform(self, X):
        if self._is_dataframe:
            return X.loc[:, self.kept_columns_]
        X_array = np.asarray(X)
        return X_array[:, self.kept_indices_]
    
def define_models():
    models = {
            "Logistic Regression": (
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("thresholding", VarianceThreshold(threshold=0.01)),
                    ("tree_selector", SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=42), threshold="median")),
                    ("scaler", StandardScaler()),
                    ("selector", FractionalSelectKBest(score_func=mutual_info_classif)),
                    ("clf", LogisticRegression(max_iter=3000, random_state=42, penalty="elasticnet", solver="saga"))
                ]),
                {
                    "selector__fraction": [0.3, 0.5, 0.7],
                    "clf__C": [0.01, 0.1, 1.0, 10.0],
                    "clf__l1_ratio": [0.25, 0.5, 0.75],
                    "clf__class_weight": [None, "balanced"],
                }
            ),
            "Random Forest": (
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("thresholding", VarianceThreshold(threshold=0.01)),
                    ("tree_selector", SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=42), threshold="median")),
                    ("scaler", StandardScaler()),
                    ("selector", FractionalSelectKBest(score_func=mutual_info_classif)),
                    ("clf", RandomForestClassifier(n_jobs = 1, random_state=42))
                ]),
                {
                    "selector__fraction": [0.3, 0.5, 0.7],
                    "clf__n_estimators": [100, 300, 500],
                    "clf__min_samples_split": [2, 5, 10],
                    "clf__min_samples_leaf": [1, 2, 4],
                    "clf__max_features": ["sqrt", "log2"]
            }),
            "LightGBM": (
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("thresholding", VarianceThreshold(threshold=0.01)),
                    ("tree_selector", SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=42), threshold="median")),
                    ("scaler", StandardScaler()),
                    ("selector", FractionalSelectKBest(score_func=mutual_info_classif)),
                    #("to_numpy", FunctionTransformer(_to_numpy_if_needed, validate=False)),
                    ("clf", LGBMClassifier(n_threads=1, verbose=-1, random_state=42))
                ]),
                {
                    "selector__fraction": [0.3, 0.5, 0.7],
                    "clf__n_estimators": [200, 500, 800],
                    "clf__learning_rate": [0.01, 0.05, 0.1],
                    "clf__num_leaves": [31, 63, 127],
                    "clf__subsample": [0.8],
                    "clf__colsample_bytree": [0.8],
                    "clf__min_child_samples": [20, 40],
                    "clf__is_unbalance": [True]
            }),
            "SVM": (
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("thresholding", VarianceThreshold(threshold=0.01)),
                    ("tree_selector", SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=42), threshold="median")),
                    ("scaler", StandardScaler()),
                    ("selector", FractionalSelectKBest(score_func=mutual_info_classif)),
                    ("clf", SVC(random_state=42, probability=True))
                ]),
                {
                    "selector__fraction": [0.3, 0.5, 0.7],
                    "clf__kernel": ["rbf", "sigmoid"],
                    "clf__C": [0.1, 1.0, 10.0],
                    "clf__gamma": ["scale", "auto", 0.01, 0.1],
                    "clf__class_weight": [None, "balanced"]
            })
        }
    return models

def _to_numpy_if_needed(X):
        return X.to_numpy() if hasattr(X, "to_numpy") else X

def train_and_evaluate_models(X, y, search="random", outer_splits=10, inner_splits=10, models_to_train=None):
    ''' Trains and evaluates multiple machine learning models using nested cross-validation.
        
        Notes: 
        - Integrate further feature selection???
        - Extract feature importance where possible 
        - Description needs to be reworked. Not up-to-date with process optimization and data leakage fixes!!!
        - Input data needs to already be grouped by subject, otherwise the CV splits will be done on individual rows, 
        which may lead to data leakage or incomplete data within the cv splits.
    '''

    # Define nested cross-validation structure
    cv_struct = {
        "outer_cv": StratifiedKFold(
            n_splits=outer_splits,
            shuffle=True,
            random_state=42
        ),
        "inner_cv": StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=42
        )
    }

    # Define models with hyperparameters to be trained and evaluated 
    models = define_models()

    # Helper function to safely index dataframes or numpy arrays
    def _safe_index(data, indices):
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        return data[indices]

    # Helper function to create searcher based on specified strategy
    def make_searcher(model, param_grid):
        if search == "grid":
            return GridSearchCV(
                estimator=clone(model),
                param_grid=param_grid,
                cv=cv_struct["inner_cv"],
                scoring="roc_auc",
                n_jobs=3,
                verbose = 3
            )
        elif search == "random":
            return RandomizedSearchCV(
                estimator=clone(model),
                param_distributions=param_grid,
                cv=cv_struct["inner_cv"],
                scoring="roc_auc",
                n_jobs=3,
                n_iter=40,
                random_state=42,
                verbose = 3
            )
        raise ValueError("Invalid search strategy. Use 'grid' or 'random'.")

    print("\nStarting model training and evaluation...")

    # Create model subset to train only selected model(s)
    if models_to_train is not None:
        if isinstance(models_to_train, str):
            selected_names = [models_to_train]
        else:
            selected_names = list(models_to_train)

        available_names = set(models.keys())
        invalid_names = [name for name in selected_names if name not in available_names]
        if invalid_names:
            raise ValueError(
                f"Unknown model name(s): {invalid_names}. "
                f"Available: {sorted(available_names)}"
            )

        models = {name: models[name] for name in selected_names}

    nested_cv_scores = {}
    for name, (model, param_grid) in models.items():
        print(f"\nEvaluating {name}...")

        oof_pred = np.full(len(y), np.nan)
        outer_scores = []
        outer_best_params = []
        outer_inner_best_scores = []

        if "subtype" in X.columns:
            strat = X["subtype"]
            X = X.drop(columns=["subtype"])
        else:
            raise ValueError("Expected 'subtype' column in features for stratification during CV. Please ensure it is included in the input data.")

        # Nested CV Setup: inner tuning inside each outer training split
        for train_idx, test_idx in tqdm(cv_struct["outer_cv"].split(X, y = strat),desc=f"Running nested CV for {name}"):
            # Safely index data for current fold
            X_train = _safe_index(X, train_idx)
            y_train = _safe_index(y, train_idx)
            X_test = _safe_index(X, test_idx)
            y_test = _safe_index(y, test_idx)
            #groups_train = _safe_index(groups, train_idx) if groups is not None else print("Warning: No groups available for training split. Grouped CV may not work properly!")
            
            fold_searcher = make_searcher(model, param_grid)
            fold_searcher.fit(X_train, y_train)  

            outer_best_params.append(fold_searcher.best_params_)
            outer_inner_best_scores.append(float(fold_searcher.best_score_))

            proba = fold_searcher.predict_proba(X_test)[:, 1]
            oof_pred[test_idx] = proba
            outer_scores.append(roc_auc_score(y_test, proba))

        nested_cv_scores[name] = {
            "mean": float(np.nanmean(outer_scores)),
            "std": float(np.nanstd(outer_scores)),
            "outer_scores": [float(s) for s in outer_scores],
            "outer_best_params": outer_best_params,
            "mean_inner_best_score": float(np.nanmean(outer_inner_best_scores))
        }
        print(f"{name} Nested CV AUC: {nested_cv_scores[name]['mean']:.4f} ± {nested_cv_scores[name]['std']:.4f}")

    # Select winner by mean OUTER-CV AUC.
    best_model_name = max(nested_cv_scores, key=lambda k: nested_cv_scores[k]["mean"])

    print(
        f"\nBest model based on nested CV: {best_model_name} "
        f"with AUC: {nested_cv_scores[best_model_name]['mean']:.4f} "
        f"± {nested_cv_scores[best_model_name]['std']:.4f}"
    )

    print("\nModel training and evaluation completed.")

    return nested_cv_scores

def train_final_model(X, y, model):
    ''' Trains the final model on the full dataset using the best hyperparameters identified from nested CV.
        1. Performs a hyperparameter search (grid or random) on the entire dataset to find the best parameters.
        2. Fits the model with the best hyperparameters on the full dataset to create a deployable model.
        3. Returns the fully trained model and its best hyperparameters.
        Note: This function should only be called after identifying the best model type from nested CV to avoid data leakage.
    '''

    all_models = define_models()
    model_to_train, param_grid = all_models[model]
        
    grouped_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
    )

    searcher = GridSearchCV(
                    estimator=model_to_train,
                    param_grid=param_grid,
                    cv=grouped_cv,
                    scoring="roc_auc",
                    n_jobs=3,
                    verbose = 3
                )
    searcher.fit(X, y)
    best_model = searcher.best_estimator_
    best_hyperparams = searcher.best_params_
    train_predictions = best_model.predict_proba(X)[:, 1]

    print(f"Best hyperparameters for final model: {best_hyperparams}")
    print(f"Best inner-CV AUC during final training: {float(searcher.best_score_):.4f}")

    return best_model, best_hyperparams, train_predictions
