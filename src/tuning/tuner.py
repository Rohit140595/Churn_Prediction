from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline as SKPipeline

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
)

# Suppress Optuna's per-trial INFO logs; progress bar from optimize() is enough.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Hyperparameter search spaces per model.
# Each entry is a tuple of (kind, *args) where kind is one of:
#   "fixed"     — use the value as-is, not suggested by Optuna
#   "int"       — suggest_int(low, high)
#   "float"     — suggest_float(low, high)
#   "float_log" — suggest_float(low, high, log=True), useful for rates/regularization
SEARCH_SPACES: dict[str, dict] = {
    "logistic": {
        "C": ("float_log", 1e-3, 100.0),  # log scale covers small regularization values densely
        # "fixed" params are not searched but must be passed to satisfy the
        # model constructor; they ensure every required argument is always present.
        "max_iter": ("fixed", 1000),
        "solver": ("fixed", "lbfgs"),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 15),
        "min_samples_leaf": ("int", 1, 20),
    },
    "xgboost": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 10),
        "learning_rate": ("float_log", 1e-3, 0.3),  # log scale important: 0.001–0.01 matters as much as 0.1–0.3
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
    },
}


def _sample_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Sample hyperparameters for ``model_name`` from the Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial.
    model_name : str
        One of ``"logistic"``, ``"random_forest"``, ``"xgboost"``.

    Returns
    -------
    dict[str, Any]
        Sampled hyperparameter values.
    """
    params: dict[str, Any] = {}
    for name, spec in SEARCH_SPACES[model_name].items():
        kind = spec[0]
        if kind == "fixed":
            params[name] = spec[1]
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
    return params


def tune(
    model_name: str,
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    n_trials: int = 50,
    cv: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], float]:
    """Run an Optuna hyperparameter search using stratified k-fold CV.

    Each trial samples a set of hyperparameters, builds a preprocessing +
    model pipeline, and scores it via cross-validated ROC-AUC on the
    training data. The test set is never touched.

    Parameters
    ----------
    model_name : str
        One of ``"logistic"``, ``"random_forest"``, ``"xgboost"``.
    X_train_df : pd.DataFrame
        Raw (unpreprocessed) training features.
    y_train : np.ndarray of shape (n_samples,)
        Training labels.
    n_trials : int, optional
        Number of Optuna trials, by default 50.
    cv : int, optional
        Number of CV folds inside each trial, by default 3.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple[dict[str, Any], float]
        ``(best_params, best_cv_roc_auc)`` — the best hyperparameters found
        and their mean cross-validated ROC-AUC score.
    """
    from src.data.preprocessor import build_preprocessor
    from src.models.registry import get_model

    present_num = [c for c in NUMERICAL_FEATURES if c in X_train_df.columns]
    present_cat = [c for c in CATEGORICAL_FEATURES if c in X_train_df.columns]
    # 3-fold (default) rather than 5-fold: acceptable variance tradeoff given
    # n_trials can be large; keeps each trial ~40% faster.
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, model_name)
        # A fresh preprocessor is created per trial so each CV fold re-fits its
        # own scaler/encoder on the fold's training data — prevents leakage
        # that would occur if a single fitted preprocessor were shared across folds.
        preprocessor = build_preprocessor(present_num, present_cat)
        # Access ._model to get the raw sklearn estimator; cross_val_score
        # needs a plain sklearn-compatible object, not the BaseModel wrapper.
        estimator = get_model(model_name, **params)._model
        pipe = SKPipeline([("preprocessor", preprocessor), ("model", estimator)])
        scores = cross_val_score(
            pipe, X_train_df, y_train, cv=cv_strategy, scoring="roc_auc"
        )
        return float(scores.mean())

    # TPE (Tree-structured Parzen Estimator) is a Bayesian method that models
    # the distribution of good vs bad hyperparameter regions to guide sampling.
    study = optuna.create_study(
        direction="maximize",  # maximise ROC-AUC
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params, study.best_value
