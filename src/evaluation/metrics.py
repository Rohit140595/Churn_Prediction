from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate as sk_cross_validate
from sklearn.pipeline import Pipeline as SKPipeline

from src.config import RANDOM_STATE


def cross_validate_model(
    X: pd.DataFrame,
    y: np.ndarray,
    estimator: Any,
    numerical_features: list[str],
    categorical_features: list[str],
    cv: int = 5,
) -> dict[str, np.ndarray]:
    """Run stratified k-fold cross-validation with preprocessing inside each fold.

    Wraps the preprocessor and estimator in an sklearn ``Pipeline`` so that
    scaling and encoding are re-fit on each training fold, preventing leakage.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        Full feature DataFrame (before any train/test split).
    y : np.ndarray of shape (n_samples,)
        Binary target labels.
    estimator : Any
        Unfitted sklearn-compatible estimator.
    numerical_features : list[str]
        Columns to pass through ``StandardScaler``.
    categorical_features : list[str]
        Columns to pass through ``OneHotEncoder``.
    cv : int, optional
        Number of stratified folds, by default 5.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``accuracy``, ``roc_auc``, ``f1``, ``precision``, ``recall``.
        Each value is an array of per-fold scores.
    """
    from src.data.preprocessor import build_preprocessor

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    pipe = SKPipeline([("preprocessor", preprocessor), ("model", estimator)])
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = sk_cross_validate(
        pipe,
        X,
        y,
        cv=cv_strategy,
        scoring=["accuracy", "roc_auc", "f1", "precision", "recall"],
    )
    return {
        metric: scores[f"test_{metric}"]
        for metric in ["accuracy", "roc_auc", "f1", "precision", "recall"]
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth binary labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities. If 2-D, column 1 is used as the positive-class score.

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``roc_auc``, ``f1``, ``precision``, ``recall``.
    """
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Plot the ROC curve for a binary classifier.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth binary labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities. If 2-D, column 1 is used.
    model_name : str, optional
        Legend label for the curve.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the ROC curve.
    """
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    if ax is None:
        _, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_proba, name=model_name, ax=ax)
    ax.set_title("ROC Curve")
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Plot a confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth binary labels.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary labels.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the confusion matrix.
    """
    if ax is None:
        _, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title("Confusion Matrix")
    return ax
