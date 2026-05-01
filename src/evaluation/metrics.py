from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
        y_proba = y_proba[:, 1]  # column 1 is the positive-class (churn) probability
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
        y_proba = y_proba[:, 1]  # extract positive-class probabilities
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
