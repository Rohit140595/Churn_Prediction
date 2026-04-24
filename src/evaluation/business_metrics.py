from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    COST_PER_OUTREACH,
    RETENTION_SUCCESS_RATE,
    REVENUE_PER_RETAINED_CUSTOMER,
)


def compute_campaign_roi(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    revenue_per_retained: float = REVENUE_PER_RETAINED_CUSTOMER,
    cost_per_outreach: float = COST_PER_OUTREACH,
    retention_success_rate: float = RETENTION_SUCCESS_RATE,
) -> dict[str, float]:
    """Compute campaign ROI at a fixed classification threshold.

    Assumes the bank contacts every predicted churner with a retention
    intervention. The net value is the revenue saved from true positives
    that are successfully retained, minus the outreach cost for all
    contacted customers (true and false positives).

    Formula::

        net_value      = TP × retention_success_rate × revenue_per_retained
        outreach_cost  = (TP + FP) × cost_per_outreach
        campaign_roi   = (net_value - outreach_cost) / outreach_cost × 100

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth binary labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities. If 2-D, column 1 is used.
    threshold : float, optional
        Decision threshold for positive class, by default 0.5.
    revenue_per_retained : float, optional
        Estimated revenue saved per successfully retained customer.
    cost_per_outreach : float, optional
        Cost of one retention intervention.
    retention_success_rate : float, optional
        Probability that an intervention retains an at-risk customer.

    Returns
    -------
    dict[str, float]
        Keys: ``threshold``, ``tp``, ``fp``, ``net_value``,
        ``outreach_cost``, ``campaign_roi_pct``.
    """
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    y_pred = (y_proba >= threshold).astype(int)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())

    net_value = tp * retention_success_rate * revenue_per_retained
    outreach_cost = (tp + fp) * cost_per_outreach
    campaign_roi_pct = (
        (net_value - outreach_cost) / outreach_cost * 100
        if outreach_cost > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "net_value": net_value,
        "outreach_cost": outreach_cost,
        "campaign_roi_pct": campaign_roi_pct,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    revenue_per_retained: float = REVENUE_PER_RETAINED_CUSTOMER,
    cost_per_outreach: float = COST_PER_OUTREACH,
    retention_success_rate: float = RETENTION_SUCCESS_RATE,
) -> Tuple[float, dict[str, float]]:
    """Find the classification threshold that maximises campaign ROI.

    Sweeps a range of thresholds and returns the one with the highest
    ``campaign_roi_pct``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth binary labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities. If 2-D, column 1 is used.
    thresholds : np.ndarray, optional
        Candidate thresholds to evaluate. Defaults to 100 values in (0.05, 0.95].
    revenue_per_retained : float, optional
        Estimated revenue saved per successfully retained customer.
    cost_per_outreach : float, optional
        Cost of one retention intervention.
    retention_success_rate : float, optional
        Probability that an intervention retains an at-risk customer.

    Returns
    -------
    Tuple[float, dict[str, float]]
        ``(optimal_threshold, roi_dict)`` where ``roi_dict`` is the output of
        :func:`compute_campaign_roi` at the optimal threshold.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 100)

    best_threshold = 0.5
    best_roi = -np.inf

    for t in thresholds:
        result = compute_campaign_roi(
            y_true, y_proba, threshold=t,
            revenue_per_retained=revenue_per_retained,
            cost_per_outreach=cost_per_outreach,
            retention_success_rate=retention_success_rate,
        )
        if result["campaign_roi_pct"] > best_roi:
            best_roi = result["campaign_roi_pct"]
            best_threshold = t

    return best_threshold, compute_campaign_roi(
        y_true, y_proba, threshold=best_threshold,
        revenue_per_retained=revenue_per_retained,
        cost_per_outreach=cost_per_outreach,
        retention_success_rate=retention_success_rate,
    )


def plot_roi_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    revenue_per_retained: float = REVENUE_PER_RETAINED_CUSTOMER,
    cost_per_outreach: float = COST_PER_OUTREACH,
    retention_success_rate: float = RETENTION_SUCCESS_RATE,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """Plot campaign ROI (%) across classification thresholds.

    Draws a vertical line at the ROI-maximising threshold.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth binary labels.
    y_proba : np.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probabilities. If 2-D, column 1 is used.
    thresholds : np.ndarray, optional
        Candidate thresholds to evaluate. Defaults to 100 values in (0.05, 0.95].
    revenue_per_retained : float, optional
        Estimated revenue saved per successfully retained customer.
    cost_per_outreach : float, optional
        Cost of one retention intervention.
    retention_success_rate : float, optional
        Probability that an intervention retains an at-risk customer.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the ROI curve.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 100)

    roi_values = [
        compute_campaign_roi(
            y_true, y_proba, threshold=t,
            revenue_per_retained=revenue_per_retained,
            cost_per_outreach=cost_per_outreach,
            retention_success_rate=retention_success_rate,
        )["campaign_roi_pct"]
        for t in thresholds
    ]

    optimal_threshold, _ = find_optimal_threshold(
        y_true, y_proba, thresholds,
        revenue_per_retained, cost_per_outreach, retention_success_rate,
    )

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(thresholds, roi_values, color="steelblue", linewidth=2)
    ax.axvline(optimal_threshold, color="crimson", linestyle="--",
               label=f"Optimal threshold = {optimal_threshold:.2f}")
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Campaign ROI (%)")
    ax.set_title("Campaign ROI vs Classification Threshold")
    ax.legend()
    return ax
