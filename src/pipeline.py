from typing import Any, Optional

from src.config import CATEGORICAL_FEATURES, DEFAULT_PARAMS, NUMERICAL_FEATURES, TARGET_COL
from src.data.loader import load_raw_data
from src.data.preprocessor import split_and_preprocess
from src.evaluation.business_metrics import compute_campaign_roi, find_optimal_threshold
from src.evaluation.metrics import compute_metrics, cross_validate_model
from src.features.engineer import add_features
from src.models.base import BaseModel
from src.models.registry import get_model
from src.tracking.mlflow_tracker import log_experiment


def run_pipeline(
    model_name: str,
    params: Optional[dict[str, Any]] = None,
    feature_selection: Optional[str] = None,
    feature_selection_k: int = 10,
    experiment_name: str = "churn_prediction",
    track: bool = True,
) -> dict[str, Any]:
    """Run the end-to-end training and evaluation pipeline.

    Steps: load → feature engineering → split & preprocess →
    (optional feature selection) → train → evaluate → (optionally) log to MLflow.

    Parameters
    ----------
    model_name : str
        Registry key for the model to train. One of ``"logistic"``,
        ``"random_forest"``, ``"xgboost"``.
    params : dict[str, Any], optional
        Hyperparameter overrides merged on top of ``DEFAULT_PARAMS[model_name]``.
    feature_selection : str, optional
        Feature selection strategy. One of ``"importance"`` (SelectFromModel
        backed by RandomForest, threshold=mean) or ``"kbest"`` (ANOVA F-score).
        No selection is applied when ``None``.
    feature_selection_k : int, optional
        Number of features to keep when ``feature_selection="kbest"``, by default 10.
    experiment_name : str, optional
        MLflow experiment name, by default ``"churn_prediction"``.
    track : bool, optional
        Whether to log the run to MLflow, by default ``True``.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``metrics`` : dict[str, float] — evaluation metrics on the test set.
        - ``cv_results`` : dict[str, np.ndarray] — per-fold scores from 5-fold CV.
        - ``model`` : BaseModel — fitted model wrapper.
        - ``preprocessor`` : ColumnTransformer — fitted preprocessor.
        - ``selector`` : sklearn transformer or None — fitted feature selector.
        - ``n_features_selected`` : int or None — features kept after selection.
        - ``roi_default`` : dict[str, float] — campaign ROI at threshold=0.5.
        - ``roi_optimal`` : dict[str, float] — campaign ROI at the ROI-maximising threshold.
        - ``run_id`` : str or None — MLflow run ID (``None`` if ``track=False``).
    """
    from src.features.selector import build_selector

    df = load_raw_data()
    df = add_features(df)

    model_params: dict[str, Any] = {
        **DEFAULT_PARAMS.get(model_name, {}),
        **(params or {}),
    }

    X_full = df.drop(columns=[TARGET_COL])
    y_full = df[TARGET_COL].values
    present_num = [c for c in NUMERICAL_FEATURES if c in X_full.columns]
    present_cat = [c for c in CATEGORICAL_FEATURES if c in X_full.columns]

    selector = (
        build_selector(feature_selection, k=feature_selection_k)
        if feature_selection
        else None
    )
    cv_results = cross_validate_model(
        X_full, y_full,
        get_model(model_name, **model_params)._model,
        present_num,
        present_cat,
        selector=build_selector(feature_selection, k=feature_selection_k) if feature_selection else None,
    )

    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df)

    if selector is not None:
        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

    model: BaseModel = get_model(model_name, **model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    n_features_selected: Optional[int] = (
        int(selector.get_support().sum()) if selector is not None else None
    )

    roi_default = compute_campaign_roi(y_test, y_proba, threshold=0.5)
    optimal_threshold, roi_optimal = find_optimal_threshold(y_test, y_proba)

    if track:
        track_params = {**model_params}
        if feature_selection:
            track_params["feature_selection"] = feature_selection
            track_params["n_features_selected"] = n_features_selected
        roi_metrics = {
            "roi_default_pct": roi_default["campaign_roi_pct"],
            "roi_optimal_pct": roi_optimal["campaign_roi_pct"],
            "optimal_threshold": optimal_threshold,
        }
        run_id = log_experiment(
            model_name, track_params, {**metrics, **roi_metrics}, model._model, experiment_name
        )
    else:
        run_id = None

    return {
        "metrics": metrics,
        "cv_results": cv_results,
        "model": model,
        "preprocessor": preprocessor,
        "selector": selector,
        "n_features_selected": n_features_selected,
        "roi_default": roi_default,
        "roi_optimal": roi_optimal,
        "run_id": run_id,
    }
