from typing import Any, Optional

from src.config import DEFAULT_PARAMS
from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess, split_data
from src.evaluation.business_metrics import compute_campaign_roi, find_optimal_threshold
from src.evaluation.metrics import compute_metrics
from src.features.engineer import add_features
from src.models.base import BaseModel
from src.models.registry import get_model
from src.tracking.mlflow_tracker import log_experiment


def run_pipeline(
    model_name: str,
    params: Optional[dict[str, Any]] = None,
    feature_selection: Optional[str] = None,
    feature_selection_k: int = 10,
    tune: bool = False,
    n_trials: int = 50,
    experiment_name: str = "churn_prediction",
    track: bool = True,
) -> dict[str, Any]:
    """Run the end-to-end training and evaluation pipeline.

    Steps: load → engineer → split → (tune) → preprocess →
    (feature selection) → fit → evaluate → ROI → (track).

    Parameters
    ----------
    model_name : str
        Registry key for the model to train. One of ``"logistic"``,
        ``"random_forest"``, ``"xgboost"``.
    params : dict[str, Any], optional
        Explicit hyperparameter overrides. Ignored when ``tune=True``.
    feature_selection : str, optional
        One of ``"importance"`` or ``"kbest"``. No selection when ``None``.
    feature_selection_k : int, optional
        Features to keep when ``feature_selection="kbest"``, by default 10.
    tune : bool, optional
        Run Optuna hyperparameter search before fitting, by default ``False``.
    n_trials : int, optional
        Number of Optuna trials when ``tune=True``, by default 50.
    experiment_name : str, optional
        MLflow experiment name, by default ``"churn_prediction"``.
    track : bool, optional
        Whether to log the run to MLflow, by default ``True``.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``metrics`` : dict[str, float] — test-set evaluation metrics.
        - ``model`` : BaseModel — fitted model wrapper.
        - ``preprocessor`` : ColumnTransformer — fitted preprocessor.
        - ``selector`` : sklearn transformer or None — fitted feature selector.
        - ``n_features_selected`` : int or None — features kept after selection.
        - ``best_params`` : dict[str, Any] — hyperparameters used to train the model.
        - ``best_cv_score`` : float or None — best ROC-AUC from tuning (``None`` if ``tune=False``).
        - ``roi_default`` : dict[str, float] — campaign ROI at threshold=0.5.
        - ``roi_optimal`` : dict[str, float] — campaign ROI at the ROI-maximising threshold.
        - ``run_id`` : str or None — MLflow run ID (``None`` if ``track=False``).
    """
    from src.features.selector import build_selector

    # --- Step 1: Load and engineer features ---
    df = load_raw_data()
    df = add_features(df)

    # --- Step 2: Train/test split ---
    # Split before tuning so the test set is invisible to the HP search entirely.
    X_train_df, X_test_df, y_train, y_test = split_data(df)

    # --- Step 3: Hyperparameter search (optional) ---
    if tune:
        from src.tuning.tuner import tune as run_tuning
        # Pass the raw DataFrame (not arrays) — the tuner builds its own preprocessor
        # inside each CV fold so it can re-fit on fold training data without leakage.
        best_params, best_cv_score = run_tuning(model_name, X_train_df, y_train, n_trials=n_trials)
    else:
        # Merge defaults with any caller-supplied overrides
        best_params = {**DEFAULT_PARAMS.get(model_name, {}), **(params or {})}
        best_cv_score = None

    # --- Step 4: Preprocessing ---
    # Fit the final preprocessor on the full training set (not on CV folds) so the
    # model benefits from all available training statistics at inference time.
    X_train, X_test, preprocessor = preprocess(X_train_df, X_test_df)

    # --- Step 5: Feature selection (optional) ---
    selector = (
        build_selector(feature_selection, k=feature_selection_k)
        if feature_selection
        else None
    )
    if selector is not None:
        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

    # --- Step 6: Train final model ---
    model: BaseModel = get_model(model_name, **best_params)
    model.fit(X_train, y_train)

    # --- Step 7: Evaluate on test set ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    n_features_selected: Optional[int] = (
        int(selector.get_support().sum()) if selector is not None else None
    )

    # --- Step 8: Business metrics ---
    roi_default = compute_campaign_roi(y_test, y_proba, threshold=0.5)
    optimal_threshold, roi_optimal = find_optimal_threshold(y_test, y_proba)

    # --- Step 9: MLflow tracking (optional) ---
    if track:
        # Bundle all run parameters, including tuning/selection metadata
        track_params = {**best_params}
        if tune:
            track_params["n_trials"] = n_trials
        if feature_selection:
            track_params["feature_selection"] = feature_selection
            track_params["n_features_selected"] = n_features_selected
        roi_metrics = {
            "roi_default_pct": roi_default["campaign_roi_pct"],
            "roi_optimal_pct": roi_optimal["campaign_roi_pct"],
            "optimal_threshold": optimal_threshold,
        }
        if best_cv_score is not None:
            roi_metrics["best_cv_roc_auc"] = best_cv_score
        run_id = log_experiment(
            model_name, track_params, {**metrics, **roi_metrics}, model._model, experiment_name
        )
    else:
        run_id = None

    return {
        "metrics": metrics,
        "model": model,
        "preprocessor": preprocessor,
        "selector": selector,
        "n_features_selected": n_features_selected,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "roi_default": roi_default,
        "roi_optimal": roi_optimal,
        "run_id": run_id,
    }
