from typing import Any, Optional

from src.config import CATEGORICAL_FEATURES, DEFAULT_PARAMS, NUMERICAL_FEATURES, TARGET_COL
from src.data.loader import load_raw_data
from src.data.preprocessor import split_and_preprocess
from src.evaluation.metrics import compute_metrics, cross_validate_model
from src.features.engineer import add_features
from src.models.base import BaseModel
from src.models.registry import get_model
from src.tracking.mlflow_tracker import log_experiment


def run_pipeline(
    model_name: str,
    params: Optional[dict[str, Any]] = None,
    experiment_name: str = "churn_prediction",
    track: bool = True,
) -> dict[str, Any]:
    """Run the end-to-end training and evaluation pipeline.

    Steps: load → feature engineering → split & preprocess → train → evaluate
    → (optionally) log to MLflow.

    Parameters
    ----------
    model_name : str
        Registry key for the model to train. One of ``"logistic"``,
        ``"random_forest"``, ``"xgboost"``.
    params : dict[str, Any], optional
        Hyperparameter overrides merged on top of ``DEFAULT_PARAMS[model_name]``.
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
        - ``run_id`` : str or None — MLflow run ID (``None`` if ``track=False``).
    """
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
    cv_results = cross_validate_model(
        X_full, y_full,
        get_model(model_name, **model_params)._model,
        present_num,
        present_cat,
    )

    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df)
    model: BaseModel = get_model(model_name, **model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    run_id: Optional[str] = None
    if track:
        run_id = log_experiment(
            model_name, model_params, metrics, model._model, experiment_name
        )

    return {
        "metrics": metrics,
        "cv_results": cv_results,
        "model": model,
        "preprocessor": preprocessor,
        "run_id": run_id,
    }
