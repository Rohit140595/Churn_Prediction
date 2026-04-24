from typing import Any

import mlflow
import mlflow.sklearn


def log_experiment(
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    estimator: Any,
    experiment_name: str = "churn_prediction",
) -> str:
    """Log a training run — params, metrics, and model artifact — to MLflow.

    Parameters
    ----------
    model_name : str
        Short identifier used as the MLflow run name (e.g. ``"xgboost"``).
    params : dict[str, Any]
        Hyperparameters to log.
    metrics : dict[str, float]
        Evaluation metrics to log.
    estimator : Any
        Fitted sklearn-compatible estimator to persist as an artifact.
    experiment_name : str, optional
        MLflow experiment name, by default ``"churn_prediction"``.

    Returns
    -------
    str
        The MLflow run ID.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(estimator, artifact_path="model")
        return run.info.run_id
