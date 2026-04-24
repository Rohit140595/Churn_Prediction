from typing import Any

from src.models.base import BaseModel
from src.models.logistic import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

# Maps CLI/config model names to their concrete BaseModel classes.
# Add a new entry here to make a model available throughout the pipeline.
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "logistic": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}


def get_model(name: str, **kwargs: Any) -> BaseModel:
    """Instantiate a registered model by name.

    Parameters
    ----------
    name : str
        Key in ``MODEL_REGISTRY`` (e.g. ``"xgboost"``).
    **kwargs
        Hyperparameters forwarded to the model constructor.

    Returns
    -------
    BaseModel
        New, unfitted model instance.

    Raises
    ------
    ValueError
        If ``name`` is not a key in ``MODEL_REGISTRY``.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)
