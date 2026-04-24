from typing import Any

import numpy as np
from xgboost import XGBClassifier

from src.config import DEFAULT_PARAMS, RANDOM_STATE
from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost classifier conforming to the BaseModel interface.

    Parameters
    ----------
    n_estimators : int, optional
        Number of boosting rounds.
    max_depth : int, optional
        Maximum tree depth.
    learning_rate : float, optional
        Step size shrinkage used in each boosting step.
    subsample : float, optional
        Fraction of training samples used per tree.
    colsample_bytree : float, optional
        Fraction of features sampled per tree.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_PARAMS["xgboost"]["n_estimators"],
        max_depth: int = DEFAULT_PARAMS["xgboost"]["max_depth"],
        learning_rate: float = DEFAULT_PARAMS["xgboost"]["learning_rate"],
        subsample: float = DEFAULT_PARAMS["xgboost"]["subsample"],
        colsample_bytree: float = DEFAULT_PARAMS["xgboost"]["colsample_bytree"],
        random_state: int = RANDOM_STATE,
    ) -> None:
        # Store params for get_params() and so the tuner/tracker can inspect them via _params.
        self._params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
        }
        # eval_metric must be set explicitly — XGBoost 2.0 changed the default and
        # prints a noisy warning on every fit if it is not provided.
        self._model = XGBClassifier(**self._params, eval_metric="logloss")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """Fit the XGBoost model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Binary target labels.

        Returns
        -------
        XGBoostModel
            Self.
        """
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels.
        """
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Class probabilities.
        """
        return self._model.predict_proba(X)

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters.

        Returns
        -------
        dict[str, Any]
            Constructor keyword arguments.
        """
        return self._params
