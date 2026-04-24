from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import DEFAULT_PARAMS, RANDOM_STATE
from src.models.base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression baseline conforming to the BaseModel interface.

    Parameters
    ----------
    C : float, optional
        Inverse of regularization strength.
    max_iter : int, optional
        Maximum number of solver iterations.
    solver : str, optional
        Algorithm for the optimization problem.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        C: float = DEFAULT_PARAMS["logistic"]["C"],
        max_iter: int = DEFAULT_PARAMS["logistic"]["max_iter"],
        solver: str = DEFAULT_PARAMS["logistic"]["solver"],
        random_state: int = RANDOM_STATE,
    ) -> None:
        # Store params for get_params() and so the tuner/tracker can inspect them via _params.
        self._params: dict[str, Any] = {
            "C": C,
            "max_iter": max_iter,
            "solver": solver,
            "random_state": random_state,
        }
        self._model = LogisticRegression(**self._params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        """Fit the logistic regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Binary target labels.

        Returns
        -------
        LogisticRegressionModel
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
