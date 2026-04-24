from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.config import DEFAULT_PARAMS, RANDOM_STATE
from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest classifier conforming to the BaseModel interface.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of each tree.
    min_samples_leaf : int, optional
        Minimum number of samples required at a leaf node.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_PARAMS["random_forest"]["n_estimators"],
        max_depth: int = DEFAULT_PARAMS["random_forest"]["max_depth"],
        min_samples_leaf: int = DEFAULT_PARAMS["random_forest"]["min_samples_leaf"],
        random_state: int = RANDOM_STATE,
    ) -> None:
        # Store params for get_params() and so the tuner/tracker can inspect them via _params.
        self._params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
        }
        self._model = RandomForestClassifier(**self._params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Fit the random forest.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Binary target labels.

        Returns
        -------
        RandomForestModel
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
