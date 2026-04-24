from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all churn prediction models.

    All concrete models must implement ``fit``, ``predict``,
    ``predict_proba``, and ``get_params``. The underlying sklearn-compatible
    estimator must be stored as ``self._model`` so the tracker can log it.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the model to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        BaseModel
            Self, to allow method chaining.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Probabilities for [class 0, class 1].
        """

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters.

        Returns
        -------
        dict[str, Any]
            Hyperparameter name-value pairs used to construct this model.
        """
