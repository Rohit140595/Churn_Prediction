from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build an unfitted sklearn ColumnTransformer.

    Scales numerical features with ``StandardScaler`` and encodes categorical
    features with ``OneHotEncoder``.

    Parameters
    ----------
    numerical_features : list[str]
        Column names to pass through ``StandardScaler``.
    categorical_features : list[str]
        Column names to pass through ``OneHotEncoder``.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer ready for ``fit_transform`` / ``transform``.
    """
    numerical_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",  # silently ignore unseen categories at inference
                sparse_output=False,      # return dense array; compatible with numpy downstream
            ),
        )
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified train/test split returning raw DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame including the target column.
    target_col : str, optional
        Name of the binary target column, by default ``TARGET_COL``.
    test_size : float, optional
        Fraction of data reserved for testing, by default ``TEST_SIZE``.
    random_state : int, optional
        Random seed for reproducibility, by default ``RANDOM_STATE``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]
        ``X_train_df``, ``X_test_df``, ``y_train``, ``y_test``.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    # stratify=y preserves the class imbalance ratio in both train and test splits.
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def preprocess(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """Fit a preprocessor on the training set and transform both splits.

    The preprocessor is fit exclusively on ``X_train_df`` to prevent leakage.

    Parameters
    ----------
    X_train_df : pd.DataFrame
        Raw training features.
    X_test_df : pd.DataFrame
        Raw test features.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, ColumnTransformer]
        ``X_train_arr``, ``X_test_arr``, fitted preprocessor.
    """
    # Config includes engineered feature names; filter to columns that actually exist
    # in case this is called before add_features() or for a different dataset.
    present_num = [c for c in NUMERICAL_FEATURES if c in X_train_df.columns]
    present_cat = [c for c in CATEGORICAL_FEATURES if c in X_train_df.columns]
    preprocessor = build_preprocessor(present_num, present_cat)
    # fit_transform on train so scaling/encoding statistics come only from training data.
    X_train_arr = preprocessor.fit_transform(X_train_df)
    # transform (not fit_transform) applies train-derived statistics to test — no leakage.
    X_test_arr = preprocessor.transform(X_test_df)
    return X_train_arr, X_test_arr, preprocessor
