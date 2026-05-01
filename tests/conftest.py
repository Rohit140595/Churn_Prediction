"""Shared pytest fixtures used across all test modules.

The synthetic DataFrame produced here mirrors the Churn Modelling dataset
schema after load_raw_data() has already dropped the identifier columns
(RowNumber, CustomerId, Surname).  Tests that exercise the pipeline patch
load_raw_data() to return this fixture so CI never needs the real CSV file.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """50-row synthetic DataFrame matching the post-load churn schema."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "CreditScore":     rng.integers(300, 900, n),
        "Geography":       rng.choice(["France", "Germany", "Spain"], n),
        "Gender":          rng.choice(["Male", "Female"], n),
        "Age":             rng.integers(18, 70, n),
        "Tenure":          rng.integers(0, 10, n),
        "Balance":         rng.uniform(0, 200_000, n),
        "NumOfProducts":   rng.integers(1, 4, n),
        "HasCrCard":       rng.integers(0, 2, n),
        "IsActiveMember":  rng.integers(0, 2, n),
        "EstimatedSalary": rng.uniform(10_000, 200_000, n),
        "Exited":          rng.integers(0, 2, n),
    })
