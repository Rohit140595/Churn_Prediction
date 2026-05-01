"""Tests for src/features/engineer.py."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import add_features


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Balance":         [100_000.0, 0.0,    50_000.0],
        "NumOfProducts":   [2,          1,       3],
        "IsActiveMember":  [1,          0,       1],
    })


class TestAddFeatures:
    def test_balance_per_product_calculated_correctly(self, base_df):
        result = add_features(base_df)
        expected = [100_000 / 2, 0.0 / 1, 50_000 / 3]
        np.testing.assert_allclose(result["BalancePerProduct"], expected)

    def test_active_products_calculated_correctly(self, base_df):
        result = add_features(base_df)
        # IsActiveMember * NumOfProducts
        expected = [1 * 2, 0 * 1, 1 * 3]
        np.testing.assert_array_equal(result["ActiveProducts"], expected)

    def test_clip_prevents_division_by_zero(self):
        # NumOfProducts=0 should not raise; denominator is clipped to 1
        df = pd.DataFrame({
            "Balance":        [5_000.0],
            "NumOfProducts":  [0],
            "IsActiveMember": [1],
        })
        result = add_features(df)
        assert result["BalancePerProduct"].iloc[0] == 5_000.0

    def test_input_dataframe_not_mutated(self, base_df):
        original_cols = set(base_df.columns)
        add_features(base_df)
        # Caller's DataFrame should be unchanged
        assert set(base_df.columns) == original_cols

    def test_output_contains_new_columns(self, base_df):
        result = add_features(base_df)
        assert "BalancePerProduct" in result.columns
        assert "ActiveProducts" in result.columns
