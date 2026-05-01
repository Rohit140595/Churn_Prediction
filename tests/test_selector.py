"""Tests for src/features/selector.py."""

import pytest
from sklearn.feature_selection import SelectFromModel, SelectKBest

from src.features.selector import build_selector


class TestBuildSelector:
    def test_importance_strategy_returns_select_from_model(self):
        selector = build_selector("importance")
        assert isinstance(selector, SelectFromModel)

    def test_kbest_strategy_returns_select_k_best(self):
        selector = build_selector("kbest", k=5)
        assert isinstance(selector, SelectKBest)

    def test_kbest_k_parameter_is_set(self):
        selector = build_selector("kbest", k=7)
        assert selector.k == 7

    def test_selector_is_unfitted(self):
        # get_support() should raise if the selector hasn't been fitted yet
        selector = build_selector("kbest", k=5)
        with pytest.raises(Exception):
            selector.get_support()

    def test_invalid_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_selector("invalid_strategy")
