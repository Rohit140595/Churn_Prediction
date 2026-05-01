"""Tests for src/serving/model_store.py."""

from unittest.mock import MagicMock

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from src.serving.model_store import build_artifact, load_artifact, save_artifact


@pytest.fixture
def mock_pipeline_result():
    """Pipeline result with mocked model wrapper — used for metadata-only tests."""
    return {
        "model":        MagicMock(),
        "preprocessor": MagicMock(),
        "selector":     None,
        "metrics":      {"accuracy": 0.87, "roc_auc": 0.86, "f1": 0.60,
                         "precision": 0.80, "recall": 0.49},
    }


@pytest.fixture
def serializable_pipeline_result():
    """Pipeline result with real sklearn objects that joblib can pickle."""
    return {
        "model":        DummyClassifier(),
        "preprocessor": StandardScaler(),
        "selector":     None,
        "metrics":      {"accuracy": 0.87, "roc_auc": 0.86, "f1": 0.60,
                         "precision": 0.80, "recall": 0.49},
    }


class TestBuildArtifact:
    def test_required_keys_present(self, mock_pipeline_result):
        artifact = build_artifact(mock_pipeline_result, model_name="xgboost")
        for key in ("model_name", "version", "trained_at", "model",
                    "preprocessor", "selector", "test_metrics"):
            assert key in artifact

    def test_model_name_stored_correctly(self, mock_pipeline_result):
        artifact = build_artifact(mock_pipeline_result, model_name="random_forest")
        assert artifact["model_name"] == "random_forest"

    def test_version_stored_correctly(self, mock_pipeline_result):
        artifact = build_artifact(mock_pipeline_result, model_name="xgboost", version="1.0.0")
        assert artifact["version"] == "1.0.0"

    def test_trained_at_is_utc_iso_string(self, mock_pipeline_result):
        artifact = build_artifact(mock_pipeline_result, model_name="xgboost")
        # ISO-8601 UTC timestamps end with "+00:00"
        assert "+00:00" in artifact["trained_at"]


class TestSaveAndLoadArtifact:
    def test_save_creates_file(self, serializable_pipeline_result, tmp_path):
        artifact = build_artifact(serializable_pipeline_result, model_name="xgboost")
        dest = tmp_path / "model.joblib"
        save_artifact(artifact, path=dest)
        assert dest.exists()

    def test_load_returns_artifact(self, serializable_pipeline_result, tmp_path):
        artifact = build_artifact(serializable_pipeline_result, model_name="xgboost")
        dest = tmp_path / "model.joblib"
        save_artifact(artifact, path=dest)
        loaded = load_artifact(path=dest)
        assert loaded["model_name"] == "xgboost"

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifact(path=tmp_path / "nonexistent.joblib")

    def test_save_creates_parent_directory(self, serializable_pipeline_result, tmp_path):
        artifact = build_artifact(serializable_pipeline_result, model_name="xgboost")
        nested_path = tmp_path / "nested" / "dir" / "model.joblib"
        save_artifact(artifact, path=nested_path)
        assert nested_path.exists()
