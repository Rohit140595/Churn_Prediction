"""Tests for src/pipeline.py.

load_raw_data() is patched in all tests so the real CSV file is never
required — CI can run these without any data files present.
"""

from unittest.mock import patch

from src.pipeline import run_pipeline


class TestRunPipeline:
    def test_returns_expected_keys(self, sample_df):
        with patch("src.pipeline.load_raw_data", return_value=sample_df):
            result = run_pipeline(model_name="logistic", track=False)

        expected_keys = {
            "metrics", "model", "preprocessor", "selector",
            "n_features_selected", "best_params", "best_cv_score",
            "roi_default", "roi_optimal", "run_id", "artifact_path",
        }
        assert expected_keys.issubset(result.keys())

    def test_metrics_contain_required_fields(self, sample_df):
        with patch("src.pipeline.load_raw_data", return_value=sample_df):
            result = run_pipeline(model_name="logistic", track=False)

        for metric in ("accuracy", "roc_auc", "f1", "precision", "recall"):
            assert metric in result["metrics"]
            assert 0.0 <= result["metrics"][metric] <= 1.0

    def test_no_tuning_returns_none_cv_score(self, sample_df):
        with patch("src.pipeline.load_raw_data", return_value=sample_df):
            result = run_pipeline(model_name="logistic", tune=False, track=False)

        assert result["best_cv_score"] is None

    def test_no_tracking_returns_none_run_id(self, sample_df):
        with patch("src.pipeline.load_raw_data", return_value=sample_df):
            result = run_pipeline(model_name="logistic", track=False)

        assert result["run_id"] is None

    def test_feature_selection_importance(self, sample_df):
        with patch("src.pipeline.load_raw_data", return_value=sample_df):
            result = run_pipeline(
                model_name="logistic",
                feature_selection="importance",
                track=False,
            )

        assert result["selector"] is not None
        assert result["n_features_selected"] is not None
        assert result["n_features_selected"] > 0

    def test_feature_selection_kbest(self, sample_df):
        with patch("src.pipeline.load_raw_data", return_value=sample_df):
            result = run_pipeline(
                model_name="logistic",
                feature_selection="kbest",
                feature_selection_k=5,
                track=False,
            )

        assert result["n_features_selected"] == 5

    def test_all_three_models_run(self, sample_df):
        for model_name in ("logistic", "random_forest", "xgboost"):
            with patch("src.pipeline.load_raw_data", return_value=sample_df):
                result = run_pipeline(model_name=model_name, track=False)
            assert result["metrics"]["roc_auc"] > 0.0

    def test_save_model_creates_artifact(self, sample_df, tmp_path):
        artifact_path = tmp_path / "churn_model.joblib"
        # Patch in model_store where MODEL_ARTIFACT_PATH is actually used
        with patch("src.pipeline.load_raw_data", return_value=sample_df), \
             patch("src.serving.model_store.MODEL_ARTIFACT_PATH", artifact_path):
            result = run_pipeline(model_name="logistic", track=False, save_model=True)

        assert artifact_path.exists()
        assert result["artifact_path"] is not None

    def test_roi_default_and_optimal_present(self, sample_df):
        with patch("src.pipeline.load_raw_data", return_value=sample_df):
            result = run_pipeline(model_name="logistic", track=False)

        assert "campaign_roi_pct" in result["roi_default"]
        assert "campaign_roi_pct" in result["roi_optimal"]
        assert "threshold" in result["roi_optimal"]
