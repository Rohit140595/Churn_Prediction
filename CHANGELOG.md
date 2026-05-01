# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — v0.5.0

> Branch: `feat/ci-cd` | PRs: #5 (pending)

### Added

- **CI workflow** (`.github/workflows/ci.yml`) — runs on every push and PR to `main`:
  - `lint` — `ruff` checks `src/`, `tests/`, `scripts/`
  - `type-check` — `mypy` validates type hints across `src/`
  - `test` — `pytest` unit tests with coverage report; `load_raw_data` is patched so no CSV is needed in CI
  - `docker` — confirms the Docker image builds cleanly on every PR
- **CD workflow** (`.github/workflows/cd.yml`) — runs on merge to `main`:
  - Builds the Docker image and tags it with the git SHA and `latest`
  - Pushes to GitHub Container Registry (`ghcr.io/rohit140595/churn-prediction`)
  - Uses layer caching from the previous push to speed up builds
- **Unit test suite** (`tests/`) — 30+ tests across 5 modules:
  - `test_schema.py` — Pydantic field validation (valid inputs, boundary violations, invalid enums)
  - `test_engineer.py` — feature engineering correctness, clip guard, input immutability
  - `test_selector.py` — selector type, k parameter, unfitted state, invalid strategy
  - `test_model_store.py` — artifact keys, save/load roundtrip, missing file error, nested directory creation
  - `test_pipeline.py` — end-to-end pipeline with mocked data (all three models, feature selection, save_model)
- **`requirements-dev.txt`** — `pytest`, `pytest-cov`, `ruff`, `mypy`, `pandas-stubs`
- **`pyproject.toml`** additions — `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]` config sections

### Changed

- `.gitignore` — added `models_output/*.joblib` so the artifact is excluded but the directory is tracked (required for the Docker build context)
- `pyproject.toml` — added `[tool.ruff]`, `[tool.mypy]`, and `[tool.pytest.ini_options]` config sections
- `README.md` — added Development section with test, lint, and CI/CD instructions; updated project structure tree with `tests/`, `.github/workflows/`, and `requirements-dev.txt`

---

## [Unreleased] — v0.4.0

> Branch: `feat/model-packaging` | PRs: #4 (pending)

### Added

- **Pydantic request/response schemas** (`src/serving/schema.py`) — `CustomerFeatures` validates all 10 raw input fields with type checks and value-range constraints; `PredictionResponse`, `ModelInfo`, and `HealthResponse` define the API output contracts.
- **Inference artifact store** (`src/serving/model_store.py`) — `build_artifact()` bundles preprocessor + selector + model + metadata into a single dict; `save_artifact()` / `load_artifact()` persist it as a `joblib` file under `models_output/churn_model.joblib`. Bundling all components prevents version skew between retraining runs.
- **FastAPI inference server** (`src/serving/app.py`) — three endpoints:
  - `GET /health` — liveness check; safe to use as a container health probe.
  - `GET /model-info` — returns model name, version, training timestamp, feature list, threshold, and test metrics.
  - `POST /predict` — accepts a validated customer record, runs feature engineering → preprocessing → (selection) → inference, returns `churn_probability` and `will_churn`.
- **`--save-model` CLI flag** (`scripts/train.py`) — serialises the artifact after training; prints the artifact path and server start command.
- **`pyproject.toml`** — makes the `src` package installable as `churn-prediction`; eliminates the `sys.path` hack needed when running outside the project root.
- **`Dockerfile`** — Python 3.11-slim base; dependencies installed before source copy to maximise layer cache hits; model artifact mounted as a volume at runtime (not baked in).
- **`.dockerignore`** — excludes `data/`, `models_output/`, `mlruns/`, `venv/`, and analysis files from the build context.

### Changed

- `src/pipeline.py` — added `save_model: bool = False` parameter and Step 10 that calls `build_artifact` + `save_artifact`; return dict now includes `artifact_path`.
- `src/config.py` — added `MODEL_ARTIFACT_PATH = MODELS_OUTPUT_DIR / "churn_model.joblib"`.
- `requirements.txt` — added `fastapi==0.110.3`, `uvicorn[standard]==0.29.0`, `pydantic==2.7.1`, `joblib==1.4.0`.

### Security note

The server speaks plain HTTP. It must be deployed behind an HTTPS-terminating reverse proxy or load balancer. See `Dockerfile` and `README` for details.

---

## [Unreleased] — v0.3.0

> Branch: `feat/feature-selection` | PRs: #3 (pending)

### Added

- **Feature selection** (`src/features/selector.py`) — `build_selector()` with two strategies:
  - `importance`: `SelectFromModel` (RandomForest, threshold=mean) — data-driven feature count
  - `kbest`: `SelectKBest` (ANOVA F-score) — fixed top-k features
- **Optuna hyperparameter tuning** (`src/tuning/tuner.py`) — TPE sampler with 3-fold stratified CV; preprocessor rebuilt per fold to prevent leakage. Search spaces for all three models.
- **Campaign ROI business metric** (`src/evaluation/business_metrics.py`):
  - `compute_campaign_roi()` — net value vs outreach cost at a fixed threshold
  - `find_optimal_threshold()` — sweeps 100 thresholds (0.05–0.95) to maximise ROI
  - `plot_roi_curve()` — ROI vs threshold with vertical line at the optimum
- **`--tune`**, **`--n-trials`**, **`--feature-selection`**, **`--feature-selection-k`** CLI flags added to `scripts/train.py`
- **Inline comments** throughout all source files — WHY-focused constraints (leakage prevention, division-by-zero guards, XGBoost 2.0 compatibility) and readability comments (section headers, formula breakdowns, inline annotations)

### Changed

- `src/data/preprocessor.py` — replaced the combined `split_and_preprocess()` wrapper with separate `split_data()` and `preprocess()` functions; tuner now receives raw DataFrames without triggering a second split
- `src/config.py` — added `colsample_bytree=0.8` to `DEFAULT_PARAMS["xgboost"]`; added business assumption constants (`REVENUE_PER_RETAINED_CUSTOMER`, `COST_PER_OUTREACH`, `RETENTION_SUCCESS_RATE`); added section headers
- `src/models/xgboost_model.py` — added `colsample_bytree` constructor parameter; set `eval_metric="logloss"` explicitly to silence XGBoost 2.0 deprecation warning
- `README.md` — added feature selection and tuning usage, hyperparameter search space table, updated sample output with tuning results, added Campaign ROI section with formula

### Removed

- `cross_validate_model()` from `src/evaluation/metrics.py` — CV now lives inside Optuna trials; standalone reporting was redundant
- `AgeGroup` engineered feature from `src/features/engineer.py` — redundant with the continuous `Age` column

---

## [0.2.0] — 2026-04-24

> PR #2: [docs: update README with project structure and usage](https://github.com/Rohit140595/Churn_Prediction/pull/2)

### Added

- `README.md` — full project documentation: directory tree, setup instructions, training commands, baseline model results table, MLflow usage, and pipeline API example

---

## [0.1.0] — 2026-04-24

> PR #1: [feat: add modular ML pipeline for churn prediction](https://github.com/Rohit140595/Churn_Prediction/pull/1)

### Added

- **Project scaffold** — `.gitignore` (excludes `data/`, `venv/`, `__pycache__`, MLflow artifacts, notebook checkpoints) and `requirements.txt` (pinned dependencies)
- **EDA notebook** (`analysis/EDA.ipynb`) — churn rate summary, correlation heatmap, univariate distribution plots, and multi-feature comparison plots
- **`src/config.py`** — single source of truth for paths, feature lists, random state, test size, and default hyperparameters
- **Data layer** (`src/data/`) — `load_raw_data()` drops identifier columns (`RowNumber`, `CustomerId`, `Surname`); `split_and_preprocess()` does stratified 80/20 split with `StandardScaler` + `OneHotEncoder` fit on training folds only
- **Feature engineering** (`src/features/engineer.py`) — `BalancePerProduct` (balance ÷ product count, floor at 1) and `ActiveProducts` (IsActiveMember × NumOfProducts)
- **Model wrappers** (`src/models/`) — `BaseModel` ABC enforcing `fit`, `predict`, `predict_proba`, `get_params`; concrete implementations for `LogisticRegressionModel`, `RandomForestModel`, `XGBoostModel`; `get_model()` registry factory
- **Evaluation** (`src/evaluation/metrics.py`) — `compute_metrics()` (accuracy, ROC-AUC, F1, precision, recall), `cross_validate_model()` (5-fold stratified CV with preprocessing per fold), `plot_roc_curve()`, `plot_confusion_matrix()`
- **MLflow tracking** (`src/tracking/mlflow_tracker.py`) — `log_experiment()` logs params, metrics, and model artifact per run
- **End-to-end pipeline** (`src/pipeline.py`) — `run_pipeline()` orchestrates all steps and returns a results dict
- **CLI** (`scripts/train.py`) — `--model`, `--experiment-name`, `--no-tracking` flags

### Baseline results (XGBoost, default hyperparameters)

| | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| CV (5-fold, mean ± std) | 0.8645 ± 0.0045 | 0.8663 ± 0.0048 | 0.5939 ± 0.0180 |
| Test set | 0.8705 | 0.8649 | 0.6058 |

---

## [0.0.0] — Initial commit

- Repository initialised with empty project structure
