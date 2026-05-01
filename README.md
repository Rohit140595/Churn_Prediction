# Churn Prediction

Binary classification project predicting whether a bank customer will churn, using the [Churn Modelling dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) (10,000 customers, 14 features).

## Project structure

```
├── analysis/
│   └── EDA.ipynb               exploratory data analysis
├── data/
│   └── Churn_Modelling.csv     raw dataset (gitignored)
├── models_output/
│   └── churn_model.joblib      serialised inference artifact (gitignored)
├── src/
│   ├── config.py               paths, constants, default hyperparameters
│   ├── data/
│   │   ├── loader.py           load_raw_data()
│   │   └── preprocessor.py     split_data(), preprocess() — leakage-free
│   ├── features/
│   │   ├── engineer.py         feature engineering (BalancePerProduct, ActiveProducts)
│   │   └── selector.py         build_selector() — importance or kbest strategies
│   ├── models/
│   │   ├── base.py             BaseModel ABC
│   │   ├── logistic.py         LogisticRegressionModel
│   │   ├── random_forest.py    RandomForestModel
│   │   ├── xgboost_model.py    XGBoostModel
│   │   └── registry.py         get_model(name) factory
│   ├── evaluation/
│   │   ├── metrics.py          compute_metrics(), plot helpers
│   │   └── business_metrics.py compute_campaign_roi(), find_optimal_threshold(), plot_roi_curve()
│   ├── tuning/
│   │   └── tuner.py            tune() — Optuna hyperparameter search (3-fold CV)
│   ├── tracking/
│   │   └── mlflow_tracker.py   MLflow experiment logging
│   ├── serving/
│   │   ├── app.py              FastAPI inference server (GET /health, GET /model-info, POST /predict)
│   │   ├── schema.py           Pydantic request / response schemas
│   │   └── model_store.py      build_artifact(), save_artifact(), load_artifact()
│   └── pipeline.py             run_pipeline() end-to-end orchestrator
├── scripts/
│   └── train.py                CLI entry point
├── Dockerfile                  container definition for the inference server
├── pyproject.toml              package metadata (makes src/ installable)
└── CHANGELOG.md                versioned change history
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: install as an editable package (removes sys.path dependency)
pip install -e .
```

## Training

```bash
# Default hyperparameters
python scripts/train.py --model xgboost

# With Optuna hyperparameter tuning
python scripts/train.py --model xgboost --tune --n-trials 50

# With feature selection
python scripts/train.py --model xgboost --feature-selection importance
python scripts/train.py --model xgboost --feature-selection kbest --feature-selection-k 8

# Disable MLflow tracking
python scripts/train.py --model xgboost --no-tracking

# Train and save the inference artifact for serving
python scripts/train.py --model xgboost --save-model
```

### Hyperparameter tuning

When `--tune` is set, Optuna runs a TPE search over the model's hyperparameter space using 3-fold stratified CV on the training split. The test set is never seen during tuning.

| Model | Search space |
|---|---|
| Logistic | `C` (log-uniform 1e-3–100) |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` |

### Feature selection strategies

| Strategy | Mechanism | Features kept |
|---|---|---|
| `importance` | `SelectFromModel` (RandomForest, threshold=mean) | Data-driven |
| `kbest` | `SelectKBest` (ANOVA F-score, top k) | Fixed via `--feature-selection-k` |

## Sample output

```
Model           : xgboost
Tuning          : Optuna (20 trials, 3-fold CV)

Best CV ROC-AUC : 0.8647
Best params     :
  n_estimators: 404
  max_depth: 4
  learning_rate: 0.0188
  subsample: 0.796
  colsample_bytree: 0.523

Test-set metrics:
  accuracy    : 0.8675
  roc_auc     : 0.8721
  f1          : 0.5904
  precision   : 0.7958
  recall      : 0.4693

Campaign ROI (assumptions: $50/outreach, $800 revenue/retained, 30% success rate):
  threshold=0.50 : +282.0%  (contacted 240, net $33,840)
  optimal=0.88   : +380.0%  (contacted 52, net $9,880)
```

## Serving

The trained model can be served as a REST API using the FastAPI inference server.

### 1. Train and save the artifact

```bash
python scripts/train.py --model xgboost --save-model --no-tracking
# → models_output/churn_model.joblib
```

### 2. Start the server

```bash
uvicorn src.serving.app:app --reload --port 8000
```

Interactive API docs are available at http://localhost:8000/docs.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — confirms service is up and model is loaded |
| `GET` | `/model-info` | Model name, version, training timestamp, feature list, test metrics |
| `POST` | `/predict` | Accepts a customer record, returns churn probability and hard label |

### Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000.0
  }'
```

```json
{
  "churn_probability": 0.083,
  "will_churn": false,
  "threshold": 0.5,
  "model_name": "xgboost",
  "model_version": "0.4.0"
}
```

### HTTPS

The server speaks plain HTTP. In production it **must** sit behind an HTTPS-terminating reverse proxy or load balancer (AWS ALB, GCP Load Balancer, nginx, Traefik). Never expose this service directly to the internet over plain HTTP.

## Docker

```bash
# Build the image
docker build -t churn-prediction:latest .

# Run — mount the model artifact as a read-only volume
docker run -p 8000:8000 \
  -v $(pwd)/models_output:/app/models_output:ro \
  churn-prediction:latest
```

The model artifact is mounted at runtime rather than baked into the image so the image does not need to be rebuilt on every retrain.

## MLflow

```bash
mlflow ui
# open http://localhost:5000
```

## Using the pipeline in code

```python
from src.pipeline import run_pipeline

# With tuning
result = run_pipeline(model_name="xgboost", tune=True, n_trials=50, track=False)

# With manual params
result = run_pipeline(
    model_name="xgboost",
    params={"n_estimators": 300, "learning_rate": 0.03},
    feature_selection="kbest",
    feature_selection_k=8,
    track=False,
)

print(result["metrics"])
print(result["best_params"])          # hyperparameters used
print(result["best_cv_score"])        # Optuna best ROC-AUC (None if tune=False)
print(result["n_features_selected"])  # int or None
print(result["roi_default"])          # ROI dict at threshold=0.5
print(result["roi_optimal"])          # ROI dict at optimal threshold
print(result["artifact_path"])        # path to saved joblib file, or None
```

## Campaign ROI

A pseudo business metric that frames model value in terms of a retention campaign:

- **Revenue per retained customer**: $800 (fictional annual account margin)
- **Cost per outreach**: $50 (call, discount, or retention offer)
- **Retention success rate**: 30% (probability intervention works)

```
net_value     = TP × success_rate × revenue_per_retained
outreach_cost = (TP + FP) × cost_per_outreach
campaign_roi  = (net_value − outreach_cost) / outreach_cost × 100
```

The pipeline reports ROI at the default threshold (0.5) and at the ROI-maximising threshold. All assumptions are configurable in `src/config.py`.

## Models

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.8145 | 0.7699 | 0.317 |
| Random Forest | 0.8640 | 0.8667 | 0.585 |
| XGBoost (default) | 0.8705 | 0.8649 | 0.606 |
| XGBoost (tuned, 20 trials) | 0.8675 | 0.8721 | 0.590 |

*Results on a single 80/20 stratified split.*
