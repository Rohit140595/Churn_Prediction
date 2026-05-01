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
│   │   └── preprocessor.py     split_and_preprocess() — leakage-free
│   ├── features/
│   │   └── engineer.py         feature engineering (BalancePerProduct, ActiveProducts)
│   ├── models/
│   │   ├── base.py             BaseModel ABC
│   │   ├── logistic.py         LogisticRegressionModel
│   │   ├── random_forest.py    RandomForestModel
│   │   ├── xgboost_model.py    XGBoostModel
│   │   └── registry.py         get_model(name) factory
│   ├── evaluation/
│   │   └── metrics.py          compute_metrics(), cross_validate_model(), plot helpers
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
# Train with default hyperparameters (XGBoost) and MLflow tracking
python scripts/train.py

# Choose a model
python scripts/train.py --model logistic
python scripts/train.py --model random_forest
python scripts/train.py --model xgboost

# Custom experiment name or disable tracking
python scripts/train.py --model xgboost --experiment-name my_experiment
python scripts/train.py --model xgboost --no-tracking

# Train and save the inference artifact for serving
python scripts/train.py --model xgboost --save-model
```

Output includes 5-fold cross-validation scores and final held-out test metrics:

```
Cross-validation (5-fold, mean ± std):
  accuracy    : 0.8645 ± 0.0045
  roc_auc     : 0.8663 ± 0.0048
  f1          : 0.5939 ± 0.0180
  precision   : 0.7621 ± 0.0153
  recall      : 0.4870 ± 0.0226

Test-set metrics:
  accuracy    : 0.8705
  roc_auc     : 0.8649
  f1          : 0.6058
  precision   : 0.7960
  recall      : 0.4889
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
# Quote the -v value — required if the project path contains spaces
docker run -p 8000:8000 \
  -v "$(pwd)/models_output:/app/models_output:ro" \
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

result = run_pipeline(
    model_name="xgboost",
    params={"n_estimators": 300, "learning_rate": 0.03},
    track=False,
)
print(result["metrics"])
print(result["cv_results"])  # per-fold arrays
```

## Models

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.8145 | 0.7699 | 0.317 |
| Random Forest | 0.8640 | 0.8667 | 0.585 |
| XGBoost | 0.8705 | 0.8649 | 0.606 |

*Results on a single 80/20 stratified split with default hyperparameters.*
