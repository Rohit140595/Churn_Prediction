# Churn Prediction

Binary classification project predicting whether a bank customer will churn, using the [Churn Modelling dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) (10,000 customers, 14 features).

## Project structure

```
├── analysis/
│   └── EDA.ipynb               exploratory data analysis
├── data/
│   └── Churn_Modelling.csv     raw dataset (gitignored)
├── src/
│   ├── config.py               paths, constants, default hyperparameters
│   ├── data/
│   │   ├── loader.py           load_raw_data()
│   │   └── preprocessor.py     split_and_preprocess() — leakage-free
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
│   │   ├── metrics.py          compute_metrics(), cross_validate_model(), plot helpers
│   │   └── business_metrics.py compute_campaign_roi(), find_optimal_threshold(), plot_roi_curve()
│   ├── tracking/
│   │   └── mlflow_tracker.py   MLflow experiment logging
│   └── pipeline.py             run_pipeline() end-to-end orchestrator
└── scripts/
    └── train.py                CLI entry point
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Train with default hyperparameters (XGBoost) and MLflow tracking
python scripts/train.py

# Choose a model
python scripts/train.py --model logistic
python scripts/train.py --model random_forest
python scripts/train.py --model xgboost

# Feature selection
python scripts/train.py --model xgboost --feature-selection importance
python scripts/train.py --model xgboost --feature-selection kbest --feature-selection-k 8

# Custom experiment name or disable tracking
python scripts/train.py --model xgboost --experiment-name my_experiment
python scripts/train.py --model xgboost --no-tracking
```

### Feature selection strategies

| Strategy | Mechanism | Features kept |
|---|---|---|
| `importance` | `SelectFromModel` (RandomForest, threshold=mean) | Data-driven |
| `kbest` | `SelectKBest` (ANOVA F-score, top k) | Fixed via `--feature-selection-k` |

Selection is applied inside each CV fold and on the final train/test split — no leakage.

Output includes 5-fold cross-validation scores, final held-out test metrics, and campaign ROI:

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

Campaign ROI (assumptions: $50/outreach, $800 revenue/retained, 30% success rate):
  threshold=0.50 : +282.1%  (contacted 250, net $35,260)
  optimal=0.90   : +372.1%  (contacted 61, net $11,350)
```

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
    feature_selection="kbest",
    feature_selection_k=8,
    track=False,
)
print(result["metrics"])
print(result["cv_results"])           # per-fold arrays
print(result["n_features_selected"])  # int or None
print(result["roi_default"])          # ROI dict at threshold=0.5
print(result["roi_optimal"])          # ROI dict at optimal threshold
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

The pipeline reports ROI at the default threshold (0.5) and at the threshold that maximises ROI. All assumptions are configurable in `src/config.py`.

## Models

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.8145 | 0.7699 | 0.317 |
| Random Forest | 0.8640 | 0.8667 | 0.585 |
| XGBoost | 0.8705 | 0.8649 | 0.606 |

*Results on a single 80/20 stratified split with default hyperparameters.*
