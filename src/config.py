from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_OUTPUT_DIR = ROOT_DIR / "models_output"

RAW_DATA_PATH = DATA_DIR / "Churn_Modelling.csv"

# --- Dataset constants ---
TARGET_COL = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]  # identifiers with no predictive value

RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80/20 train/test split

# --- Feature lists ---
# Includes engineered features — preprocessor filters to only present columns at runtime
NUMERICAL_FEATURES: list[str] = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "BalancePerProduct",
    "ActiveProducts",
]
CATEGORICAL_FEATURES: list[str] = ["Geography", "Gender"]

# --- Business assumptions ---
# All values are fictional; adjust to match real unit economics before production use.
# Revenue saved per customer successfully retained (e.g. annual account margin)
REVENUE_PER_RETAINED_CUSTOMER: float = 800.0
# Cost of one outreach intervention (call, discount, retention offer)
COST_PER_OUTREACH: float = 50.0
# Probability that an intervention actually retains an at-risk customer
RETENTION_SUCCESS_RATE: float = 0.30

# --- Default hyperparameters ---
# Used when tune=False; these are sensible starting points, not tuned values.
DEFAULT_PARAMS: dict[str, dict] = {
    "logistic": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    "random_forest": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10},
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}
