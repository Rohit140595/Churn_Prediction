from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_OUTPUT_DIR = ROOT_DIR / "models_output"

RAW_DATA_PATH = DATA_DIR / "Churn_Modelling.csv"
TARGET_COL = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

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

# --- Business assumptions (fictional, adjust to match real unit economics) ---
# Revenue saved per customer successfully retained (e.g. annual account margin)
REVENUE_PER_RETAINED_CUSTOMER: float = 800.0
# Cost of one outreach intervention (call, discount, retention offer)
COST_PER_OUTREACH: float = 50.0
# Probability that an intervention actually retains an at-risk customer
RETENTION_SUCCESS_RATE: float = 0.30

DEFAULT_PARAMS: dict[str, dict] = {
    "logistic": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
    "random_forest": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10},
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
    },
}
