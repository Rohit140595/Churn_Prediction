"""FastAPI inference server for the churn prediction model.

Endpoints
---------
GET  /health       Liveness check — confirms the service is up and the model is loaded.
GET  /model-info   Returns metadata about the deployed model artifact.
POST /predict      Accepts a customer record and returns a churn probability.

Running locally
---------------
Train and save a model first::

    python scripts/train.py --model xgboost --save-model

Then start the server::

    uvicorn src.serving.app:app --reload --port 8000

The interactive API docs are available at http://localhost:8000/docs.

Production deployment
---------------------
This server speaks plain HTTP.  In production it MUST sit behind an
HTTPS-terminating reverse proxy or load balancer (e.g. AWS ALB, GCP Load
Balancer, nginx, Traefik).  Never expose this service directly to the internet
over HTTP — customer data in the request payload would be transmitted in
plaintext.

See the project README for a Docker + reverse-proxy deployment example.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.features.engineer import add_features
from src.serving.model_store import load_artifact
from src.serving.schema import CustomerFeatures, HealthResponse, ModelInfo, PredictionResponse

logger = logging.getLogger(__name__)

# Raw input feature names in the order the API expects them.
# These match the original dataset columns (before feature engineering).
_RAW_FEATURE_NAMES: list[str] = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

# Default decision threshold — predictions with churn_probability >= this are
# labelled will_churn=True.  0.5 is a neutral starting point; tune it using
# find_optimal_threshold() from src.evaluation.business_metrics if needed.
_DEFAULT_THRESHOLD: float = 0.5

# Module-level artifact holder populated during application startup.
# Using a dict as a mutable container so the lifespan function can update it.
_state: dict[str, Any] = {"artifact": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at startup; release on shutdown."""
    try:
        _state["artifact"] = load_artifact()
        logger.info(
            "Model artifact loaded: %s v%s (trained %s)",
            _state["artifact"]["model_name"],
            _state["artifact"]["version"],
            _state["artifact"]["trained_at"],
        )
    except FileNotFoundError as exc:
        # Log the error but still start the server so /health can report the issue.
        logger.error("Failed to load model artifact: %s", exc)
    yield
    # Nothing to clean up on shutdown for a joblib-loaded model.
    _state["artifact"] = None


app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Predicts the probability that a bank customer will churn.\n\n"
        "**Note:** Deploy this service behind an HTTPS-terminating load balancer "
        "or reverse proxy.  The service itself speaks plain HTTP."
    ),
    version="0.4.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_artifact() -> dict[str, Any]:
    """Return the loaded artifact or raise 503 if it is not available."""
    if _state["artifact"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifact is not loaded.  "
                "Train and save a model first: "
                "python scripts/train.py --model xgboost --save-model"
            ),
        )
    return _state["artifact"]


def _predict(artifact: dict[str, Any], features: CustomerFeatures) -> float:
    """Run the full inference pipeline for a single customer record.

    Steps:
    1. Convert validated input to a single-row DataFrame.
    2. Apply feature engineering (BalancePerProduct, ActiveProducts).
    3. Apply the fitted preprocessor (scaling + encoding).
    4. Optionally apply the fitted feature selector.
    5. Return the positive-class (churn) probability.
    """
    # Step 1 — build a single-row DataFrame with the raw feature columns
    row = pd.DataFrame([features.model_dump()])

    # Step 2 — engineer derived features (must match training-time transformations)
    row = add_features(row)

    # Step 3 — preprocess: apply train-fit scaler/encoder, do NOT re-fit
    preprocessor = artifact["preprocessor"]
    X = preprocessor.transform(row)

    # Step 4 — feature selection (optional)
    selector = artifact["selector"]
    if selector is not None:
        X = selector.transform(X)

    # Step 5 — predict; column 1 of predict_proba is the positive-class probability
    proba = artifact["model"].predict_proba(X)[0, 1]
    return float(proba)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["Operations"],
)
def health() -> HealthResponse:
    """Return the service status and whether the model artifact is loaded.

    Use this endpoint for container health checks and load-balancer probes.
    A ``200 OK`` response with ``model_loaded: true`` means the service is
    ready to accept prediction requests.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_state["artifact"] is not None,
    )


@app.get(
    "/model-info",
    response_model=ModelInfo,
    summary="Deployed model metadata",
    tags=["Operations"],
)
def model_info() -> ModelInfo:
    """Return metadata about the currently loaded model artifact.

    Useful for auditing which model version is serving traffic without
    having to inspect the container filesystem directly.
    """
    artifact = _get_artifact()
    return ModelInfo(
        model_name=artifact["model_name"],
        model_version=artifact["version"],
        trained_at=artifact["trained_at"],
        feature_names=_RAW_FEATURE_NAMES,
        prediction_threshold=_DEFAULT_THRESHOLD,
        test_metrics=artifact["test_metrics"],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict churn probability for a single customer",
    tags=["Prediction"],
)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Predict whether a customer is likely to churn.

    Accepts a single customer record in JSON and returns:

    - ``churn_probability`` — raw model score (0–1)
    - ``will_churn`` — hard label at the default threshold (0.5)

    The threshold can be adjusted by the operator to trade precision for
    recall depending on campaign economics (see ``find_optimal_threshold``
    in ``src.evaluation.business_metrics``).

    Request body example::

        {
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
        }
    """
    artifact = _get_artifact()
    churn_prob = _predict(artifact, customer)

    return PredictionResponse(
        churn_probability=churn_prob,
        will_churn=churn_prob >= _DEFAULT_THRESHOLD,
        threshold=_DEFAULT_THRESHOLD,
        model_name=artifact["model_name"],
        model_version=artifact["version"],
    )
