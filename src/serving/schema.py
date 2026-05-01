"""Pydantic request and response schemas for the churn prediction API.

All field constraints reflect the valid ranges present in the Churn Modelling
dataset.  Tighten or loosen them to match your real customer population before
deploying to production.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Input schema for a single customer churn prediction request.

    All fields map directly to the raw dataset columns — no pre-processing or
    feature engineering is required from the caller.  The serving layer handles
    that internally.

    Examples
    --------
    >>> payload = CustomerFeatures(
    ...     CreditScore=650,
    ...     Geography="France",
    ...     Gender="Male",
    ...     Age=40,
    ...     Tenure=5,
    ...     Balance=75000.0,
    ...     NumOfProducts=2,
    ...     HasCrCard=1,
    ...     IsActiveMember=1,
    ...     EstimatedSalary=60000.0,
    ... )
    """

    CreditScore: int = Field(
        ..., ge=300, le=900,
        description="Customer credit score (300–900).",
        examples=[650],
    )
    Geography: Literal["France", "Germany", "Spain"] = Field(
        ...,
        description="Country of the customer's bank account.",
        examples=["France"],
    )
    Gender: Literal["Male", "Female"] = Field(
        ...,
        description="Customer gender.",
        examples=["Male"],
    )
    Age: int = Field(
        ..., ge=18, le=100,
        description="Customer age in years (18–100).",
        examples=[40],
    )
    Tenure: int = Field(
        ..., ge=0, le=10,
        description="Number of years the customer has been with the bank (0–10).",
        examples=[5],
    )
    Balance: float = Field(
        ..., ge=0.0,
        description="Account balance in local currency (≥ 0).",
        examples=[75000.0],
    )
    NumOfProducts: int = Field(
        ..., ge=1, le=4,
        description="Number of bank products held (1–4).",
        examples=[2],
    )
    HasCrCard: Literal[0, 1] = Field(
        ...,
        description="Whether the customer holds a credit card (0 = No, 1 = Yes).",
        examples=[1],
    )
    IsActiveMember: Literal[0, 1] = Field(
        ...,
        description="Whether the customer is an active member (0 = No, 1 = Yes).",
        examples=[1],
    )
    EstimatedSalary: float = Field(
        ..., gt=0.0,
        description="Estimated annual salary in local currency (> 0).",
        examples=[60000.0],
    )


class PredictionResponse(BaseModel):
    """Output schema returned by the ``POST /predict`` endpoint.

    Attributes
    ----------
    churn_probability : float
        Model's estimated probability that this customer will churn (0–1).
    will_churn : bool
        Hard label derived by applying ``threshold`` to ``churn_probability``.
    threshold : float
        Decision boundary used to produce ``will_churn``.
    model_name : str
        Identifier of the model that produced this prediction.
    model_version : str
        Semantic version of the deployed artifact.
    """

    # Disable Pydantic's protected-namespace check — model_name/model_version
    # are intentional field names, not accidental clashes with BaseModel internals.
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Predicted probability of churn (0–1).",
    )
    will_churn: bool = Field(
        ...,
        description="Hard churn label at the configured decision threshold.",
    )
    threshold: float = Field(
        ...,
        description="Decision threshold used to derive will_churn.",
    )
    model_name: str = Field(..., description="Model identifier (e.g. 'xgboost').")
    model_version: str = Field(..., description="Semantic version of the model artifact.")


class ModelInfo(BaseModel):
    """Response schema for the ``GET /model-info`` endpoint.

    Attributes
    ----------
    model_name : str
        Registry key of the deployed model.
    model_version : str
        Semantic version of the artifact.
    trained_at : str
        ISO-8601 UTC timestamp of when the artifact was created.
    feature_names : list[str]
        Raw input feature names expected by the API (before engineering).
    prediction_threshold : float
        Threshold used to convert probabilities to hard labels.
    test_metrics : dict[str, float]
        Held-out evaluation metrics recorded at training time.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    trained_at: str
    feature_names: list[str]
    prediction_threshold: float
    test_metrics: dict[str, float]


class HealthResponse(BaseModel):
    """Response schema for the ``GET /health`` endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="'ok' when the service is ready.")
    model_loaded: bool = Field(..., description="True when the model artifact is loaded.")
