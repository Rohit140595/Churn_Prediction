"""Tests for the Pydantic request/response schemas in src/serving/schema.py."""

import pytest
from pydantic import ValidationError

from src.serving.schema import CustomerFeatures, HealthResponse, PredictionResponse

# ---------------------------------------------------------------------------
# Valid baseline payload — reused across multiple tests
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 5,
    "Balance": 75_000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60_000.0,
}


class TestCustomerFeatures:
    def test_valid_payload_accepted(self):
        customer = CustomerFeatures(**VALID_PAYLOAD)
        assert customer.CreditScore == 650
        assert customer.Geography == "France"

    def test_credit_score_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "CreditScore": 299})

    def test_credit_score_above_maximum_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "CreditScore": 901})

    def test_invalid_geography_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "Geography": "Australia"})

    def test_invalid_gender_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "Gender": "Unknown"})

    def test_age_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "Age": 17})

    def test_age_above_maximum_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "Age": 101})

    def test_negative_balance_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "Balance": -1.0})

    def test_zero_salary_rejected(self):
        # EstimatedSalary must be > 0 (gt constraint)
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "EstimatedSalary": 0.0})

    def test_invalid_has_cr_card_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "HasCrCard": 2})

    def test_num_of_products_above_maximum_rejected(self):
        with pytest.raises(ValidationError):
            CustomerFeatures(**{**VALID_PAYLOAD, "NumOfProducts": 5})


class TestPredictionResponse:
    def test_valid_response_accepted(self):
        resp = PredictionResponse(
            churn_probability=0.25,
            will_churn=False,
            threshold=0.5,
            model_name="xgboost",
            model_version="0.4.0",
        )
        assert resp.churn_probability == 0.25
        assert resp.will_churn is False

    def test_probability_above_one_rejected(self):
        with pytest.raises(ValidationError):
            PredictionResponse(
                churn_probability=1.1,
                will_churn=True,
                threshold=0.5,
                model_name="xgboost",
                model_version="0.4.0",
            )


class TestHealthResponse:
    def test_valid_health_response(self):
        resp = HealthResponse(status="ok", model_loaded=True)
        assert resp.status == "ok"
        assert resp.model_loaded is True
