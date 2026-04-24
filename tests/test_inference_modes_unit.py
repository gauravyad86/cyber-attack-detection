from __future__ import annotations

import pytest

from cads.inference.service import _safe_float


def test_inference_service_loads_all_expected_models(inference_service) -> None:
    assert "random_forest" in inference_service.models
    assert "logistic_regression" in inference_service.models
    assert "isolation_forest" in inference_service.models


def test_inference_single_mode_returns_expected_shape(inference_service, sample_payload) -> None:
    out = inference_service.predict(sample_payload, mode="single")
    assert out.model_breakdown["mode"] == "single"
    assert isinstance(out.predicted_label, str)
    assert 0.0 <= out.confidence <= 1.0


def test_inference_single_mode_has_feature_payload(inference_service, sample_payload) -> None:
    out = inference_service.predict(sample_payload, mode="single")
    assert "protocol_id" in out.derived_features
    assert "bytes_per_second" in out.derived_features


def test_inference_compare_all_contains_breakdown(inference_service, sample_payload) -> None:
    out = inference_service.predict(sample_payload, mode="compare_all")
    models = out.model_breakdown["models"]
    assert "random_forest" in models
    assert "logistic_regression" in models
    assert "isolation_forest" in models


def test_inference_compare_all_primary_response_still_present(inference_service, sample_payload) -> None:
    out = inference_service.predict(sample_payload, mode="compare_all")
    assert isinstance(out.class_probabilities, dict)
    assert out.model_breakdown["mode"] == "compare_all"


def test_inference_ensemble_contains_ensemble_fields(inference_service, sample_payload) -> None:
    out = inference_service.predict(sample_payload, mode="ensemble")
    assert out.model_breakdown["mode"] == "ensemble"
    assert "ensemble_attack_probability" in out.class_probabilities
    assert "isolation_anomaly_flag" in out.class_probabilities


def test_inference_ensemble_confidence_range(inference_service, sample_payload) -> None:
    out = inference_service.predict(sample_payload, mode="ensemble")
    assert 0.0 <= out.confidence <= 1.0


def test_inference_invalid_mode_raises_error(inference_service, sample_payload) -> None:
    with pytest.raises(ValueError):
        inference_service.predict(sample_payload, mode="bad-mode")  # type: ignore[arg-type]


def test_safe_float_valid_value() -> None:
    assert _safe_float("1.23") == 1.23


def test_safe_float_invalid_value_uses_default() -> None:
    assert _safe_float("abc", default=9.5) == 9.5

