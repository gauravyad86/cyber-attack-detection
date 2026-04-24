from __future__ import annotations

from datetime import UTC, datetime

from cads.alerting.engine import AlertEngine
from cads.inference.schemas import InferenceInput, PredictionResult


def test_alert_engine_severity_and_score() -> None:
    engine = AlertEngine()
    req = InferenceInput(
        timestamp=datetime.now(UTC),
        src_ip="10.0.0.2",
        dst_ip="192.168.1.10",
        src_port=51512,
        dst_port=443,
        protocol="tcp",
        packet_count=200,
        byte_count=120000,
        duration=0.8,
    )
    pred = PredictionResult(
        predicted_label="dos",
        confidence=0.95,
        model_name="random_forest",
        class_probabilities={"dos": 0.95, "benign": 0.05},
        derived_features={},
    )
    decision = engine.decide(req, pred)
    assert decision.score > 0.85
    assert decision.severity == "critical"

