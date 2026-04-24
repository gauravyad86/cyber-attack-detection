from __future__ import annotations

from dataclasses import dataclass

from cads.inference.schemas import InferenceInput, PredictionResult


def _label_weight(label: str) -> float:
    weights = {
        "dos": 0.95,
        "ddos": 0.98,
        "port_scan": 0.72,
        "brute_force": 0.80,
        "anomaly_attack": 0.85,
        "benign": 0.10,
        "normal": 0.10,
        "unknown": 0.40,
    }
    return weights.get(label.lower(), 0.55)


def _severity_from_score(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


@dataclass
class AlertDecision:
    severity: str
    score: float
    evidence: dict[str, float | str | int]


class AlertEngine:
    def decide(self, request: InferenceInput, prediction: PredictionResult) -> AlertDecision:
        conf = float(prediction.confidence)
        weight = _label_weight(prediction.predicted_label)
        score = min(1.0, 0.60 * conf + 0.40 * weight)
        severity = _severity_from_score(score)
        evidence = {
            "src_port": request.src_port,
            "dst_port": request.dst_port,
            "protocol": request.protocol.upper(),
            "packet_count": request.packet_count,
            "byte_count": request.byte_count,
            "duration": request.duration,
            "model": prediction.model_name,
            "predicted_label": prediction.predicted_label,
            "confidence": round(conf, 4),
        }
        return AlertDecision(severity=severity, score=score, evidence=evidence)
