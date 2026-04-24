from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cads.config import AppConfig
from cads.data.features import engineer_features
from cads.data.preprocess import preprocess_dataframe
from cads.inference.schemas import InferenceInput, InferenceMode, PredictionResult

LOGGER = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class InferenceService:
    config: AppConfig
    best_supervised_model: str
    models: dict[str, object]
    label_encoder: object

    @classmethod
    def load(cls, config: AppConfig) -> "InferenceService":
        metrics_path = config.paths.artifacts_reports / "model_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing {metrics_path}. Run `uv run cads train-models` before starting API."
            )

        payload = json.loads(metrics_path.read_text())
        best_model = str(payload.get("best_supervised_model", "random_forest"))

        available_models: dict[str, object] = {}
        for name in ("logistic_regression", "random_forest", "isolation_forest"):
            model_path = config.paths.artifacts_models / f"{name}.joblib"
            if model_path.exists():
                available_models[name] = joblib.load(model_path)

        if best_model not in available_models:
            best_model = "random_forest" if "random_forest" in available_models else "logistic_regression"
        if best_model not in available_models:
            raise FileNotFoundError("No supervised model artifact found in artifacts/models.")

        le_path = config.paths.artifacts_models / "label_encoder.joblib"
        if not le_path.exists():
            raise FileNotFoundError(f"Missing label encoder artifact: {le_path}")
        label_encoder = joblib.load(le_path)

        LOGGER.info(
            "InferenceService loaded models=%s best_supervised=%s",
            sorted(available_models.keys()),
            best_model,
        )
        return cls(config=config, best_supervised_model=best_model, models=available_models, label_encoder=label_encoder)

    def _to_feature_frame(self, payload: InferenceInput) -> pd.DataFrame:
        raw = pd.DataFrame(
            [
                {
                    "timestamp": payload.timestamp.isoformat(),
                    "src_ip": payload.src_ip,
                    "dst_ip": payload.dst_ip,
                    "src_port": payload.src_port,
                    "dst_port": payload.dst_port,
                    "protocol": payload.protocol,
                    "packet_count": payload.packet_count,
                    "byte_count": payload.byte_count,
                    "duration": payload.duration,
                    "label": "unknown",
                }
            ]
        )
        pre = preprocess_dataframe(raw, self.config.data.canonical_columns)
        if pre.empty:
            raise ValueError("Input record invalid after preprocessing.")
        return engineer_features(pre)

    def _derived_features(self, feat: pd.DataFrame) -> dict[str, float | int]:
        feature_cols = list(self.config.data.feature_columns)
        categorical_int = {
            "src_is_private",
            "dst_is_private",
            "protocol_id",
            "same_src_dst",
            "is_well_known_dst_port",
            "hour_of_day",
        }
        out: dict[str, float | int] = {}
        for col in feature_cols:
            if col in categorical_int:
                out[col] = int(feat.iloc[0][col])
            else:
                out[col] = float(feat.iloc[0][col])
        return out

    def _predict_supervised(self, model_name: str, x: np.ndarray) -> dict[str, Any]:
        model = self.models[model_name]
        pred_idx = model.predict(x)
        pred_label = str(self.label_encoder.inverse_transform(pred_idx)[0])
        labels = list(self.label_encoder.classes_)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            prob_map = {label: float(prob) for label, prob in zip(labels, probs, strict=True)}
            confidence = float(max(prob_map.values()))
        else:
            prob_map = {pred_label: 1.0}
            confidence = 1.0

        return {
            "predicted_label": pred_label,
            "confidence": confidence,
            "class_probabilities": prob_map,
        }

    def _predict_anomaly(self, x: np.ndarray) -> dict[str, Any]:
        if "isolation_forest" not in self.models:
            return {"anomaly_flag": 0, "attack_probability": 0.0}

        model = self.models["isolation_forest"]
        raw_pred = int(model.predict(x)[0])  # -1 anomaly, 1 normal
        anomaly_flag = 1 if raw_pred == -1 else 0
        return {
            "anomaly_flag": anomaly_flag,
            "attack_probability": float(anomaly_flag),
        }

    def predict(self, payload: InferenceInput, mode: InferenceMode = "single") -> PredictionResult:
        feat = self._to_feature_frame(payload)
        x = feat[list(self.config.data.feature_columns)].to_numpy()
        derived = self._derived_features(feat)

        primary = self._predict_supervised(self.best_supervised_model, x)

        if mode == "single":
            return PredictionResult(
                predicted_label=primary["predicted_label"],
                confidence=primary["confidence"],
                model_name=self.best_supervised_model,
                class_probabilities=primary["class_probabilities"],
                derived_features=derived,
                model_breakdown={"mode": "single"},
            )

        if mode == "compare_all":
            breakdown: dict[str, Any] = {"mode": "compare_all", "models": {}}
            for model_name in ("logistic_regression", "random_forest"):
                if model_name in self.models:
                    breakdown["models"][model_name] = self._predict_supervised(model_name, x)
            breakdown["models"]["isolation_forest"] = self._predict_anomaly(x)

            return PredictionResult(
                predicted_label=primary["predicted_label"],
                confidence=primary["confidence"],
                model_name=self.best_supervised_model,
                class_probabilities=primary["class_probabilities"],
                derived_features=derived,
                model_breakdown=breakdown,
            )

        if mode == "ensemble":
            anomaly = self._predict_anomaly(x)
            primary_label = str(primary["predicted_label"])
            primary_conf = _safe_float(primary["confidence"], 0.0)
            benign_prob = _safe_float(primary["class_probabilities"].get("benign"), 0.0)
            supervised_attack_prob = max(0.0, 1.0 - benign_prob)
            ensemble_attack_prob = 0.70 * supervised_attack_prob + 0.30 * _safe_float(
                anomaly.get("attack_probability"), 0.0
            )

            final_label = primary_label
            if final_label == "benign" and ensemble_attack_prob >= 0.60:
                final_label = "anomaly_attack"

            final_conf = max(primary_conf, ensemble_attack_prob if final_label != "benign" else primary_conf)
            class_probs = dict(primary["class_probabilities"])
            class_probs["ensemble_attack_probability"] = float(ensemble_attack_prob)
            class_probs["isolation_anomaly_flag"] = float(anomaly.get("anomaly_flag", 0))

            breakdown = {
                "mode": "ensemble",
                "primary_model": self.best_supervised_model,
                "supervised": primary,
                "anomaly": anomaly,
                "ensemble": {
                    "supervised_attack_probability": supervised_attack_prob,
                    "ensemble_attack_probability": ensemble_attack_prob,
                    "final_label": final_label,
                },
            }

            return PredictionResult(
                predicted_label=final_label,
                confidence=float(min(1.0, final_conf)),
                model_name=f"{self.best_supervised_model}+isolation_forest",
                class_probabilities=class_probs,
                derived_features=derived,
                model_breakdown=breakdown,
            )

        raise ValueError(f"Unsupported mode: {mode}")

