from __future__ import annotations

import time
from datetime import UTC, datetime

import pandas as pd

from cads.alerting.engine import AlertEngine
from cads.alerting.storage import AlertInsert, AlertStore
from cads.config import AppConfig
from cads.inference.schemas import InferenceInput, InferenceMode
from cads.inference.service import InferenceService


def replay_from_test_split(config: AppConfig, limit: int = 200, mode: InferenceMode = "single") -> dict[str, int | str]:
    test_path = config.paths.data_processed / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing split: {test_path}. Run prepare-data first.")

    df = pd.read_csv(test_path).head(limit)
    inference = InferenceService.load(config)
    alert_engine = AlertEngine()
    store = AlertStore(config.paths.artifacts_reports / "alerts.db")

    inserted = 0
    for _, row in df.iterrows():
        payload = InferenceInput(
            timestamp=pd.to_datetime(row["timestamp"], utc=True),
            src_ip=str(row["src_ip"]),
            dst_ip=str(row["dst_ip"]),
            src_port=int(row["src_port"]),
            dst_port=int(row["dst_port"]),
            protocol=str(row["protocol"]),
            packet_count=float(row["packet_count"]),
            byte_count=float(row["byte_count"]),
            duration=float(row["duration"]),
        )
        prediction = inference.predict(payload, mode=mode)
        decision = alert_engine.decide(payload, prediction)
        store.insert_alert(
            AlertInsert(
                timestamp=payload.timestamp.isoformat(),
                src_ip=payload.src_ip,
                dst_ip=payload.dst_ip,
                predicted_label=prediction.predicted_label,
                confidence=prediction.confidence,
                severity=decision.severity,
                score=decision.score,
                evidence=decision.evidence,
            )
        )
        inserted += 1

    return {"inserted_alerts": inserted, "mode": mode}


def simulate_live_alerts(
    config: AppConfig,
    *,
    mode: InferenceMode = "single",
    interval_seconds: float = 5.0,
    batch_size: int = 3,
    cycles: int = 30,
) -> dict[str, int | float | str]:
    test_path = config.paths.data_processed / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing split: {test_path}. Run prepare-data first.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if cycles < 1:
        raise ValueError("cycles must be >= 1")

    df = pd.read_csv(test_path)
    if df.empty:
        raise ValueError("Processed test split is empty.")

    inference = InferenceService.load(config)
    alert_engine = AlertEngine()
    store = AlertStore(config.paths.artifacts_reports / "alerts.db")

    inserted = 0
    idx = 0
    n_rows = len(df)

    for cycle in range(cycles):
        for _ in range(batch_size):
            row = df.iloc[idx % n_rows]
            idx += 1
            payload = InferenceInput(
                timestamp=datetime.now(UTC),
                src_ip=str(row["src_ip"]),
                dst_ip=str(row["dst_ip"]),
                src_port=int(row["src_port"]),
                dst_port=int(row["dst_port"]),
                protocol=str(row["protocol"]),
                packet_count=float(row["packet_count"]),
                byte_count=float(row["byte_count"]),
                duration=float(row["duration"]),
            )
            prediction = inference.predict(payload, mode=mode)
            decision = alert_engine.decide(payload, prediction)
            store.insert_alert(
                AlertInsert(
                    timestamp=payload.timestamp.isoformat(),
                    src_ip=payload.src_ip,
                    dst_ip=payload.dst_ip,
                    predicted_label=prediction.predicted_label,
                    confidence=prediction.confidence,
                    severity=decision.severity,
                    score=decision.score,
                    evidence={**decision.evidence, "live_cycle": cycle + 1},
                )
            )
            inserted += 1

        if cycle < cycles - 1:
            time.sleep(interval_seconds)

    return {
        "inserted_alerts": inserted,
        "mode": mode,
        "interval_seconds": interval_seconds,
        "batch_size": batch_size,
        "cycles": cycles,
    }
