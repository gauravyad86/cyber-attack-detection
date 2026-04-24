from __future__ import annotations

import json
import time

import pandas as pd

from cads.config import AppConfig
from cads.inference.schemas import InferenceInput
from cads.inference.service import InferenceService


def main() -> None:
    config = AppConfig()
    test_path = config.paths.data_processed / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("Missing data/processed/test.csv. Run data prep first.")

    df = pd.read_csv(test_path).head(500)
    svc = InferenceService.load(config)

    latencies_ms: list[float] = []
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
        t0 = time.perf_counter()
        svc.predict(payload)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latencies_ms.sort()
    n = len(latencies_ms)
    result = {
        "samples": n,
        "latency_ms_avg": round(sum(latencies_ms) / max(n, 1), 4),
        "latency_ms_p50": round(latencies_ms[int(0.50 * (n - 1))], 4),
        "latency_ms_p95": round(latencies_ms[int(0.95 * (n - 1))], 4),
        "latency_ms_p99": round(latencies_ms[int(0.99 * (n - 1))], 4),
    }
    out = config.paths.artifacts_reports / "performance_report.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"performance_report": str(out), **result}, indent=2))


if __name__ == "__main__":
    main()

