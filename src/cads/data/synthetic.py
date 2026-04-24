from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _rand_ip(rng: np.random.Generator, private: bool = True) -> str:
    if private:
        return f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
    return f"{rng.integers(11, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"


def generate_sample_dataset(output_dir: Path, rows: int = 5000, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    base_time = datetime.now(UTC) - timedelta(days=2)

    labels = rng.choice(
        ["benign", "dos", "port_scan", "brute_force"],
        size=rows,
        p=[0.70, 0.12, 0.10, 0.08],
    )
    protocols = rng.choice(["TCP", "UDP", "ICMP"], size=rows, p=[0.72, 0.23, 0.05])

    packet_count = rng.poisson(lam=20, size=rows).clip(min=1)
    byte_count = (packet_count * rng.normal(loc=520, scale=140, size=rows)).clip(min=50).astype(int)
    duration = rng.gamma(shape=2.1, scale=1.4, size=rows).clip(min=0.01)
    src_port = rng.integers(1024, 65535, size=rows)
    dst_port = rng.choice([22, 53, 80, 443, 445, 3389, 8080, 3306], size=rows)

    # Inject attack signatures into distributions so models can learn meaningful patterns.
    dos_mask = labels == "dos"
    packet_count[dos_mask] = rng.integers(200, 2200, size=int(dos_mask.sum()))
    byte_count[dos_mask] = (packet_count[dos_mask] * rng.integers(700, 1200, size=int(dos_mask.sum()))).astype(int)
    duration[dos_mask] = rng.uniform(0.1, 2.0, size=int(dos_mask.sum()))
    dst_port[dos_mask] = rng.choice([80, 443], size=int(dos_mask.sum()))

    scan_mask = labels == "port_scan"
    packet_count[scan_mask] = rng.integers(6, 40, size=int(scan_mask.sum()))
    byte_count[scan_mask] = (packet_count[scan_mask] * rng.integers(40, 120, size=int(scan_mask.sum()))).astype(int)
    duration[scan_mask] = rng.uniform(0.01, 0.4, size=int(scan_mask.sum()))
    dst_port[scan_mask] = rng.integers(1, 65535, size=int(scan_mask.sum()))

    brute_mask = labels == "brute_force"
    packet_count[brute_mask] = rng.integers(40, 180, size=int(brute_mask.sum()))
    byte_count[brute_mask] = (packet_count[brute_mask] * rng.integers(80, 300, size=int(brute_mask.sum()))).astype(int)
    duration[brute_mask] = rng.uniform(0.2, 4.0, size=int(brute_mask.sum()))
    dst_port[brute_mask] = rng.choice([22, 3389], size=int(brute_mask.sum()))

    df = pd.DataFrame(
        {
            "timestamp": [base_time + timedelta(seconds=int(x)) for x in rng.integers(0, 172800, size=rows)],
            "src_ip": [_rand_ip(rng, private=True) for _ in range(rows)],
            "dst_ip": [_rand_ip(rng, private=False) for _ in range(rows)],
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocols,
            "packet_count": packet_count,
            "byte_count": byte_count,
            "duration": duration.round(5),
            "label": labels,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "synthetic_network_traffic.csv"
    df.to_csv(out_path, index=False)
    LOGGER.info("Synthetic dataset written to %s (%s rows)", out_path, rows)
    return out_path

