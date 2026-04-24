from __future__ import annotations

import ipaddress
import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _as_private_flag(value: str) -> int:
    try:
        return int(ipaddress.ip_address(value).is_private)
    except ValueError:
        return 0


def _ip_low16(value: str) -> int:
    try:
        ip_int = int(ipaddress.ip_address(value))
        return ip_int & 0xFFFF
    except ValueError:
        return 0


def preprocess_dataframe(df: pd.DataFrame, required_columns: tuple[str, ...]) -> pd.DataFrame:
    work = df.copy()

    for col in required_columns:
        if col not in work.columns:
            work[col] = np.nan

    work = work[list(required_columns)]
    work = work.drop_duplicates()

    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work["src_ip"] = work["src_ip"].astype(str)
    work["dst_ip"] = work["dst_ip"].astype(str)
    work["protocol"] = work["protocol"].astype(str).str.upper().fillna("UNKNOWN")

    numeric_cols = ["src_port", "dst_port", "packet_count", "byte_count", "duration"]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["label"] = work["label"].astype(str).str.strip().str.lower()
    work["label"] = work["label"].replace({"nan": "unknown", "": "unknown"})

    work = work.dropna(subset=["timestamp", "src_port", "dst_port", "packet_count", "byte_count", "duration"])
    work = work[work["duration"] > 0]
    work = work[work["packet_count"] > 0]
    work = work[work["byte_count"] >= 0]

    work["src_is_private"] = work["src_ip"].map(_as_private_flag)
    work["dst_is_private"] = work["dst_ip"].map(_as_private_flag)
    work["src_ip_low16"] = work["src_ip"].map(_ip_low16)
    work["dst_ip_low16"] = work["dst_ip"].map(_ip_low16)
    work["hour_of_day"] = work["timestamp"].dt.hour.astype(int)

    if len(work) > 1:
        LOGGER.info("Preprocessed dataframe shape: %s", work.shape)
    return work
