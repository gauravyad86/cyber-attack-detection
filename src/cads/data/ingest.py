from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cads.data.schema import CANONICAL_ALIASES

LOGGER = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_cols = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        normalized_cols[col] = CANONICAL_ALIASES.get(key, key)
    return df.rename(columns=normalized_cols)


def load_raw_csv_files(raw_dir: Path, pattern: str = "*.csv") -> pd.DataFrame:
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir} with pattern '{pattern}'")

    frames: list[pd.DataFrame] = []
    for file in files:
        LOGGER.info("Reading raw file: %s", file)
        frame = pd.read_csv(file)
        frame = _normalize_columns(frame)
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    LOGGER.info("Loaded %s records from %s file(s)", len(merged), len(files))
    return merged

