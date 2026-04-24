from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from cads.config import AppConfig
from cads.data.features import engineer_features
from cads.data.ingest import load_raw_csv_files
from cads.data.preprocess import preprocess_dataframe

LOGGER = logging.getLogger(__name__)


def _class_distribution(df: pd.DataFrame) -> dict[str, int]:
    return {str(label): int(count) for label, count in df["label"].value_counts().to_dict().items()}


def _safe_stratify_series(series: pd.Series) -> pd.Series | None:
    vc = series.value_counts()
    if vc.empty:
        return None
    return series if vc.min() >= 2 else None


def prepare_data(config: AppConfig) -> dict[str, object]:
    raw_df = load_raw_csv_files(config.paths.data_raw, config.data.raw_glob)
    clean_df = preprocess_dataframe(raw_df, config.data.canonical_columns)
    feat_df = engineer_features(clean_df)

    stratify_series = _safe_stratify_series(feat_df["label"])
    train_df, test_df = train_test_split(
        feat_df,
        test_size=config.data.test_size,
        random_state=config.data.random_seed,
        stratify=stratify_series,
    )

    val_ratio_of_train = config.data.val_size / (1.0 - config.data.test_size)
    stratify_train = _safe_stratify_series(train_df["label"])
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_ratio_of_train,
        random_state=config.data.random_seed,
        stratify=stratify_train,
    )

    config.paths.data_processed.mkdir(parents=True, exist_ok=True)
    train_path = config.paths.data_processed / "train.csv"
    val_path = config.paths.data_processed / "val.csv"
    test_path = config.paths.data_processed / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    report = {
        "total_records": int(len(feat_df)),
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "class_distribution": {
            "overall": _class_distribution(feat_df),
            "train": _class_distribution(train_df),
            "val": _class_distribution(val_df),
            "test": _class_distribution(test_df),
        },
        "feature_columns": list(config.data.feature_columns),
        "config": asdict(config.data),
    }

    report_path = config.paths.artifacts_reports / "data_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    LOGGER.info("Data preparation complete. Report saved to %s", report_path)
    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "report_path": str(report_path),
    }


def load_processed_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")
    return pd.read_csv(path)

