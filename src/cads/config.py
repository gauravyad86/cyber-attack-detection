from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    artifacts_models: Path = Path("artifacts/models")
    artifacts_reports: Path = Path("artifacts/reports")
    artifacts_plots: Path = Path("artifacts/plots")


@dataclass(frozen=True)
class DataConfig:
    test_size: float = 0.20
    val_size: float = 0.20
    random_seed: int = 42

    raw_glob: str = "*.csv"
    canonical_columns: tuple[str, ...] = (
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "packet_count",
        "byte_count",
        "duration",
        "label",
    )
    feature_columns: tuple[str, ...] = (
        "src_port",
        "dst_port",
        "packet_count",
        "byte_count",
        "duration",
        "bytes_per_packet",
        "packets_per_second",
        "bytes_per_second",
        "is_well_known_dst_port",
        "same_src_dst",
        "protocol_id",
        "src_is_private",
        "dst_is_private",
        "src_ip_low16",
        "dst_ip_low16",
        "hour_of_day",
    )
    benign_labels: tuple[str, ...] = ("benign", "normal")


@dataclass(frozen=True)
class ModelConfig:
    random_seed: int = 42
    models_to_train: tuple[str, ...] = ("logistic_regression", "random_forest", "isolation_forest")
    class_weight: str = "balanced"

    # IsolationForest
    iforest_contamination: str = "auto"
    iforest_estimators: int = 250


@dataclass(frozen=True)
class AppConfig:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def ensure_directories(paths: Paths) -> None:
    paths.data_raw.mkdir(parents=True, exist_ok=True)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    paths.artifacts_models.mkdir(parents=True, exist_ok=True)
    paths.artifacts_reports.mkdir(parents=True, exist_ok=True)
    paths.artifacts_plots.mkdir(parents=True, exist_ok=True)

