from __future__ import annotations

import json
from pathlib import Path

from cads.config import AppConfig, DataConfig, ModelConfig, Paths, ensure_directories
from cads.data.pipeline import prepare_data
from cads.data.synthetic import generate_sample_dataset


def _test_config(tmp_path: Path) -> AppConfig:
    paths = Paths(
        root=tmp_path,
        data_raw=tmp_path / "data/raw",
        data_processed=tmp_path / "data/processed",
        artifacts_models=tmp_path / "artifacts/models",
        artifacts_reports=tmp_path / "artifacts/reports",
        artifacts_plots=tmp_path / "artifacts/plots",
    )
    return AppConfig(paths=paths, data=DataConfig(random_seed=7), model=ModelConfig(random_seed=7))


def test_prepare_data_generates_splits_and_report(tmp_path: Path) -> None:
    config = _test_config(tmp_path)
    ensure_directories(config.paths)
    generate_sample_dataset(config.paths.data_raw, rows=1200, seed=7)

    output = prepare_data(config)

    assert Path(output["train_path"]).exists()
    assert Path(output["val_path"]).exists()
    assert Path(output["test_path"]).exists()
    report_path = Path(output["report_path"])
    assert report_path.exists()

    report = json.loads(report_path.read_text())
    assert report["total_records"] > 0
    assert report["split_sizes"]["train"] > 0
    assert "feature_columns" in report

