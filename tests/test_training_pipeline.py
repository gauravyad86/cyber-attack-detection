from __future__ import annotations

import json
from pathlib import Path

from cads.config import AppConfig, DataConfig, ModelConfig, Paths, ensure_directories
from cads.data.pipeline import prepare_data
from cads.data.synthetic import generate_sample_dataset
from cads.models.train import train_and_evaluate


def _test_config(tmp_path: Path) -> AppConfig:
    paths = Paths(
        root=tmp_path,
        data_raw=tmp_path / "data/raw",
        data_processed=tmp_path / "data/processed",
        artifacts_models=tmp_path / "artifacts/models",
        artifacts_reports=tmp_path / "artifacts/reports",
        artifacts_plots=tmp_path / "artifacts/plots",
    )
    return AppConfig(paths=paths, data=DataConfig(random_seed=11), model=ModelConfig(random_seed=11))


def test_train_models_writes_artifacts_and_metrics(tmp_path: Path) -> None:
    config = _test_config(tmp_path)
    ensure_directories(config.paths)
    generate_sample_dataset(config.paths.data_raw, rows=1800, seed=11)
    prepare_data(config)

    output = train_and_evaluate(config)

    metrics_path = Path(output["metrics_path"])
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "best_supervised_model" in metrics
    assert "random_forest" in metrics["metrics"]
    assert (config.paths.artifacts_models / "random_forest.joblib").exists()
    assert (config.paths.artifacts_models / "label_encoder.joblib").exists()

