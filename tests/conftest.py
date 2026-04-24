from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cads.api.app import create_app
from cads.config import AppConfig, DataConfig, ModelConfig, Paths, ensure_directories
from cads.data.pipeline import prepare_data
from cads.data.synthetic import generate_sample_dataset
from cads.inference.schemas import InferenceInput
from cads.inference.service import InferenceService
from cads.models.train import train_and_evaluate


@pytest.fixture(scope="session")
def trained_config(tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    root = tmp_path_factory.mktemp("trained_stack")
    paths = Paths(
        root=root,
        data_raw=root / "data/raw",
        data_processed=root / "data/processed",
        artifacts_models=root / "artifacts/models",
        artifacts_reports=root / "artifacts/reports",
        artifacts_plots=root / "artifacts/plots",
    )
    config = AppConfig(paths=paths, data=DataConfig(random_seed=21), model=ModelConfig(random_seed=21))
    ensure_directories(config.paths)
    generate_sample_dataset(config.paths.data_raw, rows=1200, seed=21)
    prepare_data(config)
    train_and_evaluate(config)
    return config


@pytest.fixture()
def inference_service(trained_config: AppConfig) -> InferenceService:
    return InferenceService.load(trained_config)


@pytest.fixture()
def sample_payload() -> InferenceInput:
    return InferenceInput(
        timestamp="2026-04-23T12:00:00Z",
        src_ip="10.0.0.5",
        dst_ip="203.0.113.9",
        src_port=50001,
        dst_port=443,
        protocol="tcp",
        packet_count=120.0,
        byte_count=85000.0,
        duration=1.4,
    )


@pytest.fixture()
def api_client(trained_config: AppConfig) -> TestClient:
    app = create_app(config=trained_config)
    return TestClient(app)

