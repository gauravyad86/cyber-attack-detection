from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cads.api.app import create_app
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
    return AppConfig(paths=paths, data=DataConfig(random_seed=13), model=ModelConfig(random_seed=13))


def test_predict_and_alert_endpoints(tmp_path: Path) -> None:
    config = _test_config(tmp_path)
    ensure_directories(config.paths)
    generate_sample_dataset(config.paths.data_raw, rows=1000, seed=13)
    prepare_data(config)
    train_and_evaluate(config)

    app = create_app(config=config)
    client = TestClient(app)

    payload = {
        "timestamp": "2026-04-23T12:30:00Z",
        "src_ip": "10.0.0.5",
        "dst_ip": "203.0.113.9",
        "src_port": 51000,
        "dst_port": 443,
        "protocol": "TCP",
        "packet_count": 120.0,
        "byte_count": 80000.0,
        "duration": 1.2,
    }

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "single"
    assert "prediction" in body
    assert "alert" in body
    assert "id" in body["alert"]

    compare_response = client.post("/predict?mode=compare_all&persist=false", json=payload)
    assert compare_response.status_code == 200
    compare_body = compare_response.json()
    assert compare_body["mode"] == "compare_all"
    assert "models" in compare_body["prediction"]["model_breakdown"]

    ensemble_response = client.post("/predict?mode=ensemble&persist=false", json=payload)
    assert ensemble_response.status_code == 200
    ensemble_body = ensemble_response.json()
    assert ensemble_body["mode"] == "ensemble"
    assert "ensemble" in ensemble_body["prediction"]["model_breakdown"]

    alerts = client.get("/alerts?limit=10")
    assert alerts.status_code == 200
    assert alerts.json()["count"] >= 1

    metrics = client.get("/alerts/metrics")
    assert metrics.status_code == 200
    assert "by_label" in metrics.json()
    assert "by_severity" in metrics.json()
