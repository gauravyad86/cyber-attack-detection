from __future__ import annotations


def _valid_payload() -> dict[str, object]:
    return {
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


def test_predict_invalid_payload_returns_422(api_client) -> None:
    bad = _valid_payload()
    bad["duration"] = -1
    response = api_client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_mode_returns_422(api_client) -> None:
    response = api_client.post("/predict?mode=wrong-mode", json=_valid_payload())
    assert response.status_code == 422


def test_alerts_limit_validation_returns_422(api_client) -> None:
    response = api_client.get("/alerts?limit=9999")
    assert response.status_code == 422

