from __future__ import annotations

from pathlib import Path

from cads.alerting.storage import AlertInsert, AlertStore


def _record(idx: int, label: str, severity: str) -> AlertInsert:
    return AlertInsert(
        timestamp=f"2026-04-23T12:00:{idx:02d}Z",
        src_ip=f"10.0.0.{idx}",
        dst_ip="203.0.113.1",
        predicted_label=label,
        confidence=0.8,
        severity=severity,
        score=0.75,
        evidence={"k": idx},
    )


def test_alert_store_initializes_and_empty_aggregates(tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "alerts.db")
    agg = store.aggregate_counts()
    assert agg["by_label"] == {}
    assert agg["by_severity"] == {}


def test_alert_store_insert_returns_incrementing_ids(tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "alerts.db")
    id1 = store.insert_alert(_record(1, "dos", "critical"))
    id2 = store.insert_alert(_record(2, "benign", "low"))
    assert id1 == 1
    assert id2 == 2


def test_alert_store_fetch_recent_respects_limit_and_order(tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "alerts.db")
    for i in range(1, 6):
        store.insert_alert(_record(i, "dos", "high"))
    rows = store.fetch_recent(limit=3)
    assert len(rows) == 3
    assert rows[0]["id"] > rows[1]["id"] > rows[2]["id"]


def test_alert_store_aggregate_counts_by_label_and_severity(tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "alerts.db")
    store.insert_alert(_record(1, "dos", "critical"))
    store.insert_alert(_record(2, "dos", "high"))
    store.insert_alert(_record(3, "benign", "low"))
    agg = store.aggregate_counts()
    assert agg["by_label"]["dos"] == 2
    assert agg["by_label"]["benign"] == 1
    assert agg["by_severity"]["critical"] == 1
    assert agg["by_severity"]["high"] == 1
    assert agg["by_severity"]["low"] == 1


def test_alert_store_evidence_roundtrip(tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "alerts.db")
    payload = _record(1, "brute_force", "high")
    payload.evidence = {"mode": "ensemble", "confidence": 0.9}
    store.insert_alert(payload)
    rows = store.fetch_recent(limit=1)
    assert rows[0]["evidence"]["mode"] == "ensemble"
    assert rows[0]["evidence"]["confidence"] == 0.9

