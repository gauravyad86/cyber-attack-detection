from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  src_ip TEXT NOT NULL,
  dst_ip TEXT NOT NULL,
  predicted_label TEXT NOT NULL,
  confidence REAL NOT NULL,
  severity TEXT NOT NULL,
  score REAL NOT NULL,
  evidence_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class AlertInsert:
    timestamp: str
    src_ip: str
    dst_ip: str
    predicted_label: str
    confidence: float
    severity: str
    score: float
    evidence: dict[str, object]


class AlertStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(SCHEMA_SQL)
            conn.commit()

    def insert_alert(self, record: AlertInsert) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO alerts (
                  timestamp, src_ip, dst_ip, predicted_label, confidence, severity, score, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp,
                    record.src_ip,
                    record.dst_ip,
                    record.predicted_label,
                    float(record.confidence),
                    record.severity,
                    float(record.score),
                    json.dumps(record.evidence),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def fetch_recent(self, limit: int = 100) -> list[dict[str, object]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, timestamp, src_ip, dst_ip, predicted_label, confidence, severity, score, evidence_json
                FROM alerts
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out: list[dict[str, object]] = []
        for row in rows:
            out.append(
                {
                    "id": int(row["id"]),
                    "timestamp": str(row["timestamp"]),
                    "src_ip": str(row["src_ip"]),
                    "dst_ip": str(row["dst_ip"]),
                    "predicted_label": str(row["predicted_label"]),
                    "confidence": float(row["confidence"]),
                    "severity": str(row["severity"]),
                    "score": float(row["score"]),
                    "evidence": json.loads(row["evidence_json"]),
                }
            )
        return out

    def aggregate_counts(self) -> dict[str, dict[str, int]]:
        with sqlite3.connect(self.db_path) as conn:
            label_rows = conn.execute(
                "SELECT predicted_label, COUNT(*) FROM alerts GROUP BY predicted_label"
            ).fetchall()
            severity_rows = conn.execute(
                "SELECT severity, COUNT(*) FROM alerts GROUP BY severity"
            ).fetchall()
        return {
            "by_label": {str(label): int(count) for label, count in label_rows},
            "by_severity": {str(sev): int(count) for sev, count in severity_rows},
        }

