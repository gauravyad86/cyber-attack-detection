from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query

from cads.alerting.engine import AlertEngine
from cads.alerting.storage import AlertInsert, AlertStore
from cads.config import AppConfig, ensure_directories
from cads.inference.schemas import InferenceInput, InferenceMode
from cads.inference.service import InferenceService
from cads.logging_utils import setup_logging


@dataclass
class ServiceContainer:
    inference: InferenceService
    alerts: AlertEngine
    store: AlertStore


def _build_config() -> AppConfig:
    return AppConfig()


def create_app(config: AppConfig | None = None) -> FastAPI:
    setup_logging()
    app = FastAPI(title="Cyber Attack Detection API", version="0.1.0")
    cfg = config or _build_config()
    ensure_directories(cfg.paths)
    db_path = Path(os.getenv("CADS_DB_PATH", str(cfg.paths.artifacts_reports / "alerts.db")))
    services = ServiceContainer(
        inference=InferenceService.load(cfg),
        alerts=AlertEngine(),
        store=AlertStore(db_path),
    )
    app.state.services = services

    def get_services() -> ServiceContainer:
        return app.state.services

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(
        payload: InferenceInput,
        persist: bool = Query(default=True),
        mode: InferenceMode = Query(default="single"),
        svc: ServiceContainer = Depends(get_services),
    ) -> dict[str, object]:
        try:
            prediction = svc.inference.predict(payload, mode=mode)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        decision = svc.alerts.decide(payload, prediction)
        out = {
            "mode": mode,
            "prediction": prediction.model_dump(),
            "alert": {
                "severity": decision.severity,
                "score": round(decision.score, 6),
                "evidence": decision.evidence,
            },
        }

        if persist:
            alert_id = svc.store.insert_alert(
                AlertInsert(
                    timestamp=payload.timestamp.isoformat(),
                    src_ip=payload.src_ip,
                    dst_ip=payload.dst_ip,
                    predicted_label=prediction.predicted_label,
                    confidence=prediction.confidence,
                    severity=decision.severity,
                    score=decision.score,
                    evidence={**decision.evidence, "mode": mode, "model_breakdown": prediction.model_breakdown},
                )
            )
            out["alert"]["id"] = alert_id

        return out

    @app.get("/alerts")
    def alerts(limit: int = Query(default=50, ge=1, le=500), svc: ServiceContainer = Depends(get_services)) -> dict[str, object]:
        items = svc.store.fetch_recent(limit=limit)
        return {"items": items, "count": len(items)}

    @app.get("/alerts/metrics")
    def alert_metrics(svc: ServiceContainer = Depends(get_services)) -> dict[str, object]:
        return svc.store.aggregate_counts()

    return app


try:
    app = create_app()
except Exception as exc:  # pragma: no cover
    app = FastAPI(title="Cyber Attack Detection API", version="0.1.0")
    app.state.init_error = str(exc)

    @app.get("/health")
    def degraded_health() -> dict[str, str]:
        return {"status": "degraded", "reason": app.state.init_error}


def run() -> None:
    uvicorn.run("cads.api.app:app", host="0.0.0.0", port=8000, reload=False)
