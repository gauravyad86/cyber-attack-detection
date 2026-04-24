from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class InferenceInput(BaseModel):
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int = Field(ge=0, le=65535)
    dst_port: int = Field(ge=0, le=65535)
    protocol: str
    packet_count: float = Field(gt=0)
    byte_count: float = Field(ge=0)
    duration: float = Field(gt=0)


class PredictionResult(BaseModel):
    predicted_label: str
    confidence: float
    model_name: str
    class_probabilities: dict[str, float]
    derived_features: dict[str, float | int]
    model_breakdown: dict[str, Any] = {}


InferenceMode = Literal["single", "compare_all", "ensemble"]


class AlertRecord(BaseModel):
    id: int
    timestamp: str
    src_ip: str
    dst_ip: str
    predicted_label: str
    confidence: float
    severity: str
    score: float
    evidence: dict[str, Any]
