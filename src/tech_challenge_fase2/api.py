"""API stateless preparada para múltiplas réplicas e autoscaling."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from .explainability import prediction_context
from .llm import ExplanationProvider, provider_from_environment
from .observability import configure_logging


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = PROJECT_ROOT / "artifacts" / "models" / "logistic_regression_best.joblib"
REQUEST_COUNT = Counter("diagnosis_requests_total", "Total de predições", ["status"])
REQUEST_LATENCY = Histogram("diagnosis_request_seconds", "Latência da predição")
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(min_length=1)
    include_explanation: bool = True


class PredictionResponse(BaseModel):
    predicted_class: str
    malignant_probability: float
    model_name: str
    explanation_provider: str | None = None
    explanation: dict[str, Any] | None = None


def _load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = joblib.load(path)
    required = {"model", "feature_names", "model_name"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"Artefato inválido: {path}.")
    return payload


def create_app(
    *,
    artifact_path: str | Path | None = None,
    explanation_provider: ExplanationProvider | None = None,
) -> FastAPI:
    configure_logging()
    configured_path = Path(
        artifact_path or os.getenv("MODEL_ARTIFACT", str(DEFAULT_ARTIFACT))
    ).resolve()
    artifact = _load_artifact(configured_path)
    provider = explanation_provider or provider_from_environment()

    application = FastAPI(
        title="Tech Challenge Fase 2 - Diagnóstico",
        version="0.1.0",
        description="API educacional; não substitui diagnóstico médico.",
    )
    application.mount("/metrics", make_asgi_app())

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    @application.get("/ready")
    def ready() -> dict[str, Any]:
        if artifact is None:
            raise HTTPException(status_code=503, detail="model_not_ready")
        return {
            "status": "ready",
            "artifact": str(configured_path),
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        started = time.perf_counter()
        if artifact is None:
            REQUEST_COUNT.labels(status="unavailable").inc()
            raise HTTPException(
                status_code=503,
                detail="Modelo ainda não treinado. Execute scripts/run_experiments.py.",
            )
        expected = list(artifact["feature_names"])
        missing = sorted(set(expected) - set(request.features))
        extra = sorted(set(request.features) - set(expected))
        if missing or extra:
            REQUEST_COUNT.labels(status="invalid").inc()
            raise HTTPException(
                status_code=422,
                detail={"missing_features": missing, "extra_features": extra},
            )
        row = pd.DataFrame([[request.features[name] for name in expected]], columns=expected)
        context = prediction_context(
            artifact["model"],
            row,
            model_name=artifact["model_name"],
        )
        explanation = provider.explain(context) if request.include_explanation else None
        duration = time.perf_counter() - started
        REQUEST_LATENCY.observe(duration)
        REQUEST_COUNT.labels(status="ok").inc()
        logger.info(
            "prediction_completed",
            extra={"event": "prediction_completed", "duration_ms": round(duration * 1000, 2)},
        )
        return PredictionResponse(
            predicted_class=context.predicted_class,
            malignant_probability=context.malignant_probability,
            model_name=context.model_name,
            explanation_provider=provider.name if explanation else None,
            explanation=explanation.model_dump() if explanation else None,
        )

    return application


app = create_app()
