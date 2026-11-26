"""FastAPI service around the trained model, called by the Fastify backend.

Start (from the project folder):
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request

from leescoach import config
from leescoach.data_cleaning import load_processed
from leescoach.features import StudentProfile
from leescoach.hybrid import HybridRecommender

from .mapping import feature_labels, to_history, to_output, to_student_profile
from .problems import ProblemError, register_problem_handlers
from .schemas import Health, ModelInfo, RecommendationRequest, RecommendationResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(config.MODEL_FILE)))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = HybridRecommender.load(MODEL_PATH) if MODEL_PATH.exists() else None
    app.state.demo = None
    if config.STUDENTS_CSV.exists() and config.INTERACTIONS_CSV.exists():
        _, students, interactions = load_processed()
        app.state.demo = (students, interactions)
    yield


app = FastAPI(
    title="Smart Reading Coach AI",
    version="1.0.0",
    description="Geeft 3 tot 5 onderbouwde leesadviezen op basis van leesprofiel en leesgeschiedenis.",
    lifespan=lifespan,
)
register_problem_handlers(app)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """When AI_API_KEY is set, only the backend that knows the key may call the model."""
    expected = os.getenv("AI_API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise ProblemError(401, "Ontbrekende of ongeldige API sleutel.")


def get_model(request: Request) -> HybridRecommender:
    model: HybridRecommender | None = request.app.state.model
    if model is None:
        raise ProblemError(503, "Het model is nog niet getraind. Voer eerst 'python -m leescoach.train' uit.")
    return model


Model = Annotated[HybridRecommender, Depends(get_model)]


@app.get("/api/health", response_model=Health)
def health(request: Request) -> Health:
    model: HybridRecommender | None = request.app.state.model
    return Health(status="ok" if model else "geen model", modelVersion=model.version if model else None)


@app.get("/api/model", response_model=ModelInfo, dependencies=[Depends(require_api_key)])
def model_info(model: Model) -> ModelInfo:
    """Transparency for teachers: which signals weigh most in the advice."""
    return ModelInfo(
        modelVersion=model.version,
        materials=len(model.materials),
        features=feature_labels(model),
        explanation=(
            "Een Random Forest voorspelt per materiaal de kans dat een student het leest, liket of bewaart. "
            "De getallen tonen hoe zwaar elk kenmerk meeweegt (samen 1)."
        ),
    )


@app.post("/api/recommendations", response_model=RecommendationResponse, dependencies=[Depends(require_api_key)])
def recommend(body: RecommendationRequest, model: Model) -> RecommendationResponse:
    profile = to_student_profile(body.profile)
    history = to_history(body.history, model)
    recommendations = model.recommend(profile, history, n=body.count)
    if len(recommendations) < 3:
        raise ProblemError(404, "Er is te weinig passend leesmateriaal gevonden voor dit profiel.")
    return RecommendationResponse(modelVersion=model.version, recommendations=[to_output(r) for r in recommendations])


@app.get(
    "/api/demo/students/{student_id}/recommendations",
    response_model=RecommendationResponse,
    dependencies=[Depends(require_api_key)],
)
def demo_recommend(student_id: str, request: Request, model: Model, count: int = 5) -> RecommendationResponse:
    """Advice for a student from the client dataset (S001 to S100), handy for demos and tests."""
    if request.app.state.demo is None:
        raise ProblemError(503, "De opgeschoonde dataset ontbreekt. Voer eerst de training uit.")
    students, interactions = request.app.state.demo
    match = students[students["student_id"] == student_id.upper()]
    if match.empty:
        raise ProblemError(404, f"Student {student_id} bestaat niet in de dataset.")
    own = interactions[interactions["student_id"] == student_id.upper()]
    history = dict(zip(own["cataloguscode"], own["voorkeurscore"].astype(float)))
    profile = StudentProfile.from_dataset_row(match.iloc[0])
    recommendations = model.recommend(profile, history, n=max(3, min(count, 5)))
    return RecommendationResponse(modelVersion=model.version, recommendations=[to_output(r) for r in recommendations])

