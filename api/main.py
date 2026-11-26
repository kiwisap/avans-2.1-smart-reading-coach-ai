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

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, Request

from reading_coach import config
from reading_coach.data_cleaning import load_processed
from reading_coach.hybrid import HybridRecommender
from reading_coach.sources import MongoSettings

from .mapping import feature_labels, to_history, to_output, to_student_profile
from .problems import ProblemError, register_problem_handlers
from .schemas import Health, ModelInfo, RecommendationRequest, RecommendationResponse, RetrainStarted
from .training import Trainer

# Settings from .env in the project folder, when it is there. Real environment variables win.
load_dotenv(config.PROJECT_ROOT / ".env")

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(config.MODEL_FILE)))


def load_demo() -> tuple | None:
    if config.STUDENTS_CSV.exists() and config.INTERACTIONS_CSV.exists():
        _, students, interactions = load_processed()
        return students, interactions
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = HybridRecommender.load(MODEL_PATH) if MODEL_PATH.exists() else None
    app.state.demo = load_demo()

    def use_model(model: HybridRecommender) -> None:
        app.state.model = model
        app.state.demo = load_demo()

    app.state.trainer = Trainer(model_path=MODEL_PATH, on_model=use_model)
    # The model in the image was trained at build time. With MongoDB the catalog may have changed
    # since, so the service trains once more on what the backend imported. Until that is done,
    # the model from the build answers.
    if MongoSettings.from_env() and os.getenv("TRAIN_ON_START", "true").lower() != "false":
        app.state.trainer.start("startup, catalog from MongoDB")
    yield
    await app.state.trainer.wait()


app = FastAPI(
    title="Smart Reading Coach AI",
    version="2.1.0",
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
        raise ProblemError(503, "Het model is nog niet getraind. Voer eerst 'python -m reading_coach.train' uit.")
    return model


Model = Annotated[HybridRecommender, Depends(get_model)]


@app.get("/api/health", response_model=Health)
def health(request: Request) -> Health:
    model: HybridRecommender | None = request.app.state.model
    return Health(
        status="ok" if model else "no_model",
        modelVersion=model.version if model else None,
        training=request.app.state.trainer.running,
    )


@app.post(
    "/api/admin/retrain",
    response_model=RetrainStarted,
    status_code=202,
    dependencies=[Depends(require_api_key)],
)
async def retrain(request: Request) -> RetrainStarted:
    """Trains a new model on the current catalog, after the backend imported a new one.

    The answer comes right away; the current model keeps answering until the new one is ready.
    """
    if not request.app.state.trainer.start("requested by the backend"):
        raise ProblemError(409, "Er wordt al een nieuw model getraind. Probeer het straks opnieuw.")
    return RetrainStarted(status="started")


@app.get("/api/model", response_model=ModelInfo, dependencies=[Depends(require_api_key)])
def model_info(model: Model) -> ModelInfo:
    """Transparency for teachers: which signals weigh most in the advice."""
    return ModelInfo(
        modelVersion=model.version,
        materials=len(model.materials),
        materialTypes=model.vocabulary.types,
        genres=model.vocabulary.genres,
        features=feature_labels(model),
        explanation=(
            "Een Random Forest voorspelt per materiaal de kans dat een student het leest, liket of bewaart. "
            "De getallen tonen hoe zwaar elk kenmerk meeweegt (samen 1)."
        ),
    )


@app.post("/api/recommendations", response_model=RecommendationResponse, dependencies=[Depends(require_api_key)])
def recommend(body: RecommendationRequest, model: Model) -> RecommendationResponse:
    profile = to_student_profile(body.profile, model)
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
    history = dict(zip(own["catalog_code"], own["preference_score"].astype(float)))
    profile = model.profile_of(match.iloc[0])
    recommendations = model.recommend(profile, history, n=max(3, min(count, 5)))
    return RecommendationResponse(modelVersion=model.version, recommendations=[to_output(r) for r in recommendations])

