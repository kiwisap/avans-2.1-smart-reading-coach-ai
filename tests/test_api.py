from __future__ import annotations

import copy
import threading

import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture()
def client(model, monkeypatch, tmp_path):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("MONGO_URL", raising=False)
    # Do not load models/hybrid_model.joblib from disk: the tests use the freshly trained fixture model.
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "no_model.joblib")
    with TestClient(main.app) as test_client:
        main.app.state.model = model
        yield test_client


PROFILE = {
    "languageLevel": "2F",
    "materialTypes": ["boek"],
    "topics": ["vriendschap", "humor"],
    "desiredLength": "medium",
    "readingGoal": "enjoyment",
}


def test_health(client) -> None:
    assert client.get("/api/health").json()["status"] == "ok"


def test_recommendations(client) -> None:
    response = client.post("/api/recommendations", json={"profile": PROFILE, "count": 3})
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 3
    assert all(item["reasons"] and item["motivation"] for item in body["recommendations"])


def test_history_by_app_key(client, model) -> None:
    key = model.materials["key"].iloc[0]
    response = client.post("/api/recommendations", json={"profile": PROFILE, "history": [{"key": key, "status": "read", "rating": 5}]})
    assert response.status_code == 200
    assert key not in {item["key"] for item in response.json()["recommendations"]}


def test_validation_error_is_problem_details(client) -> None:
    response = client.post("/api/recommendations", json={"profile": {**PROFILE, "languageLevel": "9F"}})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["title"] == "Ongeldige invoer"
    assert "languageLevel" in body["detail"]


def test_api_key_required_when_configured(client, monkeypatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "secret")
    assert client.post("/api/recommendations", json={"profile": PROFILE}).status_code == 401
    ok = client.post("/api/recommendations", json={"profile": PROFILE}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_unknown_material_types_are_ignored(client) -> None:
    profile = {**PROFILE, "materialTypes": ["boek", "hologram"]}
    response = client.post("/api/recommendations", json={"profile": profile, "count": 3})
    assert response.status_code == 200


def test_model_info_lists_the_learned_vocabulary(client, model) -> None:
    body = client.get("/api/model").json()
    assert body["materialTypes"] == model.vocabulary.types
    assert body["genres"] == model.vocabulary.genres


def test_retrain_swaps_in_the_new_model(client, model, monkeypatch) -> None:
    from api import training
    from reading_coach.train import TrainResult

    retrained = copy.copy(model)
    monkeypatch.setattr(training, "train", lambda *args, **kwargs: TrainResult(retrained))
    response = client.post("/api/admin/retrain")
    assert response.status_code == 202
    assert response.json() == {"status": "started"}
    client.portal.call(main.app.state.trainer.wait)
    assert main.app.state.model is retrained
    assert client.get("/api/health").json()["training"] is False


def test_retrain_twice_at_once_is_refused(client, monkeypatch) -> None:
    from api import training

    started = threading.Event()
    release = threading.Event()

    def slow_train(*args, **kwargs):
        started.set()
        release.wait(5)
        raise RuntimeError("stopped by the test")

    monkeypatch.setattr(training, "train", slow_train)
    assert client.post("/api/admin/retrain").status_code == 202
    started.wait(5)
    second = client.post("/api/admin/retrain")
    assert second.status_code == 409
    assert second.headers["content-type"].startswith("application/problem+json")
    assert client.get("/api/health").json()["training"] is True
    release.set()
    client.portal.call(main.app.state.trainer.wait)
    # A failed training keeps the model that was there.
    assert client.get("/api/health").json()["status"] == "ok"


def test_retrain_needs_the_api_key(client, monkeypatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "geheim")
    assert client.post("/api/admin/retrain").status_code == 401
