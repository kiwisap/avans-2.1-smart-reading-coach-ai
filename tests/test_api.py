from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main


@pytest.fixture()
def client(model, monkeypatch, tmp_path):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    # Do not load models/hybride_model.joblib from disk: the tests use the freshly trained fixture model.
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "geen_model.joblib")
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
    assert all(item["reasons"] for item in body["recommendations"])


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
    monkeypatch.setenv("AI_API_KEY", "geheim")
    assert client.post("/api/recommendations", json={"profile": PROFILE}).status_code == 401
    ok = client.post("/api/recommendations", json={"profile": PROFILE}, headers={"X-API-Key": "geheim"})
    assert ok.status_code == 200
