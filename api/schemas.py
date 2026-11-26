"""Request and response models. Field names follow the JSON style of the Fastify app (camelCase)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LanguageLevel = Literal["2F", "3F", "3F+"]
MaterialType = Literal["boek", "online-artikel", "blogpost", "dichtbundel", "tijdschrift", "krant"]
DesiredLength = Literal["short", "medium", "long"]
ReadingGoal = Literal["enjoyment", "learn", "language", "work"]
HistoryStatus = Literal[
    "unread", "read", "afgewezen", "niet_afgemaakt", "bekeken", "opgeslagen", "gelezen", "geliket"
]


class Profile(BaseModel):
    """Same shape as ProfileData in smart-reading-coach-api/src/profile/profileOptions.ts."""

    model_config = ConfigDict(extra="ignore")

    languageLevel: LanguageLevel
    materialTypes: list[MaterialType] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list, max_length=10)
    desiredLength: DesiredLength = "medium"
    readingGoal: ReadingGoal = "enjoyment"
    language: str = "Nederlands"


class HistoryItem(BaseModel):
    """A material the student already interacted with. Identify it by cataloguscode or by app key."""

    cataloguscode: str | None = None
    key: str | None = None
    status: HistoryStatus = "read"
    rating: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def needs_identifier(self) -> "HistoryItem":
        if not self.cataloguscode and not self.key:
            raise ValueError("cataloguscode of key is verplicht")
        return self


class RecommendationRequest(BaseModel):
    profile: Profile
    history: list[HistoryItem] = Field(default_factory=list, max_length=500)
    count: int = Field(default=5, ge=3, le=5)


class RecommendationOut(BaseModel):
    cataloguscode: str
    key: str
    title: str
    author: str | None
    type: str
    genre: str
    level: str
    score: float
    reasons: list[str]
    summary: str


class RecommendationResponse(BaseModel):
    modelVersion: str
    recommendations: list[RecommendationOut]


class ModelInfo(BaseModel):
    modelVersion: str
    materials: int
    features: dict[str, float]
    explanation: str


class Health(BaseModel):
    status: Literal["ok", "geen model"]
    modelVersion: str | None
