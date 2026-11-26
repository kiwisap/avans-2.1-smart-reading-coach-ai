"""Request and response models. Field names follow the JSON style of the Fastify app (camelCase).

Material types and topics are plain strings: the catalog decides which ones exist, the model
knows them from its training data and ignores values it does not know.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DesiredLength = Literal["short", "medium", "long"]
ReadingGoal = Literal["enjoyment", "learn", "language", "work"]


class Profile(BaseModel):
    """Same shape as ProfileData in smart-reading-coach-api/src/profile/profileOptions.ts."""

    model_config = ConfigDict(extra="ignore")

    # A Dutch reference level such as 2F, 3F+ or 2F-3F.
    languageLevel: str = Field(pattern=r"^[1-4]F\+?(-[1-4]F\+?)?$", max_length=10)
    materialTypes: list[str] = Field(default_factory=list, max_length=20)
    topics: list[str] = Field(default_factory=list, max_length=10)
    desiredLength: DesiredLength = "medium"
    readingGoal: ReadingGoal = "enjoyment"
    language: str | None = Field(default=None, max_length=40)


class HistoryItem(BaseModel):
    """A material the student already interacted with. Identified by catalog code or by app key.

    The status is 'read' or 'unread' from the app, or an interaction type from the dataset.
    """

    catalogCode: str | None = None
    key: str | None = None
    status: str = Field(default="read", max_length=40)
    rating: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def needs_identifier(self) -> "HistoryItem":
        if not self.catalogCode and not self.key:
            raise ValueError("catalogCode of key is verplicht")
        return self


class RecommendationRequest(BaseModel):
    profile: Profile
    history: list[HistoryItem] = Field(default_factory=list, max_length=500)
    count: int = Field(default=5, ge=3, le=5)


class RecommendationOut(BaseModel):
    catalogCode: str
    key: str
    title: str
    author: str | None
    type: str
    genre: str
    level: str
    score: float
    reasons: list[str]
    motivation: str


class RecommendationResponse(BaseModel):
    modelVersion: str
    recommendations: list[RecommendationOut]


class ModelInfo(BaseModel):
    modelVersion: str
    materials: int
    materialTypes: list[str]
    genres: list[str]
    features: dict[str, float]
    explanation: str


class Health(BaseModel):
    status: Literal["ok", "no_model"]
    modelVersion: str | None
    training: bool = False


class RetrainStarted(BaseModel):
    status: Literal["started"]
