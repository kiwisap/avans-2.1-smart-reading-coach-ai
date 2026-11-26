"""Maps the profile and reading list of the app to the model input, and the advice back to JSON.

The mappings below translate the words of the app into the words of the model. They are part
of the contract between the two, not of the catalog: which genres and types exist still comes
from the data, and a value the model does not know simply counts as 'no preference'.
"""

from __future__ import annotations

from reading_coach.features import FEATURE_LABELS, StudentProfile
from reading_coach.hybrid import HybridRecommender, Recommendation

from .schemas import HistoryItem, Profile, RecommendationOut

# The reading list of the app knows 'unread' (saved for later) and 'read'. These are the
# interaction types in the dataset that mean the same.
STATUS_TO_INTERACTION = {"unread": "opgeslagen", "read": "gelezen"}

# How long a text may take, per answer in the reading profile. None means no limit.
DESIRED_LENGTH_TO_MINUTES: dict[str, int | None] = {"short": 20, "medium": 90, "long": None}

# The reading goal of the app points to the kind of text that suits it: reading for fun
# fits stories, learning and work fit informative texts. For language practice anything goes.
READING_GOAL_TO_GENRE: dict[str, str | None] = {
    "enjoyment": "fictie",
    "learn": "informatief",
    "language": None,
    "work": "informatief",
}


def to_student_profile(profile: Profile, model: HybridRecommender) -> StudentProfile:
    vocabulary = model.vocabulary
    types = [kind for kind in dict.fromkeys(profile.materialTypes) if kind in vocabulary.types]
    genre = READING_GOAL_TO_GENRE.get(profile.readingGoal)
    return StudentProfile(
        reading_level=profile.languageLevel,
        interests=[topic.strip().lower() for topic in profile.topics if topic.strip()],
        material_types=types,
        preferred_genre=genre if genre in vocabulary.genres else None,
        max_minutes=DESIRED_LENGTH_TO_MINUTES[profile.desiredLength],
        preferred_language=profile.language or vocabulary.default_language,
    )


def to_history(items: list[HistoryItem], model: HybridRecommender) -> dict[str, float]:
    """Preference score per catalog code, on the same scale (1 to 5) as during training."""
    code_by_key = dict(zip(model.materials["key"], model.materials["catalog_code"]))
    history: dict[str, float] = {}
    for item in items:
        code = item.catalogCode if item.catalogCode in model.content.index else code_by_key.get(item.key or "")
        if not code:
            continue
        interaction = STATUS_TO_INTERACTION.get(item.status, item.status)
        type_score = model.vocabulary.interaction_score(interaction)
        history[code] = (type_score + item.rating) / 2 if item.rating else type_score
    return history


def to_output(recommendation: Recommendation) -> RecommendationOut:
    return RecommendationOut(
        catalogCode=recommendation.catalog_code,
        key=recommendation.key,
        title=recommendation.title,
        author=recommendation.author,
        type=recommendation.type,
        genre=recommendation.genre,
        level=recommendation.level,
        score=recommendation.score,
        reasons=recommendation.reasons,
        motivation=recommendation.motivation,
    )


def feature_labels(model: HybridRecommender) -> dict[str, float]:
    return {FEATURE_LABELS[name]: round(float(value), 4) for name, value in model.feature_importance.items()}
