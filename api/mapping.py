"""Maps the app profile and reading list to the model input, and model output back to JSON."""

from __future__ import annotations

from leescoach import config
from leescoach.features import FEATURE_LABELS, StudentProfile
from leescoach.hybrid import HybridRecommender, Recommendation

from .schemas import HistoryItem, Profile, RecommendationOut

# The app reading list only knows 'unread' (saved) and 'read'.
STATUS_TO_INTERACTION = {"unread": "opgeslagen", "read": "gelezen"}


def to_student_profile(profile: Profile) -> StudentProfile:
    types = list(dict.fromkeys(profile.materialTypes)) or list(config.APP_MATERIAL_TYPES)
    return StudentProfile(
        leesniveau=profile.languageLevel,
        interesses=[topic.strip().lower() for topic in profile.topics if topic.strip()],
        materiaaltypes=types,
        voorkeur_genre=config.READING_GOAL_TO_GENRE[profile.readingGoal],
        max_leestijd_min=config.DESIRED_LENGTH_TO_MINUTES[profile.desiredLength],
        voorkeur_taal=profile.language,
    )


def to_history(items: list[HistoryItem], model: HybridRecommender) -> dict[str, float]:
    """Preference score per cataloguscode, the same scale (1 to 5) as used during training."""
    code_by_key = dict(zip(model.materials["key"], model.materials["cataloguscode"]))
    history: dict[str, float] = {}
    for item in items:
        code = item.cataloguscode if item.cataloguscode in model.content.index else code_by_key.get(item.key or "")
        if not code:
            continue
        interaction = STATUS_TO_INTERACTION.get(item.status, item.status)
        type_score = config.INTERACTION_SCORE[interaction]
        history[code] = (type_score + item.rating) / 2 if item.rating else type_score
    return history


def to_output(recommendation: Recommendation) -> RecommendationOut:
    return RecommendationOut(
        cataloguscode=recommendation.cataloguscode,
        key=recommendation.key,
        title=recommendation.titel,
        author=recommendation.auteur,
        type=recommendation.type,
        genre=recommendation.genre,
        level=recommendation.niveau,
        score=recommendation.score,
        reasons=recommendation.redenen,
        summary=recommendation.samenvatting,
    )


def feature_labels(model: HybridRecommender) -> dict[str, float]:
    return {FEATURE_LABELS[name]: round(float(value), 4) for name, value in model.feature_importance.items()}
