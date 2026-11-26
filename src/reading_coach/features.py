"""The student profile and the features that describe how well one material fits one student."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .vocabulary import Vocabulary, parse_level

FEATURE_COLUMNS = [
    "level_diff",
    "level_distance",
    "type_match",
    "genre_match",
    "interest_sim",
    "interest_hits",
    "history_sim",
    "time_ratio_log",
    "fits_time",
    "language_match",
    "avg_rating",
    "popularity",
    "cf_knn",
    "complexity_score",
]

# Dutch names for plots and for the teacher view.
FEATURE_LABELS = {
    "level_diff": "Niveauverschil (materiaal min student)",
    "level_distance": "Afstand tot eigen niveau",
    "type_match": "Past bij voorkeur materiaaltype",
    "genre_match": "Past bij voorkeur genre",
    "interest_sim": "Tekstovereenkomst met interesses",
    "interest_hits": "Aantal interesses in de tekst",
    "history_sim": "Lijkt op eerder gewaardeerd materiaal",
    "time_ratio_log": "Leestijd ten opzichte van maximum (log)",
    "fits_time": "Binnen maximale leestijd",
    "language_match": "Juiste taal",
    "avg_rating": "Gemiddelde beoordeling",
    "popularity": "Populair bij andere studenten",
    "cf_knn": "Gewaardeerd door studenten met dezelfde smaak",
    "complexity_score": "Leescomplexiteit",
}

DEFAULT_LEVEL = 2.0


@dataclass
class StudentProfile:
    """What the model needs to know about a student. Comes from the dataset or from the app.

    Empty preferences mean 'no preference': no type list, no genre, no language.
    """

    reading_level: str
    interests: list[str]
    material_types: list[str] = field(default_factory=list)
    preferred_genre: str | None = None
    max_minutes: float | None = None
    preferred_language: str | None = None
    student_id: str | None = None

    @property
    def level_num(self) -> float:
        value = parse_level(self.reading_level)
        return DEFAULT_LEVEL if np.isnan(value) else value

    @classmethod
    def from_dataset_row(cls, row: pd.Series, vocabulary: Vocabulary) -> "StudentProfile":
        interests = [value for value in str(row["interests"]).split(";") if value]
        # The preferred type of a dataset student, like 'online artikel', also covers the types
        # with the same dominant genre. A value that is not a type ('gemengd') means no preference.
        preferred = str(row["preferred_type"]).strip().lower().replace(" ", "-")
        types = vocabulary.related_types(preferred) if preferred in vocabulary.types else []
        return cls(
            student_id=str(row["student_id"]),
            reading_level=str(row["reading_level"]),
            interests=interests,
            material_types=types,
            preferred_genre=str(row["preferred_genre"]).lower(),
            max_minutes=float(row["max_minutes"]) if pd.notna(row["max_minutes"]) else None,
            preferred_language=str(row["preferred_language"]),
        )


def static_features(profile: StudentProfile, materials: pd.DataFrame, vocabulary: Vocabulary) -> pd.DataFrame:
    """Features that only depend on the profile and the metadata of the materials (no interactions)."""
    level_difference = materials["level_num"].to_numpy() - profile.level_num
    types = set(profile.material_types) & set(vocabulary.types)
    if 0 < len(types) < len(vocabulary.types):
        type_match = materials["type"].isin(types).astype(float).to_numpy()
    else:
        type_match = np.full(len(materials), 0.5)
    if profile.preferred_genre in vocabulary.genres:
        genre_match = (materials["genre"] == profile.preferred_genre).astype(float).to_numpy()
    else:
        genre_match = np.full(len(materials), 0.5)
    minutes = materials["reading_minutes"].to_numpy(dtype=float)
    if profile.max_minutes:
        ratio = minutes / profile.max_minutes
        fits = (minutes <= profile.max_minutes).astype(float)
    else:
        ratio = np.ones(len(materials))
        fits = np.ones(len(materials))
    language = (profile.preferred_language or vocabulary.default_language).lower()
    rating = materials["avg_rating"].astype(float)
    return pd.DataFrame(
        {
            "level_diff": level_difference,
            "level_distance": np.abs(level_difference),
            "type_match": type_match,
            "genre_match": genre_match,
            "time_ratio_log": np.log1p(ratio),
            "fits_time": fits,
            "language_match": (materials["language"].str.lower() == language).astype(float).to_numpy(),
            "avg_rating": rating.fillna(rating.mean()).to_numpy(),
            "complexity_score": materials["complexity_score"].astype(float).to_numpy(),
        },
        index=materials["catalog_code"].to_numpy(),
    )
