"""Everything the model needs to know about the words in the catalog, learned from the data.

The catalog grows: new books, maybe a new kind of material or a new genre. So none of these
values are written in the code. Material types, their labels, genres, languages and the
weight of each interaction type all come from the cleaned data, and the trained model keeps
its own copy (Vocabulary) so the API uses exactly what the model was trained on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

LEVEL_PATTERN = re.compile(r"^([1-9])F(\+?)$")


def parse_level(label: object) -> float:
    """Dutch reference level as a number: 2F is 2, 3F+ is 3.5, a range like 2F-3F is the middle.

    Works for every level of the framework, also ones that are not in the catalog yet.
    Returns NaN for anything that is not a level.
    """
    if label is None or pd.isna(label):
        return np.nan
    parts = [part.strip() for part in str(label).upper().replace(" ", "").split("-") if part.strip()]
    values = []
    for part in parts:
        match = LEVEL_PATTERN.match(part)
        if not match:
            return np.nan
        values.append(float(match.group(1)) + (0.5 if match.group(2) else 0.0))
    return float(np.mean(values)) if values else np.nan


def parse_cefr(label: object) -> float:
    if label is None or pd.isna(label):
        return np.nan
    return config.CEFR_TO_LEVEL.get(str(label).strip().upper(), np.nan)


# Material types

def _base_type(raw: object) -> tuple[str | None, str | None]:
    """'Boek - thriller' gives ('boek', 'thriller'), 'krant (papier)' gives ('krant', None)."""
    if raw is None or pd.isna(raw) or not str(raw).strip():
        return None, None
    text = re.sub(r"\s+", " ", str(raw).strip().lower())
    base, _, subgenre = text.partition(" - ")
    base = re.sub(r"\([^)]*\)", "", base).strip()
    return base, (subgenre.strip() or None)


def type_mapping(raw_types: pd.Series) -> dict[str, tuple[str, str, str | None]]:
    """Maps every raw type value to (type, label, subgenre), using only the values in the data.

    A qualifier in front of a type that also exists on its own is dropped: when the data has
    'krant (papier)', then 'digitale krant' is a 'krant' as well. The type itself is the label
    with dashes, the same as in the Fastify app ('online artikel' becomes 'online-artikel').
    """
    bases = {raw: _base_type(raw) for raw in raw_types.dropna().unique()}
    known = {base for base, _ in bases.values() if base}
    mapping: dict[str, tuple[str, str, str | None]] = {}
    for raw, (base, subgenre) in bases.items():
        if not base:
            continue
        words = base.split(" ")
        if len(words) > 1 and words[-1] in known:
            base = words[-1]
        mapping[raw] = (base.replace(" ", "-"), base, subgenre)
    return mapping


# Languages

def _is_subsequence(short: str, long: str) -> bool:
    letters = iter(long)
    return all(letter in letters for letter in short)


def language_mapping(values: pd.Series) -> dict[str, str]:
    """Maps spellings like 'NL', 'NED' and 'nederlands' to the full name used most in the data.

    Full names are the values of at least four letters. A short value belongs to the full name
    that starts with the same letter and contains its letters in the same order.
    """
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    full_names = cleaned[cleaned.str.len() >= 4].str.lower().value_counts()
    mapping: dict[str, str] = {}
    for value in cleaned.unique():
        lower = value.lower()
        if lower in full_names.index:
            mapping[value] = lower.capitalize()
            continue
        matches = [name for name in full_names.index if name[0] == lower[0] and _is_subsequence(lower, name)]
        mapping[value] = matches[0].capitalize() if matches else value.capitalize()
    return mapping


# Interactions

def learn_interaction_scores(interactions: pd.DataFrame) -> dict[str, float]:
    """How positive each interaction type is: the average rating students gave with it (1 to 5).

    In the data a 'geliket' comes with a rating of 5 and an 'afgewezen' with about 2, so the
    data itself says how strong each signal is. A type without any rating is neutral.
    """
    ratings = interactions.groupby("interaction")["rating"].mean()
    scores: dict[str, float] = {}
    for kind in interactions["interaction"].dropna().unique():
        value = ratings.get(kind)
        scores[str(kind)] = float(value) if value is not None and pd.notna(value) else config.NEUTRAL_SCORE
    return scores


def positive_interactions(scores: dict[str, float]) -> set[str]:
    return {kind for kind, score in scores.items() if score >= config.POSITIVE_SCORE}


@dataclass
class Vocabulary:
    """The catalog words the model was trained with."""

    types: list[str] = field(default_factory=list)
    type_labels: dict[str, str] = field(default_factory=dict)
    genres: list[str] = field(default_factory=list)
    genre_by_type: dict[str, str] = field(default_factory=dict)
    languages: list[str] = field(default_factory=list)
    default_language: str = ""
    interaction_scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def learn(cls, materials: pd.DataFrame, interactions: pd.DataFrame) -> "Vocabulary":
        genre_by_type = (
            materials.dropna(subset=["genre"]).groupby("type")["genre"].agg(lambda s: s.mode().iat[0]).to_dict()
        )
        labels = materials.dropna(subset=["type_label"]).groupby("type")["type_label"].agg(lambda s: s.mode().iat[0])
        return cls(
            types=sorted(materials["type"].dropna().unique()),
            type_labels=labels.to_dict(),
            genres=sorted(materials["genre"].dropna().unique()),
            genre_by_type=genre_by_type,
            languages=sorted(materials["language"].dropna().unique()),
            default_language=str(materials["language"].mode().iat[0]),
            interaction_scores=learn_interaction_scores(interactions),
        )

    @property
    def positive_interactions(self) -> set[str]:
        return positive_interactions(self.interaction_scores)

    def label(self, material_type: str) -> str:
        return self.type_labels.get(material_type, material_type.replace("-", " "))

    def related_types(self, material_type: str) -> list[str]:
        """Types with the same dominant genre, for a student who prefers 'online artikel': news
        papers, magazines and blogs are informative texts as well."""
        genre = self.genre_by_type.get(material_type)
        if genre is None:
            return [material_type] if material_type in self.types else []
        return [kind for kind in self.types if self.genre_by_type.get(kind) == genre]

    def interaction_score(self, interaction: str) -> float:
        return self.interaction_scores.get(interaction, config.NEUTRAL_SCORE)
