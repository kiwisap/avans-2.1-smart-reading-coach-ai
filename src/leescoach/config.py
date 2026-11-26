"""Central configuration: paths, vocabularies and mappings used across the project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_DATASET = RAW_DIR / "leescatalogus_recommender.xlsx"
MATERIALS_CSV = PROCESSED_DIR / "materialen.csv"
STUDENTS_CSV = PROCESSED_DIR / "studenten.csv"
INTERACTIONS_CSV = PROCESSED_DIR / "interacties.csv"
MODEL_FILE = MODELS_DIR / "hybride_model.joblib"

SHEET_MATERIALS = "Leesmateriaal"
SHEET_STUDENTS = "Studenten"
SHEET_INTERACTIONS = "Interacties"

RANDOM_STATE = 42

# Reading levels as numbers so differences can be computed. "2F-3F" sits in between.
LEVEL_TO_NUMBER: dict[str, float] = {
    "1F": 1.0,
    "2F": 2.0,
    "2F-3F": 2.5,
    "3F": 3.0,
    "3F+": 3.5,
    "4F": 4.0,
}

# CEFR language level (relevant for NT2 students) translated to the same scale.
CEFR_TO_NUMBER: dict[str, float] = {
    "A1": 1.0,
    "A2": 1.5,
    "B1": 2.0,
    "B2": 3.0,
    "C1": 3.5,
    "C2": 4.0,
}

# Raw "Type materiaal" values mapped to the material types used by the Fastify app
# (see smart-reading-coach-api/src/profile/profileOptions.ts) plus an optional subgenre.
MATERIAL_TYPE_MAP: dict[str, tuple[str, str | None]] = {
    "boek": ("boek", None),
    "boek - thriller": ("boek", "thriller"),
    "boek - roman": ("boek", "roman"),
    "online artikel": ("online-artikel", None),
    "blogpost": ("blogpost", None),
    "dichtbundel": ("dichtbundel", None),
    "tijdschrift": ("tijdschrift", None),
    "krant (papier)": ("krant", None),
    "digitale krant": ("krant", None),
}
APP_MATERIAL_TYPES = ["boek", "online-artikel", "blogpost", "dichtbundel", "tijdschrift", "krant"]

# Student preference "voorkeur_materiaaltype" mapped to the app material types it covers.
STUDENT_TYPE_PREFERENCE: dict[str, list[str]] = {
    "boek": ["boek"],
    "online artikel": ["online-artikel", "blogpost", "krant", "tijdschrift"],
    "gemengd": list(APP_MATERIAL_TYPES),
}

# When the genre is missing it can be derived from the material type.
GENRE_BY_TYPE: dict[str, str] = {
    "boek": "fictie",
    "dichtbundel": "poëzie",
    "online-artikel": "informatief",
    "blogpost": "informatief",
    "krant": "informatief",
    "tijdschrift": "informatief",
}
GENRES = ["fictie", "informatief", "poëzie"]

LANGUAGE_MAP: dict[str, str] = {
    "nederlands": "Nederlands",
    "nl": "Nederlands",
    "ned": "Nederlands",
    "engels": "Engels",
    "en": "Engels",
}

# Interaction types ordered from negative to positive with an implicit score (1 to 5).
INTERACTION_SCORE: dict[str, float] = {
    "afgewezen": 1.0,
    "niet_afgemaakt": 2.0,
    "bekeken": 3.0,
    "opgeslagen": 4.0,
    "gelezen": 4.0,
    "geliket": 5.0,
}
POSITIVE_INTERACTIONS = {"gelezen", "geliket", "opgeslagen"}

# Student interests use other words than the catalog. Each interest is expanded with
# words that appear in the descriptions, themes and urls so that text matching works.
INTEREST_SYNONYMS: dict[str, list[str]] = {
    "kunstmatige intelligentie": ["ai", "algoritmes", "digitalisering", "technologie", "privacy"],
    "technologie": ["tech", "digitaal", "digitale", "digitalisering", "online", "cybercrime", "ai"],
    "relaties": ["liefde", "relatie", "relaties", "romantisch", "partner", "familie"],
    "vriendschap": ["vriendschap", "vrienden", "verbondenheid", "groepsdruk"],
    "reizen": ["reizen", "reis", "reisverhalen", "roadtrip", "wereld"],
    "persoonlijke ontwikkeling": ["groei", "doorzetten", "zelfvertrouwen", "identiteit", "opgroeien", "veerkracht", "ambitie", "zelfreflectie"],
    "misdaad": ["misdaad", "criminaliteit", "jeugdcriminaliteit", "moord", "detective", "thriller", "cybercrime"],
    "spanning": ["spanning", "spannend", "spannende", "thriller", "mysterie", "geheimen"],
    "geschiedenis": ["geschiedenis", "oorlog", "woii", "wereldoorlog", "verleden", "onderduik"],
    "maatschappij": ["samenleving", "maatschappelijke", "burgerschap", "rechtsstaat", "democratie", "economie", "ongelijkheid"],
    "politiek": ["politiek", "overheid", "kabinet", "minister", "democratie", "wetgeving", "verordening"],
    "gezondheid": ["gezondheid", "zorg", "ziekte", "welzijn", "eten", "fysiotherapie"],
    "klimaat": ["klimaat", "duurzaam", "energie", "biodiversiteit", "natuur"],
    "natuur": ["natuur", "biodiversiteit", "dieren", "tuin", "nachtvlinders"],
    "cultuur": ["cultuur", "kunst", "migratie", "creativiteit", "taal", "identiteit"],
    "muziek": ["muziek", "band", "zingen"],
    "sport": ["sport", "voetbal", "training"],
    "humor": ["humor", "humoristische", "grappig"],
}

# Mapping from the Fastify app profile (profileOptions.ts) to the model profile.
DESIRED_LENGTH_TO_MINUTES: dict[str, int | None] = {"short": 20, "medium": 90, "long": None}
READING_GOAL_TO_GENRE: dict[str, str] = {
    "enjoyment": "fictie",
    "learn": "informatief",
    "language": "geen voorkeur",
    "work": "informatief",
}
