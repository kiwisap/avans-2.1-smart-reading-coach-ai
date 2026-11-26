"""Paths and model settings.

Nothing in here describes the catalog itself: material types, genres, languages, themes and
the weight of each interaction type are learned from the data (see vocabulary.py), so new
books or new kinds of material work without changing the code.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_DATASET = RAW_DIR / "leescatalogus_recommender.xlsx"
MATERIALS_CSV = PROCESSED_DIR / "materials.csv"
STUDENTS_CSV = PROCESSED_DIR / "students.csv"
INTERACTIONS_CSV = PROCESSED_DIR / "interactions.csv"
MODEL_FILE = MODELS_DIR / "hybrid_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"

# Sheet names in the dataset of the client.
SHEET_MATERIALS = "Leesmateriaal"
SHEET_STUDENTS = "Studenten"
SHEET_INTERACTIONS = "Interacties"

RANDOM_STATE = 42

# An interaction counts as positive when the average rating of its type (learned from the data)
# is at least this high. On a scale of 1 to 5, 3 is neutral.
POSITIVE_SCORE = 3.5

# Neutral point of the preference scale (1 to 5). Collaborative filtering centres around it.
NEUTRAL_SCORE = 3.0

# CEFR levels (relevant for NT2 students) on the scale of the Dutch reference levels (1F to 4F).
# This is the published alignment of both frameworks, not something from the catalog.
CEFR_TO_LEVEL: dict[str, float] = {"A1": 1.0, "A2": 1.5, "B1": 2.0, "B2": 3.0, "C1": 3.5, "C2": 4.0}
