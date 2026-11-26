"""Loading and cleaning the dataset of the client (sheets Leesmateriaal, Studenten, Interacties).

Every step is a small function, so the notebooks can show them one by one, while clean_all
runs the whole pipeline for training and for the API. The raw sheets have Dutch column
names; they are renamed to English here, once, so the rest of the code is English.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .vocabulary import (
    language_mapping,
    learn_interaction_scores,
    parse_cefr,
    parse_level,
    positive_interactions,
    type_mapping,
)

# Raw column names of the client sheets and their name in this project.
MATERIAL_COLUMNS = {
    "Titel": "title",
    "Auteur": "author",
    "Korte omschrijving": "description",
    "Type materiaal": "raw_type",
    "Niveau": "level",
    "Themas_boek": "book_themes",
    "materiaal_id": "material_id",
    "genre": "genre",
    "genre_kopie": "genre_copy",
    "onderwerp": "subject",
    "trefwoorden": "keywords",
    "taal": "language",
    "aantal_paginas": "pages",
    "geschatte_leestijd_min": "reading_minutes",
    "bron_url": "source_url",
    "doelgroep": "target_group",
    "doelgroep_extra": "target_group_extra",
    "woordenaantal_omschrijving": "description_words",
    "gem_woordlengte": "avg_word_length",
    "gem_zinslengte_woorden": "avg_sentence_length",
    "aandeel_lange_woorden_pct": "long_words_pct",
    "aantal_zinnen": "sentence_count",
    "leescomplexiteit_score": "complexity_score",
    "interessecluster": "interest_cluster",
    "publicatiejaar": "publication_year",
    "beschikbaar_formaat": "format",
    "cataloguscode": "catalog_code",
    "Gemiddelde beoordeling": "avg_rating",
}
# Administrative columns that say nothing about the material.
ADMIN_COLUMNS = ["Unnamed: 3", "interne_batchcode", "administratieve_prioriteit", "gegevensstatus", "uitgever"]

STUDENT_COLUMNS = {
    "student_id": "student_id",
    "leeftijdscategorie": "age_group",
    "leesniveau": "reading_level",
    "taalniveau": "cefr_level",
    "interesses": "interests",
    "voorkeur_materiaaltype": "preferred_type",
    "voorkeur_genre": "preferred_genre",
    "max_leestijd_min": "max_minutes",
    "voorkeur_taal": "preferred_language",
}

INTERACTION_COLUMNS = {
    "interactie_id": "interaction_id",
    "student_id": "student_id",
    "cataloguscode": "catalog_code",
    "interactietype": "interaction",
    "beoordeling": "rating",
    "leespercentage": "read_pct",
    "leestijd_min": "minutes_read",
    "datum": "date",
}

CURRENT_YEAR = pd.Timestamp.now().year
MAX_PLAUSIBLE_WORD_LENGTH = 15.0
FIRST_YEAR = 1900


@dataclass
class CleaningReport:
    """Collects what was changed, so the notebook can show it and the docs can describe it."""

    steps: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.steps.append(message)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"step": range(1, len(self.steps) + 1), "what": self.steps})


@dataclass
class CleanData:
    materials: pd.DataFrame
    students: pd.DataFrame
    interactions: pd.DataFrame
    report: CleaningReport


# Generic helpers

def slug(value: object) -> str:
    """Same normalisation as buildKey in the Fastify app (normalizeRow.ts)."""
    text = "" if value is None or pd.isna(value) else str(value)
    text = unicodedata.normalize("NFD", text.strip().lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def build_app_key(title: object, author: object, material_type: str) -> str:
    """The key the Fastify app uses to identify a catalog item: title|author|type."""
    return "|".join([slug(title), slug(author), material_type])


def to_number(series: pd.Series) -> pd.Series:
    """Turns strings like '26.7%', '3,5' or 'onbekend' into floats (NaN when that is impossible)."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def split_terms(*values: object) -> list[str]:
    """Merges theme strings like 'WOII;onderduik' and 'WOII; onderduik' into one unique list."""
    seen: dict[str, None] = {}
    for value in values:
        if value is None or pd.isna(value):
            continue
        for part in re.split(r"[;,]", str(value)):
            term = part.strip().strip("[]").lower()
            if term and "." not in term:
                seen[term] = None
    return list(seen)


def clean_url(value: object) -> str | None:
    """'https://site.nl/page [site.nl]' becomes 'https://site.nl/page'."""
    if value is None or pd.isna(value):
        return None
    url = re.sub(r"\s*\[[^\]]*\]\s*$", "", str(value).strip())
    return url if re.match(r"^https?://", url, flags=re.IGNORECASE) else None


def url_words(url: object) -> str:
    """Words from the path of a url, for example 'tech technologie' from quest.nl/tech/technologie/..."""
    if url is None or pd.isna(url) or not str(url):
        return ""
    path = re.sub(r"^https?://[^/]+", "", str(url))
    words = re.findall(r"[a-zA-Z]{3,}", path)
    return " ".join(word.lower() for word in words if not word.isupper())


def looks_like_url(value: object) -> bool:
    return value is not None and not pd.isna(value) and bool(re.match(r"^\s*https?://", str(value), flags=re.IGNORECASE))


def looks_like_term_list(value: object, max_words: int = 4) -> bool:
    """'AI, privacy, deepfakes' is a list of themes; 'Een spannende thriller over ...' is a sentence."""
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    parts = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
    return len(parts) >= 2 and not text.endswith(".") and all(len(part.split()) <= max_words for part in parts)


def mode(series: pd.Series) -> object:
    values = series.dropna()
    return values.mode().iat[0] if not values.empty else pd.NA


# Loading

def load_raw(path: Path | str = config.RAW_DATASET) -> dict[str, pd.DataFrame]:
    """Reads the three sheets. Materials are read as text, so no value is converted silently."""
    path = Path(path)
    return {
        "materials": pd.read_excel(path, sheet_name=config.SHEET_MATERIALS, dtype=str),
        "students": pd.read_excel(path, sheet_name=config.SHEET_STUDENTS),
        "interactions": pd.read_excel(path, sheet_name=config.SHEET_INTERACTIONS),
    }


# Materials

def clean_materials(raw: pd.DataFrame, report: CleaningReport | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Cleans the catalog. Returns the clean frame and a map from every catalog code to the code kept."""
    report = report or CleaningReport()
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    for column in df.columns:
        df[column] = df[column].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})

    dropped = [column for column in ADMIN_COLUMNS if column in df.columns]
    df = df.drop(columns=dropped).rename(columns=MATERIAL_COLUMNS)
    report.add(f"Dropped administrative or empty columns: {', '.join(dropped)}")

    # Material type: one spelling per type, learned from the values in the data, plus the subgenre.
    mapping = type_mapping(df["raw_type"])
    mapped = df["raw_type"].map(lambda value: mapping.get(value) if pd.notna(value) else None)
    df["type"] = mapped.map(lambda value: value[0] if value else pd.NA)
    df["type_label"] = mapped.map(lambda value: value[1] if value else pd.NA)
    df["subgenre"] = mapped.map(lambda value: value[2] if value else pd.NA)
    # A missing type becomes the most common type, like the catalog import of the Fastify app
    # does, so the catalog keys of the model and of the app stay the same.
    missing_type = df["type"].isna()
    most_common = mode(df["type"])
    df.loc[missing_type, "type"] = most_common
    df.loc[missing_type, "type_label"] = mode(df.loc[df["type"] == most_common, "type_label"])
    report.add(
        f"{df['raw_type'].nunique()} spellings of the material type brought back to {df['type'].nunique()} types, "
        f"{int(missing_type.sum())} missing type set to the most common type ({most_common})"
    )

    # Genre: the copy column first, then the most common genre of the same type.
    genre = df["genre"].str.lower()
    if "genre_copy" in df.columns:
        genre = genre.fillna(df["genre_copy"].str.lower())
    genre_by_type = genre.groupby(df["type"]).agg(mode)
    missing_genre = genre.isna()
    genre = genre.fillna(df["type"].map(genre_by_type)).fillna(mode(genre))
    df["genre"] = genre
    df = df.drop(columns=[column for column in ["genre_copy"] if column in df.columns])
    report.add(f"Genre filled from genre_copy, then from the most common genre of the type ({int(missing_genre.sum())} rows)")

    # Language and target group.
    languages = language_mapping(df["language"])
    df["language"] = df["language"].map(lambda value: languages.get(value, value) if pd.notna(value) else pd.NA)
    df["language"] = df["language"].fillna(mode(df["language"]))
    target = df["target_group"].str.upper().str.replace(" ", "", regex=False)
    if "target_group_extra" in df.columns:
        target = target.fillna(df["target_group_extra"].str.upper().str.replace(" ", "", regex=False))
        df = df.drop(columns=["target_group_extra"])
    df["target_group"] = target.fillna(mode(target))
    report.add(f"Language spellings merged ({', '.join(sorted(set(languages.values())))}), target group in one spelling")

    df["format"] = df["format"].str.lower()
    df["interest_cluster"] = df["interest_cluster"].str.lower()

    # Numbers.
    pages = to_number(df["pages"])
    df["pages"] = pages.where(pages > 0)
    df["reading_minutes"] = to_number(df["reading_minutes"]).where(lambda s: s > 0)
    df["avg_word_length"] = to_number(df["avg_word_length"]).where(lambda s: s <= MAX_PLAUSIBLE_WORD_LENGTH)
    df["long_words_pct"] = to_number(df["long_words_pct"])
    df["description_words"] = to_number(df["description_words"])
    df["sentence_count"] = to_number(df["sentence_count"])
    df["avg_sentence_length"] = to_number(df["avg_sentence_length"]).fillna(
        df["description_words"] / df["sentence_count"].replace(0, np.nan)
    )
    df["complexity_score"] = to_number(df["complexity_score"])
    year = to_number(df["publication_year"])
    df["publication_year"] = year.where(year.between(FIRST_YEAR, CURRENT_YEAR))
    df["avg_rating"] = to_number(df["avg_rating"])
    report.add(
        "Numbers converted ('26.7%', 'onbekend'), impossible values emptied: "
        f"pages <= 0, word length > {MAX_PLAUSIBLE_WORD_LENGTH:.0f}, year after {CURRENT_YEAR}"
    )

    # Pages and reading time per material type with the median, with a flag.
    df["pages_imputed"] = df["pages"].isna()
    df["pages"] = df["pages"].fillna(df.groupby("type")["pages"].transform("median")).fillna(df["pages"].median())
    minutes_per_page = (df["reading_minutes"] / df["pages"]).groupby(df["type"]).transform("median")
    minutes_per_page = minutes_per_page.fillna((df["reading_minutes"] / df["pages"]).median())
    df["minutes_imputed"] = df["reading_minutes"].isna()
    df["reading_minutes"] = df["reading_minutes"].fillna((df["pages"] * minutes_per_page).round())
    for column in ["avg_word_length", "avg_sentence_length", "long_words_pct"]:
        df[column] = df[column].fillna(df[column].median())
    report.add("Pages and reading time filled with the median of the type (flag *_imputed)")

    # Level: one spelling, a number, and an estimate from the complexity score when it is missing.
    df["level"] = df["level"].str.upper().str.replace(" ", "", regex=False)
    df["level_num"] = df["level"].map(parse_level)
    df.loc[df["level_num"].isna(), "level"] = pd.NA
    df["level_imputed"] = df["level"].isna()
    complexity_by_level = (
        df.dropna(subset=["level", "complexity_score"]).groupby("level")["complexity_score"].median().sort_values()
    )

    def guess_level(row: pd.Series) -> str:
        score = row["complexity_score"]
        if pd.isna(score) or complexity_by_level.empty:
            return str(mode(df.loc[df["type"] == row["type"], "level"]))
        return str((complexity_by_level - score).abs().idxmin())

    missing_level = df["level"].isna()
    df.loc[missing_level, "level"] = df[missing_level].apply(guess_level, axis=1)
    df["level_num"] = df["level"].map(parse_level)
    df["complexity_score"] = df["complexity_score"].fillna(df.groupby("level")["complexity_score"].transform("median"))
    report.add(f"{int(missing_level.sum())} missing levels estimated from the complexity score")

    # The sheet mixes two layouts: for articles, blogs and poetry the description column holds a
    # list of themes and the theme column holds the url. That is detected from the values
    # themselves, not from the type, so a new kind of material is handled the same way.
    url_in_theme_column = df["book_themes"].map(looks_like_url)
    df["source_url"] = df["source_url"].where(~url_in_theme_column | df["source_url"].notna(), df["book_themes"])
    df.loc[url_in_theme_column, "book_themes"] = pd.NA
    themes_in_description = df["description"].map(looks_like_term_list)
    description_themes = df["description"].where(themes_in_description)
    df.loc[themes_in_description, "description"] = pd.NA
    report.add(
        f"{int(themes_in_description.sum())} rows with themes in the description column and "
        f"{int(url_in_theme_column.sum())} with the url in the theme column put in the right column"
    )

    # Themes from four columns, a clean url and one text column for NLP.
    df["themes"] = [
        ";".join(split_terms(a, b, c, e))
        for a, b, c, e in zip(df["book_themes"], df["keywords"], df["subject"], description_themes)
    ]
    df["source_url"] = df["source_url"].map(clean_url)
    df["text"] = (
        df["title"].fillna("")
        + ". "
        + df["description"].fillna("")
        + ". "
        + df["themes"].str.replace(";", " ", regex=False)
        + " "
        + df["subgenre"].fillna("")
        + " "
        + df["source_url"].map(url_words)
    ).str.strip()
    df = df.drop(columns=["book_themes", "keywords", "subject", "raw_type"])
    report.add("Themes merged from book_themes, keywords, subject and the theme lists, url cleaned, one text column made")

    # Duplicates: the same article appears as 'online artikel' and as 'blogpost' with the same url.
    # material_id is not unique (different books share an id), so catalog_code is the key.
    df["key"] = [build_app_key(t, a, ty) for t, a, ty in zip(df["title"], df["author"], df["type"])]
    duplicate_key = df["source_url"].fillna(df["key"])
    kept_index = duplicate_key.drop_duplicates(keep="first").index
    kept = df.loc[kept_index]
    first_code = dict(zip(duplicate_key.loc[kept_index], kept["catalog_code"]))
    code_map = dict(zip(df["catalog_code"], duplicate_key.map(first_code)))
    report.add(
        f"{len(df) - len(kept)} duplicate materials (same url, or same title, author and type) merged; "
        f"material_id is not unique ({int(df['material_id'].duplicated().sum())} times), so catalog_code is the key"
    )

    kept = kept.astype({"title": "string", "author": "string"}).reset_index(drop=True)
    ordered = [
        "catalog_code", "material_id", "key", "title", "author", "type", "type_label", "subgenre", "genre",
        "level", "level_num", "language", "target_group", "format", "interest_cluster", "themes", "description",
        "text", "source_url", "pages", "reading_minutes", "publication_year", "avg_rating", "description_words",
        "avg_word_length", "avg_sentence_length", "long_words_pct", "sentence_count", "complexity_score",
        "pages_imputed", "minutes_imputed", "level_imputed",
    ]
    return kept[ordered], code_map


# Students

def clean_students(raw: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    report = report or CleaningReport()
    df = raw.rename(columns=STUDENT_COLUMNS).copy()
    df["reading_level"] = df["reading_level"].astype(str).str.upper().str.strip()
    df["reading_level_num"] = df["reading_level"].map(parse_level)
    df["cefr_level"] = df["cefr_level"].astype(str).str.upper().str.strip()
    df["cefr_level_num"] = df["cefr_level"].map(parse_cefr)
    df["interests"] = df["interests"].map(lambda value: ";".join(split_terms(value)))
    df["preferred_type"] = df["preferred_type"].astype(str).str.lower().str.strip()
    df["preferred_genre"] = df["preferred_genre"].astype(str).str.lower().str.strip()
    languages = language_mapping(df["preferred_language"])
    df["preferred_language"] = df["preferred_language"].map(lambda value: languages.get(value, value))
    df["max_minutes"] = pd.to_numeric(df["max_minutes"], errors="coerce")
    report.add("Students: levels as numbers, interests as a list, preferences in lowercase")
    return df


# Interactions

def clean_interactions(
    raw: pd.DataFrame,
    code_map: dict[str, str],
    known_students: set[str],
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    report = report or CleaningReport()
    df = raw.rename(columns=INTERACTION_COLUMNS).copy()
    df["interaction"] = df["interaction"].astype(str).str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["catalog_code"] = df["catalog_code"].map(code_map)
    before = len(df)
    df = df.dropna(subset=["catalog_code"])
    df = df[df["student_id"].isin(known_students)]
    report.add(f"Interactions linked to the merged materials, {before - len(df)} without a valid material or student removed")

    # The data says how positive each interaction type is: the average rating that comes with it.
    scores = learn_interaction_scores(df)
    df["type_score"] = df["interaction"].map(scores)
    df["preference_score"] = df[["type_score", "rating"]].mean(axis=1)
    positive = positive_interactions(scores)
    df["positive"] = df["interaction"].isin(positive).astype(int)
    learned = ", ".join(f"{kind} {score:.1f}" for kind, score in sorted(scores.items(), key=lambda item: item[1]))
    report.add(
        f"Score per interaction type learned from the ratings ({learned}); positive = score of at least "
        f"{config.POSITIVE_SCORE} ({', '.join(sorted(positive))}); preference_score = mean of that score and the rating"
    )

    # One row per student and material: the most recent interaction. After merging duplicate
    # materials a student can have two interactions with what is now one material.
    before = len(df)
    df = df.sort_values("date").drop_duplicates(subset=["student_id", "catalog_code"], keep="last")
    report.add(f"{before - len(df)} duplicate interactions (same student and material) reduced to the most recent one")
    return df.reset_index(drop=True)


# Whole pipeline

def clean_all(path: Path | str = config.RAW_DATASET, catalog: pd.DataFrame | None = None) -> CleanData:
    """Cleans the three sheets of the dataset.

    `catalog` replaces the materials sheet, with the same columns: the rows the backend imported
    in MongoDB (see sources.py). Students and interactions always come from the dataset.
    """
    raw = load_raw(path)
    report = CleaningReport()
    if catalog is not None:
        report.add(f"Catalog read from MongoDB ({len(catalog)} rows) instead of the materials sheet")
    materials, code_map = clean_materials(raw["materials"] if catalog is None else catalog, report)
    students = clean_students(raw["students"], report)
    interactions = clean_interactions(raw["interactions"], code_map, set(students["student_id"]), report)
    return CleanData(materials, students, interactions, report)


def clean_with_processed(catalog: pd.DataFrame, directory: Path = config.PROCESSED_DIR) -> CleanData:
    """A new catalog with the students and interactions of the last training.

    On the server the dataset of the client is removed after the build; only the cleaned files
    stay. Retraining on a newly imported catalog then reuses those and links the interactions
    to the new catalog.
    """
    report = CleaningReport()
    report.add(f"Catalog read from MongoDB ({len(catalog)} rows); students and interactions from the last training")
    materials, code_map = clean_materials(catalog, report)
    _, students, interactions = load_processed(directory)
    before = len(interactions)
    interactions = interactions.assign(catalog_code=interactions["catalog_code"].map(code_map))
    interactions = interactions.dropna(subset=["catalog_code"])
    interactions = interactions.sort_values("date").drop_duplicates(subset=["student_id", "catalog_code"], keep="last")
    report.add(f"Interactions linked to the new catalog, {before - len(interactions)} without a material removed")
    return CleanData(materials, students, interactions.reset_index(drop=True), report)


def save_processed(data: CleanData, directory: Path = config.PROCESSED_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data.materials.to_csv(directory / config.MATERIALS_CSV.name, index=False)
    data.students.to_csv(directory / config.STUDENTS_CSV.name, index=False)
    data.interactions.to_csv(directory / config.INTERACTIONS_CSV.name, index=False)


def load_processed(directory: Path = config.PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    materials = pd.read_csv(directory / config.MATERIALS_CSV.name, keep_default_na=True)
    materials["themes"] = materials["themes"].fillna("")
    students = pd.read_csv(directory / config.STUDENTS_CSV.name)
    interactions = pd.read_csv(directory / config.INTERACTIONS_CSV.name, parse_dates=["date"])
    return materials, students, interactions
