"""Loading and cleaning the client dataset (sheets Leesmateriaal, Studenten and Interacties).

Every cleaning step is a small function so the notebooks can show the steps one by one,
while `clean_all` runs the full pipeline for training and for the API.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ADMIN_COLUMNS = [
    "Unnamed: 3",
    "interne_batchcode",
    "administratieve_prioriteit",
    "gegevensstatus",
    "uitgever",
]

CURRENT_YEAR = pd.Timestamp.now().year
MAX_PLAUSIBLE_WORD_LENGTH = 15.0


@dataclass
class CleaningReport:
    """Collects what was changed so the notebook can show it and the README can document it."""

    steps: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.steps.append(message)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"stap": range(1, len(self.steps) + 1), "wat": self.steps})


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
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def build_app_key(title: object, author: object, material_type: str) -> str:
    """Key the Fastify app uses to identify a catalog item: title|author|type."""
    return "|".join([slug(title), slug(author), material_type])


def to_number(series: pd.Series) -> pd.Series:
    """Converts strings like '26.7%', '3,5' or 'onbekend' to floats (NaN when impossible)."""
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
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
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
    """Words from the url path, for example 'tech technologie' from quest.nl/tech/technologie/..."""
    if url is None or pd.isna(url) or not str(url):
        return ""
    url = str(url)
    path = re.sub(r"^https?://[^/]+", "", url)
    words = re.findall(r"[a-zA-Z]{3,}", path)
    return " ".join(word.lower() for word in words if not word.isupper())


# Loading

def load_raw(path: Path | str = config.RAW_DATASET) -> dict[str, pd.DataFrame]:
    """Reads the three sheets. Materials are read as text so no value is silently converted."""
    path = Path(path)
    return {
        "materials": pd.read_excel(path, sheet_name=config.SHEET_MATERIALS, dtype=str),
        "students": pd.read_excel(path, sheet_name=config.SHEET_STUDENTS),
        "interactions": pd.read_excel(path, sheet_name=config.SHEET_INTERACTIONS),
    }


# Materials

def clean_materials(raw: pd.DataFrame, report: CleaningReport | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Cleans the catalog. Returns the clean frame and a map old cataloguscode to kept cataloguscode."""
    report = report or CleaningReport()
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    for column in df.columns:
        df[column] = df[column].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})

    # Column names in snake case, without the administrative noise.
    df = df.rename(
        columns={
            "Titel": "titel",
            "Auteur": "auteur",
            "Korte omschrijving": "omschrijving",
            "Type materiaal": "type_ruw",
            "Niveau": "niveau",
            "Themas_boek": "themas_boek",
            "Gemiddelde beoordeling": "gem_beoordeling",
        }
    )
    dropped = [column for column in ADMIN_COLUMNS if column in df.columns]
    df = df.drop(columns=dropped)
    report.add(f"Administratieve of lege kolommen verwijderd: {', '.join(dropped)}")

    # Material type: unify spelling, map to the app types and keep the subgenre.
    type_key = df["type_ruw"].str.lower().str.strip()
    mapped = type_key.map(lambda value: config.MATERIAL_TYPE_MAP.get(value) if pd.notna(value) else None)
    df["type"] = mapped.map(lambda value: value[0] if value else pd.NA)
    df["subgenre"] = mapped.map(lambda value: value[1] if value else pd.NA)
    pages = to_number(df["aantal_paginas"])
    # A missing type becomes "boek", exactly like the catalog import of the Fastify app, so the
    # catalog keys of the model and the app stay the same.
    missing_type = df["type"].isna()
    df.loc[missing_type, "type"] = "boek"
    report.add(
        f"Type materiaal genormaliseerd naar {len(config.APP_MATERIAL_TYPES)} app types, "
        f"{int(missing_type.sum())} ontbrekend type wordt 'boek' (zoals de import van de app)"
    )

    # Genre: use the duplicate column first, then derive it from the type.
    genre = df["genre"].str.lower()
    if "genre_kopie" in df.columns:
        genre = genre.fillna(df["genre_kopie"].str.lower())
    filled_from_type = genre.isna()
    df["genre"] = genre.fillna(df["type"].map(config.GENRE_BY_TYPE))
    df = df.drop(columns=[column for column in ["genre_kopie"] if column in df.columns])
    report.add(f"Genre aangevuld uit genre_kopie en daarna uit het type ({int(filled_from_type.sum())} rijen)")

    # Language and target group.
    df["taal"] = df["taal"].str.lower().map(lambda value: config.LANGUAGE_MAP.get(value, value) if pd.notna(value) else "Nederlands")
    target = df["doelgroep"].str.upper().str.replace(" ", "", regex=False)
    extra = df["doelgroep_extra"].str.upper().str.replace(" ", "", regex=False) if "doelgroep_extra" in df.columns else pd.Series(pd.NA, index=df.index)
    df["doelgroep"] = target.fillna(extra).fillna("MBO/NT2")
    df["geschikt_nt2"] = df["doelgroep"].str.contains("NT2") | extra.fillna("").str.contains("NT2")
    df = df.drop(columns=[column for column in ["doelgroep_extra"] if column in df.columns])
    report.add("Taal (NL, NED, nederlands) en doelgroep (mbo / nt2) gelijkgetrokken, doelgroep_extra samengevoegd")

    # Format.
    df["formaat"] = (
        df["beschikbaar_formaat"].str.lower().replace({"pdf": "digitaal", "online": "digitaal"}).fillna("onbekend")
    )
    df = df.drop(columns=["beschikbaar_formaat"])

    # Interest cluster: only casing differs.
    df["interessecluster"] = df["interessecluster"].str.lower()

    # Numbers.
    df["aantal_paginas"] = pages.where(pages > 0)
    df["geschatte_leestijd_min"] = to_number(df["geschatte_leestijd_min"]).where(lambda s: s > 0)
    df["gem_woordlengte"] = to_number(df["gem_woordlengte"]).where(lambda s: s <= MAX_PLAUSIBLE_WORD_LENGTH)
    df["aandeel_lange_woorden_pct"] = to_number(df["aandeel_lange_woorden_pct"])
    df["woordenaantal_omschrijving"] = to_number(df["woordenaantal_omschrijving"])
    df["aantal_zinnen"] = to_number(df["aantal_zinnen"])
    df["gem_zinslengte_woorden"] = to_number(df["gem_zinslengte_woorden"]).fillna(
        df["woordenaantal_omschrijving"] / df["aantal_zinnen"].replace(0, np.nan)
    )
    df["leescomplexiteit_score"] = to_number(df["leescomplexiteit_score"])
    year = to_number(df["publicatiejaar"])
    df["publicatiejaar"] = year.where(year.between(1900, CURRENT_YEAR))
    df["gem_beoordeling"] = to_number(df["gem_beoordeling"])
    report.add(
        "Getallen omgezet ('26.7%', 'onbekend'), onmogelijke waarden leeg gemaakt: "
        f"pagina's <= 0, woordlengte > {MAX_PLAUSIBLE_WORD_LENGTH:.0f}, jaartal na {CURRENT_YEAR}"
    )

    # Impute pages and reading time per material type with the median, keep a flag.
    df["paginas_geimputeerd"] = df["aantal_paginas"].isna()
    df["aantal_paginas"] = df["aantal_paginas"].fillna(df.groupby("type")["aantal_paginas"].transform("median"))
    minutes_per_page = (df["geschatte_leestijd_min"] / df["aantal_paginas"]).groupby(df["type"]).transform("median")
    df["leestijd_geimputeerd"] = df["geschatte_leestijd_min"].isna()
    df["geschatte_leestijd_min"] = df["geschatte_leestijd_min"].fillna((df["aantal_paginas"] * minutes_per_page).round())
    for column in ["gem_woordlengte", "gem_zinslengte_woorden", "aandeel_lange_woorden_pct"]:
        df[column] = df[column].fillna(df[column].median())
    report.add("Pagina's en leestijd per type aangevuld met de mediaan (met vlag *_geimputeerd)")

    # Level: normalise, impute from the complexity score when missing.
    df["niveau"] = df["niveau"].str.upper().str.replace(" ", "", regex=False)
    df["niveau_geimputeerd"] = df["niveau"].isna()
    df["niveau_num"] = df["niveau"].map(config.LEVEL_TO_NUMBER)
    level_by_complexity = (
        df.dropna(subset=["niveau_num", "leescomplexiteit_score"])
        .groupby("niveau")["leescomplexiteit_score"]
        .median()
        .sort_values()
    )
    def guess_level(row: pd.Series) -> str:
        score = row["leescomplexiteit_score"]
        if pd.isna(score) or level_by_complexity.empty:
            return df.loc[df["type"] == row["type"], "niveau"].mode().iat[0]
        return str((level_by_complexity - score).abs().idxmin())
    missing_level = df["niveau"].isna()
    df.loc[missing_level, "niveau"] = df[missing_level].apply(guess_level, axis=1)
    df["niveau_num"] = df["niveau"].map(config.LEVEL_TO_NUMBER)
    df["leescomplexiteit_score"] = df["leescomplexiteit_score"].fillna(df.groupby("niveau")["leescomplexiteit_score"].transform("median"))
    report.add(f"{int(missing_level.sum())} ontbrekende niveaus geschat op basis van de leescomplexiteit_score")

    # Themes from three sources, urls and one text column for NLP.
    df["themas"] = [
        ";".join(split_terms(a, b, c)) for a, b, c in zip(df["themas_boek"], df["trefwoorden"], df["onderwerp"])
    ]
    df["bron_url"] = df["bron_url"].map(clean_url)
    df["tekst"] = (
        df["titel"].fillna("")
        + ". "
        + df["omschrijving"].fillna("")
        + ". "
        + df["themas"].str.replace(";", " ", regex=False)
        + " "
        + df["subgenre"].fillna("")
        + " "
        + df["bron_url"].map(url_words)
    ).str.strip()
    df = df.drop(columns=["themas_boek", "trefwoorden", "onderwerp", "type_ruw"])
    report.add("Thema's uit themas_boek, trefwoorden en onderwerp samengevoegd, url opgeschoond, tekstkolom gemaakt")

    # Duplicates: the same article appears as 'online artikel' and as 'blogpost' with the same url.
    # materiaal_id is not unique (different books share an id), so cataloguscode is the primary key.
    df["key"] = [build_app_key(t, a, ty) for t, a, ty in zip(df["titel"], df["auteur"], df["type"])]
    duplicate_key = df["bron_url"].fillna(df["key"])
    kept_index = duplicate_key.drop_duplicates(keep="first").index
    kept = df.loc[kept_index]
    first_code = dict(zip(duplicate_key.loc[kept_index], kept["cataloguscode"]))
    code_map = dict(zip(df["cataloguscode"], duplicate_key.map(first_code)))
    report.add(
        f"{len(df) - len(kept)} dubbele materialen (zelfde url of zelfde titel, auteur en type) samengevoegd; "
        f"materiaal_id is niet uniek ({int(df['materiaal_id'].duplicated().sum())} keer dubbel), "
        "daarom is cataloguscode de sleutel"
    )

    kept = kept.astype({"titel": "string", "auteur": "string"}).reset_index(drop=True)
    ordered = [
        "cataloguscode", "materiaal_id", "key", "titel", "auteur", "type", "subgenre", "genre", "niveau",
        "niveau_num", "taal", "doelgroep", "geschikt_nt2", "formaat", "interessecluster", "themas",
        "omschrijving", "tekst", "bron_url", "aantal_paginas", "geschatte_leestijd_min", "publicatiejaar",
        "gem_beoordeling", "woordenaantal_omschrijving", "gem_woordlengte", "gem_zinslengte_woorden",
        "aandeel_lange_woorden_pct", "aantal_zinnen", "leescomplexiteit_score", "paginas_geimputeerd",
        "leestijd_geimputeerd", "niveau_geimputeerd",
    ]
    return kept[ordered], code_map


# Students

def clean_students(raw: pd.DataFrame, report: CleaningReport | None = None) -> pd.DataFrame:
    report = report or CleaningReport()
    df = raw.copy()
    df["leesniveau"] = df["leesniveau"].astype(str).str.upper().str.strip()
    df["leesniveau_num"] = df["leesniveau"].map(config.LEVEL_TO_NUMBER)
    df["taalniveau"] = df["taalniveau"].astype(str).str.upper().str.strip()
    df["taalniveau_num"] = df["taalniveau"].map(config.CEFR_TO_NUMBER)
    df["interesses"] = df["interesses"].map(lambda value: ";".join(split_terms(value)))
    df["voorkeur_materiaaltype"] = df["voorkeur_materiaaltype"].astype(str).str.lower().str.strip()
    df["voorkeur_genre"] = df["voorkeur_genre"].astype(str).str.lower().str.strip()
    df["voorkeur_taal"] = df["voorkeur_taal"].astype(str).str.lower().map(lambda v: config.LANGUAGE_MAP.get(v, v.title()))
    df["max_leestijd_min"] = pd.to_numeric(df["max_leestijd_min"], errors="coerce")
    report.add("Studenten: niveaus naar getallen, interesses als lijst, voorkeuren in kleine letters")
    return df


# Interactions

def clean_interactions(
    raw: pd.DataFrame,
    code_map: dict[str, str],
    known_students: set[str],
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    report = report or CleaningReport()
    df = raw.copy()
    df["interactietype"] = df["interactietype"].astype(str).str.lower().str.strip()
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df["cataloguscode"] = df["cataloguscode"].map(code_map)
    before = len(df)
    df = df.dropna(subset=["cataloguscode"])
    df = df[df["student_id"].isin(known_students)]
    report.add(f"Interacties gekoppeld aan samengevoegde materialen, {before - len(df)} zonder geldig materiaal of student verwijderd")

    df["type_score"] = df["interactietype"].map(config.INTERACTION_SCORE)
    df["voorkeurscore"] = df[["type_score", "beoordeling"]].mean(axis=1)
    df["positief"] = df["interactietype"].isin(config.POSITIVE_INTERACTIONS).astype(int)
    report.add(
        "Doelvariabele gemaakt: positief = gelezen, geliket of opgeslagen; voorkeurscore = gemiddelde van "
        "interactiescore en beoordeling (beoordeling ontbreekt soms)"
    )

    # One row per student and material: keep the most recent interaction. After merging duplicate
    # materials a student can have two interactions with what is now the same material.
    before = len(df)
    df = df.sort_values("datum").drop_duplicates(subset=["student_id", "cataloguscode"], keep="last")
    report.add(f"{before - len(df)} dubbele interacties (zelfde student en materiaal) teruggebracht tot de meest recente")
    return df.reset_index(drop=True)


# Full pipeline

def clean_all(path: Path | str = config.RAW_DATASET) -> CleanData:
    raw = load_raw(path)
    report = CleaningReport()
    materials, code_map = clean_materials(raw["materials"], report)
    students = clean_students(raw["students"], report)
    interactions = clean_interactions(raw["interactions"], code_map, set(students["student_id"]), report)
    return CleanData(materials, students, interactions, report)


def save_processed(data: CleanData, directory: Path = config.PROCESSED_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    data.materials.to_csv(directory / config.MATERIALS_CSV.name, index=False)
    data.students.to_csv(directory / config.STUDENTS_CSV.name, index=False)
    data.interactions.to_csv(directory / config.INTERACTIONS_CSV.name, index=False)


def load_processed(directory: Path = config.PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    materials = pd.read_csv(directory / config.MATERIALS_CSV.name, keep_default_na=True)
    materials["themas"] = materials["themas"].fillna("")
    students = pd.read_csv(directory / config.STUDENTS_CSV.name)
    interactions = pd.read_csv(directory / config.INTERACTIONS_CSV.name, parse_dates=["datum"])
    return materials, students, interactions
