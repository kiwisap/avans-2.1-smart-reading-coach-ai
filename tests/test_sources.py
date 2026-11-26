"""MongoDB as the source of the catalog, tested against an in-memory MongoDB (mongomock)."""

from __future__ import annotations

import mongomock
import pandas as pd
import pytest

from reading_coach import config
from reading_coach.data_cleaning import clean_all, load_raw
from reading_coach.sources import CatalogUnavailable, catalog_rows, save_synonyms
from reading_coach.synonyms import Related


def imported_books(materials: pd.DataFrame) -> list[dict]:
    """The books as the Fastify import stores them: every sheet row as text, empty cells left out."""
    rows = [
        {column: str(value) for column, value in row.items() if pd.notna(value) and str(value).strip()}
        for _, row in materials.iterrows()
    ]
    # Reversed, so the test also shows that the order of the sheet is restored.
    return [{"key": f"book-{number}", "sources": [row]} for number, row in enumerate(reversed(rows))]


@pytest.fixture()
def database():
    return mongomock.MongoClient()["smart_reading_coach_files"]


def test_an_empty_database_says_what_to_do(database) -> None:
    with pytest.raises(CatalogUnavailable, match="MONGO_DB"):
        catalog_rows(database)


def test_books_from_an_older_import_ask_for_a_new_import(database) -> None:
    database.books.insert_one({"key": "oud|boek", "title": "Oud"})
    with pytest.raises(CatalogUnavailable, match="older version"):
        catalog_rows(database)


def test_the_catalog_from_mongodb_cleans_to_the_same_materials(database) -> None:
    if not config.RAW_DATASET.exists():
        pytest.skip("The dataset is missing in data/raw")
    sheet = load_raw()["materials"]
    database.books.insert_many(imported_books(sheet))
    from_sheet = clean_all()
    from_mongo = clean_all(catalog=catalog_rows(database))
    pd.testing.assert_frame_equal(from_sheet.materials, from_mongo.materials, check_dtype=False)
    assert len(from_mongo.interactions) == len(from_sheet.interactions)


def test_synonyms_are_replaced_by_the_new_model(database) -> None:
    save_synonyms(database, {"oud": [Related("x", "synonym")], "maatschappij": [Related("cultuur", "narrower")]}, "1")
    save_synonyms(database, {"maatschappij": [Related("samenleving", "synonym")]}, "2")
    stored = list(database.topic_synonyms.find({}, {"_id": 0, "updatedAt": 0}))
    assert stored == [
        {
            "term": "maatschappij",
            "related": [{"term": "samenleving", "relation": "synonym"}],
            "source": "wordnet",
            "modelVersion": "2",
        }
    ]
