"""Where the data comes from, and where the learned synonyms go.

The catalog lives in MongoDB: the Fastify backend imports the spreadsheet there (collection
books) and keeps every original row in the field `sources`. Training on those rows means the
model always knows exactly the titles the app shows, also after new books are imported.

Students and interactions are only in the dataset of the client (the xlsx). Without MONGO_URL
everything comes from the xlsx, so the notebooks and the tests work without a database.

MongoDB is only read, except for the collection topic_synonyms: after training the learned
synonyms are written there, so the Fastify backend can use the same links.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .data_cleaning import MATERIAL_COLUMNS
from .synonyms import Related

BOOKS = "books"
TOPIC_SYNONYMS = "topic_synonyms"
DEFAULT_DATABASE = "smart_reading_coach_files"  # the same default as the Fastify backend


class CatalogUnavailable(RuntimeError):
    """MongoDB is configured but holds no imported catalog."""


@dataclass(frozen=True)
class MongoSettings:
    url: str
    database: str = DEFAULT_DATABASE

    @staticmethod
    def from_env() -> "MongoSettings | None":
        url = os.getenv("MONGO_URL", "").strip()
        if not url:
            return None
        return MongoSettings(url, os.getenv("MONGO_DB", "").strip() or DEFAULT_DATABASE)

    def connect(self):
        """A pymongo database. A short timeout, so a missing database fails fast instead of hanging."""
        from pymongo import MongoClient

        client = MongoClient(self.url, serverSelectionTimeoutMS=5000, appname="smart-reading-coach-ai")
        return client[self.database]


# The raw column that holds the catalog code, the id the interactions refer to.
RAW_CATALOG_CODE = next(raw for raw, name in MATERIAL_COLUMNS.items() if name == "catalog_code")


def catalog_rows(database) -> pd.DataFrame:
    """The original spreadsheet rows of every imported book, as text, in the order of the sheet."""
    rows = [
        row
        for book in database[BOOKS].find({}, {"sources": 1, "_id": 0})
        for row in book.get("sources") or []
    ]
    if not rows:
        books = database[BOOKS].count_documents({})
        where = f"database '{database.name}'"
        if books == 0:
            raise CatalogUnavailable(
                f"There are no books in {where}. Check that MONGO_DB is the same as in the backend, "
                "and run 'npm run import:catalog' in the backend."
            )
        raise CatalogUnavailable(
            f"The {books} books in {where} were imported by an older version of the backend, without the "
            "spreadsheet rows. Run 'npm run import:catalog' in the backend again."
        )
    frame = pd.DataFrame(rows, dtype="string")
    if RAW_CATALOG_CODE in frame.columns:
        frame = frame.sort_values(RAW_CATALOG_CODE, kind="stable")
    return frame.reset_index(drop=True)


def save_synonyms(database, synonyms: dict[str, list[Related]], model_version: str) -> int:
    """Replaces the stored synonyms with the ones of the new model. Returns the number of terms."""
    collection = database[TOPIC_SYNONYMS]
    now = datetime.now(timezone.utc)
    for term, related in synonyms.items():
        collection.replace_one(
            {"term": term},
            {
                "term": term,
                "related": [{"term": item.term, "relation": item.relation} for item in related],
                "source": "wordnet",
                "modelVersion": model_version,
                "updatedAt": now,
            },
            upsert=True,
        )
    # Terms that no longer have synonyms (for example a theme that left the catalog) go away.
    collection.delete_many({"term": {"$nin": list(synonyms)}})
    collection.create_index("term", unique=True, name="term_unique")
    return len(synonyms)
