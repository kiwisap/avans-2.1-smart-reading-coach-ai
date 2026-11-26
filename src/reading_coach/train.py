"""Training: clean the data, evaluate, train on everything, save the model and the synonyms.

Usage (from the project folder):
    python -m reading_coach.train
    python -m reading_coach.train --skip-evaluation
    python -m reading_coach.train --catalog xlsx      # ignore MongoDB, use the sheet in the xlsx

The API calls train() as well, when the backend asks for a new model (POST /api/admin/retrain).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .data_cleaning import CleanData, clean_all, clean_with_processed, save_processed
from .evaluation import catalog_coverage, evaluate_rankings, make_scorers, temporal_split
from .hybrid import HybridRecommender
from .sources import CatalogUnavailable, MongoSettings, catalog_rows, save_synonyms

Log = Callable[[str], None]


@dataclass
class TrainResult:
    model: HybridRecommender
    metrics: dict = field(default_factory=dict)


def load_data(dataset: Path, mongo: MongoSettings | None, log: Log) -> tuple[CleanData, str, object | None]:
    """The clean data, where the catalog came from, and the MongoDB database (or None)."""
    database = mongo.connect() if mongo else None
    catalog = catalog_rows(database) if database is not None else None
    if dataset.exists():
        return clean_all(dataset, catalog), "mongodb" if catalog is not None else "xlsx", database
    if catalog is not None and config.STUDENTS_CSV.exists() and config.INTERACTIONS_CSV.exists():
        log("    The dataset is not here, students and interactions come from the last training")
        return clean_with_processed(catalog), "mongodb", database
    raise FileNotFoundError(f"The dataset {dataset} is missing and there is no catalog in MongoDB to train on")


def train(
    dataset: Path | str = config.RAW_DATASET,
    mongo: MongoSettings | None = None,
    evaluate: bool = True,
    model_path: Path | str = config.MODEL_FILE,
    log: Log = print,
) -> TrainResult:
    log("1/4 Cleaning the data")
    data, catalog_source, database = load_data(Path(dataset), mongo, log)
    save_processed(data)
    for step in data.report.steps:
        log(f"    {step}")

    metrics: dict = {}
    if evaluate:
        log("2/4 Evaluating with a split in time per student")
        train_part, test_part = temporal_split(data.interactions)
        evaluation_model = HybridRecommender().fit(data.materials, data.students, train_part)
        scorers = make_scorers(evaluation_model, train_part, data.students)
        table = evaluate_rankings(scorers, evaluation_model, data.students, train_part, test_part)
        log(table.to_string())
        metrics["evaluation"] = table.reset_index().to_dict(orient="records")
        metrics["catalog_coverage_top5"] = round(catalog_coverage(evaluation_model, data.students, train_part), 3)
    else:
        log("2/4 Evaluation skipped")

    log("3/4 Training the final model on all interactions")
    model = HybridRecommender().fit(data.materials, data.students, data.interactions)
    path = model.save(model_path)

    log("4/4 Saving")
    metrics.update(
        {
            "model_version": model.version,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "catalog_source": catalog_source,
            "materials": len(data.materials),
            "students": len(data.students),
            "interactions": len(data.interactions),
            "synonym_terms": len(model.synonyms),
            "interaction_scores": {kind: round(score, 2) for kind, score in model.vocabulary.interaction_scores.items()},
            "feature_importance": model.feature_importance.round(4).to_dict(),
        }
    )
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"    Model saved in {path}")
    if database is not None:
        stored = save_synonyms(database, model.synonyms, model.version)
        log(f"    Synonyms of {stored} terms stored in MongoDB")
    return TrainResult(model, metrics)


def main() -> None:
    from dotenv import load_dotenv

    # MONGO_URL and MONGO_DB from .env in the project folder; real environment variables win.
    load_dotenv(config.PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Train the Smart Reading Coach recommendation model")
    parser.add_argument("--dataset", default=str(config.RAW_DATASET), help="Path to the xlsx of the client")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip the offline evaluation")
    parser.add_argument(
        "--catalog",
        choices=["auto", "xlsx", "mongodb"],
        default="auto",
        help="Where the catalog comes from; auto uses MongoDB when MONGO_URL is set",
    )
    args = parser.parse_args()

    mongo = None if args.catalog == "xlsx" else MongoSettings.from_env()
    if args.catalog == "mongodb" and mongo is None:
        parser.error("--catalog mongodb needs MONGO_URL")
    try:
        train(args.dataset, mongo, evaluate=not args.skip_evaluation)
    except (CatalogUnavailable, FileNotFoundError) as error:
        raise SystemExit(f"Training stopped: {error}") from None


if __name__ == "__main__":
    main()
