"""Command line training: clean the data, evaluate, train on everything and save the model.

Usage (from the project folder):
    python -m leescoach.train
    python -m leescoach.train --skip-evaluation
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from . import config
from .data_cleaning import clean_all, save_processed
from .evaluation import catalog_coverage, evaluate_rankings, make_scorers, temporal_split
from .hybrid import HybridRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Train het Smart Reading Coach aanbevelingsmodel")
    parser.add_argument("--dataset", default=str(config.RAW_DATASET), help="Pad naar de xlsx van de opdrachtgever")
    parser.add_argument("--skip-evaluation", action="store_true", help="Sla de offline evaluatie over")
    args = parser.parse_args()

    print("1/4 Data opschonen")
    data = clean_all(args.dataset)
    save_processed(data)
    for step in data.report.steps:
        print(f"    {step}")

    metrics: dict = {}
    if not args.skip_evaluation:
        print("2/4 Evalueren met een tijdsplitsing per student")
        train, test = temporal_split(data.interactions)
        evaluation_model = HybridRecommender().fit(data.materials, data.students, train)
        table = evaluate_rankings(make_scorers(evaluation_model, train, data.students), data.students, train, test)
        print(table.to_string())
        metrics["evaluatie"] = table.reset_index().to_dict(orient="records")
        metrics["catalogusdekking_top5"] = round(catalog_coverage(evaluation_model, data.students, train), 3)

    print("3/4 Eindmodel trainen op alle interacties")
    model = HybridRecommender().fit(data.materials, data.students, data.interactions)
    path = model.save()

    print("4/4 Opslaan")
    metrics.update(
        {
            "modelversie": model.version,
            "getraind_op": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "aantal_materialen": len(data.materials),
            "aantal_studenten": len(data.students),
            "aantal_interacties": len(data.interactions),
            "feature_importance": model.feature_importance.round(4).to_dict(),
        }
    )
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    Model opgeslagen in {path}")


if __name__ == "__main__":
    main()
