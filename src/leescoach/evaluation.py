"""Offline evaluation: temporal split per student, ranking metrics and comparison with baselines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .collaborative import SVDRecommender, user_item_matrix
from .features import StudentProfile
from .hybrid import HybridRecommender

Scorer = Callable[[StudentProfile, dict[str, float]], pd.Series]


def temporal_split(interactions: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per student the most recent interactions are the test set: the model predicts the future."""
    ordered = interactions.sort_values(["student_id", "datum"])
    position = ordered.groupby("student_id").cumcount()
    size = ordered.groupby("student_id")["datum"].transform("size")
    is_test = position >= np.ceil(size * (1 - test_fraction))
    return ordered[~is_test].copy(), ordered[is_test].copy()


def precision_recall_ndcg(ranked: list[str], relevant: set[str], k: int) -> tuple[float, float, float, float]:
    top = ranked[:k]
    hits = [1.0 if code in relevant else 0.0 for code in top]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant) if relevant else 0.0
    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    ndcg = dcg / ideal if ideal else 0.0
    return precision, recall, ndcg, float(any(hits))


def make_scorers(model: HybridRecommender, train: pd.DataFrame, students: pd.DataFrame) -> dict[str, Scorer]:
    """Baselines and model variants that all return a score per material for one student."""
    codes = model.materials["cataloguscode"].tolist()
    rng = np.random.default_rng(config.RANDOM_STATE)
    svd = SVDRecommender().fit(user_item_matrix(train, codes, students["student_id"].tolist()))

    def random_scorer(profile: StudentProfile, history: dict[str, float]) -> pd.Series:
        return pd.Series(rng.random(len(codes)), index=codes)

    def popularity_scorer(profile: StudentProfile, history: dict[str, float]) -> pd.Series:
        return model.popularity.copy()

    def rules_scorer(profile: StudentProfile, history: dict[str, float]) -> pd.Series:
        # Roughly the hand written scoring in the current Fastify app (advice/scoring.ts).
        features = model.pair_features(profile, history)
        score = (
            3 * (features["niveau_afstand"] < 0.01)
            + 1 * (features["niveau_verschil"] < 0)
            + 3 * features["interesse_treffers"]
            + 2 * (features["type_match"] == 1.0)
            + 1 * features["leestijd_past"]
        ).astype(float)
        score[features["niveau_verschil"] > 0.01] = -1.0
        return score + 1e-6 * rng.random(len(score))

    def content_scorer(profile: StudentProfile, history: dict[str, float]) -> pd.Series:
        liked = [code for code, value in history.items() if value >= 4]
        return pd.Series(model.content.scores(profile.interesses, liked), index=model.content.codes)

    def knn_scorer(profile: StudentProfile, history: dict[str, float]) -> pd.Series:
        return pd.Series(model.knn.scores(history), index=model.knn.codes)

    def svd_scorer(profile: StudentProfile, history: dict[str, float]) -> pd.Series:
        return pd.Series(svd.scores(history), index=svd.codes)

    def hybrid_scorer(profile: StudentProfile, history: dict[str, float]) -> pd.Series:
        return model.score(profile, history)

    return {
        "Willekeurig": random_scorer,
        "Populariteit": popularity_scorer,
        "Regels (huidige app)": rules_scorer,
        "Content based (TF-IDF)": content_scorer,
        "Item KNN": knn_scorer,
        "Matrix factorisatie (SVD)": svd_scorer,
        "Hybride model (Random Forest)": hybrid_scorer,
    }


def evaluate_rankings(
    scorers: dict[str, Scorer],
    students: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """Two views per method.

    1. Top k over the whole catalog: do the materials the student liked later appear in the advice?
    2. AUC within the test interactions: does the method rank liked above disliked materials?
    """
    train_by_student = train.groupby("student_id")
    test_by_student = test.groupby("student_id")
    results = []
    for name, scorer in scorers.items():
        metrics: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        predictions: list[float] = []
        per_student_auc: list[float] = []
        for _, student in students.iterrows():
            sid = student["student_id"]
            if sid not in test_by_student.groups:
                continue
            profile = StudentProfile.from_dataset_row(student)
            own_train = train_by_student.get_group(sid) if sid in train_by_student.groups else train.iloc[0:0]
            own_test = test_by_student.get_group(sid)
            history = dict(zip(own_train["cataloguscode"], own_train["voorkeurscore"]))
            scores = scorer(profile, history)

            candidates = scores.drop(index=[code for code in history if code in scores.index])
            ranked = candidates.sort_values(ascending=False).index.tolist()
            relevant = set(own_test.loc[own_test["positief"] == 1, "cataloguscode"])
            if relevant:
                metrics.append(precision_recall_ndcg(ranked, relevant, k))

            test_scores = scores.reindex(own_test["cataloguscode"]).fillna(scores.min()).to_numpy()
            labels.extend(own_test["positief"].tolist())
            predictions.extend(test_scores.tolist())
            if own_test["positief"].nunique() == 2:
                per_student_auc.append(roc_auc_score(own_test["positief"], test_scores))

        values = np.array(metrics)
        results.append(
            {
                "methode": name,
                f"precision@{k}": values[:, 0].mean(),
                f"recall@{k}": values[:, 1].mean(),
                f"ndcg@{k}": values[:, 2].mean(),
                f"hitrate@{k}": values[:, 3].mean(),
                "auc_test_interacties": roc_auc_score(labels, predictions),
                "auc_per_student": float(np.mean(per_student_auc)),
            }
        )
    return pd.DataFrame(results).set_index("methode").round(3)


def catalog_coverage(model: HybridRecommender, students: pd.DataFrame, interactions: pd.DataFrame, n: int = 5) -> float:
    """Share of the catalog that appears in at least one advice: a measure of variety."""
    shown: set[str] = set()
    by_student = interactions.groupby("student_id")
    for _, student in students.iterrows():
        own = by_student.get_group(student["student_id"]) if student["student_id"] in by_student.groups else interactions.iloc[0:0]
        history = dict(zip(own["cataloguscode"], own["voorkeurscore"]))
        for recommendation in model.recommend(StudentProfile.from_dataset_row(student), history, n=n):
            shown.add(recommendation.cataloguscode)
    return len(shown) / len(model.materials)
