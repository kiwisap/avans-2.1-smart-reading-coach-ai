"""Hybrid recommender: content based and collaborative signals combined by a trained classifier.

For every (student, material) pair features are computed (level fit, type and genre fit,
text similarity with the interests, similarity with liked materials, collaborative score,
popularity, reading time). A Random Forest learns from the interactions which combination
predicts a positive interaction (read, liked or saved). The predicted probability ranks the
catalog; a diversity step keeps the advice varied, and every advice gets an explanation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier

from . import config
from .collaborative import ItemKNN, popularity, user_item_matrix
from .content import ContentModel
from .explain import explain
from .features import FEATURE_COLUMNS, StudentProfile, static_features

MODEL_VERSION = "1.0.0"


def default_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


@dataclass
class Recommendation:
    cataloguscode: str
    key: str
    titel: str
    auteur: str | None
    type: str
    genre: str
    niveau: str
    score: float
    redenen: list[str]
    samenvatting: str
    kenmerken: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HybridRecommender:
    classifier: ClassifierMixin = field(default_factory=default_classifier)
    knn_neighbours: int = 20
    max_level_gap: float = 1.0
    diversity: float = 0.25
    materials: pd.DataFrame = field(init=False)
    content: ContentModel = field(init=False)
    knn: ItemKNN = field(init=False)
    popularity: pd.Series = field(init=False)
    item_similarity: np.ndarray = field(init=False)
    feature_importance: pd.Series = field(init=False)
    version: str = MODEL_VERSION

    # Training

    def fit(self, materials: pd.DataFrame, students: pd.DataFrame, interactions: pd.DataFrame) -> "HybridRecommender":
        self.fit_components(materials, students, interactions)
        training = self.training_frame(students, interactions)
        self.classifier = clone(self.classifier)
        self.classifier.fit(training[FEATURE_COLUMNS], training["positief"])
        importances = getattr(self.classifier, "feature_importances_", None)
        if importances is not None:
            self.feature_importance = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)
        else:
            self.feature_importance = pd.Series(dtype=float)
        return self

    def fit_components(self, materials: pd.DataFrame, students: pd.DataFrame, interactions: pd.DataFrame) -> None:
        """Fits everything except the classifier (content vectors, KNN, popularity)."""
        self.materials = materials.reset_index(drop=True)
        self.content = ContentModel().fit(self.materials)
        self.item_similarity = self.content.item_similarity()
        codes = self.materials["cataloguscode"].tolist()
        matrix = user_item_matrix(interactions, codes, students["student_id"].tolist())
        self.knn = ItemKNN(k=self.knn_neighbours).fit(matrix)
        self.popularity = popularity(interactions, codes)

    def training_frame(self, students: pd.DataFrame, interactions: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
        """One row per interaction, with features that never see the label of that row.

        The collaborative score and the popularity are computed out of fold: the students are
        split in groups and for each group the KNN and popularity are fitted on the other groups.
        Within the own history the item itself is left out as well.
        """
        codes = self.materials["cataloguscode"].tolist()
        student_ids = students["student_id"].to_numpy()
        rng = np.random.default_rng(config.RANDOM_STATE)
        fold_of = dict(zip(student_ids, rng.permutation(len(student_ids)) % folds))
        by_student = interactions.groupby("student_id")
        rows: list[pd.DataFrame] = []
        for fold in range(folds):
            other_ids = [sid for sid in student_ids if fold_of[sid] != fold]
            other = interactions[interactions["student_id"].isin(other_ids)]
            knn = ItemKNN(k=self.knn_neighbours).fit(user_item_matrix(other, codes, other_ids))
            pop = popularity(other, codes)
            for _, student in students[students["student_id"].map(fold_of) == fold].iterrows():
                if student["student_id"] not in by_student.groups:
                    continue
                own = by_student.get_group(student["student_id"])
                profile = StudentProfile.from_dataset_row(student)
                history = dict(zip(own["cataloguscode"], own["voorkeurscore"]))
                features = self.pair_features(
                    profile, history, codes=own["cataloguscode"].tolist(), leave_one_out=True, knn=knn, pop=pop
                )
                features["positief"] = own["positief"].to_numpy()
                features["student_id"] = student["student_id"]
                rows.append(features)
        return pd.concat(rows).rename_axis("cataloguscode").reset_index()

    # Features

    def _history_similarity(self, history: dict[str, float], positions: np.ndarray, leave_one_out: bool) -> np.ndarray:
        liked = np.array([self.content.index[c] for c, s in history.items() if s >= 4 and c in self.content.index], dtype=int)
        if liked.size == 0:
            return np.zeros(len(positions))
        block = self.item_similarity[np.ix_(positions, liked)].copy()
        if leave_one_out:
            block[positions[:, None] == liked[None, :]] = np.nan
        counts = (~np.isnan(block)).sum(axis=1)
        sums = np.nansum(block, axis=1)
        return np.divide(sums, counts, out=np.zeros(len(positions)), where=counts > 0)

    def pair_features(
        self,
        profile: StudentProfile,
        history: dict[str, float],
        codes: list[str] | None = None,
        leave_one_out: bool = False,
        knn: ItemKNN | None = None,
        pop: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Feature table for the given materials (all materials when codes is None)."""
        all_static = static_features(profile, self.materials)
        codes = codes if codes is not None else self.materials["cataloguscode"].tolist()
        positions = np.array([self.content.index[code] for code in codes], dtype=int)
        frame = all_static.iloc[positions].copy()

        interest_scores = self.content.scores(profile.interesses)
        frame["interesse_sim"] = interest_scores[positions]
        frame["interesse_treffers"] = [len(self.content.interest_matches(profile.interesses, code)) for code in codes]
        frame["geschiedenis_sim"] = self._history_similarity(history, positions, leave_one_out)

        knn = knn or self.knn
        pop = pop if pop is not None else self.popularity
        if leave_one_out:
            frame["cf_knn"] = [knn.score_leave_one_out(history, code) for code in codes]
        else:
            frame["cf_knn"] = knn.scores(history)[[knn.index[code] for code in codes]]
        frame["populariteit"] = pop.reindex(codes).to_numpy()
        return frame[FEATURE_COLUMNS]

    # Scoring and recommending

    def score(self, profile: StudentProfile, history: dict[str, float] | None = None) -> pd.Series:
        """Probability of a positive interaction for every material."""
        features = self.pair_features(profile, history or {})
        probabilities = self.classifier.predict_proba(features)[:, 1]
        return pd.Series(probabilities, index=features.index)

    def recommend(
        self,
        profile: StudentProfile,
        history: dict[str, float] | None = None,
        n: int = 5,
        exclude_seen: bool = True,
    ) -> list[Recommendation]:
        history = history or {}
        features = self.pair_features(profile, history)
        scores = pd.Series(self.classifier.predict_proba(features)[:, 1], index=features.index)

        allowed = features["niveau_verschil"] <= self.max_level_gap
        if exclude_seen:
            allowed &= ~features.index.isin(list(history))
        candidates = scores[allowed].sort_values(ascending=False)
        chosen = self._diversify(candidates, n)

        by_code = self.materials.set_index("cataloguscode")
        results: list[Recommendation] = []
        for code in chosen:
            material = by_code.loc[code]
            hits = self.content.interest_matches(profile.interesses, code)
            reasons, summary = explain(profile, material, features.loc[code], hits)
            results.append(
                Recommendation(
                    cataloguscode=code,
                    key=str(material["key"]),
                    titel=str(material["titel"]),
                    auteur=None if pd.isna(material["auteur"]) else str(material["auteur"]),
                    type=str(material["type"]),
                    genre=str(material["genre"]),
                    niveau=str(material["niveau"]),
                    score=round(float(scores[code]), 4),
                    redenen=reasons,
                    samenvatting=summary,
                    kenmerken={name: round(float(value), 4) for name, value in features.loc[code].items()},
                )
            )
        return results

    def _diversify(self, candidates: pd.Series, n: int) -> list[str]:
        """Maximal marginal relevance: high score, but not too similar to what is already chosen.

        The client wants students to read more varied material, so near duplicates are pushed down.
        """
        pool = candidates.head(max(n * 8, 30))
        chosen: list[str] = []
        while pool.size and len(chosen) < n:
            if not chosen:
                best = pool.index[0]
            else:
                chosen_positions = [self.content.index[code] for code in chosen]
                pool_positions = [self.content.index[code] for code in pool.index]
                max_similarity = self.item_similarity[np.ix_(pool_positions, chosen_positions)].max(axis=1)
                adjusted = (1 - self.diversity) * pool.to_numpy() - self.diversity * max_similarity
                best = pool.index[int(np.argmax(adjusted))]
            chosen.append(best)
            pool = pool.drop(best)
        return chosen

    # Persistence

    def save(self, path: Path | str = config.MODEL_FILE) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | str = config.MODEL_FILE) -> "HybridRecommender":
        model = joblib.load(path)
        if not isinstance(model, HybridRecommender):
            raise TypeError(f"{path} bevat geen HybridRecommender")
        return model
