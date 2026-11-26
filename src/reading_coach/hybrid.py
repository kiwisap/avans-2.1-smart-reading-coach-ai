"""Hybrid recommender: content based and collaborative signals combined by a trained classifier.

For every (student, material) pair features are computed (level fit, type and genre fit,
text similarity with the interests, similarity with liked materials, collaborative score,
popularity, reading time). A Random Forest learns from the interactions which combination
predicts a positive interaction (the interaction types the data rates as positive). The predicted probability ranks the
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
from .synonyms import NARROWER_DEPTH, Related, as_terms, learn_synonyms
from .vocabulary import Vocabulary

MODEL_VERSION = "2.1.0"


def catalog_themes(materials: pd.DataFrame) -> list[str]:
    return sorted({term for value in materials["themes"].fillna("") for term in str(value).split(";") if term})


def student_interests(students: pd.DataFrame) -> list[str]:
    return sorted({term for value in students["interests"].fillna("") for term in str(value).split(";") if term})


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
    catalog_code: str
    key: str
    title: str
    author: str | None
    type: str
    genre: str
    level: str
    score: float
    reasons: list[str]  # loose reasons, most important first
    motivation: str  # the same reasons as a short story for the student
    features: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HybridRecommender:
    classifier: ClassifierMixin = field(default_factory=default_classifier)
    knn_neighbours: int = 20
    max_level_gap: float = 1.0
    diversity: float = 0.25
    narrower_depth: int = NARROWER_DEPTH
    materials: pd.DataFrame = field(init=False)
    vocabulary: Vocabulary = field(init=False)
    synonyms: dict[str, list[Related]] = field(init=False)
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
        self.classifier.fit(training[FEATURE_COLUMNS], training["positive"])
        importances = getattr(self.classifier, "feature_importances_", None)
        if importances is not None:
            self.feature_importance = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)
        else:
            self.feature_importance = pd.Series(dtype=float)
        return self

    def fit_components(self, materials: pd.DataFrame, students: pd.DataFrame, interactions: pd.DataFrame) -> None:
        """Fits everything except the classifier (vocabulary, content vectors, KNN, popularity)."""
        self.materials = materials.reset_index(drop=True)
        self.vocabulary = Vocabulary.learn(self.materials, interactions)
        self.synonyms = learn_synonyms(
            [*catalog_themes(self.materials), *student_interests(students)],
            catalog_themes(self.materials),
            narrower_depth=self.narrower_depth,
        )
        self.content = ContentModel().fit(self.materials, as_terms(self.synonyms))
        self.item_similarity = self.content.item_similarity()
        codes = self.materials["catalog_code"].tolist()
        matrix = user_item_matrix(interactions, codes, students["student_id"].tolist())
        self.knn = ItemKNN(k=self.knn_neighbours).fit(matrix)
        self.popularity = popularity(interactions, codes)

    def training_frame(self, students: pd.DataFrame, interactions: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
        """One row per interaction, with features that never see the label of that row.

        The collaborative score and the popularity are computed out of fold: the students are
        split in groups and for each group the KNN and popularity are fitted on the other groups.
        Within the own history the item itself is left out as well.
        """
        codes = self.materials["catalog_code"].tolist()
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
                profile = self.profile_of(student)
                history = dict(zip(own["catalog_code"], own["preference_score"]))
                features = self.pair_features(
                    profile, history, codes=own["catalog_code"].tolist(), leave_one_out=True, knn=knn, pop=pop
                )
                features["positive"] = own["positive"].to_numpy()
                features["student_id"] = student["student_id"]
                rows.append(features)
        return pd.concat(rows).rename_axis("catalog_code").reset_index()

    def profile_of(self, student: pd.Series) -> StudentProfile:
        """The profile of a student from the dataset, read with the vocabulary of this model."""
        return StudentProfile.from_dataset_row(student, self.vocabulary)

    # Features

    def _history_similarity(self, history: dict[str, float], positions: np.ndarray, leave_one_out: bool) -> np.ndarray:
        liked = np.array(
            [self.content.index[code] for code, score in history.items() if score >= 4 and code in self.content.index],
            dtype=int,
        )
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
        all_static = static_features(profile, self.materials, self.vocabulary)
        codes = codes if codes is not None else self.materials["catalog_code"].tolist()
        positions = np.array([self.content.index[code] for code in codes], dtype=int)
        frame = all_static.iloc[positions].copy()

        interest_scores = self.content.scores(profile.interests)
        frame["interest_sim"] = interest_scores[positions]
        frame["interest_hits"] = [len(self.content.interest_matches(profile.interests, code)) for code in codes]
        frame["history_sim"] = self._history_similarity(history, positions, leave_one_out)

        knn = knn or self.knn
        pop = pop if pop is not None else self.popularity
        if leave_one_out:
            frame["cf_knn"] = [knn.score_leave_one_out(history, code) for code in codes]
        else:
            frame["cf_knn"] = knn.scores(history)[[knn.index[code] for code in codes]]
        frame["popularity"] = pop.reindex(codes).to_numpy()
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

        allowed = features["level_diff"] <= self.max_level_gap
        if exclude_seen:
            allowed &= ~features.index.isin(list(history))
        candidates = scores[allowed].sort_values(ascending=False)
        chosen = self._diversify(candidates, n)

        by_code = self.materials.set_index("catalog_code", drop=False)
        results: list[Recommendation] = []
        for code in chosen:
            material = by_code.loc[code]
            hits = self.content.interest_matches(profile.interests, code)
            explanation = explain(
                profile,
                material,
                features.loc[code],
                hits,
                self.vocabulary,
                self.related_themes(profile.interests, material),
            )
            results.append(
                Recommendation(
                    catalog_code=code,
                    key=str(material["key"]),
                    title=str(material["title"]),
                    author=None if pd.isna(material["author"]) else str(material["author"]),
                    type=str(material["type"]),
                    genre=str(material["genre"]),
                    level=str(material["level"]),
                    score=round(float(scores[code]), 4),
                    reasons=explanation.facts,
                    motivation=explanation.story,
                    features={name: round(float(value), 4) for name, value in features.loc[code].items()},
                )
            )
        return results

    def related_themes(self, interests: list[str], material: pd.Series) -> dict[str, list[str]]:
        """Per interest, the themes of this material that the catalog links to that interest."""
        themes = set(str(material.get("themes") or "").split(";"))
        return {
            interest: [term for term in self.content.expander.related_terms(interest) if term in themes]
            for interest in interests
        }

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
        # Write next to the target and swap, so a running service never reads half a file.
        partial = path.with_name(f"{path.name}.partial")
        joblib.dump(self, partial)
        partial.replace(path)
        return path

    @staticmethod
    def load(path: Path | str = config.MODEL_FILE) -> "HybridRecommender":
        model = joblib.load(path)
        if not isinstance(model, HybridRecommender):
            raise TypeError(f"{path} does not contain a HybridRecommender")
        return model
