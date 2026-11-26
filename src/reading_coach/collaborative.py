"""Collaborative filtering: item based KNN and matrix factorisation (SVD) on the interactions.

Interactions are converted to a preference score from 1 to 5 and centred around 3, so a
rejected material counts as a negative signal and a liked material as a positive one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from .config import NEUTRAL_SCORE


def user_item_matrix(
    interactions: pd.DataFrame,
    codes: list[str],
    students: list[str],
    value_column: str = "preference_score",
) -> pd.DataFrame:
    """Students as rows, materials as columns, centred preference as value and 0 when unknown."""
    pivot = interactions.pivot_table(index="student_id", columns="catalog_code", values=value_column, aggfunc="mean")
    pivot = pivot.reindex(index=students, columns=codes)
    return (pivot - NEUTRAL_SCORE).fillna(0.0)


def history_to_vector(history: dict[str, float], index: dict[str, int], size: int) -> np.ndarray:
    vector = np.zeros(size)
    for code, score in history.items():
        position = index.get(code)
        if position is not None:
            vector[position] = score - NEUTRAL_SCORE
    return vector


@dataclass
class ItemKNN:
    """Scores a material by how similar it is to materials the student rated well or badly."""

    k: int = 20
    similarity: np.ndarray = field(init=False)
    codes: list[str] = field(init=False)
    index: dict[str, int] = field(init=False)

    def fit(self, matrix: pd.DataFrame) -> "ItemKNN":
        self.codes = list(matrix.columns)
        self.index = {code: position for position, code in enumerate(self.codes)}
        similarity = cosine_similarity(matrix.to_numpy().T)
        np.fill_diagonal(similarity, 0.0)
        # Keep only the k strongest neighbours per material to reduce noise.
        if self.k and self.k < similarity.shape[0]:
            weakest = np.argsort(np.abs(similarity), axis=1)[:, : -self.k]
            np.put_along_axis(similarity, weakest, 0.0, axis=1)
        self.similarity = similarity
        return self

    def scores(self, history: dict[str, float]) -> np.ndarray:
        """Predicted centred preference (about -2 to 2) for every material, 0 when there is no evidence."""
        ratings = history_to_vector(history, self.index, len(self.codes))
        rated = ratings != 0
        if not rated.any():
            return np.zeros(len(self.codes))
        weights = self.similarity[:, rated]
        numerator = weights @ ratings[rated]
        denominator = np.abs(weights).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denominator > 0, numerator / denominator, 0.0)
        return result

    def score_leave_one_out(self, history: dict[str, float], code: str) -> float:
        """Score for one material without using the interaction with that same material (no leakage)."""
        reduced = {other: value for other, value in history.items() if other != code}
        position = self.index.get(code)
        if position is None or not reduced:
            return 0.0
        return float(self.scores(reduced)[position])


@dataclass
class SVDRecommender:
    """Matrix factorisation: students and materials in a small number of latent taste dimensions."""

    n_components: int = 12
    random_state: int = 42
    model: TruncatedSVD = field(init=False)
    codes: list[str] = field(init=False)
    index: dict[str, int] = field(init=False)

    def fit(self, matrix: pd.DataFrame) -> "SVDRecommender":
        self.codes = list(matrix.columns)
        self.index = {code: position for position, code in enumerate(self.codes)}
        components = min(self.n_components, min(matrix.shape) - 1)
        self.model = TruncatedSVD(n_components=components, random_state=self.random_state)
        self.model.fit(matrix.to_numpy())
        return self

    def scores(self, history: dict[str, float]) -> np.ndarray:
        """Fold in: projects the history of any (also a new) student on the latent space and back."""
        vector = history_to_vector(history, self.index, len(self.codes))
        latent = self.model.transform(vector.reshape(1, -1))
        return self.model.inverse_transform(latent).ravel()


def popularity(interactions: pd.DataFrame, codes: list[str], smoothing: float = 5.0) -> pd.Series:
    """Share of positive interactions per material, smoothed towards the overall mean for rare items."""
    prior = interactions["positive"].mean()
    stats = interactions.groupby("catalog_code")["positive"].agg(["sum", "count"])
    smoothed = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
    return smoothed.reindex(codes).fillna(prior)
