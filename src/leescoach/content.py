"""Content based part of the recommender (workshop 06): TF-IDF vectors and cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import INTEREST_SYNONYMS
from .text import expand_interests, preprocess, stem, tokenize


@dataclass
class ContentModel:
    """Represents every material as a TF-IDF vector of its title, description, themes and url words."""

    min_df: int = 1
    ngram_range: tuple[int, int] = (1, 2)
    vectorizer: TfidfVectorizer = field(init=False)
    matrix: sparse.csr_matrix = field(init=False)
    codes: list[str] = field(init=False)
    index: dict[str, int] = field(init=False)
    stem_sets: list[set[str]] = field(init=False)

    def fit(self, materials: pd.DataFrame) -> "ContentModel":
        documents = materials["tekst"].fillna("").map(preprocess)
        self.vectorizer = TfidfVectorizer(min_df=self.min_df, ngram_range=self.ngram_range, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(documents).tocsr()
        self.codes = materials["cataloguscode"].tolist()
        self.index = {code: position for position, code in enumerate(self.codes)}
        self.stem_sets = [set(document.split()) for document in documents]
        return self

    # Vectors

    def text_vector(self, text: str) -> sparse.csr_matrix:
        return self.vectorizer.transform([preprocess(text)])

    def profile_vector(
        self,
        interests: list[str],
        liked_codes: list[str] | None = None,
        history_weight: float = 0.4,
    ) -> sparse.csr_matrix:
        """Interests (with synonyms) and, when available, the mean vector of materials the student liked."""
        vector = self.text_vector(expand_interests(interests))
        liked = [self.index[code] for code in (liked_codes or []) if code in self.index]
        if liked:
            history = sparse.csr_matrix(self.matrix[liked].mean(axis=0))
            vector = (1 - history_weight) * vector + history_weight * history
        return sparse.csr_matrix(vector)

    # Scores

    def scores(self, interests: list[str], liked_codes: list[str] | None = None) -> np.ndarray:
        """Cosine similarity between the student profile and every material (0 to 1)."""
        profile = self.profile_vector(interests, liked_codes)
        if profile.nnz == 0:
            return np.zeros(len(self.codes))
        return cosine_similarity(profile, self.matrix).ravel()

    def interest_matches(self, interests: list[str], code: str) -> list[str]:
        """Which interests of the student literally occur (stemmed, with synonyms) in a material."""
        position = self.index.get(code)
        if position is None:
            return []
        words = self.stem_sets[position]
        hits: list[str] = []
        for interest in interests:
            interest = interest.strip().lower()
            candidates = [interest, *INTEREST_SYNONYMS.get(interest, [])]
            stems = {token for candidate in candidates for token in stem(tokenize(candidate))}
            if stems & words:
                hits.append(interest)
        return hits

    def item_similarity(self) -> np.ndarray:
        return cosine_similarity(self.matrix)

    def similar_items(self, code: str, top_n: int = 5) -> list[tuple[str, float]]:
        position = self.index[code]
        similarities = cosine_similarity(self.matrix[position], self.matrix).ravel()
        similarities[position] = -1
        best = np.argsort(similarities)[::-1][:top_n]
        return [(self.codes[i], float(similarities[i])) for i in best]

    def search(self, query: str, top_n: int = 5) -> list[tuple[str, float]]:
        similarities = cosine_similarity(self.text_vector(query), self.matrix).ravel()
        best = np.argsort(similarities)[::-1][:top_n]
        return [(self.codes[i], float(similarities[i])) for i in best]
