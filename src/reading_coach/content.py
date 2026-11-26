"""Content based part of the recommender (workshop content based filtering): TF-IDF and cosine similarity."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import preprocess, stems_of, stems_overlap


@dataclass
class InterestExpander:
    """Finds catalog themes that belong to an interest, learned instead of typed in.

    Two sources, both outside the code:

    * synonyms: themes with the same or a narrower meaning according to the wordnet
      (synonyms.py), for example 'kunstmatige intelligentie' and 'ai';
    * the catalog itself: a student who likes 'technologie' should also find articles about
      'digitalisering' and 'privacy'. The expander looks at the materials that mention the
      interest (or one of its synonyms) and takes the themes that occur clearly more often there
      than in the catalog as a whole (lift).

    New books bring their own themes along automatically.
    """

    min_count: int = 1
    min_lift: float = 1.5  # chosen with the offline evaluation (train.py)
    max_terms: int = 6
    theme_stems: list[set[str]] = field(init=False)
    text_stems: list[set[str]] = field(init=False)
    themes: list[list[str]] = field(init=False)
    theme_counts: Counter = field(init=False)
    synonyms: dict[str, list[str]] = field(init=False, default_factory=dict)
    cache: dict[str, list[str]] = field(init=False, default_factory=dict)

    def fit(
        self,
        materials: pd.DataFrame,
        text_stems: list[set[str]],
        synonyms: dict[str, list[str]] | None = None,
    ) -> "InterestExpander":
        self.themes = [[term for term in str(value).split(";") if term] for value in materials["themes"].fillna("")]
        self.theme_stems = [stems_of(" ".join(terms)) for terms in self.themes]
        self.text_stems = text_stems
        self.theme_counts = Counter(term for terms in self.themes for term in set(terms))
        self.synonyms = synonyms or {}
        self.cache = {}
        return self

    def synonyms_of(self, interest: str) -> list[str]:
        return self.synonyms.get(interest.strip().lower(), [])

    def matching(self, interest: str) -> list[int]:
        """Positions of the materials that mention the interest, or one of its synonyms."""
        wanted = [stems for stems in map(stems_of, [interest, *self.synonyms_of(interest)]) if stems]
        if not wanted:
            return []
        return [
            position
            for position, (text, themes) in enumerate(zip(self.text_stems, self.theme_stems))
            if any(stems_overlap(stems, text | themes) for stems in wanted)
        ]

    def related_terms(self, interest: str) -> list[str]:
        interest = interest.strip().lower()
        if interest not in self.cache:
            positions = self.matching(interest)
            total = len(self.themes)
            counts = Counter(term for position in positions for term in set(self.themes[position]))
            scored = []
            for term, count in counts.items():
                if count < self.min_count or term == interest:
                    continue
                lift = (count / len(positions)) / (self.theme_counts[term] / total)
                if lift >= self.min_lift:
                    scored.append((lift * count, term))
            synonyms = self.synonyms_of(interest)
            learned = [term for _, term in sorted(scored, reverse=True) if term not in synonyms]
            self.cache[interest] = synonyms + learned[: self.max_terms]
        return self.cache[interest]

    def expand(self, interests: list[str]) -> str:
        """The interests plus their related catalog themes, as one text."""
        words: list[str] = []
        for interest in interests:
            interest = interest.strip().lower()
            if interest:
                words.append(interest)
                words.extend(self.related_terms(interest))
        return " ".join(words)


@dataclass
class ContentModel:
    """Every material as a TF-IDF vector of its title, description, themes and url words."""

    min_df: int = 1
    max_df: float = 0.5
    ngram_range: tuple[int, int] = (1, 2)
    vectorizer: TfidfVectorizer = field(init=False)
    matrix: sparse.csr_matrix = field(init=False)
    codes: list[str] = field(init=False)
    index: dict[str, int] = field(init=False)
    stem_sets: list[set[str]] = field(init=False)
    expander: InterestExpander = field(init=False)

    def fit(self, materials: pd.DataFrame, synonyms: dict[str, list[str]] | None = None) -> "ContentModel":
        documents = materials["text"].fillna("").map(preprocess)
        self.vectorizer = TfidfVectorizer(
            min_df=self.min_df, max_df=self.max_df, ngram_range=self.ngram_range, sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(documents).tocsr()
        self.codes = materials["catalog_code"].tolist()
        self.index = {code: position for position, code in enumerate(self.codes)}
        self.stem_sets = [set(document.split()) for document in documents]
        self.expander = InterestExpander().fit(materials, self.stem_sets, synonyms)
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
        """The interests (with related themes) and, when known, the mean vector of liked materials."""
        vector = self.text_vector(self.expander.expand(interests))
        liked = [self.index[code] for code in (liked_codes or []) if code in self.index]
        if liked:
            history = sparse.csr_matrix(self.matrix[liked].mean(axis=0))
            vector = (1 - history_weight) * vector + history_weight * history
        return sparse.csr_matrix(vector)

    # Scores

    def scores(self, interests: list[str], liked_codes: list[str] | None = None) -> np.ndarray:
        """Cosine similarity between the profile and every material (0 to 1)."""
        profile = self.profile_vector(interests, liked_codes)
        if profile.nnz == 0:
            return np.zeros(len(self.codes))
        return cosine_similarity(profile, self.matrix).ravel()

    def interest_matches(self, interests: list[str], code: str) -> list[str]:
        """Which interests of the student the text of a material really mentions, by name or synonym."""
        position = self.index.get(code)
        if position is None:
            return []
        words = self.stem_sets[position]
        return [
            interest
            for interest in interests
            if interest.strip()
            and any(
                stems_overlap(stems_of(term), words) for term in [interest, *self.expander.synonyms_of(interest)]
            )
        ]

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
