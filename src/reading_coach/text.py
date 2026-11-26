"""Dutch text preprocessing (workshop NLP): cleaning, tokenising, stopwords and stemming.

The stopword list is a general Dutch list, included here so no NLTK download is needed. Words
that are common in this particular catalog ('boek', 'verhaal') are not listed by hand: the
TF-IDF vectorizer drops words that occur in a large part of the documents (max_df), so that
keeps working when the catalog changes. The Snowball stemmer for Dutch ships with NLTK.
"""

from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem.snowball import SnowballStemmer

DUTCH_STOPWORDS: frozenset[str] = frozenset(
    """
    aan al alle alles als altijd andere ben bij daar dan dat de der deze die dit doch doen door dus een
    eens en er ge geen geweest haar had heb hebben heeft hem het hier hij hoe hun iemand iets ik in is ja
    je kan kon kunnen maar me meer men met mij mijn moet na naar niet niets nog nu of om omdat onder ons
    ook op over reeds te tegen toch toen tot u uit uw van veel voor want waren was wat werd wezen wie wil
    worden wordt zal ze zelf zich zij zijn zo zonder zou hen waarin welke wel binnen tussen vaak steeds
    komt komen gaat maken laat staan staat elkaar twee eigen echter tijdens vanuit waarbij
    """.split()
)

TOKEN_PATTERN = re.compile(r"[a-zà-ÿ]+")


@lru_cache(maxsize=1)
def _stemmer() -> SnowballStemmer:
    return SnowballStemmer("dutch")


def normalize(text: str) -> str:
    """Lowercase, without numbers and punctuation (letters with accents stay)."""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize(text))


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in DUTCH_STOPWORDS and len(token) > 1]


def stem(tokens: list[str]) -> list[str]:
    stemmer = _stemmer()
    return [stemmer.stem(token) for token in tokens]


def preprocess(text: str) -> str:
    """The whole pipeline the vectorizers use: normalise, tokenise, drop stopwords, stem."""
    return " ".join(stem(remove_stopwords(tokenize(text or ""))))


def stems_of(text: str) -> set[str]:
    return set(preprocess(text).split())


def stems_overlap(first: set[str], second: set[str], min_prefix: int = 5) -> bool:
    """True when two sets of stems share a stem, or two stems start with the same five letters.

    The Snowball stemmer keeps 'spanning' and 'spannend' apart; the shared start catches that.
    """
    if first & second:
        return True
    prefixes = {word[:min_prefix] for word in first if len(word) >= min_prefix}
    return any(word[:min_prefix] in prefixes for word in second if len(word) >= min_prefix)
