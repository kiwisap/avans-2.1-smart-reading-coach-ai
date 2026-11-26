"""Dutch text preprocessing (workshop 05 NLP): cleaning, tokenisation, stopwords and stemming.

The Dutch stopword list is included here so no NLTK download is needed. The Snowball
stemmer for Dutch ships with NLTK itself and works offline.
"""

from __future__ import annotations

import re
from functools import lru_cache

from nltk.stem.snowball import SnowballStemmer

from .config import INTEREST_SYNONYMS

DUTCH_STOPWORDS: frozenset[str] = frozenset(
    """
    aan al alle alles als altijd andere ben bij daar dan dat de der deze die dit doch doen door dus een
    eens en er ge geen geweest haar had heb hebben heeft hem het hier hij hoe hun iemand iets ik in is ja
    je kan kon kunnen maar me meer men met mij mijn moet na naar niet niets nog nu of om omdat onder ons
    ook op over reeds te tegen toch toen tot u uit uw van veel voor want waren was wat werd wezen wie wil
    worden wordt zal ze zelf zich zij zijn zo zonder zou hun hen waarin welke wel binnen tussen vaak
    steeds komt komen gaat maken laat staan staat elkaar twee eigen echter tijdens vanuit waarbij
    verhaal verhalen boek boeken artikel artikelen centraal herkenbare herkenbaar toegankelijk toegankelijke
    """.split()
)

TOKEN_PATTERN = re.compile(r"[a-zà-ÿ]+")


@lru_cache(maxsize=1)
def _stemmer() -> SnowballStemmer:
    return SnowballStemmer("dutch")


def normalize(text: str) -> str:
    """Lowercase, remove numbers and punctuation (keeps letters with accents)."""
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
    """Full pipeline used by the vectorizers: normalise, tokenise, drop stopwords, stem."""
    return " ".join(stem(remove_stopwords(tokenize(text or ""))))


def expand_interests(interests: list[str]) -> str:
    """Interests plus their synonyms as one text, so 'kunstmatige intelligentie' also matches 'AI'."""
    words: list[str] = []
    for interest in interests:
        interest = interest.strip().lower()
        if not interest:
            continue
        words.append(interest)
        words.extend(INTEREST_SYNONYMS.get(interest, []))
    return " ".join(words)
