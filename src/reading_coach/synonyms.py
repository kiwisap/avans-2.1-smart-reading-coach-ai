"""Terms that mean the same, learned from a dictionary instead of typed into the code.

A student who picks 'kunstmatige intelligentie' should find the articles with the theme 'ai',
and 'maatschappij' should also find 'samenleving'. The catalog alone cannot tell that: those
words never occur together. The Open Multilingual Wordnet can. It groups words with the same
meaning in synsets, for Dutch (from Open Dutch WordNet) and for English. Two terms are related
when they share a synset, or when the catalog term is a narrower meaning of the interest
(misdaad: cybercrime). The English words of a synset count too, because the catalog uses
loanwords such as 'ai'.

Only terms that really occur as a theme in the catalog are kept, so every link leads to material.
The result is stored with the model and in MongoDB (collection topic_synonyms), where the Fastify
backend uses it as well. When the wordnet data is not installed, there simply are no synonyms
and the catalog based expansion (content.InterestExpander) still works.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

log = logging.getLogger(__name__)

WORDNET_PACKAGES = ("wordnet", "omw-2.0")  # English wordnet plus the Open Multilingual Wordnet
DUTCH = "nld"
NARROWER_DEPTH = 1  # chosen with the offline evaluation (train.py)


@dataclass(frozen=True)
class Related:
    term: str
    relation: str  # "synonym" or "narrower"


@lru_cache(maxsize=1)
def _wordnet():
    """The wordnet corpus reader, or None when the data is not installed."""
    try:
        from nltk.corpus import wordnet

        # The Dutch part loads lazily; a common word tells whether it is there.
        if not wordnet.synsets("boek", lang=DUTCH):
            raise LookupError("the Open Multilingual Wordnet has no Dutch words")
        return wordnet
    except LookupError as error:
        log.warning("Wordnet data is not installed, no synonyms are learned: %s", error)
        return None


def wordnet_available() -> bool:
    return _wordnet() is not None


def download_wordnet(directory: str | None = None) -> bool:
    """Installs the wordnet data (about 40 MB). The Docker build does this once."""
    import nltk

    ok = all(nltk.download(package, download_dir=directory, quiet=True) for package in WORDNET_PACKAGES)
    _wordnet.cache_clear()
    return ok


def _synsets(wordnet, term: str) -> set:
    lemma = term.strip().lower().replace(" ", "_")
    found = set(wordnet.synsets(lemma, lang=DUTCH))
    # Loanwords such as 'ai' or 'blog' are often only in the English part.
    return found or set(wordnet.synsets(lemma))


def _narrower(synsets: set, depth: int) -> set:
    found: set = set()
    for synset in synsets:
        found.update(synset.closure(lambda s: s.hyponyms(), depth=depth))
    return found


def learn_synonyms(
    terms: Iterable[str],
    catalog_themes: Iterable[str],
    narrower_depth: int = NARROWER_DEPTH,
) -> dict[str, list[Related]]:
    """Per term, the catalog themes with the same meaning (synonym) or a narrower one."""
    wordnet = _wordnet()
    if wordnet is None:
        return {}
    themes = sorted({theme.strip().lower() for theme in catalog_themes if theme and theme.strip()})
    theme_synsets = {theme: _synsets(wordnet, theme) for theme in themes}
    result: dict[str, list[Related]] = {}
    for term in sorted({term.strip().lower() for term in terms if term and term.strip()}):
        own = theme_synsets.get(term) or _synsets(wordnet, term)
        if not own:
            continue
        narrower = _narrower(own, narrower_depth) if narrower_depth > 0 else set()
        related: list[Related] = []
        for theme, synsets in theme_synsets.items():
            if theme == term or not synsets:
                continue
            if own & synsets:
                related.append(Related(theme, "synonym"))
            elif narrower & synsets:
                related.append(Related(theme, "narrower"))
        if related:
            result[term] = related
    return result


def as_terms(synonyms: dict[str, list[Related]]) -> dict[str, list[str]]:
    """Synonyms first, then narrower terms, as plain strings."""
    order = {"synonym": 0, "narrower": 1}
    return {
        term: [item.term for item in sorted(related, key=lambda item: (order[item.relation], item.term))]
        for term, related in synonyms.items()
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not download_wordnet(target):
        raise SystemExit("Downloading the wordnet data failed")
    print("Wordnet data installed")
