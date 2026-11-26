"""Turns the features of one advice into a short story in Dutch, the way Bo would tell it.

The student sees one small paragraph ("waarom dit bij jou past") instead of a list of loose
facts. The story is built from the same facts the model used: the interests found in the
text, the level, the reading time, earlier favourites and students with a similar taste.
The separate facts are returned as well, ordered by importance, for teachers and tests.

The sentences are Dutch because the app is Dutch. Which phrasing is used depends on the
material, so five advices next to each other do not all start the same way.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import pandas as pd

from .features import StudentProfile
from .vocabulary import Vocabulary

STRONG_TEXT_MATCH = 0.08
STRONG_HISTORY_MATCH = 0.12
STRONG_TASTE_MATCH = 0.3
HIGH_RATING = 3.7
MAX_FACTS = 4


@dataclass
class Explanation:
    facts: list[str]  # loose reasons, most important first
    story: str  # the same reasons as one paragraph


def join_words(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " en " + items[-1]


def describe_minutes(value: float) -> str:
    if value >= 90:
        return f"ongeveer {round(value / 60)} uur"
    return f"ongeveer {int(math.ceil(value / 5.0) * 5)} minuten"


def pick(options: list[str], seed: str) -> str:
    """The same material always gets the same phrasing, different materials get different ones."""
    return options[zlib.crc32(seed.encode("utf-8")) % len(options)]


def explain(
    profile: StudentProfile,
    material: pd.Series,
    features: pd.Series,
    interest_hits: list[str],
    vocabulary: Vocabulary,
    related_themes: dict[str, list[str]] | None = None,
) -> Explanation:
    """Builds the explanation of one advice.

    related_themes: per interest of the student, the themes of this material that the catalog
    links to that interest (InterestExpander), for example 'thriller' for 'spanning'.
    """
    code = str(material["catalog_code"])
    themes = [term for term in str(material.get("themes") or "").split(";") if term]
    links = {interest: terms for interest, terms in (related_themes or {}).items() if terms}
    kind = vocabulary.label(str(material["type"]))
    level = str(material["level"])
    own_level = profile.reading_level
    facts: list[tuple[float, str]] = []

    # 1. Why it is interesting: the subject, earlier favourites or students like this one.
    if interest_hits:
        topics = join_words(interest_hits)
        opening = pick(
            [
                f"Je houdt van {topics}, en precies daar gaat dit over.",
                f"Omdat je van {topics} houdt, denk ik dat dit echt iets voor jou is.",
                f"Het gaat over {topics}, en daar lees jij graag over.",
            ],
            code,
        )
        facts.append((3.0 + len(interest_hits), f"Het gaat over {topics}, iets waar je interesse in hebt."))
    elif links:
        interest, terms = next(iter(links.items()))
        opening = f"Je houdt van {interest}, en dit gaat over {join_words(terms[:2])}: dat past daar mooi bij."
        facts.append((2.5, f"Het gaat over {join_words(terms[:2])}, iets wat vaak samengaat met {interest}."))
    elif features["history_sim"] > STRONG_HISTORY_MATCH:
        opening = "Dit lijkt op wat je eerder goed vond."
        facts.append((2.0, "Het lijkt op materiaal dat je eerder goed vond."))
    elif features["interest_sim"] > STRONG_TEXT_MATCH:
        opening = "Het onderwerp sluit aan bij waar jij in geïnteresseerd bent."
        facts.append((2.0, "Het onderwerp sluit aan bij je interesses."))
    elif features["cf_knn"] > STRONG_TASTE_MATCH:
        opening = "Lezers met een smaak zoals die van jou waren hier enthousiast over."
        facts.append((1.8, "Studenten met een vergelijkbare smaak waardeerden dit hoog."))
    elif themes:
        opening = f"Dit gaat over {join_words(themes[:2])}, en ik denk dat het goed bij je past."
    else:
        opening = "Ik denk dat dit goed bij je past."

    # 2. Whether it is doable: level and reading time in one sentence.
    difference = features["level_diff"]
    if abs(difference) < 0.01:
        level_part = f"het is een {kind} op {level}, precies jouw niveau"
        facts.append((2.5, f"Het niveau ({level}) past precies bij jouw leesniveau ({own_level})."))
    elif 0 < difference <= 0.5:
        level_part = f"het is een {kind} op {level}, net iets uitdagender dan {own_level} en dus goed om van te groeien"
        facts.append((2.2, f"Het niveau ({level}) is net iets uitdagender dan jouw niveau ({own_level})."))
    elif difference < 0:
        level_part = f"het is een {kind} op {level}, iets makkelijker dan {own_level}, dus het leest lekker vlot"
        facts.append((1.0, f"Het niveau ({level}) is wat makkelijker dan jouw niveau ({own_level})."))
    else:
        level_part = f"het is een {kind} op {level}, en dat is pittig voor {own_level}"
        facts.append((0.5, f"Het niveau ({level}) is pittig voor jouw niveau ({own_level})."))

    minutes = float(material["reading_minutes"])
    time_part = ""
    if profile.max_minutes:
        if features["fits_time"] == 1.0:
            time_part = f"met {describe_minutes(minutes)} leestijd past het in de tijd die je hebt"
            facts.append((1.0, f"De leestijd is {describe_minutes(minutes)}, dat past binnen jouw {int(profile.max_minutes)} minuten."))
        elif minutes > profile.max_minutes:
            sessions = max(2, math.ceil(minutes / profile.max_minutes))
            time_part = f"al is het langer dan je wilde, in ongeveer {sessions} keer heb je het uit"
            facts.append((0.4, f"Het is langer dan je maximale leestijd; je kunt het in ongeveer {sessions} keer lezen."))
    middle = level_part[0].upper() + level_part[1:]
    middle += f", en {time_part}." if time_part else "."

    # 3. A last nudge: students with the same taste, earlier favourites or a high rating.
    closing = ""
    if (interest_hits or links) and features["cf_knn"] > STRONG_TASTE_MATCH:
        closing = "Bovendien vonden lezers met een smaak zoals die van jou het goed."
        facts.append((1.8, "Studenten met een vergelijkbare smaak waardeerden dit hoog."))
    elif (interest_hits or links) and features["history_sim"] > STRONG_HISTORY_MATCH:
        closing = "Het lijkt ook op wat je eerder goed vond."
        facts.append((2.0, "Het lijkt op materiaal dat je eerder goed vond."))
    rating = material.get("avg_rating")
    if rating is not None and not pd.isna(rating) and float(rating) >= HIGH_RATING:
        rating_text = f"{float(rating):.1f}".replace(".", ",")
        facts.append((0.8, f"Andere lezers geven het een hoge gemiddelde beoordeling ({rating_text})."))
        if not closing:
            closing = f"Andere lezers geven het gemiddeld een {rating_text}."
    if features["type_match"] == 1.0:
        facts.append((1.5, f"Het is een {kind}, het soort tekst dat je graag leest."))
    if features["genre_match"] == 1.0:
        facts.append((1.2, f"Het genre ({material['genre']}) is jouw voorkeur."))

    story = " ".join(part for part in [opening, middle, closing] if part)
    ordered = [text for _, text in sorted(facts, key=lambda item: item[0], reverse=True)]
    return Explanation(facts=list(dict.fromkeys(ordered))[:MAX_FACTS], story=story)
