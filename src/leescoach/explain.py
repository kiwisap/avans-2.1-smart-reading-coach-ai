"""Turns model features into short Dutch explanations for the student (and the teacher)."""

from __future__ import annotations

import math

import pandas as pd

from .features import StudentProfile

TYPE_LABELS = {
    "boek": "boek",
    "online-artikel": "online artikel",
    "blogpost": "blogpost",
    "dichtbundel": "dichtbundel",
    "tijdschrift": "tijdschrift",
    "krant": "krant",
}


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " en " + items[-1]


def _minutes(value: float) -> str:
    if value >= 90:
        hours = round(value / 60)
        return f"ongeveer {hours} uur"
    return f"ongeveer {int(math.ceil(value / 5.0) * 5)} minuten"


def explain(
    profile: StudentProfile,
    material: pd.Series,
    features: pd.Series,
    interest_hits: list[str],
) -> tuple[list[str], str]:
    """Returns (reasons, summary). Reasons are ordered from most to least important."""
    reasons: list[tuple[float, str]] = []
    material_type = TYPE_LABELS.get(str(material["type"]), str(material["type"]))

    if interest_hits:
        reasons.append((3.0 + len(interest_hits), f"Het gaat over {_join(interest_hits)}, iets waar je interesse in hebt."))
    elif features["interesse_sim"] > 0.08:
        reasons.append((2.0, "Het onderwerp sluit aan bij je interesses."))

    difference = features["niveau_verschil"]
    level = material["niveau"]
    if abs(difference) < 0.01:
        reasons.append((2.5, f"Het niveau ({level}) past precies bij jouw leesniveau ({profile.leesniveau})."))
    elif 0 < difference <= 0.5:
        reasons.append((2.2, f"Het niveau ({level}) is net iets uitdagender dan jouw niveau ({profile.leesniveau}), goed om te groeien."))
    elif difference < 0:
        reasons.append((1.0, f"Het niveau ({level}) is wat makkelijker dan jouw niveau ({profile.leesniveau}), dus het leest vlot."))
    else:
        reasons.append((0.5, f"Let op: het niveau ({level}) is pittig voor jouw niveau ({profile.leesniveau})."))

    if features["geschiedenis_sim"] > 0.12:
        reasons.append((2.0, "Het lijkt op materiaal dat je eerder goed vond."))
    if features["cf_knn"] > 0.3:
        reasons.append((1.8, "Studenten met een vergelijkbare smaak waardeerden dit hoog."))

    if features["type_match"] == 1.0:
        reasons.append((1.5, f"Het is een {material_type}, dat is het soort materiaal dat je graag leest."))
    if features["genre_match"] == 1.0:
        reasons.append((1.2, f"Het genre ({material['genre']}) is jouw voorkeur."))

    minutes = float(material["geschatte_leestijd_min"])
    if profile.max_leestijd_min:
        if features["leestijd_past"] == 1.0:
            reasons.append((1.0, f"De leestijd is {_minutes(minutes)}, dat past binnen jouw {int(profile.max_leestijd_min)} minuten."))
        elif material["type"] in {"boek", "dichtbundel"}:
            sessions = max(2, math.ceil(minutes / profile.max_leestijd_min))
            reasons.append((0.4, f"Het is langer dan je maximale leestijd; je kunt het in ongeveer {sessions} keer lezen."))

    if features["taal_match"] == 0.0:
        reasons.append((0.3, f"Er is nog weinig materiaal in het {profile.voorkeur_taal}; dit is een Nederlandse tekst."))

    rating = material.get("gem_beoordeling")
    if rating is not None and not pd.isna(rating) and float(rating) >= 3.7:
        reasons.append((0.8, f"Andere lezers geven het een hoge gemiddelde beoordeling ({float(rating):.1f})."))

    ordered = [text for _, text in sorted(reasons, key=lambda item: item[0], reverse=True)]
    top = ordered[:4]
    summary = top[0] if top else "Dit materiaal past bij je profiel."
    return top, summary
