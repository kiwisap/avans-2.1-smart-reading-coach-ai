"""Student profile and the features that describe how well one material fits one student."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

FEATURE_COLUMNS = [
    "niveau_verschil",
    "niveau_afstand",
    "type_match",
    "genre_match",
    "interesse_sim",
    "interesse_treffers",
    "geschiedenis_sim",
    "leestijd_ratio_log",
    "leestijd_past",
    "taal_match",
    "gem_beoordeling",
    "populariteit",
    "cf_knn",
    "leescomplexiteit_score",
]

# Readable Dutch names for plots and for the teacher view.
FEATURE_LABELS = {
    "niveau_verschil": "Niveauverschil (materiaal min student)",
    "niveau_afstand": "Afstand tot eigen niveau",
    "type_match": "Past bij voorkeur materiaaltype",
    "genre_match": "Past bij voorkeur genre",
    "interesse_sim": "Tekstovereenkomst met interesses",
    "interesse_treffers": "Aantal interesses in de tekst",
    "geschiedenis_sim": "Lijkt op eerder gewaardeerd materiaal",
    "leestijd_ratio_log": "Leestijd ten opzichte van maximum (log)",
    "leestijd_past": "Binnen maximale leestijd",
    "taal_match": "Juiste taal",
    "gem_beoordeling": "Gemiddelde beoordeling",
    "populariteit": "Populair bij andere studenten",
    "cf_knn": "Gewaardeerd door studenten met dezelfde smaak",
    "leescomplexiteit_score": "Leescomplexiteit",
}


@dataclass
class StudentProfile:
    """What the model needs to know about a student. Can come from the dataset or from the app."""

    leesniveau: str
    interesses: list[str]
    materiaaltypes: list[str] = field(default_factory=lambda: list(config.APP_MATERIAL_TYPES))
    voorkeur_genre: str = "geen voorkeur"
    max_leestijd_min: float | None = None
    voorkeur_taal: str = "Nederlands"
    student_id: str | None = None

    @property
    def niveau_num(self) -> float:
        return config.LEVEL_TO_NUMBER.get(self.leesniveau.upper(), 2.0)

    @property
    def heeft_type_voorkeur(self) -> bool:
        return 0 < len(set(self.materiaaltypes)) < len(config.APP_MATERIAL_TYPES)

    @classmethod
    def from_dataset_row(cls, row: pd.Series) -> "StudentProfile":
        interests = [value for value in str(row["interesses"]).split(";") if value]
        types = config.STUDENT_TYPE_PREFERENCE.get(str(row["voorkeur_materiaaltype"]).lower(), config.APP_MATERIAL_TYPES)
        return cls(
            student_id=str(row["student_id"]),
            leesniveau=str(row["leesniveau"]),
            interesses=interests,
            materiaaltypes=list(types),
            voorkeur_genre=str(row["voorkeur_genre"]).lower(),
            max_leestijd_min=float(row["max_leestijd_min"]) if pd.notna(row["max_leestijd_min"]) else None,
            voorkeur_taal=str(row["voorkeur_taal"]),
        )


def static_features(profile: StudentProfile, materials: pd.DataFrame) -> pd.DataFrame:
    """Features that only depend on the profile and the material metadata (no interactions)."""
    level_difference = materials["niveau_num"].to_numpy() - profile.niveau_num
    if profile.heeft_type_voorkeur:
        type_match = materials["type"].isin(profile.materiaaltypes).astype(float).to_numpy()
    else:
        type_match = np.full(len(materials), 0.5)
    if profile.voorkeur_genre in config.GENRES:
        genre_match = (materials["genre"] == profile.voorkeur_genre).astype(float).to_numpy()
    else:
        genre_match = np.full(len(materials), 0.5)
    minutes = materials["geschatte_leestijd_min"].to_numpy(dtype=float)
    if profile.max_leestijd_min:
        ratio = minutes / profile.max_leestijd_min
        fits = (minutes <= profile.max_leestijd_min).astype(float)
    else:
        ratio = np.ones(len(materials))
        fits = np.ones(len(materials))
    rating = materials["gem_beoordeling"].astype(float)
    return pd.DataFrame(
        {
            "niveau_verschil": level_difference,
            "niveau_afstand": np.abs(level_difference),
            "type_match": type_match,
            "genre_match": genre_match,
            "leestijd_ratio_log": np.log1p(ratio),
            "leestijd_past": fits,
            "taal_match": (materials["taal"].str.lower() == profile.voorkeur_taal.lower()).astype(float).to_numpy(),
            "gem_beoordeling": rating.fillna(rating.mean()).to_numpy(),
            "leescomplexiteit_score": materials["leescomplexiteit_score"].astype(float).to_numpy(),
        },
        index=materials["cataloguscode"].to_numpy(),
    )
