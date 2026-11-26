from __future__ import annotations

import pandas as pd

from leescoach import config
from leescoach.data_cleaning import build_app_key, clean_url, split_terms, to_number


def test_to_number_handles_percentages_and_text() -> None:
    result = to_number(pd.Series(["26.7%", "3,5", "onbekend", None]))
    assert result.iloc[0] == 26.7
    assert result.iloc[1] == 3.5
    assert result.iloc[2:].isna().all()


def test_split_terms_merges_and_deduplicates() -> None:
    assert split_terms("WOII;onderduik", "WOII; onderduik", None) == ["woii", "onderduik"]


def test_clean_url_strips_domain_label() -> None:
    assert clean_url("https://prodemos.nl/kennis/ [prodemos.nl]") == "https://prodemos.nl/kennis/"
    assert clean_url("geen url") is None


def test_app_key_matches_fastify_normalisation() -> None:
    assert build_app_key("Het dagboek van Anne Frank", "Anne Frank", "boek") == "het dagboek van anne frank|anne frank|boek"
    assert build_app_key("Café Één", None, "boek") == "cafe een||boek"


def test_clean_materials_has_valid_values(data) -> None:
    materials = data.materials
    assert materials["cataloguscode"].is_unique
    assert set(materials["type"]) <= set(config.APP_MATERIAL_TYPES)
    assert set(materials["niveau"]) <= set(config.LEVEL_TO_NUMBER)
    assert set(materials["genre"]) <= set(config.GENRES)
    assert materials["niveau_num"].notna().all()
    assert (materials["aantal_paginas"] > 0).all()
    assert materials["publicatiejaar"].dropna().le(pd.Timestamp.now().year).all()


def test_interactions_point_to_existing_materials(data) -> None:
    assert data.interactions["cataloguscode"].isin(data.materials["cataloguscode"]).all()
    assert set(data.interactions["positief"].unique()) <= {0, 1}
