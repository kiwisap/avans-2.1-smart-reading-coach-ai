from __future__ import annotations

import numpy as np
import pandas as pd

from reading_coach.data_cleaning import build_app_key, clean_url, split_terms, to_number
from reading_coach.vocabulary import language_mapping, learn_interaction_scores, parse_level, type_mapping


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


def test_missing_author_gives_empty_key_part() -> None:
    assert build_app_key("Libello voor mannen", pd.NA, "tijdschrift") == "libello voor mannen||tijdschrift"


def test_levels_are_parsed_without_a_fixed_list() -> None:
    assert parse_level("2F") == 2.0
    assert parse_level("3f+") == 3.5
    assert parse_level("2F-3F") == 2.5
    assert parse_level("4F") == 4.0  # not in the catalog yet, still understood
    assert np.isnan(parse_level("B1"))
    assert np.isnan(parse_level(None))


def test_material_types_come_from_the_data() -> None:
    raw = pd.Series(["Boek", "Boek - thriller", "online artikel", "krant (papier)", "digitale krant", "Podcast", None])
    mapping = type_mapping(raw)
    assert mapping["Boek - thriller"] == ("boek", "boek", "thriller")
    assert mapping["online artikel"] == ("online-artikel", "online artikel", None)
    assert mapping["digitale krant"][0] == "krant"
    # A new kind of material in the catalog simply becomes a new type.
    assert mapping["Podcast"][0] == "podcast"


def test_language_spellings_go_to_the_most_used_full_name() -> None:
    mapping = language_mapping(pd.Series(["Nederlands", "Nederlands", "NL", "NED", "nederlands", "Engels", "EN"]))
    assert {mapping[value] for value in ["NL", "NED", "nederlands", "Nederlands"]} == {"Nederlands"}
    assert mapping["EN"] == "Engels"


def test_interaction_scores_are_learned_from_the_ratings() -> None:
    frame = pd.DataFrame(
        {"interaction": ["liked", "liked", "rejected", "viewed"], "rating": [5, 4, 1, np.nan]}
    )
    scores = learn_interaction_scores(frame)
    assert scores == {"liked": 4.5, "rejected": 1.0, "viewed": 3.0}


def test_clean_materials_has_valid_values(data) -> None:
    materials = data.materials
    assert materials["catalog_code"].is_unique
    assert materials["type"].notna().all() and materials["genre"].notna().all()
    assert materials["level_num"].notna().all()
    assert (materials["pages"] > 0).all()
    assert materials["publication_year"].dropna().le(pd.Timestamp.now().year).all()
    # Every type has a readable label for the explanations.
    assert materials["type_label"].notna().all()


def test_interactions_point_to_existing_materials(data) -> None:
    assert data.interactions["catalog_code"].isin(data.materials["catalog_code"]).all()
    assert set(data.interactions["positive"].unique()) <= {0, 1}
