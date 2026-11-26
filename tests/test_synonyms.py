from __future__ import annotations

import pytest

from reading_coach.synonyms import Related, as_terms, learn_synonyms, wordnet_available

needs_wordnet = pytest.mark.skipif(
    not wordnet_available(), reason="wordnet data missing: python -m reading_coach.synonyms"
)


@needs_wordnet
def test_a_dutch_interest_finds_the_english_loanword_in_the_catalog() -> None:
    synonyms = learn_synonyms(["kunstmatige intelligentie"], ["ai", "liefde"])
    assert synonyms == {"kunstmatige intelligentie": [Related("ai", "synonym")]}


@needs_wordnet
def test_only_themes_of_the_catalog_are_kept() -> None:
    synonyms = learn_synonyms(["maatschappij"], ["samenleving", "voetbal"])
    assert [item.term for item in synonyms["maatschappij"]] == ["samenleving"]


@needs_wordnet
def test_a_narrower_theme_counts_as_related() -> None:
    synonyms = learn_synonyms(["misdaad"], ["cybercrime"])
    assert synonyms["misdaad"] == [Related("cybercrime", "narrower")]


def test_synonyms_come_before_narrower_terms() -> None:
    related = {"maatschappij": [Related("cultuur", "narrower"), Related("samenleving", "synonym")]}
    assert as_terms(related) == {"maatschappij": ["samenleving", "cultuur"]}


@needs_wordnet
def test_the_model_uses_the_synonyms(model) -> None:
    from reading_coach.features import StudentProfile

    profile = StudentProfile(reading_level="3F", interests=["kunstmatige intelligentie"], material_types=["online-artikel"])
    advice = model.recommend(profile, n=3)
    assert any("ai" in item.title.lower().split() for item in advice)
    assert "kunstmatige intelligentie" in advice[0].motivation
