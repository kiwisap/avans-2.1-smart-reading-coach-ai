from __future__ import annotations

from reading_coach.evaluation import HYBRID, RANDOM, evaluate_rankings, make_scorers, temporal_split
from reading_coach.features import StudentProfile
from reading_coach.hybrid import HybridRecommender


def test_recommends_three_to_five_with_a_story(model) -> None:
    profile = StudentProfile(reading_level="2F", interests=["vriendschap", "sport"], material_types=["boek"])
    recommendations = model.recommend(profile, n=5)
    assert len(recommendations) == 5
    for recommendation in recommendations:
        assert recommendation.reasons, "every advice has its reasons"
        # The motivation is a small story of a few sentences, not a list.
        assert recommendation.motivation.count(".") >= 2
        assert "\n" not in recommendation.motivation
        assert 0.0 <= recommendation.score <= 1.0


def test_story_mentions_the_interest_and_the_level(model) -> None:
    profile = StudentProfile(reading_level="2F", interests=["oorlog"], material_types=["boek"], max_minutes=90)
    first = model.recommend(profile, n=3)[0]
    assert first.level in first.motivation
    if first.features["interest_hits"] > 0:
        assert "oorlog" in first.motivation


def test_never_recommends_material_far_above_level(model) -> None:
    profile = StudentProfile(reading_level="2F", interests=["politiek"])
    for recommendation in model.recommend(profile, n=5):
        assert recommendation.features["level_diff"] <= model.max_level_gap


def test_history_is_excluded_and_changes_advice(model) -> None:
    profile = StudentProfile(reading_level="3F", interests=["misdaad"])
    first = model.recommend(profile, n=5)
    history = {first[0].catalog_code: 5.0}
    second = model.recommend(profile, history, n=5)
    assert first[0].catalog_code not in {r.catalog_code for r in second}


def test_advice_is_varied(model) -> None:
    profile = StudentProfile(reading_level="3F", interests=["kunstmatige intelligentie"], material_types=["online-artikel"])
    titles = [r.title for r in model.recommend(profile, n=5)]
    assert len(set(titles)) == len(titles)


def test_vocabulary_is_learned_from_the_catalog(model, data) -> None:
    vocabulary = model.vocabulary
    assert set(vocabulary.types) == set(data.materials["type"])
    assert set(vocabulary.genres) == set(data.materials["genre"])
    # The data itself rates a positive interaction higher than a negative one.
    scores = vocabulary.interaction_scores
    assert max(scores.values()) > 4 > 2.5 > min(scores.values())


def test_interests_are_expanded_with_themes_from_the_catalog(model) -> None:
    related = model.content.expander.related_terms("spanning")
    assert related, "related themes are found in the catalog"
    assert "spanning" not in related


def test_model_beats_random_on_held_out_interactions(data) -> None:
    train, test = temporal_split(data.interactions)
    model = HybridRecommender().fit(data.materials, data.students, train)
    scorers = make_scorers(model, train, data.students)
    table = evaluate_rankings({name: scorers[name] for name in [RANDOM, HYBRID]}, model, data.students, train, test)
    assert table.loc[HYBRID, "auc_test_interactions"] > table.loc[RANDOM, "auc_test_interactions"] + 0.1
