from __future__ import annotations

from leescoach.features import StudentProfile


def test_recommends_three_to_five_with_reasons(model) -> None:
    profile = StudentProfile(leesniveau="2F", interesses=["vriendschap", "sport"], materiaaltypes=["boek"])
    recommendations = model.recommend(profile, n=5)
    assert len(recommendations) == 5
    for recommendation in recommendations:
        assert recommendation.redenen, "elke aanbeveling heeft uitleg"
        assert recommendation.samenvatting
        assert 0.0 <= recommendation.score <= 1.0


def test_never_recommends_material_far_above_level(model) -> None:
    profile = StudentProfile(leesniveau="2F", interesses=["politiek"])
    for recommendation in model.recommend(profile, n=5):
        assert recommendation.kenmerken["niveau_verschil"] <= model.max_level_gap


def test_history_is_excluded_and_changes_advice(model, data) -> None:
    profile = StudentProfile(leesniveau="3F", interesses=["misdaad"])
    first = model.recommend(profile, n=5)
    history = {first[0].cataloguscode: 5.0}
    second = model.recommend(profile, history, n=5)
    assert first[0].cataloguscode not in {r.cataloguscode for r in second}


def test_advice_is_varied(model) -> None:
    profile = StudentProfile(leesniveau="3F", interesses=["kunstmatige intelligentie"], materiaaltypes=["online-artikel"])
    titles = [r.titel for r in model.recommend(profile, n=5)]
    assert len(set(titles)) == len(titles)


def test_model_beats_random_on_held_out_interactions(data) -> None:
    from leescoach.evaluation import evaluate_rankings, make_scorers, temporal_split
    from leescoach.hybrid import HybridRecommender

    train, test = temporal_split(data.interactions)
    model = HybridRecommender().fit(data.materials, data.students, train)
    scorers = make_scorers(model, train, data.students)
    table = evaluate_rankings(
        {name: scorers[name] for name in ["Willekeurig", "Hybride model (Random Forest)"]}, data.students, train, test
    )
    assert table.loc["Hybride model (Random Forest)", "auc_test_interacties"] > table.loc["Willekeurig", "auc_test_interacties"] + 0.1
