"""Shared fixtures: the model is trained once per test session on the dataset of the client."""

from __future__ import annotations

import pytest

from reading_coach import config
from reading_coach.data_cleaning import CleanData, clean_all
from reading_coach.hybrid import HybridRecommender


@pytest.fixture(scope="session")
def data() -> CleanData:
    if not config.RAW_DATASET.exists():
        pytest.skip("The dataset is missing in data/raw")
    return clean_all()


@pytest.fixture(scope="session")
def model(data: CleanData) -> HybridRecommender:
    return HybridRecommender().fit(data.materials, data.students, data.interactions)
