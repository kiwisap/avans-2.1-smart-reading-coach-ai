"""Shared fixtures: the model is trained once per test session on the client dataset."""

from __future__ import annotations

import pytest

from leescoach import config
from leescoach.data_cleaning import CleanData, clean_all
from leescoach.hybrid import HybridRecommender


@pytest.fixture(scope="session")
def data() -> CleanData:
    if not config.RAW_DATASET.exists():
        pytest.skip("Dataset ontbreekt in data/raw")
    return clean_all()


@pytest.fixture(scope="session")
def model(data: CleanData) -> HybridRecommender:
    return HybridRecommender().fit(data.materials, data.students, data.interactions)
