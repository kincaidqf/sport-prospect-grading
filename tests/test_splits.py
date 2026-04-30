"""Tests for train/test split strategies."""
from __future__ import annotations

import pandas as pd
import pytest

from src.training.splits import (
    get_chronological_split,
    get_random_split,
    get_repeated_stratified_cv,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "draft_year": list(range(2010, 2025)) * 5,
        "value": range(75),
        "label": [i % 4 for i in range(75)],
    })


def test_random_split_sizes(sample_df):
    train, test = get_random_split(sample_df, test_size=0.2)
    assert len(train) + len(test) == len(sample_df)
    assert len(test) == pytest.approx(len(sample_df) * 0.2, abs=2)


def test_random_split_no_overlap(sample_df):
    train, test = get_random_split(sample_df, test_size=0.2)
    assert set(train.index).isdisjoint(set(test.index))


def test_chronological_split_year_boundaries(sample_df):
    train, val, test = get_chronological_split(
        sample_df, train_end_year=2018, val_years=(2019, 2020), test_years=(2021, 2022)
    )
    assert train["draft_year"].max() <= 2018
    assert set(val["draft_year"].unique()).issubset({2019, 2020})
    assert set(test["draft_year"].unique()).issubset({2021, 2022})


def test_chronological_split_empty_raises(sample_df):
    with pytest.raises(ValueError):
        get_chronological_split(sample_df, train_end_year=2000, test_years=(1990,))


def test_repeated_stratified_cv_returns_splitter():
    from sklearn.model_selection import RepeatedStratifiedKFold
    cv = get_repeated_stratified_cv(n_splits=5, n_repeats=3)
    assert isinstance(cv, RepeatedStratifiedKFold)
    assert cv.n_repeats == 3
    assert cv.cvargs.get("n_splits") == 5
