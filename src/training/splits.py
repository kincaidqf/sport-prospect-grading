"""Evaluation split strategies for NBA draft prospect classification."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


def get_random_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    stratify_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = df[stratify_col] if stratify_col else None
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    return train, test


def get_chronological_split(
    df: pd.DataFrame,
    train_end_year: int = 2018,
    val_years: tuple[int, ...] = (2019, 2020),
    test_years: tuple[int, ...] = (2021, 2022, 2023),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) split by draft year — no temporal leakage."""
    train = df[df["draft_year"] <= train_end_year].copy()
    val = df[df["draft_year"].isin(val_years)].copy()
    test = df[df["draft_year"].isin(test_years)].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Chronological split produced empty train ({len(train)}) or test ({len(test)}). "
            "Check that draft_year values exist for the requested ranges."
        )
    return train, val, test


def get_repeated_stratified_cv(
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
