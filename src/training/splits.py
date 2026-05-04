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
    seed: int = 42,
    stratify_col: str | None = "prospect_tier",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return random (train, val, test) split sized to prior year buckets.

    The split is random-sampled for experimentation while preserving the same
    train/val/test counts implied by the historical year-based boundaries.
    """
    train_by_year = df[df["draft_year"] <= train_end_year]
    val_by_year = df[df["draft_year"].isin(val_years)]
    test_by_year = df[df["draft_year"].isin(test_years)]
    if train_by_year.empty or test_by_year.empty:
        raise ValueError(
            f"Reference year split produced empty train ({len(train_by_year)}) or test ({len(test_by_year)}). "
            "Check that draft_year values exist for the requested ranges."
        )

    total_n = len(df)
    test_size = len(test_by_year) / total_n
    remainder_n = len(train_by_year) + len(val_by_year)
    val_size_within_remainder = len(val_by_year) / remainder_n if remainder_n else 0.0

    has_stratify = bool(stratify_col and stratify_col in df.columns)
    stratify_all = df[stratify_col] if has_stratify else None
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_all,
    )

    stratify_train_val = train_val[stratify_col] if has_stratify else None
    train, val = train_test_split(
        train_val,
        test_size=val_size_within_remainder,
        random_state=seed,
        stratify=stratify_train_val,
    )
    return train.copy(), val.copy(), test.copy()


def get_repeated_stratified_cv(
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
