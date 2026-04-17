"""Feature engineering and train/val/test splitting."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# NCAA stats columns used as model features (update as needed)
STAT_FEATURE_COLS: list[str] = [
    # TODO: populate after EDA — data/scouting/players.csv column audit
]

# Scouting numeric rating columns
RATING_COLS: list[str] = [
    # TODO: populate from players.csv (athleticism, defense, strength, etc.)
]


class StatPreprocessor:
    """Scales numerical NCAA stat features."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "StatPreprocessor":
        self.feature_cols = feature_cols
        self.scaler.fit(df[feature_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return out

    def fit_transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df)


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames stratified by draft year."""
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed
    )
    adjusted_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=adjusted_val, random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_and_merge(
    ncaa_path: str,
    nba_path: str,
    target_col: str = "PLUS_MINUS",
) -> pd.DataFrame:
    """Merge NCAA features with NBA target labels into a single DataFrame."""
    ncaa = pd.read_csv(ncaa_path)
    nba = pd.read_csv(nba_path)
    # TODO: define merge key after EDA (likely player name + draft year)
    merged = ncaa.merge(nba[["player_name", "draft_year", target_col]], on=["player_name", "draft_year"], how="inner")
    merged = merged.dropna(subset=[target_col])
    return merged
