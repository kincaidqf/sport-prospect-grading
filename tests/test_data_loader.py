"""Tests for the shared data loader and feature engineering."""
from __future__ import annotations

import pandas as pd
import pytest


# ── Raw CSV integrity ──────────────────────────────────────────────────────────

def test_ncaa_csv_readable(ncaa_df):
    assert isinstance(ncaa_df, pd.DataFrame)
    assert len(ncaa_df) > 0


def test_nba_csv_readable(nba_df):
    assert isinstance(nba_df, pd.DataFrame)
    assert len(nba_df) > 0


def test_ncaa_has_required_columns(ncaa_df):
    for col in ("Name", "draft_year"):
        assert col in ncaa_df.columns, f"NCAA CSV missing column: {col}"


def test_nba_has_required_columns(nba_df):
    for col in ("player_name", "draft_year", "PLUS_MINUS"):
        assert col in nba_df.columns, f"NBA CSV missing column: {col}"


def test_ncaa_draft_year_is_numeric(ncaa_df):
    assert pd.api.types.is_numeric_dtype(ncaa_df["draft_year"])


def test_nba_draft_year_is_numeric(nba_df):
    nba_df["draft_year"] = pd.to_numeric(nba_df["draft_year"], errors="coerce")
    assert nba_df["draft_year"].notna().any()


# ── Merged DataFrame ───────────────────────────────────────────────────────────

def test_merged_df_not_empty(merged_df):
    assert len(merged_df) > 0


def test_merged_df_has_target_columns(merged_df):
    for col in ("PLUS_MINUS", "composite_score", "nba_role_zscore", "prospect_tier"):
        assert col in merged_df.columns, f"Merged df missing expected column: {col}"


def test_merged_df_prospect_tier_valid_values(merged_df):
    tiers = set(merged_df["prospect_tier"].dropna().unique())
    assert tiers.issubset({0, 1, 2, 3}), f"Unexpected tier values: {tiers}"


def test_merged_df_draft_year_coverage(merged_df):
    years = set(merged_df["draft_year"].dropna().unique())
    # Expect at least a decade of data
    assert len(years) >= 10, f"Only {len(years)} draft years found"


def test_merged_df_numeric_stats_present(merged_df):
    for col in ("pts_pg", "reb_pg", "ast_pg"):
        assert col in merged_df.columns, f"Merged df missing stat column: {col}"


# ── build_feature_matrix ───────────────────────────────────────────────────────

def test_build_feature_matrix_returns_preprocessor(merged_df):
    from src.data.loader import build_feature_matrix
    result = build_feature_matrix(merged_df)
    preprocessor, numeric_cols, categorical_cols, ordinal_cols, passthrough_cols = result
    assert preprocessor is not None
    assert len(numeric_cols) > 0


def test_build_feature_matrix_with_pos_categorical(merged_df):
    from src.data.loader import build_feature_matrix, POSITION_FEATURE
    preprocessor, numeric_cols, categorical_cols, ordinal_cols, passthrough_cols = build_feature_matrix(
        merged_df, use_pos_categorical=True
    )
    assert POSITION_FEATURE in categorical_cols


def test_target_col_map_covers_all_modes():
    from src.data.loader import TARGET_COL
    expected_modes = {"plus_minus", "composite_score", "nba_role_zscore", "prospect_tier", "became_starter"}
    assert expected_modes.issubset(set(TARGET_COL.keys()))
