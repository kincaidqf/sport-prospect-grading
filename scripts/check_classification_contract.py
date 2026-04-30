"""Contract tests for the classification pipeline.

Checks:
  1. No leakage columns in classification feature list when use_draft_pick=False.
  2. Engineered feature columns exist after load_data.
  3. prospect_tier has exactly 3 classes (0, 1, 2).
  4. Chronological split has no train year >= first validation year.
  5. predict_proba_stats output rows match input and probabilities sum to ~1.

Run:
  uv run python scripts/check_classification_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.loader import (
    CLASSIFICATION_ENGINEERED_NUMERIC,
    CLASSIFICATION_EXCLUDED_NUMERIC,
    DRAFT_PICK_FEATURE,
    build_feature_matrix,
    load_data,
)
from src.training.splits import get_chronological_split

# Avoid importing classification_model (which loads XGBoost at import time).
# Inline predict_proba_stats directly here.
def _predict_proba_stats(df, pipeline, feature_cols):
    from src.models.classification_inference import predict_proba_stats
    return predict_proba_stats(df, pipeline, feature_cols)


_PASS = "[PASS]"
_FAIL = "[FAIL]"


def check(condition: bool, msg: str) -> bool:
    tag = _PASS if condition else _FAIL
    print(f"  {tag}  {msg}")
    return condition


def main() -> int:
    failures = 0

    print("\n=== Classification Contract Checks ===\n")

    cfg          = _load_config()
    composite_cfg = (cfg.get("model") or {}).get("composite_score") or {}
    tier_pct     = tuple(composite_cfg.get("tier_percentiles", (50, 80)))
    print(f"[config] tier_percentiles = {tier_pct}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    df = load_data(composite_cfg=composite_cfg)
    print(f"  Loaded {len(df)} rows\n")

    # 1. No leakage columns
    print("1. Leakage check (use_draft_pick=False)")
    _, num_cols, cat_cols, ord_cols = build_feature_matrix(
        df,
        use_draft_pick=False,
        exclude_features=CLASSIFICATION_EXCLUDED_NUMERIC,
        use_engineered_features=True,
        use_pos_categorical=True,
    )
    all_cols = set(num_cols + cat_cols + ord_cols)
    leakage_cols = {"draft_pick", "big_board_rank", "actual_pick", "nba_team_drafted_by",
                    "PLUS_MINUS", "MIN", "GP", "became_starter", "composite_score"}
    found_leakage = leakage_cols & all_cols
    failures += not check(not found_leakage, f"no leakage columns in feature list (found: {found_leakage or 'none'})")

    # 2. Engineered features exist
    print("\n2. Engineered feature existence")
    missing_eng = [f for f in CLASSIFICATION_ENGINEERED_NUMERIC if f not in df.columns]
    failures += not check(not missing_eng, f"all engineered features present (missing: {missing_eng or 'none'})")
    for feat in CLASSIFICATION_ENGINEERED_NUMERIC:
        n_null = df[feat].isna().sum()
        pct = 100 * n_null / len(df)
        ok = pct < 50
        failures += not check(ok, f"  {feat}: {n_null} nulls ({pct:.1f}%)")

    # 3. prospect_tier has exactly 3 classes
    print("\n3. prospect_tier class count")
    tier_classes = sorted(int(c) for c in df["prospect_tier"].unique())
    failures += not check(tier_classes == [0, 1, 2], f"prospect_tier classes = {tier_classes}")
    for cls, name in enumerate(["Bust", "Contributor", "Star"]):
        n = (df["prospect_tier"] == cls).sum()
        check(n > 0, f"  class {cls} ({name}): {n} players")

    # 4. Chronological split — no train year >= first val year
    print("\n4. Chronological split temporal ordering")
    train_df, val_df, test_df = get_chronological_split(df)
    max_train_year = int(train_df["draft_year"].max())
    min_val_year   = int(val_df["draft_year"].min())
    min_test_year  = int(test_df["draft_year"].min())
    max_val_year   = int(val_df["draft_year"].max())
    failures += not check(max_train_year < min_val_year, f"max train year ({max_train_year}) < min val year ({min_val_year})")
    failures += not check(max_val_year < min_test_year,  f"max val year ({max_val_year}) < min test year ({min_test_year})")
    failures += not check(len(train_df) > 0, f"train split non-empty: {len(train_df)} rows")
    failures += not check(len(val_df) > 0,   f"val split non-empty: {len(val_df)} rows")
    failures += not check(len(test_df) > 0,  f"test split non-empty: {len(test_df)} rows")

    # 5. predict_proba_stats shape and probability sums
    print("\n5. predict_proba_stats output contract")
    _, num_cols, cat_cols, ord_cols = build_feature_matrix(
        df,
        use_draft_pick=False,
        exclude_features=CLASSIFICATION_EXCLUDED_NUMERIC,
        use_engineered_features=True,
        use_pos_categorical=True,
    )
    feature_cols = num_cols + cat_cols + ord_cols

    # Fit a tiny surrogate pipeline just for the contract test
    preprocessor, _, _, _ = build_feature_matrix(
        df,
        use_draft_pick=False,
        exclude_features=CLASSIFICATION_EXCLUDED_NUMERIC,
        use_engineered_features=True,
        use_pos_categorical=True,
    )
    surrogate = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])
    surrogate.fit(df[feature_cols], df["prospect_tier"])

    proba_df = _predict_proba_stats(df, surrogate, feature_cols)
    failures += not check(len(proba_df) == len(df), f"output rows ({len(proba_df)}) match input rows ({len(df)})")

    prob_cols = [c for c in proba_df.columns if c.startswith("p_")]
    prob_sums = proba_df[prob_cols].sum(axis=1)
    all_close = np.allclose(prob_sums, 1.0, atol=1e-4)
    failures += not check(all_close, f"probabilities sum to ~1.0 (max deviation: {(prob_sums - 1.0).abs().max():.2e})")

    no_draft_pick = "draft_pick" not in proba_df.columns
    failures += not check(no_draft_pick, "output does not contain draft_pick column")

    # Summary
    print(f"\n{'='*40}")
    if failures == 0:
        print(f"  All checks PASSED")
    else:
        print(f"  {failures} check(s) FAILED")
    print(f"{'='*40}\n")

    return failures


if __name__ == "__main__":
    sys.exit(main())
