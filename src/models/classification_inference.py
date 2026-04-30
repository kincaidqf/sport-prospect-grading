"""Stable inference utility for the stats-based prospect classification pipeline.

Usage
-----
After running classification training, load the saved sklearn pipeline and call
predict_proba_stats() to produce calibrated 3-class probabilities.  The output
can be used directly as stats-branch input for multimodal experiments.

No post-draft fields (draft_pick, big_board_rank, NBA outcomes) are ever read
or written here.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from src.data.loader import (
    TARGET_COL,
    PROSPECT_CONTEXT_MODE,
    CLASSIFICATION_EXCLUDED_NUMERIC,
    build_feature_matrix,
    load_data,
)

TIER_CLASS_NAMES = ["bust", "contributor", "star"]
TARGET_MODE = "prospect_tier"


def load_pipeline(model_path: str | Path):
    """Load a fitted sklearn pipeline from a local pickle or MLflow artifact URI."""
    path = str(model_path)
    if path.startswith("runs:/") or path.startswith("mlflow-artifacts:/"):
        return mlflow.sklearn.load_model(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba_stats(
    df: pd.DataFrame,
    pipeline,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return per-class probabilities and predicted tier for each player.

    Output columns: p_bust, p_contributor, p_star, pred_tier, confidence.
    Rows match df exactly (same index).  Never includes draft_pick or scouting rank.
    """
    proba = pipeline.predict_proba(df[feature_cols])
    n_classes = proba.shape[1]
    names = TIER_CLASS_NAMES[:n_classes]

    out = pd.DataFrame(
        {f"p_{name}": proba[:, i] for i, name in enumerate(names)},
        index=df.index,
    )
    out["pred_tier"]  = np.argmax(proba, axis=1)
    out["confidence"] = proba.max(axis=1)
    return out


def export_stats_embeddings_or_proba(
    df: pd.DataFrame,
    pipeline,
    feature_cols: list[str],
    output_path: str | Path,
) -> pd.DataFrame:
    """Write per-player stat probabilities to CSV for multimodal experiments.

    The CSV includes player identifiers (Name, draft_year) plus probability columns.
    It intentionally excludes NBA outcomes and scouting-derived fields.
    """
    proba_df = predict_proba_stats(df, pipeline, feature_cols)
    id_cols  = [c for c in ("Name", "draft_year") if c in df.columns]
    out      = pd.concat([df[id_cols].reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[inference] Wrote {len(out)} rows to {output_path}")
    return out


def run_inference_on_merged_data(
    pipeline,
    output_path: str | Path = "outputs/stat_proba.csv",
    composite_cfg: dict | None = None,
    prospect_context_mode: str = PROSPECT_CONTEXT_MODE,
    input_normalization_mode: str = "global",
) -> pd.DataFrame:
    """Load merged NCAA+NBA data, run inference, and export probabilities.

    Convenience wrapper for multimodal experiments.  No draft_pick in output.
    """
    df = load_data(composite_cfg=composite_cfg)
    _, numeric_cols, categorical_cols, ordinal_cols, passthrough_cols = build_feature_matrix(
        df,
        use_draft_pick=False,
        exclude_features=CLASSIFICATION_EXCLUDED_NUMERIC,
        prospect_context_mode=prospect_context_mode,
        use_engineered_features=True,
        use_pos_categorical=True,
        input_normalization_mode=input_normalization_mode,
    )
    feature_cols = numeric_cols + categorical_cols + ordinal_cols + passthrough_cols
    return export_stats_embeddings_or_proba(df, pipeline, feature_cols, output_path)


# ── Shared label helper ────────────────────────────────────────────────────────
# Text, stats, and multimodal models should all use this to map composite_score
# to prospect_tier labels, so the label definition stays in one place.

def get_prospect_tier_labels(df: pd.DataFrame, target_col: str = "prospect_tier") -> pd.Series:
    """Return the prospect_tier column from a loaded DataFrame.

    This is the canonical label used by stats, text, and multimodal models.
    Text model TODO: switch target from PLUS_MINUS to this column.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Column '{target_col}' not found.  Load data via load_data() which computes it."
        )
    return df[target_col]
