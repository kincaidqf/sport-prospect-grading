from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.probability import PROBA_COLUMNS, align_text_tier_proba_to_meta_cols, load_text_tier_proba_table


def test_load_text_tier_proba_table_roundtrip(tmp_path):
    p = tmp_path / "tier.csv"
    rows = pd.DataFrame(
        {
            "Name": [" A ", "B"],
            "draft_year": [2020, 2021],
            **{c: [0.25, 0.25] for c in PROBA_COLUMNS},
        },
    )
    rows.to_csv(p, index=False)
    loaded = load_text_tier_proba_table(str(p))
    assert len(loaded) == 2
    assert loaded["Name"].tolist() == ["A", "B"]


def test_align_text_tier_proba_fills_missing_with_uniform():
    lookup = pd.DataFrame(
        {
            "Name": ["A"],
            "draft_year": [2020],
            "p_bust": [0.4],
            "p_bench": [0.3],
            "p_starter": [0.2],
            "p_star": [0.1],
        },
    )
    df = pd.DataFrame({"Name": ["A", "B"], "draft_year": [2020, 2021]})
    out = align_text_tier_proba_to_meta_cols(df, lookup, "scouting")
    assert out.shape == (2, 4)
    assert np.allclose(out.iloc[1].to_numpy(), 0.25)
    matched = out.iloc[0][f"text__scouting__{PROBA_COLUMNS[0]}"]
    assert abs(float(matched) - 0.4) < 1e-6
