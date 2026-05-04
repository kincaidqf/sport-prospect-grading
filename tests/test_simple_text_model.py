"""Tests for shallow lexicon text bundle and multimodal text stack (parallel to clf/reg)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.multimodal import MultimodalProspectModel
from src.models.multimodal_text_towers import parse_text_towers, resolve_text_stack
from src.models.simple_text_model import (
    ShallowLexiconConfig,
    ShallowTextTierBundle,
    fit_shallow_text_tier_bundle_for_multimodal,
    handcrafted_feature_matrix,
    merge_shallow_cfg,
)
from src.models.text_model import UniformTextTierBundle


def test_parse_text_towers_legacy_fallback():
    towers = parse_text_towers({"text_meta_key": "scouting"})
    assert len(towers) == 1
    assert towers[0]["meta_key"] == "scouting"
    assert towers[0]["type"] == "transformer"


def test_parse_text_towers_multi():
    mm = {
        "text_towers": [
            {"meta_key": "a", "type": "transformer"},
            {"meta_key": "b", "type": "shallow_lexicon", "shallow": {"ridge_alpha": 2.0}},
        ],
    }
    towers = parse_text_towers(mm)
    assert [t["meta_key"] for t in towers] == ["a", "b"]
    assert towers[1]["shallow"]["ridge_alpha"] == 2.0


def test_resolve_text_stack_base_models_matches_clf_reg_pattern():
    mm = {
        "base_models": {
            "classification": ["logistic_l2"],
            "regression": ["ridge"],
            "text": ["scout_deep", "scout_lex"],
        },
        "text_backends": {
            "scout_deep": "transformer",
            "scout_lex": "shallow_lexicon",
        },
        "text_shallow_overrides": {"scout_lex": {"ridge_alpha": 3.0}},
    }
    keys, specs = resolve_text_stack(mm)
    assert keys == ["scout_deep", "scout_lex"]
    assert specs[0] == {"meta_key": "scout_deep", "type": "transformer", "shallow": {}}
    assert specs[1]["meta_key"] == "scout_lex"
    assert specs[1]["type"] == "shallow_lexicon"
    assert specs[1]["shallow"]["ridge_alpha"] == 3.0


def test_merge_shallow_cfg_tower_overrides_sentiment():
    g = {"sentiment": {"enabled": True, "use_pos_neg_neu": True}}
    t = {"sentiment": {"enabled": False}}
    sc = merge_shallow_cfg(g, t)
    assert sc.sentiment_enabled is False


def test_handcrafted_feature_matrix_shape_sentiment_disabled():
    cfg = ShallowLexiconConfig(sentiment_enabled=False, sentiment_use_pos_neg_neu=False)
    texts = ["elite versatile player"]
    X = handcrafted_feature_matrix(texts, cfg)
    n_lex = (
        1
        + 2 * len(cfg.success_words)
        + 2 * len(cfg.red_flag_words)
        + 2 * len(cfg.red_flag_phrases)
    )
    assert X.shape == (1, n_lex)


def test_shallow_bundle_predict_tier_proba_and_missing_text():
    from sklearn.linear_model import Ridge

    cfg_feat = ShallowLexiconConfig(sentiment_enabled=True, tfidf_enabled=False)
    texts_train = ["elite versatile high motor", "raw limited upside concerns", "steady role player"]
    X = handcrafted_feature_matrix(texts_train, cfg_feat).astype(np.float64)
    y = np.array([1.5, -0.5, 0.2], dtype=np.float64)
    ridge = Ridge(alpha=1.0).fit(X, y)

    tier_std = float(
        np.std(y - ridge.predict(X), ddof=0) + 1e-8,
    )
    bundle = ShallowTextTierBundle(ridge=ridge, tier_residual_std=tier_std, feat_cfg=cfg_feat, tfidf=None)

    df = pd.DataFrame(
        {
            "text": ["elite shooter", "", None],
        },
        index=[10, 11, 12],
    )
    out = bundle.predict_tier_proba(df)
    assert list(out.columns) == ["p_bust", "p_bench", "p_starter", "p_star"]
    assert len(out) == 3
    empty_mask = pd.Series([False, True, True], index=[10, 11, 12])
    probs_empty = out.loc[empty_mask].to_numpy(dtype=float)
    assert np.allclose(probs_empty, 0.25)
    sums = out.to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(sums, 1.0), sums
    assert not np.any(np.isnan(out.to_numpy()))


def test_fit_shallow_below_min_rows_uniform():
    from src.data.loader import TARGET_COL

    z_col = TARGET_COL["nba_role_zscore"]
    fold = pd.DataFrame(
        {"text": ["a"], z_col: [0.1], "prospect_tier": [1]},
    )
    cfg = {"model": {"multimodal": {"text_min_train_rows": 8}}}
    bundle = fit_shallow_text_tier_bundle_for_multimodal(fold, cfg, shallow_cfg={}, silent=True)
    assert isinstance(bundle, UniformTextTierBundle)


def test_fit_shallow_returns_bundle_when_enough_rows():
    from src.data.loader import TARGET_COL

    z_col = TARGET_COL["nba_role_zscore"]
    rows = [{"text": f"elite prospect number {i} versatile", z_col: 0.1 * i - 1, "prospect_tier": 1} for i in range(12)]
    fold = pd.DataFrame(rows)
    cfg = {"model": {"multimodal": {"text_min_train_rows": 8}, "text_shallow": {"tfidf": {"enabled": False}}}}
    bundle = fit_shallow_text_tier_bundle_for_multimodal(fold, cfg, shallow_cfg={}, silent=True)
    assert isinstance(bundle, ShallowTextTierBundle)
    preds = bundle.predict_tier_proba(fold[["text"]])
    assert preds.shape == (12, 4)


def test_multimodal_meta_cols_counts_two_text_models_via_base_models():
    cfg = {
        "model": {
            "multimodal": {
                "base_models": {
                    "classification": ["logistic_l2"],
                    "regression": ["ridge"],
                    "text": ["scout_a", "scout_b"],
                },
                "text_backends": {
                    "scout_a": "transformer",
                    "scout_b": "shallow_lexicon",
                },
            },
        },
    }
    mm = MultimodalProspectModel(cfg)
    assert mm.text_models == ["scout_a", "scout_b"]
    expected = (1 + 1 + 2) * 4
    assert len(mm.meta_cols) == expected
