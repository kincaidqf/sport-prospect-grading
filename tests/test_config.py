"""Tests for config.yaml structure and values."""
from __future__ import annotations


def test_config_loads(cfg):
    assert isinstance(cfg, dict)


def test_config_top_level_keys(cfg):
    for key in ("model", "training", "data", "logging", "output"):
        assert key in cfg, f"Missing top-level config key: {key}"


def test_model_type_is_valid(cfg):
    valid_types = {"regression", "classification", "text"}
    assert cfg["model"]["type"] in valid_types


def test_regression_config_present(cfg):
    reg = cfg["model"]["regression"]
    assert "target_mode" in reg
    assert reg["target_mode"] in {"plus_minus", "composite_score", "nba_role_zscore"}


def test_classification_config_present(cfg):
    cls = cfg["model"]["classification"]
    assert "target_mode" in cls
    assert cls["target_mode"] in {"became_starter", "prospect_tier"}


def test_text_config_present(cfg):
    text = cfg["model"]["text"]
    assert "pretrained" in text
    assert "output_dim" in text


def test_composite_score_weights_sum_to_one(cfg):
    c = cfg["model"]["composite_score"]
    total = c["w_min"] + c["w_gp"] + c["w_plus_minus"]
    assert abs(total - 1.0) < 1e-6, f"composite_score weights sum to {total}, expected 1.0"


def test_composite_score_tier_percentiles(cfg):
    tiers = cfg["model"]["composite_score"]["tier_percentiles"]
    assert len(tiers) == 2
    assert tiers[0] < tiers[1]


def test_output_config_present(cfg):
    assert "dir" in cfg["output"]
    assert "plots_dir" in cfg["output"]


def test_training_has_seed(cfg):
    assert "seed" in cfg["training"]
    assert isinstance(cfg["training"]["seed"], int)
