"""Tests for the main.py CLI entry point."""
from __future__ import annotations

import sys
from unittest.mock import patch


def test_parse_args_defaults():
    from src.main import parse_args
    with patch.object(sys, "argv", ["main.py"]):
        args = parse_args()
    assert args.config == "src/config/config.yaml"
    assert args.model is None
    assert args.epochs is None
    assert args.output_dir is None
    assert args.run_name is None
    assert args.tracking_uri is None


def test_parse_args_model_choices():
    from src.main import parse_args
    for model in ("regression", "classification", "text"):
        with patch.object(sys, "argv", ["main.py", "--model", model]):
            args = parse_args()
        assert args.model == model


def test_parse_args_rejects_unknown_model():
    from src.main import parse_args
    import pytest
    with pytest.raises(SystemExit):
        with patch.object(sys, "argv", ["main.py", "--model", "multimodal"]):
            parse_args()


def test_load_config_returns_dict():
    from src.main import load_config
    cfg = load_config("src/config/config.yaml")
    assert isinstance(cfg, dict)
    assert "model" in cfg


def test_main_dispatches_to_regression(merged_df, cfg, tmp_path):
    """Smoke test: main() routes regression config to regression_model.run()."""
    import src.models.regression_model as rm

    run_calls = []

    def fake_run(**kwargs):
        run_calls.append(kwargs)

    import sys
    from unittest.mock import patch

    cfg_copy = {**cfg}
    cfg_copy["model"] = {**cfg["model"], "type": "regression"}

    with patch.object(sys, "argv", ["main.py", "--model", "regression"]):
        with patch("src.models.regression_model.run", side_effect=fake_run):
            with patch("src.main.load_config", return_value=cfg_copy):
                with patch("src.main._ensure_macos_openmp_for_xgboost"):
                    with patch("src.main.get_device"), patch("src.main.log_device_info"):
                        from src.main import main
                        main()

    assert len(run_calls) == 1
    assert "target_mode" in run_calls[0]
    assert "cfg" in run_calls[0]


def test_main_dispatches_to_classification(cfg):
    """Smoke test: main() routes classification config to classification_model.run()."""
    import sys
    from unittest.mock import patch

    cfg_copy = {**cfg}
    cfg_copy["model"] = {**cfg["model"], "type": "classification"}

    run_calls = []

    def fake_run(**kwargs):
        run_calls.append(kwargs)

    with patch.object(sys, "argv", ["main.py", "--model", "classification"]):
        with patch("src.models.classification_model.run", side_effect=fake_run):
            with patch("src.main.load_config", return_value=cfg_copy):
                with patch("src.main._ensure_macos_openmp_for_xgboost"):
                    with patch("src.main.get_device"), patch("src.main.log_device_info"):
                        from src.main import main
                        main()

    assert len(run_calls) == 1
    assert "target_mode" in run_calls[0]
    assert "cfg" in run_calls[0]
