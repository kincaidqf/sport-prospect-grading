"""Tests for model module imports and public API contracts."""
from __future__ import annotations

import inspect


def test_regression_model_imports():
    import src.models.regression_model as rm
    assert callable(rm.run)
    assert callable(rm.train_and_evaluate)


def test_classification_model_imports():
    import src.models.classification_model as cm
    assert callable(cm.run)
    assert callable(cm.train_and_evaluate)


def test_text_model_imports():
    import src.models.text_model as tm
    assert callable(tm.train_and_evaluate_text_model)
    assert callable(tm.load_text_data)


def test_regression_run_signature():
    import src.models.regression_model as rm
    sig = inspect.signature(rm.run)
    params = set(sig.parameters.keys())
    assert "target_mode" in params
    assert "use_draft_pick" in params
    assert "cfg" in params
    assert "run_name" in params


def test_classification_run_signature():
    import src.models.classification_model as cm
    sig = inspect.signature(cm.run)
    params = set(sig.parameters.keys())
    assert "target_mode" in params
    assert "use_draft_pick" in params
    assert "cfg" in params
    assert "run_name" in params


def test_text_model_run_signature():
    import src.models.text_model as tm
    sig = inspect.signature(tm.train_and_evaluate_text_model)
    params = set(sig.parameters.keys())
    assert "pretrained" in params
    assert "epochs" in params
    assert "cfg" in params
    assert "run_name" in params


def test_regression_target_modes():
    import src.models.regression_model as rm
    expected = {"plus_minus", "composite_score", "nba_role_zscore"}
    assert expected == rm.REGRESSION_TARGETS


def test_classification_target_modes():
    import src.models.classification_model as cm
    expected = {"became_starter", "prospect_tier"}
    assert expected == cm.CLASSIFICATION_TARGETS


def test_loader_imports():
    from src.data.loader import load_data, build_feature_matrix, TARGET_COL
    assert callable(load_data)
    assert callable(build_feature_matrix)
    assert isinstance(TARGET_COL, dict)


def test_classification_inference_imports():
    from src.models.classification_inference import (
        load_pipeline,
        run_inference_on_merged_data,
        get_prospect_tier_labels,
    )
    assert callable(load_pipeline)
    assert callable(run_inference_on_merged_data)
    assert callable(get_prospect_tier_labels)


def test_mlflow_utils_imports():
    from src.utils.mlflow_utils import (
        build_mlflow_context,
        managed_run,
        log_common_params,
        log_config_dict,
    )
    assert callable(build_mlflow_context)
    assert callable(managed_run)


def test_plotting_utils_imports():
    from src.utils.plotting import plot_feature_importance, plot_model_summary, save_and_log
    assert callable(plot_feature_importance)
    assert callable(plot_model_summary)
    assert callable(save_and_log)
