from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.multimodal_reporting import write_multimodal_report
from src.models.probability import PROBA_COLUMNS, TIER_CLASS_NAMES


class FakeBundle:
    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_tier_proba(self, df):
        return pd.DataFrame(self._proba[:len(df)], columns=PROBA_COLUMNS, index=df.index)


class FakeStacker:
    classes_ = np.array([0, 1, 2, 3])

    def __init__(self, n_features):
        self.coef_ = np.arange(1, 4 * n_features + 1, dtype=float).reshape(4, n_features) / 100.0


class FakeModel:
    clf_models = ["logistic_l1", "xgboost"]
    reg_models = ["lasso"]
    _mm_cfg: dict = {}

    def __init__(self, meta_features):
        self.meta_cols = list(meta_features.columns)
        self.stacker = FakeStacker(len(self.meta_cols))
        self.final_clf_bundles = {
            "logistic_l1": FakeBundle([
                [0.70, 0.20, 0.05, 0.05],
                [0.10, 0.70, 0.10, 0.10],
                [0.10, 0.20, 0.60, 0.10],
                [0.05, 0.10, 0.20, 0.65],
            ]),
            "xgboost": FakeBundle([
                [0.60, 0.20, 0.10, 0.10],
                [0.20, 0.50, 0.20, 0.10],
                [0.05, 0.20, 0.55, 0.20],
                [0.05, 0.10, 0.25, 0.60],
            ]),
        }
        self.final_reg_bundles = {
            "lasso": FakeBundle([
                [0.50, 0.30, 0.10, 0.10],
                [0.20, 0.50, 0.20, 0.10],
                [0.10, 0.20, 0.55, 0.15],
                [0.10, 0.15, 0.25, 0.50],
            ]),
        }
        self._meta_features = meta_features

    def meta_features(self, df):
        out = self._meta_features.iloc[:len(df)].copy()
        out.index = df.index
        return out


def _fake_inputs():
    test_df = pd.DataFrame({
        "Name": ["A", "B", "C", "D"],
        "draft_year": [2020, 2021, 2022, 2023],
        "draft_pick": [5, 20, 15, 45],
    })
    y_test = pd.Series([0, 1, 2, 3])
    test_proba = np.array([
        [0.70, 0.20, 0.05, 0.05],
        [0.20, 0.55, 0.15, 0.10],
        [0.10, 0.20, 0.55, 0.15],
        [0.05, 0.10, 0.15, 0.70],
    ])
    y_pred = np.argmax(test_proba, axis=1)

    meta_cols = []
    for prefix in ["classification__logistic_l1", "classification__xgboost", "regression__lasso"]:
        for col in PROBA_COLUMNS:
            meta_cols.append(f"{prefix}__{col}")
    meta_features = pd.DataFrame(np.full((len(test_df), len(meta_cols)), 0.25), columns=meta_cols)
    return FakeModel(meta_features), test_df, y_test, y_pred, test_proba


def test_write_multimodal_report_creates_expected_outputs(tmp_path):
    model, test_df, y_test, y_pred, test_proba = _fake_inputs()

    write_multimodal_report(
        model=model,
        test_df=test_df,
        y_test=y_test,
        y_pred=y_pred,
        test_proba=test_proba,
        out_dir=str(tmp_path),
    )

    expected_files = {
        "test_predictions.csv",
        "test_meta_features.csv",
        "base_model_summary.csv",
        "model_summary.csv",
        "worst_misses.csv",
        "ordinal_error_distribution.csv",
        "confusion_matrix.csv",
        "probability_mass_by_true_class.csv",
        "stacker_contributions.csv",
        "model_summary.png",
        "ordinal_confusion_matrix.png",
        "ordinal_error_distribution.png",
        "expected_vs_true.png",
        "probability_mass_by_true_class.png",
        "worst_misses_probability_bars.png",
        "lottery_bust_hits.csv",
        "late_star_hits.csv",
        "best_hits_probability_bars.png",
        "stacker_contribution_heatmap.png",
    }
    actual_files = {p.name for p in tmp_path.iterdir()}
    assert expected_files <= actual_files
    assert all("outputs/plots" not in str(p) for p in tmp_path.rglob("*"))


def test_multimodal_report_tables_preserve_class_order_and_summary_rows(tmp_path):
    model, test_df, y_test, y_pred, test_proba = _fake_inputs()
    write_multimodal_report(model, test_df, y_test, y_pred, test_proba, str(tmp_path))

    confusion = pd.read_csv(tmp_path / "confusion_matrix.csv", index_col=0)
    probability_mass = pd.read_csv(tmp_path / "probability_mass_by_true_class.csv", index_col=0)
    model_summary = pd.read_csv(tmp_path / "model_summary.csv")

    assert list(confusion.index) == TIER_CLASS_NAMES
    assert list(confusion.columns) == TIER_CLASS_NAMES
    assert list(probability_mass.index) == TIER_CLASS_NAMES
    assert list(probability_mass.columns) == PROBA_COLUMNS
    assert set(model_summary["model"]) == {
        "multimodal__stacker",
        "classification__logistic_l1",
        "classification__xgboost",
        "regression__lasso",
    }
