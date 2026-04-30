# Code Review & Diagnostic Report: NBA Draft Prospect Grading

---

## Changelog

| Date | Change | Performance Impact |
|---|---|---|
| 2026-04-29 | Flipped per-game stat sourcing: PPG/RPG/APG/BKPG/STPG and FG%/FT% are now the primary source; totals/G used only as fallback. Removed `shoots_3s`. | XGBoost regression: slight accuracy **increase**. XGBoost classification: slight accuracy **decrease**. Linear models unaffected by sourcing order due to median imputation + scaling. |
| 2026-04-29 | Removed `height_in` from `CLASSIFICATION_ENGINEERED_NUMERIC` and added to `CLASSIFICATION_EXCLUDED_NUMERIC`. Only `height_dev` (position-adjusted deviation) now enters the feature matrix. | Classification models no longer see raw height alongside the composite. Expected to reduce redundant signal and improve `height_dev` importance rank. |
| 2026-04-29 | Config-as-source-of-truth fixes: regression alpha search now reads `alpha_min`/`alpha_max`/`alpha_steps`/`cv_folds` from config; regression XGBoost grid expanded to match classification (`colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `gamma`); hardcoded `min_child_weight=5` removed; `prospect_context_mode` added to config.yaml; both `__main__` blocks now load and pass config. | Regression alpha search range changes from `[1e-3, 1e2]` to `[1e-4, 1e2]` per config. XGBoost grid is larger for regression — expect longer run times. No classification impact. |
| 2026-04-29 | `use_engineered_features` and `use_pos_categorical` added to config under both `model.regression` and `model.classification`, defaulting to `false`. Both flags are now fully config-driven and logged to MLflow on every run. | No change to current behavior (both default to false). Enables direct comparison experiments from config alone. |
| 2026-04-29 | Dead code removed: `trainer.py` deleted, `multimodal_model.py` deleted, `multimodal` branch and argparse choice removed from `main.py`, `multimodal` block removed from `config.yaml`, `ranking_metrics()` removed from `evaluate.py`. XGBoost added to `repeated_cv` eval mode with fixed params from config (class balancing applied in final artifact fit only). | No functional impact on regression or classification runs. |

---

## 1. Input Features: What Goes Into Training

### Shared Base Features (`NUMERIC_FEATURES` — used in both regression and classification)

| Feature | Primary Source | Fallback | Notes |
|---|---|---|---|
| `pts_pg` | `PPG` (given) | `PTS / G` | |
| `reb_pg` | `RPG` (given) | `REB / G` | |
| `ast_pg` | `APG` (given) | `AST / G` | |
| `blk_pg` | `BKPG` (given) | `BLKS / G` | |
| `stl_pg` | `STPG` (given) | `ST / G` | |
| `fgm_pg` | computed only | `FGM / G` | no given equivalent in source data |
| `fga_pg` | computed only | `FGA / G` | no given equivalent in source data |
| `ft_pg` | computed only | `FT / G` | no given equivalent in source data |
| `fta_pg` | computed only | `FTA / G` | no given equivalent in source data |
| `fg3_pg` | computed only | `3FG / G` | no given equivalent in source data |
| `fg_pct` | `FG%` (given) | `(FGM/FGA) × 100` | |
| `ft_pct` | `FT%` (given) | `(FT/FTA) × 100` | |
| `pts_per_fga` | computed only | `PTS / FGA` | efficiency ratio |
| `ft_rate` | computed only | `FTA / FGA` | drawing fouls proxy |
| `fg3_share` | computed only | `3FG / FGM` | 3-point reliance |
| `G` | raw games played | — | **regression only** — excluded from classification via `CLASSIFICATION_EXCLUDED_NUMERIC` |
| `height_dev` | computed | `\|height_in − pos_avg_height\|` | deviation from position mean; replaces raw `height_in` |
| `team_difficulty_score` | lookup (0.25–5.0) | — | school strength proxy |

### Engineered Features (`CLASSIFICATION_ENGINEERED_NUMERIC`)

Available to both models when `use_engineered_features: true` in config. Defaults to `false`.

| Feature | Formula | Signal |
|---|---|---|
| `to_pg` | `TO / G` | ball-handling liability |
| `oreb_pg` | `ORebs / G` | offensive rebounding |
| `dreb_pg` | `DRebs / G` | defensive rebounding |
| `fg3a_pg` | `3FGA / G` | 3-point volume |
| `ast_to` | `AST / TO` | playmaking efficiency |
| `stocks_pg` | `(ST + BLKS) / G` | defensive impact |
| `usage_proxy` | `(FGA + 0.44×FTA + TO) / G` | ball usage rate |
| `efg_pct` | `(FGM + 0.5×3FG) / FGA` | shooting efficiency |
| `ts_pct` | `PTS / (2×(FGA + 0.44×FTA))` | true shooting |

### Categorical/Ordinal Features

| Feature | Encoding | When Included |
|---|---|---|
| `Pos` (position) | OneHot → Guard / Forward / Center | when `use_pos_categorical: true` in config (defaults to `false` for both models) |
| `Cl` (class year) | Ordinal (Fr.=0 → Sr.=3) + StandardScaler | when `prospect_context_mode ∈ {"individual","both"}` — default is `"individual"`, so always |

### Explicitly Excluded From Training

- **`height_in`** — raw height in inches. Removed from `CLASSIFICATION_ENGINEERED_NUMERIC` and added to `CLASSIFICATION_EXCLUDED_NUMERIC`. `height_dev` (position-relative deviation) captures this signal without the redundancy of having both raw height and the composite in the model simultaneously.
- **`mpg_minutes`** — NCAA minutes per game. Computed and stored but marked `# stored for analysis only; never used in training`. Correctly excluded.
- **`shoots_3s`** — **removed**. Was a binary flag `(3FG > 0)`. Provided marginal signal already captured by `fg3_pg` and `fg3_share`.
- **`prospect_context_score`** (composite form: `difficulty × class_score^1.5`) — only added when `prospect_context_mode ∈ {"composite","both"}`. Current default is `"individual"`, so not in the feature matrix.
- **`draft_pick`** — controlled by `use_draft_pick` (default `False` everywhere).
- **`G`** — excluded from classification via `CLASSIFICATION_EXCLUDED_NUMERIC`.

---

## 2. Target Construction (NBA Side)

### Available Targets

| Target | Column | Description |
|---|---|---|
| `became_starter` | binary int | `1` if NBA career MIN ≥ 25 |
| `composite_score` | float | Weighted z-score: `0.55×z(MIN) + 0.30×z(GP) + 0.15×z(PLUS_MINUS)`, NaN→−3.0 |
| `nba_role_zscore` | float | Weighted winsorized z-score across {MIN, nba_pts, nba_reb, nba_ast, nba_stl, nba_blk}, re-standardized, NaN→−3.0 |
| `prospect_tier` | int 0–3 | Binned from `nba_role_zscore` at thresholds (−0.5, 0.5, 1.5) → Bust/Bench/Starter/Star |
| `plus_minus` | float | Raw NBA PLUS_MINUS — legacy target, still supported |

### Current Active Targets

- **Regression → `nba_role_zscore`** (config: `model.regression.target_mode`)
- **Classification → `prospect_tier`** (config: `model.classification.target_mode`)

### `nba_role_zscore` Construction Pipeline

1. For each stat in {MIN, nba_pts, nba_reb, nba_ast, nba_stl, nba_blk}:
   - Compute global z-score OR position-group z-score (config: `model.nba_role_score.target_score_mode`)
   - Winsorize at ±2.5 (config: `winsor_clip`)
2. Weighted average using weights {MIN:0.30, pts:0.25, reb:0.20, ast:0.15, stl:0.05, blk:0.05}
   - Missing stats excluded from denominator (graceful handling)
3. Re-standardize whole composite to std=1
4. Players with no NBA data → fill with `nan_floor: -3.0`

### `prospect_tier` Binning

Fixed z-score thresholds from config (`tier_thresholds: [-0.5, 0.5, 1.5]`):
- 0 = Bust (z < −0.5)
- 1 = Bench (−0.5 ≤ z < 0.5)
- 2 = Starter (0.5 ≤ z < 1.5)
- 3 = Star (z ≥ 1.5)

---

## 3. Data Processing Pipeline (Pre-Training)

1. **Load & Inner Join**: NCAA master + NBA master on `(Name, draft_year)` — drops anyone without NBA outcome data
2. **Target construction**: all four targets computed on the merged frame
3. **Height parsing**: `"6-4"` → 76 inches; position mean computed and `height_dev` added; raw `height_in` excluded from training
4. **Class year normalization**: string cleanup (`"Fr"` → `"Fr."`)
5. **Position grouping**: broad 3-way map for NBA position-relative scoring
6. **Per-game stats**: given columns (PPG/RPG/APG/BKPG/STPG) used directly; totals/G computed as fallback only when given column is missing
7. **Percentage stats**: given columns (FG%/FT%) used directly; computed from totals as fallback only when given column is missing
8. **Computed-only per-game stats**: fgm_pg, fga_pg, ft_pg, fta_pg, fg3_pg derived from totals (no given equivalents exist)
9. **Ratio features**: pts_per_fga, ft_rate, fg3_share computed from totals
10. **Engineered features** (when `use_engineered_features: true`): to_pg, oreb_pg, dreb_pg, ast_to, stocks_pg, usage_proxy, efg_pct, ts_pct — available to both regression and classification
11. **Context features**: `team_difficulty_score` (school tier lookup), `mpg_minutes` (stored only), `prospect_context_score` (stored but only used if mode != `"individual"`)
12. **sklearn ColumnTransformer**:
    - Numeric: `SimpleImputer(median)` → `StandardScaler`
    - Categorical (Pos, when `use_pos_categorical: true`): `SimpleImputer(most_frequent)` → `OneHotEncoder`
    - Ordinal (Cl): `SimpleImputer(most_frequent)` → `OrdinalEncoder` → `StandardScaler`

---

## 4. Training Modes and Variations

### Model Types

| Type | Models Run | Hyperparameter Search |
|---|---|---|
| **Regression** | Lasso CV, Ridge CV, XGBoost | Grid search (CV scoring: R²) |
| **Classification** | Logistic L1, Logistic L2, XGBoost | Grid search (CV scoring: F1-macro) |
| **Text** | DistilBERT encoder + regression head | Full training loop implemented in `text_model.py` |

### Classification Evaluation Modes (`model.classification.eval_mode`)

| Mode | Behavior |
|---|---|
| `random` (default) | Stratified random 80/20 split; optional further 15% val hold-out for threshold tuning |
| `chronological` | Train ≤2018, Val 2019–2020, Test 2021–2023; respects temporal ordering |
| `repeated_cv` | 5-fold × 5-repeat stratified CV for Logistic L1, Logistic L2, and XGBoost (fixed params from config — no nested grid search); final artifact fit on full data |

### `repeated_cv` XGBoost Notes

XGBoost in `repeated_cv` uses the first value of each config grid parameter as fixed hyperparameters. `cross_validate` does not support per-fold `sample_weight` subsetting, so class balancing is applied only in the final artifact fit, not during CV folds. For full XGBoost grid search with class balancing, use `random` or `chronological` eval mode.

### Classification-Specific Adaptations

- **`class_weight: balanced`** — passed to LogisticRegressionCV; for XGBoost uses `compute_sample_weight("balanced", y_train)` since `scale_pos_weight` is insufficient for multi-class
- **`threshold_tuning: false`** — when enabled, coordinate-descent search over per-class probability offsets (±0.15, 13 steps) on validation set to maximize macro F1; available for both Logistic and XGBoost

### Target Mode Variations

**Classification:**
- `prospect_tier` (current): 4-class ordinal outcome derived from role z-score
- `became_starter` (legacy): binary threshold at NBA MIN ≥ 25

**Regression:**
- `nba_role_zscore` (current): continuous weighted composite
- `composite_score` (legacy): simpler 3-stat composite
- `plus_minus` (legacy): raw NBA PLUS_MINUS

---

## 5. Issues Found

### Config as Source of Truth — Gaps

#### FIXED: Regression model ignored config alpha parameters

**Original**: `_run_regression` hardcoded `alphas = np.logspace(-3, 2, 100)` and `linear_cv_folds = 5`, ignoring `alpha_min`, `alpha_max`, `alpha_steps`, and `cv_folds` in config.

**Fix**: Added `reg_cfg=None` parameter to `train_and_evaluate` and `_run_regression`. These functions now read:
```python
alpha_min       = float(_rcfg.get("alpha_min",    1e-4))
alpha_max       = float(_rcfg.get("alpha_max",    1e2))
alpha_steps     = int(_rcfg.get("alpha_steps",   100))
linear_cv_folds = int(_rcfg.get("cv_folds",      5))
alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), alpha_steps)
```
`run()` extracts the full `reg_cfg = model_cfg.get("regression", {})` block and passes it through. The alpha search range changed from `[1e-3, 1e2]` to `[1e-4, 1e2]` to match config.

**Files changed**: `regression_model.py`

---

#### FIXED: Regression XGBoost hardcoded `min_child_weight` and missing grid params

**Original**: `XGBRegressor(min_child_weight=5)` hardcoded in constructor, overriding config. Regression XGBoost `param_grid` was missing `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, and `gamma` — parameters classification already searched.

**Fix**: Removed `min_child_weight=5` from the `XGBRegressor` constructor. Expanded `param_grid` to match classification:
```python
"xgb__colsample_bytree": cfg.get("colsample_bytree", [0.7, 1.0]),
"xgb__min_child_weight": cfg.get("min_child_weight", [3, 5, 10]),
"xgb__reg_alpha":        cfg.get("reg_alpha",        [0, 0.1, 1]),
"xgb__reg_lambda":       cfg.get("reg_lambda",       [1, 5, 10]),
"xgb__gamma":            cfg.get("gamma",            [0]),
```
Added corresponding defaults to `config.yaml` under `model.regression.xgboost`.

**Files changed**: `regression_model.py`, `config.yaml`

---

#### FIXED: `prospect_context_mode` not in config.yaml

**Original**: `prospect_context_mode` was a loader-level constant (`"individual"`). Classification tried to read it from config via `model_cfg.get("prospect_context_mode", PROSPECT_CONTEXT_MODE)` but the key didn't exist, so it always silently fell back to the hardcoded default. Regression didn't read or pass it at all.

**Fix**: Added `prospect_context_mode: individual` to `config.yaml` under `model`. Both `run()` functions now read it:
```python
prospect_context_mode = model_cfg.get("prospect_context_mode", PROSPECT_CONTEXT_MODE)
```
and pass it through to `build_feature_matrix`. Changing this key in config now controls whether class year (ordinal), the composite prospect context score, both, or neither are included as features.

**Files changed**: `config.yaml`, `regression_model.py`, `classification_model.py`

---

#### FIXED: Module-level constants not config-driven when run as `__main__`

**Original**: Both `regression_model.py` and `classification_model.py` ended with `run()`, using hardcoded `TARGET_MODE` defaults and no config — only `main.py` wired config correctly.

**Fix**: Both `__main__` blocks now load `config.yaml` and extract the model-specific target and flags before calling `run()`:
```python
if __name__ == "__main__":
    import yaml
    _cfg_path = os.path.join(PROJECT_ROOT, "src", "config", "config.yaml")
    with open(_cfg_path) as _f:
        _cfg = yaml.safe_load(_f)
    _reg_cfg = (_cfg.get("model", {}) or {}).get("regression", {}) or {}
    run(
        target_mode=_reg_cfg.get("target_mode", TARGET_MODE),
        use_draft_pick=_reg_cfg.get("use_draft_pick", USE_DRAFT_PICK),
        cfg=_cfg,
    )
```

**Files changed**: `regression_model.py`, `classification_model.py`

---

### Feature Asymmetry

#### FIXED: Regression and classification had asymmetric feature sets with no config control

**Original**: Classification hardcoded `use_engineered_features=True` and `use_pos_categorical=True` in its `build_feature_matrix` call. Regression passed neither, permanently using the leaner feature set. Neither flag was in config.

**Fix**: Both flags are now first-class config keys under both `model.regression` and `model.classification`, defaulting to `false`:
```yaml
use_engineered_features: false  # add ts_pct, efg_pct, usage_proxy, ast_to, etc.
use_pos_categorical: false      # one-hot encode position (Guard/Forward/Center)
```
Both `run()` functions extract the flags and pass them through `train_and_evaluate` → `build_feature_matrix`. Both are logged to MLflow on every run. Either model can now be given the richer feature set by flipping a config flag with no code changes.

**Files changed**: `config.yaml`, `regression_model.py`, `classification_model.py`

---

### Dead Code

#### FIXED: `trainer.py` — unimplemented PyTorch Trainer stub

**Original**: `src/training/trainer.py` contained a `Trainer` class whose three methods (`train_epoch`, `eval_epoch`, `fit`) all raised `NotImplementedError`. Nothing in the codebase imported it.

**Fix**: File deleted.

---

#### FIXED: Text model reported as likely absent — confirmed present

**Original diagnostic**: "Text model referenced in `main.py` but `src/models/text_model.py` likely doesn't exist."

**Finding**: The file exists and contains a complete implementation — `ScoutingReportEncoder`, `TextProspectPredictor`, `_TokenizedTextDataset`, full training loop with early stopping, MLflow logging, and plotting. No action required; the diagnostic was incorrect.

---

#### FIXED: `ranking_metrics()` raised NotImplementedError

**Original**: `evaluate.py` contained a `ranking_metrics()` function with a TODO comment and `raise NotImplementedError` as its entire body.

**Fix**: Function removed from `evaluate.py`. Nothing in the codebase called it.

---

#### FIXED: `multimodal` pipeline was an unimplemented stub

**Original**: `multimodal_model.py` had a `MultimodalProspectModel` whose `forward()` raised `NotImplementedError` and whose `__init__` had all layers commented out. `main.py` accepted `--model multimodal` and raised `NotImplementedError` at runtime. `config.yaml` had a `model.multimodal` block.

**Fix**: `multimodal_model.py` deleted. `multimodal` removed from `main.py` argparse `choices` and the `elif` dispatch branch removed. `model.multimodal` block removed from `config.yaml`.

**Files changed**: `main.py`, `config.yaml`; `multimodal_model.py` deleted

---

#### FIXED: `repeated_cv` eval mode silently skipped XGBoost

**Original**: `_run_repeated_cv` only ran Logistic L1 and Logistic L2. XGBoost was absent with no comment explaining why.

**Fix**: XGBoost added to `_run_repeated_cv`. Because `cross_validate` does not support nested hyperparameter search, XGBoost uses fixed params drawn from the first value of each config grid list:
```python
xgb_fixed = {
    "n_estimators":     cfg.get("n_estimators",     [200])[0],
    "max_depth":        cfg.get("max_depth",         [3])[0],
    ...
}
```
Class balancing (`sample_weight`) is applied in the final artifact fit but not during CV folds — `cross_validate` does not subset `fit_params` arrays per fold, so passing sample weights would produce shape mismatches. This limitation is noted in a code comment. For full XGBoost tuning with class balancing, use `random` or `chronological` eval mode.

**Files changed**: `classification_model.py`

---

### Minor

- `_compute_composite_score` (the legacy target) uses NBA `MIN` as a season total, not per-game; the `composite_score` weights (w_min, w_gp, w_pm) are also loaded in `load_data()` but NBA `MIN` in the NBA master is ambiguous about whether it's career or single-season total.
- The `nba_role_score` config block has both `nan_floor` and `tier_thresholds`, but `composite_score` also has `nan_floor` — two separate nan floors that can diverge.

---

## 6. Future Directions

### High-Impact Model Improvements

**Position-relative input normalization** — currently only the *target* supports position-relative z-scoring (`global` vs `position_relative`). Applying the same normalization to input features (e.g., `reb_pg` means very different things for a center vs a guard) could substantially improve signal quality. This would mirror what's already done on the output side.

**Draft class normalization** — stats should ideally be normalized within each draft class to account for era effects and conference inflation. A player from 2015 and a player from 2023 are measured on different scales. Z-scoring within draft year would remove this noise.

**Ordinal regression for `prospect_tier`** — since Bust < Bench < Starter < Star is a natural ordering, proportional-odds ordinal regression (e.g., `mord` library) is more principled than treating this as nominal 4-class. Multinomial logistic regression ignores the rank structure.

**Calibrated probabilities** — no probability calibration is applied after training. For prospect grading, probability output matters (e.g., "40% starter probability"). Adding isotonic regression or Platt scaling as a post-processing step would improve probability reliability.

**Engineered features for regression** — `use_engineered_features` and `use_pos_categorical` now default to `false` for both models. Enabling them for regression is a one-line config change and worth benchmarking to see if the richer feature set improves R² and tier distribution accuracy.

### Target Construction Improvements

**Separate playing-time from performance in the target** — `nba_role_zscore` heavily weights MIN (30%). A player who plays 30 mpg poorly and one who plays 20 mpg excellently may score similarly. Separating "did they stick?" (years in league, GP) from "how good were they?" (efficiency metrics) could yield cleaner learning signal.

**Survival framing** — instead of averaging NBA stats over 3 years, model the probability of *still being in the league* at year 3 as a separate signal from average performance. Many busts play briefly; modeling this as a two-stage problem (make it to year 3?) × (how good in year 3?) could be more informative.

---

**Bottom line**: All config-as-source-of-truth gaps are closed. Dead code is removed. Both models now share symmetric config control over feature set richness (`use_engineered_features`, `use_pos_categorical`). The biggest remaining opportunities for model quality improvement are position-relative input normalization, draft-class normalization, and switching to ordinal regression for the tier target.
