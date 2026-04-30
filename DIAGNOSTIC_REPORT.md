# Code Review & Diagnostic Report: NBA Draft Prospect Grading

---

## Changelog

| Date | Change | Performance Impact |
|---|---|---|
| 2026-04-29 | Flipped per-game stat sourcing: PPG/RPG/APG/BKPG/STPG and FG%/FT% are now the primary source; totals/G used only as fallback. Removed `shoots_3s`. | XGBoost regression: slight accuracy **increase**. XGBoost classification: slight accuracy **decrease**. Linear models unaffected by sourcing order due to median imputation + scaling. |

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
| `G` | raw games played | — | **regression only** — excluded from classification |
| `height_dev` | computed | `\|height_in − pos_avg_height\|` | deviation from position mean |
| `team_difficulty_score` | lookup (0.25–5.0) | — | school strength proxy |

### Classification-Exclusive Engineered Features (`CLASSIFICATION_ENGINEERED_NUMERIC`)

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
| `height_in` | raw inches | classification gets both raw height and deviation |

### Categorical/Ordinal Features

| Feature | Encoding | When Included |
|---|---|---|
| `Pos` (position) | OneHot → Guard / Forward / Center | classification only (`use_pos_categorical=True`) |
| `Cl` (class year) | Ordinal (Fr.=0 → Sr.=3) + StandardScaler | when `prospect_context_mode ∈ {"individual","both"}` — default is `"individual"`, so always |

### Explicitly Excluded From Training

- **`mpg_minutes`** — NCAA minutes per game. Computed and stored but marked `# stored for analysis only; never used in training`. Correctly excluded.
- **`shoots_3s`** — **removed**. Was a binary flag `(3FG > 0)`. Provided marginal signal already captured by `fg3_pg` and `fg3_share`.
- **`prospect_context_score`** (composite form: `difficulty × class_score^1.5`) — only added when `prospect_context_mode ∈ {"composite","both"}`. Current default is `"individual"`, so not in the feature matrix.
- **`draft_pick`** — controlled by `use_draft_pick` (default `False` everywhere).

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
3. **Height parsing**: `"6-4"` → 76 inches; position mean computed and deviation added
4. **Class year normalization**: string cleanup (`"Fr"` → `"Fr."`)
5. **Position grouping**: broad 3-way map for NBA position-relative scoring
6. **Per-game stats**: given columns (PPG/RPG/APG/BKPG/STPG) used directly; totals/G computed as fallback only when given column is missing
7. **Percentage stats**: given columns (FG%/FT%) used directly; computed from totals as fallback only when given column is missing
8. **Computed-only per-game stats**: fgm_pg, fga_pg, ft_pg, fta_pg, fg3_pg derived from totals (no given equivalents exist)
9. **Ratio features**: pts_per_fga, ft_rate, fg3_share computed from totals
10. **Classification extras**: to_pg, oreb_pg, dreb_pg, ast_to, stocks_pg, usage_proxy, efg_pct, ts_pct
11. **Context features**: `team_difficulty_score` (school tier lookup), `mpg_minutes` (stored only), `prospect_context_score` (stored but only used if mode != `"individual"`)
12. **sklearn ColumnTransformer**:
    - Numeric: `SimpleImputer(median)` → `StandardScaler`
    - Categorical (Pos): `SimpleImputer(most_frequent)` → `OneHotEncoder`
    - Ordinal (Cl): `SimpleImputer(most_frequent)` → `OrdinalEncoder` → `StandardScaler`

---

## 4. Training Modes and Variations

### Model Types

| Type | Models Run | Hyperparameter Search |
|---|---|---|
| **Regression** | Lasso CV, Ridge CV, XGBoost | Grid search (CV scoring: R²) |
| **Classification** | Logistic L1, Logistic L2, XGBoost | Grid search (CV scoring: F1-macro) |
| **Text** | DistilBERT encoder | Configured but model file absent |
| **Multimodal** | fusion head | Stub — raises NotImplementedError |

### Classification Evaluation Modes (`model.classification.eval_mode`)

| Mode | Behavior |
|---|---|
| `random` (default) | Stratified random 80/20 split; optional further 15% val hold-out for threshold tuning |
| `chronological` | Train ≤2018, Val 2019–2020, Test 2021–2023; respects temporal ordering |
| `repeated_cv` | 5-fold × 5-repeat stratified CV; fits final artifact on random split; **only runs Logistic models** (XGBoost excluded in `_run_repeated_cv`) |

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

**Regression model ignores config alpha parameters entirely** (`regression_model.py:118-119`):
```python
alphas = np.logspace(-3, 2, 100)  # hardcoded
linear_cv_folds = 5               # hardcoded
```
Config has `alpha_min: 1e-4`, `alpha_max: 1e2`, `alpha_steps: 100`, `cv_folds: 5` that are never read.

**Regression XGBoost has hardcoded `min_child_weight=5`** (`regression_model.py:226`) — overrides anything in config, and the regression XGBoost grid search is also missing `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `gamma` parameters that classification uses.

**`prospect_context_mode` is not in config.yaml** — it's a loader-level constant (`"individual"`) that the classification model tries to read from config but the key doesn't exist, so it always silently falls back to the hardcoded default. This means the class year ordinal feature is always included but the composite prospect_context_score never is, and there's no way to change this from config alone.

**Module-level constants not config-driven**: When run as `__main__` directly (not via `main.py`), both model files use their hardcoded `TARGET_MODE` defaults rather than reading config. Only the `main.py` entrypoint properly wires config values to the `run()` call.

### Feature Asymmetry (Likely Intentional but Worth Flagging)

Regression uses a notably weaker feature set than classification — it doesn't get `use_engineered_features=True`, so it misses `ts_pct`, `efg_pct`, `usage_proxy`, `ast_to`, etc. It also doesn't get position as a categorical feature. This may be intentional (simpler interpretable model for regression), but it means the regression model is handicapped relative to classification.

### Dead Code

- `trainer.py` — the entire PyTorch Trainer class is stub with `raise NotImplementedError` on all methods. Never called.
- Text model referenced in `main.py` but `src/models/text_model.py` likely doesn't exist
- `ranking_metrics()` in `evaluate.py` raises NotImplementedError; there's a TODO comment
- `multimodal` pipeline raises NotImplementedError
- `repeated_cv` eval mode runs **only** Logistic models — XGBoost is silently skipped. No comment explains why.

### Minor

- `_compute_composite_score` (the legacy target) uses NBA `MIN` as a season total, not per-game; the `composite_score` weights (w_min, w_gp, w_pm) are also loaded in `load_data()` but NBA `MIN` in the NBA master is ambiguous about whether it's career or single-season total.
- The `nba_role_score` config block has both `nan_floor` and `tier_thresholds`, but `composite_score` also has `nan_floor` — two separate nan floors that can diverge.

---

## 6. Future Directions

### High-Impact Model Improvements

**Position-relative input normalization** — currently only the *target* supports position-relative z-scoring (`global` vs `position_relative`). Applying the same normalization to input features (e.g., `reb_pg` means very different things for a center vs a guard) could substantially improve signal quality. This would mirror what's already done on the output side.

**Draft class normalization** — stats should ideally be normalized within each draft class to account for era effects and conference inflation. A player from 2015 and a player from 2023 are measured on different scales. Z-scoring within draft year would remove this noise.

**Multi-season trajectory features** — the model only uses the final NCAA season. Year-over-year improvement rate (sophomore jump, junior consistency) is a well-known scouting signal. If the data supports it, adding delta features (e.g., `pts_pg_delta`) could add predictive power.

**Ordinal regression for `prospect_tier`** — since Bust < Bench < Starter < Star is a natural ordering, proportional-odds ordinal regression (e.g., `mord` library) is more principled than treating this as nominal 4-class. Multinomial logistic regression ignores the rank structure.

**Calibrated probabilities** — no probability calibration is applied after training. For prospect grading, probability output matters (e.g., "40% starter probability"). Adding isotonic regression or Platt scaling as a post-processing step would improve probability reliability.

### Target Construction Improvements

**Separate playing-time from performance in the target** — `nba_role_zscore` heavily weights MIN (30%). A player who plays 30 mpg poorly and one who plays 20 mpg excellently may score similarly. Separating "did they stick?" (years in league, GP) from "how good were they?" (efficiency metrics) could yield cleaner learning signal.

**Survival framing** — instead of averaging NBA stats over 3 years, model the probability of *still being in the league* at year 3 as a separate signal from average performance. Many busts play briefly; modeling this as a two-stage problem (make it to year 3?) × (how good in year 3?) could be more informative.

### Code Quality / Config Hygiene

1. Fix regression model to read `alpha_min`/`alpha_max`/`alpha_steps`/`cv_folds` from config instead of hardcoding
2. Add `prospect_context_mode` to `config.yaml` so it's controllable
3. Give regression XGBoost the same hyperparameter search space as classification (add `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `gamma` to grid)
4. Remove or complete `trainer.py` — it adds confusion with no function
5. Add XGBoost to `repeated_cv` mode or document why it's excluded
6. Consider giving regression `use_engineered_features=True` to level the playing field and see if the richer features help

---

**Bottom line**: The core pipeline is well-structured and the config-as-source-of-truth goal is mostly met, with the key exceptions being hardcoded regression alpha/CV parameters and the missing `prospect_context_mode` key in config. NCAA minutes are correctly excluded from training. The biggest opportunities for model quality improvement are position-relative input normalization, draft-class normalization, and switching to ordinal regression for the tier target.
