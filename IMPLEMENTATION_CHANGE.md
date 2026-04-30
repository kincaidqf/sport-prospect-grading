# Implementation Plan: NBA Role Score Redesign

## Overview

Replace the current PLUS_MINUS-weighted composite score with a standardized NBA role score built
from MIN, PTS, REB, AST, STL, BLK. Add position-relative z-score mode switchable via config.
Regression predicts continuous z-score; classification predicts one of four tier buckets.

**New tiers (4-class)**:
- 0 = Bust    (z < −0.5)
- 1 = Bench   (−0.5 ≤ z < 0.5)
- 2 = Starter (0.5 ≤ z < 1.5)
- 3 = Star    (z ≥ 1.5)

---

## Phase 1: Fix NBA Season Selection — COMPLETE

### Files changed

- `data/scripts/fetch_nba_stats.py`
- `data/scripts/recover_nba_players.py`

### What was changed

**`fetch_nba_stats.py`**

Added a module-level constant:
```python
ROLE_STATS = ["MIN", "PTS", "REB", "AST", "STL", "BLK"]
```

Replaced the PLUS_MINUS season selection block with a min-max composite:
```python
# For each role stat, min-max scale within this player's candidate seasons,
# then average the scaled values. NaN stats contribute 0 (missing stat
# doesn't disqualify a season; it just doesn't help it).
# Fall back to highest PTS if all role stats are NaN across all seasons.
candidates_df = pd.DataFrame(candidate_rows)
for col in ROLE_STATS:
    candidates_df[col] = pd.to_numeric(candidates_df[col], errors="coerce")

has_role_data = candidates_df[ROLE_STATS].notna().any().any()

if has_role_data:
    scaled = pd.DataFrame(index=candidates_df.index)
    for col in ROLE_STATS:
        col_min = candidates_df[col].min()
        col_max = candidates_df[col].max()
        if pd.notna(col_min) and pd.notna(col_max) and col_max > col_min:
            scaled[col] = (candidates_df[col] - col_min) / (col_max - col_min)
        else:
            scaled[col] = 0.0  # identical or all-NaN stat — contributes nothing
    candidates_df["_selection_score"] = scaled.fillna(0.0).mean(axis=1)
    best = candidates_df.loc[candidates_df["_selection_score"].idxmax()]
    best = best.drop("_selection_score")  # prevent temp column leaking into output CSV
else:
    candidates_df["PTS"] = pd.to_numeric(candidates_df["PTS"], errors="coerce")
    best = candidates_df.loc[candidates_df["PTS"].idxmax()]
```

**`recover_nba_players.py`**

The recovery script has its own `get_best_season()` function that fetches stats for players whose
names weren't resolved by the main fetch. It also used PLUS_MINUS selection and was updated with
the same min-max composite logic and a matching `ROLE_STATS` constant, so recovered players use
identical selection criteria to the main fetch.

### Pipeline run order

Three scripts must be run in this exact sequence. Running reconcile before recover caused
`ncaa_master.csv` to be overwritten with a reduced player set (764 instead of 787), requiring a
`git checkout` of `ncaa_master.csv` to recover. The correct order is:

```
1. uv run python data/scripts/fetch_nba_stats.py
   → regenerates data/nba/nba_stats_best_season.csv (1795 rows, 1090 ok)

2. uv run python data/scripts/recover_nba_players.py
   → patches nba_stats_best_season.csv with fuzzy-matched players (1213 ok + recovered total)

3. uv run python data/scripts/reconcile_from_ncaa_master.py
   → rebuilds data/nba/nba_master.csv and trims data/ncaa/ncaa_master.csv to the intersection
```

### Verification results

| Check | Result |
|---|---|
| `nba_master.csv` row count | 787 (unchanged) |
| `ncaa_master.csv` row count | 787 (unchanged) |
| `nba_stats_best_season.csv` ok + recovered | 1213 (unchanged) |
| `_selection_score` column in output | absent (correctly dropped) |
| Player sets match between masters | ✓ |

**Spot-check: season selection vs. old PLUS_MINUS logic**

| Player | New selection | Old selection | Why new is correct |
|---|---|---|---|
| Blake Griffin | 2010-11 (22.5 pts, 12.1 reb, 37.9 min) | 2011-12 (20.7 pts, 10.9 reb, higher PM) | Rookie of Year season was clearly his peak production |
| Karl-Anthony Towns | 2016-17 (25.1 pts, 12.3 reb) | 2017-18 (21.3 pts, PM=+4.5) | KAT's breakout year was higher across all scoring/rebounding stats |
| John Wall | 2010-11 (37.8 min, 8.3 ast, 1.8 stl) | 2012-13 (18.5 pts, PM=+0.9) | Highest minutes + assists season correctly wins over PM-driven pick |
| Damian Lillard | 2014-15 (21.0 pts, 4.6 reb, 6.2 ast) | 2013-14 (PM=+4.4) | Most complete production year beats PM-inflated pick |
| James Harden | 2011-12 | 2011-12 | Same — both methods agree |
| Anthony Davis | 2014-15 | 2014-15 | Same — both methods agree |

---

## Phase 2: Position Mapping Utility — COMPLETE

### File changed

`src/data/loader.py`

### Discovery during implementation

The plan assumed the NCAA `Pos` column contained written-out strings ("Point Guard", etc.). The
actual data shows `Pos` is abbreviated (G / F / C), with two players (Josh Jackson, Anthony
Edwards) having `Pos = "-"` due to missing scraped data. The `position` column — sourced from
`players_list.txt` via the reconcile pipeline — contains the full written-out strings and is
identical between `ncaa_master.csv` and `nba_master.csv`. `pos_group` is derived from `position`
rather than `Pos` to ensure complete, accurate coverage.

### What was added

A `_POS_BROAD_MAP` dict and `_map_pos_group()` helper added in a new
`# ── Position mapping ──` section of `loader.py`, between the `_MPG_CAP` constant and the
`# ── Helpers ──` section:

```python
_POS_BROAD_MAP: dict[str, str] = {
    "Point Guard":    "Guard",
    "Shooting Guard": "Guard",
    "Guard":          "Guard",
    "Small Forward":  "Forward",
    "Power Forward":  "Forward",
    "Forward":        "Forward",
    "Center":         "Center",
}

def _map_pos_group(pos_str: str) -> str:
    """Map a written-out position string to Guard / Forward / Center.

    Splits on '/' and uses only the primary (first) position, so
    'Shooting Guard/Small Forward' → 'Guard'.
    Unknown strings (including '-') fall back to 'Forward'.
    Handles both the full written-out NCAA/NBA format and the abbreviated
    G / F / C values present in the NCAA Pos column.
    """
    primary = str(pos_str).strip().split("/")[0].strip()
    if primary == "G":
        return "Guard"
    if primary == "C":
        return "Center"
    if primary in ("F", "-"):
        return "Forward"
    return _POS_BROAD_MAP.get(primary, "Forward")
```

In `load_data()`, after `df[POSITION_FEATURE] = df[POSITION_FEATURE].str.strip()`:

```python
df["pos_group"] = df["position"].apply(_map_pos_group)
```

`pos_group` is a derived column on the merged DataFrame. It is not added to the model feature
matrix. It will be used in Phase 3 for position-relative z-score computation on both the NCAA
side (player grouping) and the NBA side (when the NBA `position` column is added to the merge).

### Verification results

| Check | Result |
|---|---|
| `pos_group` null count | 0 |
| Guard count | 403 |
| Forward count | 320 |
| Center count | 64 |
| Total | 787 |
| Josh Jackson (Pos="-") | Guard (from `position` = "Shooting Guard/Small Forward") |
| Anthony Edwards (Pos="-") | Guard (from `position` = "Shooting Guard") |

**All 9 unique position values and their mappings**:

| position | pos_group |
|---|---|
| Point Guard | Guard |
| Point Guard/Shooting Guard | Guard |
| Shooting Guard | Guard |
| Shooting Guard/Small Forward | Guard |
| Small Forward | Forward |
| Small Forward/Power Forward | Forward |
| Power Forward | Forward |
| Power Forward/Center | Forward |
| Center | Center |

---

## Phase 3: New Target — NBA Role Z-Score

**File**: `src/data/loader.py`

### 3a. Update the NBA merge in `load_data()` — COMPLETE

**File changed**: `src/data/loader.py`

**Discovery during implementation**: `PTS`, `REB`, and `AST` exist in both `ncaa_master.csv` and
`nba_master.csv`. Merging them without renaming would cause pandas to suffix both sides (`_x`,
`_y`), breaking all downstream references to the NCAA stats. The `position` column is likewise
present in both CSVs. To prevent any collision, the NBA role stats and position are renamed
immediately on selection, before the merge.

**What was changed**:

```python
_nba_cols = nba[["player_name", "draft_year", "PLUS_MINUS", "MIN", "GP", "player_id",
                  "PTS", "REB", "AST", "STL", "BLK", "position"]].rename(columns={
    "PTS":      "nba_pts",
    "REB":      "nba_reb",
    "AST":      "nba_ast",
    "STL":      "nba_stl",
    "BLK":      "nba_blk",
    "position": "nba_position",
})
df = ncaa.merge(
    _nba_cols,
    left_on=["Name", "draft_year"],
    right_on=["player_name", "draft_year"],
    how="inner",
)
```

All existing references to `df["PTS"]`, `df["REB"]`, `df["AST"]` (NCAA stats) are unaffected.
The NBA role stat columns in the merged DataFrame are `nba_pts`, `nba_reb`, `nba_ast`, `nba_stl`,
`nba_blk`. The config `weights` keys in Phase 4 use these same names (`MIN` is unambiguous and
kept as-is).

### 3b. Why weighted z-scores with winsorizing — COMPLETE (design rationale, no code)

**Problem with equal-weight z-scores**: z-scoring normalizes each stat's variance to 1 regardless
of basketball significance. STL has a real-world std dev of ~0.4/game; PTS has ~6/game. Equal
weighting means a 1.6-steal difference (+4.0 z) outscores a 6-point difference (+1.0 z) in the
composite. That is wrong for role size.

**Why not sum raw stats**: MIN ranges 10–38/game; BLK ranges 0–3/game. A raw sum is almost
entirely a minutes counter — the other stats contribute noise-level variation relative to MIN.
Z-scores are still necessary to handle unit/scale differences. The fix is unequal weights, not
removing z-scores.

**Solution: weighted z-scores with per-stat winsorizing**

Weights reflect each stat's contribution to NBA role size:

| Stat | Weight | Rationale |
|------|--------|-----------|
| MIN  | 0.30   | Floor time is role size by definition |
| PTS  | 0.25   | Primary offensive role signal |
| REB  | 0.20   | Strong role signal; less position-skewed than BLK |
| AST  | 0.15   | Role signal, especially for guards |
| STL  | 0.05   | Low variance, noisy, position-influenced |
| BLK  | 0.05   | Low variance, strongly center-skewed |

All weights are configurable in `config.yaml`. Running with equal weights (0.167 each) is a
one-line config change, enabling direct empirical comparison of weighted vs. equal-weight runs
via MLflow.

**Winsorizing**: clip each per-stat z-score at ±2.5 before taking the weighted average. This
prevents outlier players in low-variance stats (e.g., a historically high-steal or high-block
season) from producing extreme z-scores that distort the composite for every other player.

### 3c. New function `_compute_nba_role_score(df, mode, weights, winsor_clip, nan_floor)` — COMPLETE

**File changed**: `src/data/loader.py`

**What was added** (placed after `_assign_tier`, before `# ── Data loading ──`):

```python
def _compute_nba_role_score(df, mode, weights, winsor_clip=2.5, nan_floor=-3.0):
    stat_cols = list(weights.keys())
    z_parts = {}

    for col in stat_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if mode == "position_relative":
            z = df.groupby("nba_pos_group")[col].transform(
                lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
            )
        else:
            mu, sigma = s.mean(), s.std()
            z = (s - mu) / (sigma if sigma > 0 else 1.0)
        z_parts[col] = z.clip(-winsor_clip, winsor_clip)

    # Weighted average; missing stats contribute 0 to numerator, excluded from denominator
    numerator   = sum(weights[c] * z_parts[c].fillna(0.0) for c in stat_cols)
    denominator = sum(weights[c] * z_parts[c].notna().astype(float) for c in stat_cols)
    combined = numerator / denominator.replace(0.0, np.nan)

    # Re-standardize so std=1 regardless of mode or weight set
    mu_c, sigma_c = combined.mean(), combined.std()
    role_zscore = (combined - mu_c) / (sigma_c if sigma_c > 0 else 1.0)

    return role_zscore.fillna(nan_floor)
```

**Column naming**: Because 3a renamed NBA stats to `nba_pts`, `nba_reb`, `nba_ast`, `nba_stl`,
`nba_blk`, the `weights` dict keys use those names (e.g. `{"MIN": 0.30, "nba_pts": 0.25, ...}`).
`position_relative` mode groups on `nba_pos_group`, which is set in `load_data()` before this
function is called.

### 3d. New function `_assign_tier_thresholded(zscore, thresholds)` — COMPLETE

**File changed**: `src/data/loader.py`

Added immediately after `_compute_nba_role_score`:

```python
def _assign_tier_thresholded(zscore, thresholds=(-0.5, 0.5, 1.5)):
    """Bin z-scores into 4 fixed tiers: 0=Bust, 1=Bench, 2=Starter, 3=Star."""
    lo, mid, hi = thresholds
    bins   = [-np.inf, lo, mid, hi, np.inf]
    labels = [0, 1, 2, 3]
    return pd.cut(zscore, bins=bins, labels=labels).astype(int)
```

The old `_assign_tier` (percentile-based, 3-class) is kept intact for backward compatibility
with any caller using `composite_score`. `prospect_tier` now uses this new function exclusively.

### 3e. Update `load_data()` — COMPLETE

**File changed**: `src/data/loader.py`

**What was added** after the merge and composite-score config block:

```python
# nba_role_score config
_role_cfg         = _full_cfg.get("nba_role_score") or {}
_role_mode        = _role_cfg.get("target_score_mode", "global")
_role_weights     = _role_cfg.get("weights") or {
    "MIN": 0.30, "nba_pts": 0.25, "nba_reb": 0.20,
    "nba_ast": 0.15, "nba_stl": 0.05, "nba_blk": 0.05,
}
_role_winsor      = float(_role_cfg.get("winsor_clip", 2.5))
_role_thresholds  = tuple(_role_cfg.get("tier_thresholds", (-0.5, 0.5, 1.5)))
_role_nan_floor   = float(_role_cfg.get("nan_floor", -3.0))

# Derived targets — backward-compat targets first, then new role-score targets
df["became_starter"]  = (df["MIN"] >= 25).astype(int)
df["composite_score"] = _compute_composite_score(...)

# NBA position group for position-relative mode
df["nba_pos_group"] = df["nba_position"].apply(_map_pos_group)

df["nba_role_zscore"] = _compute_nba_role_score(
    df, mode=_role_mode, weights=_role_weights,
    winsor_clip=_role_winsor, nan_floor=_role_nan_floor,
)
df["prospect_tier"] = _assign_tier_thresholded(df["nba_role_zscore"], thresholds=_role_thresholds)
```

`_full_cfg` is now loaded once at the top of the config block (previously `_load_project_config()`
was called inline only for the composite path). `became_starter` and `composite_score` are
unchanged; `prospect_tier` now derives from `nba_role_zscore` instead of `composite_score`.

### Verification results

| Check | Result |
|---|---|
| Row count | 787 (unchanged) |
| `nba_role_zscore` null count | 0 |
| `prospect_tier` 0 (Bust) | 275 (35%) |
| `prospect_tier` 1 (Bench) | 264 (34%) |
| `prospect_tier` 2 (Starter) | 183 (23%) |
| `prospect_tier` 3 (Star) | 65 (8%) |
| `nba_pos_group` Guard | 403 |
| `nba_pos_group` Forward | 320 |
| `nba_pos_group` Center | 64 |

**Note on tier distribution**: The plan estimated ~19%/40%/26%/15%. The actual distribution
(35%/34%/23%/8%) matches the theoretical normal-distribution probabilities for thresholds at
−0.5/0.5/1.5σ, which is correct behaviour. The plan's estimates were inaccurate.

### 3f. Update `TARGET_COL` — COMPLETE

**File changed**: `src/data/loader.py`

```python
TARGET_COL = {
    "plus_minus":      "PLUS_MINUS",
    "became_starter":  "became_starter",
    "prospect_tier":   "prospect_tier",       # 4-class: 0=bust 1=bench 2=starter 3=star
    "composite_score": "composite_score",      # preserved for backward compat
    "nba_role_zscore": "nba_role_zscore",      # new regression target
}
```

---

## Phase 4: Config Changes

**File**: `src/config/config.yaml`

Add a new `nba_role_score` section under `model:`. Keep the old `composite_score` section to avoid
breaking any existing runs that reference it.

```yaml
model:
  nba_role_score:
    target_score_mode: global       # global | position_relative
    tier_thresholds: [-0.5, 0.5, 1.5]
    nan_floor: -3.0
    winsor_clip: 2.5                # clip each per-stat z-score at ±this value before combining
    weights:                        # relative weights for the weighted z-score composite
      MIN: 0.30                     # floor time = role size by definition
      PTS: 0.25                     # primary offensive role signal
      REB: 0.20                     # role signal; less position-skewed than BLK
      AST: 0.15                     # role signal, especially for guards
      STL: 0.05                     # low variance, noisy, position-influenced
      BLK: 0.05                     # low variance, strongly center-skewed
```

To run equal-weight comparison, set all weights to 0.167 (or any equal value — they are
normalized to sum to 1 before use). MLflow logs the full config on every run, so any weight
experiment is automatically tracked.

**How the toggle works**:
- `target_score_mode: global` → z-scores computed across all players regardless of position
- `target_score_mode: position_relative` → z-scores computed within Guard / Forward / Center groups

`load_data()` reads `nba_role_score` from config (the same auto-load pattern already used for
`composite_score`). No model file changes are needed to switch modes — only the config value changes.

The mode is logged to MLflow via `log_config_dict(cfg)` which already runs at the top of each
model's `run()` function, making it trivially easy to compare runs in the MLflow UI.

---

## Phase 5: Comparing global vs. position_relative

No new code is needed for the comparison. The workflow is:

1. Set `target_score_mode: global` in config → run `regression_model.py` → MLflow run A
2. Set `target_score_mode: position_relative` → run again → MLflow run B
3. In MLflow UI, compare R², RMSE, MAE (regression) or F1-macro, balanced accuracy (classification)

Because `log_config_dict(cfg)` logs the full config on every run, filtering by
`target_score_mode` in the MLflow UI gives a clean apples-to-apples comparison.

---

## Phase 6: regression_model.py Updates

**File**: `src/models/regression_model.py`

Minimal changes to preserve existing structure (Lasso / Ridge / XGBoost suite, GridSearchCV,
MLflow logging, `plot_results`):

1. Add `"nba_role_zscore"` to `REGRESSION_TARGETS`.
2. Update `TARGET_MODE = "nba_role_zscore"` as the new default.
3. In `_plot_regression()`, add a 3rd subplot row showing the bucket distribution comparison:
   - Apply `_assign_tier_thresholded()` to `y_test` (actual z-scores) and `y_pred` (predicted)
   - Bar chart: actual tier counts vs. predicted tier counts (same style as in classification)
   - This lets you visually inspect whether the regression model is calibrated across tiers
   - Label the 4 tiers: Bust / Bench / Starter / Star
4. Log `target_score_mode` as an explicit MLflow param (extract from `cfg`).

No changes to Lasso, Ridge, or XGBoost architectures, hyperparameter grids, or CV setup.

---

## Phase 7: classification_model.py Updates

**File**: `src/models/classification_model.py`

The `prospect_tier` target now carries 4 classes (0–3) instead of 3. Changes required:

1. Update tier labels:
   ```python
   TIER_NAMES       = ["Bust", "Bench", "Starter", "Star"]
   TIER_CLASS_NAMES = ["bust", "bench", "starter", "star"]
   ```
2. Update XGBoost multiclass config: `"num_class": len(TIER_NAMES)` (evaluates to 4).
   This is already parameterized via `len(TIER_NAMES)` — no hardcoded value to change.
3. Update `CLASSIFICATION_TARGETS` to also accept `"nba_role_tier"` as an alias if desired,
   or keep as `"prospect_tier"` (simpler — the tier column just has 4 values now).
4. Log `target_score_mode` as an explicit MLflow param (extract from `cfg`).
5. The confusion matrix, bar chart, precision/recall-per-class print loop, and
   `_tune_thresholds` all iterate over `TIER_NAMES` dynamically — no further changes needed.

---

## Phase 8: Validation Checklist

After all changes are implemented and `fetch_nba_stats.py` re-run:

- [ ] Verify `nba_master.csv` — check that `note == "ok"` count is stable vs. pre-change
- [ ] Spot-check season selection for 5 known players: confirm peak box-score season was selected
- [ ] Print tier distribution from `load_data()` — should roughly match:
      bust ~19%, bench ~40%, starter ~26%, star ~15% (at ±0.5/0.5/1.5 thresholds on N=787)
- [ ] Run `regression_model.py` with `target_score_mode: global` — confirm MLflow run completes,
      plots save, R² is logged
- [ ] Run `regression_model.py` with `target_score_mode: position_relative` — compare R² and
      RMSE in MLflow UI
- [ ] Run `classification_model.py` — confirm 4-class confusion matrix renders correctly,
      F1-macro logged, all tier names display
- [ ] Run both models with `target_score_mode: position_relative` — compare classification
      F1-macro vs. global mode in MLflow UI

---

## File Change Summary

| File | Nature of Change |
|---|---|
| `data/scripts/fetch_nba_stats.py` | Replace PLUS_MINUS season selection with composite stat score (min-max scaled avg of MIN, PTS, REB, AST, STL, BLK) |
| `src/config/config.yaml` | Add `nba_role_score` block with `target_score_mode`, `tier_thresholds`, `nan_floor`, `winsor_clip`, and per-stat `weights` |
| `src/data/loader.py` | Add `_POS_BROAD_MAP`, `_map_pos_group()`; add `_compute_nba_role_score()` with weighted z-scores and winsorizing; add `_assign_tier_thresholded()`; update NBA merge columns; update `load_data()` to compute new targets; extend `TARGET_COL` |
| `src/models/regression_model.py` | Add `nba_role_zscore` to targets and set as default; add tier bucket bar chart row to `_plot_regression()`; log `target_score_mode` param |
| `src/models/classification_model.py` | Update `TIER_NAMES`/`TIER_CLASS_NAMES` to 4-class; log `target_score_mode` param |
