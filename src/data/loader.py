"""
Shared data loading and feature engineering for NBA draft prospect models.

Loads NCAA and NBA master CSVs, merges them, engineers features, and builds
sklearn ColumnTransformer preprocessors.
"""

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NCAA_PATH    = os.path.join(PROJECT_ROOT, "data", "ncaa", "ncaa_master.csv")
NBA_PATH     = os.path.join(PROJECT_ROOT, "data", "nba", "nba_master.csv")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "data", "nba", "season_cache")
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "src", "config", "config.yaml")

# ── Feature config ─────────────────────────────────────────────────────────────

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Maps TARGET_MODE strings to DataFrame column names
TARGET_COL = {
    "plus_minus":      "PLUS_MINUS",
    "became_starter":  "became_starter",
    "prospect_tier":   "prospect_tier",       # 4-class: 0=bust 1=bench 2=starter 3=star
    "composite_score": "composite_score",      # preserved for backward compat
    "nba_role_zscore": "nba_role_zscore",      # new regression target
}

NUMERIC_FEATURES = [
    # core per-game (given columns primary; backfill from totals/G if missing)
    "pts_pg", "reb_pg", "ast_pg", "blk_pg", "stl_pg",
    # computed per-game (no given equivalents)
    "fgm_pg", "fga_pg", "ft_pg", "fta_pg", "fg3_pg",
    # percentages (given columns primary; backfill from totals if missing)
    "fg_pct", "ft_pct",
    # derived efficiency
    "pts_per_fga", "ft_rate", "fg3_share",
    # other
    "G",
]

# Additional engineered features available for classification only.
CLASSIFICATION_ENGINEERED_NUMERIC: list[str] = [
    "to_pg",        # turnovers per game
    "oreb_pg",      # offensive rebounds per game
    "dreb_pg",      # defensive rebounds per game
    "fg3a_pg",      # three-point attempts per game
    "ast_to",       # assist-to-turnover ratio
    "stocks_pg",    # steals + blocks per game
    "usage_proxy",  # (FGA + 0.44*FTA + TO) / G
    "efg_pct",      # effective field goal %
    "ts_pct",       # true shooting %
    # height_in excluded — height_dev (position-relative deviation) is used instead
]

# Raw counting-stat equivalents excluded from classification (per user spec:
# PTS, FGA, 3PG, REB, AST, BLKS, ST — Ratio and MP are not in the model).
# height_in excluded because height_dev (position-relative deviation) subsumes it.
CLASSIFICATION_EXCLUDED_NUMERIC: frozenset[str] = frozenset({
    "G",
    "height_in",
})

DRAFT_PICK_FEATURE      = "draft_pick"
HEIGHT_FEATURE          = "Ht"
HEIGHT_DEV_FEATURE      = "height_dev"
CLASS_FEATURE           = "Cl"
CLASS_ORDER             = [["Fr.", "So.", "Jr.", "Sr."]]
POSITION_FEATURE        = "Pos"

PROSPECT_CONTEXT_FEATURE  = "prospect_context_score"
TEAM_DIFFICULTY_FEATURE   = "team_difficulty_score"
MPG_MINUTES_FEATURE       = "mpg_minutes"
PROSPECT_CONTEXT_MODE     = "individual"  # "composite" | "individual" | "both" | "none"

# ── Prospect Context Score config ──────────────────────────────────────────────

_TIER_1_TEAMS = {"Duke", "Kentucky", "Kansas", "North Carolina", "UCLA", "Arizona", "Michigan St."}
_TIER_2_TEAMS = {"Villanova", "Gonzaga", "Virginia", "Texas", "Baylor", "Florida", "Oregon", "Louisville", "Indiana"}
_TIER_3_TEAMS = {"Arkansas", "Auburn", "Alabama", "Tennessee", "Ohio St.", "Wisconsin", "Illinois",
                 "Texas Tech", "Houston", "Connecticut", "Marquette", "Creighton", "Xavier"}
_TIER_4_TEAMS = {"Seton Hall", "Providence", "Butler", "Saint Mary's", "VCU", "San Diego St.",
                 "Memphis", "Cincinnati", "BYU", "Dayton"}
_TIER_5_TEAMS = {"Georgia Tech", "Boston College", "Wake Forest", "Nebraska", "Minnesota", "DePaul", "Washington St."}

_TEAM_DIFFICULTY: dict[str, float] = {}
for _t, _score in [
    (_TIER_1_TEAMS, 5.00), (_TIER_2_TEAMS, 4.00), (_TIER_3_TEAMS, 3.00),
    (_TIER_4_TEAMS, 2.00), (_TIER_5_TEAMS, 1.00),
]:
    _TEAM_DIFFICULTY.update({team: _score for team in _t})

# ESPN box-score data uses full mascot names; map them to the canonical short
# names used in the tier sets above so both forms resolve to the same score.
_TEAM_ALIASES: dict[str, str] = {
    # Tier 1
    "Duke Blue Devils":           "Duke",
    "Kentucky Wildcats":          "Kentucky",
    "Kansas Jayhawks":            "Kansas",
    "North Carolina Tar Heels":   "North Carolina",
    "UCLA Bruins":                "UCLA",
    "Arizona Wildcats":           "Arizona",
    "Michigan State Spartans":    "Michigan St.",
    # Tier 2
    "Villanova Wildcats":         "Villanova",
    "Gonzaga Bulldogs":           "Gonzaga",
    "Virginia Cavaliers":         "Virginia",
    "Texas Longhorns":            "Texas",
    "Baylor Bears":               "Baylor",
    "Florida Gators":             "Florida",
    "Oregon Ducks":               "Oregon",
    "Louisville Cardinals":       "Louisville",
    "Indiana Hoosiers":           "Indiana",
    # Tier 3
    "Arkansas Razorbacks":        "Arkansas",
    "Auburn Tigers":              "Auburn",
    "Alabama Crimson Tide":       "Alabama",
    "Tennessee Volunteers":       "Tennessee",
    "Ohio State Buckeyes":        "Ohio St.",
    "Wisconsin Badgers":          "Wisconsin",
    "Illinois Fighting Illini":   "Illinois",
    "Texas Tech Red Raiders":     "Texas Tech",
    "Houston Cougars":            "Houston",
    "UConn Huskies":              "Connecticut",
    "Marquette Golden Eagles":    "Marquette",
    "Creighton Bluejays":         "Creighton",
    "Xavier Musketeers":          "Xavier",
    # Tier 4
    "Seton Hall Pirates":         "Seton Hall",
    "Providence Friars":          "Providence",
    "Butler Bulldogs":            "Butler",
    "Saint Mary's Gaels":         "Saint Mary's",
    "VCU Rams":                   "VCU",
    "San Diego State Aztecs":     "San Diego St.",
    "Memphis Tigers":             "Memphis",
    "Cincinnati Bearcats":        "Cincinnati",
    "BYU Cougars":                "BYU",
    "Dayton Flyers":              "Dayton",
    # Tier 5
    "Georgia Tech Yellow Jackets": "Georgia Tech",
    "Boston College Eagles":       "Boston College",
    "Wake Forest Demon Deacons":   "Wake Forest",
    "Nebraska Cornhuskers":        "Nebraska",
    "Minnesota Golden Gophers":    "Minnesota",
    "DePaul Blue Demons":          "DePaul",
    "Washington State Cougars":    "Washington St.",
}


def _team_difficulty(team: str) -> float:
    t = str(team).strip()
    canonical = _TEAM_ALIASES.get(t, t)
    return _TEAM_DIFFICULTY.get(canonical, 0.25)

_CLASS_SCORE: dict[str, float] = {
    "Fr.": 1.00, "Fr": 1.00,
    "So.": 0.75, "So": 0.75,
    "Jr.": 0.50, "Jr": 0.50,
    "Sr.": 0.25, "Sr": 0.25,
}

_MPG_CAP = 30.0

# ── Position mapping ───────────────────────────────────────────────────────────

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_height(ht_str):
    try:
        parts = str(ht_str).strip().split("-")
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return int(parts[0]) * 12 + int(parts[1])
    except Exception:
        pass
    return np.nan


def _parse_mpg_to_minutes(val) -> float:
    """Convert a raw MPG cell to float minutes/game.

    The NCAA CSV stores MPG in two formats depending on the data source:
      - Numeric string of seconds/game (e.g. '2283.6' → 38.1 min)
      - 'MM:SS' string (e.g. '35:56' → 35.93 min)
    Returns np.nan when the value is missing or unparseable.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    if not s or s == "nan":
        return np.nan
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(s) / 60.0  # stored as seconds/game
    except ValueError:
        return np.nan


def compute_prospect_context_score(team: str, cl: str) -> float:
    """Multiplicative scalar: school difficulty × class standing."""
    difficulty  = _team_difficulty(team)
    class_score = _CLASS_SCORE.get(str(cl).strip(), np.nan)
    if np.isnan(class_score):
        return np.nan
    return difficulty * (class_score ** 1.5)


def _compute_composite_score(df, w_min=0.55, w_gp=0.3, w_pm=0.15, nan_floor=-3.0):
    """Weighted z-score composite of NBA MIN, GP, and PLUS_MINUS.

    Players with no NBA data (all three stats NaN) receive nan_floor so they
    reliably land in the lowest tier after percentile binning.
    """
    def _zscore(s):
        mu, sigma = s.mean(), s.std()
        return (s - mu) / (sigma if sigma > 0 else 1.0)

    composite = w_min * _zscore(df["MIN"]) + w_gp * _zscore(df["GP"]) + w_pm * _zscore(df["PLUS_MINUS"])
    return composite.fillna(nan_floor)


def _load_project_config() -> dict:
    """Load src/config/config.yaml. Returns empty dict if the file is missing."""
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _assign_tier(composite, percentiles=(50, 80)):
    """Bin composite scores into 0=Bust, 1=Contributor, 2=Star.

    Cut points are derived from the given percentiles of the composite
    distribution, making it easy to shift tier boundaries via config.
    """
    cuts = np.percentile(composite, list(percentiles))
    return pd.Series(np.digitize(composite.values, cuts), index=composite.index, dtype=int)


def _compute_nba_role_score(df, mode, weights, winsor_clip=2.5, nan_floor=-3.0):
    """Weighted, winsorized z-score composite of NBA role stats.

    Each stat is z-scored (globally or within pos group), clipped at ±winsor_clip,
    then combined as a weighted average (missing stats excluded from denominator).
    The composite is re-standardized to std=1 so tier thresholds are mode-invariant.
    Players with no NBA data receive nan_floor.
    """
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
    denominator = sum(
        weights[c] * z_parts[c].notna().astype(float) for c in stat_cols
    )
    combined = numerator / denominator.replace(0.0, np.nan)

    # Re-standardize so std=1 regardless of mode or weight set
    mu_c, sigma_c = combined.mean(), combined.std()
    role_zscore = (combined - mu_c) / (sigma_c if sigma_c > 0 else 1.0)

    return role_zscore.fillna(nan_floor)


def _assign_tier_thresholded(zscore, thresholds=(-0.5, 0.5, 1.5)):
    """Bin z-scores into 4 fixed tiers: 0=Bust, 1=Bench, 2=Starter, 3=Star."""
    lo, mid, hi = thresholds
    bins   = [-np.inf, lo, mid, hi, np.inf]
    labels = [0, 1, 2, 3]
    return pd.cut(zscore, bins=bins, labels=labels).astype(int)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(composite_cfg=None):
    ncaa = pd.read_csv(NCAA_PATH)
    nba  = pd.read_csv(NBA_PATH)

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

    # Config is the source of truth. Auto-load when caller passes nothing or empty dict.
    _full_cfg = _load_project_config().get("model") or {}
    if not composite_cfg:
        composite_cfg = _full_cfg.get("composite_score") or {}
    _cfg             = composite_cfg
    w_min            = _cfg.get("w_min", 0.55)
    w_gp             = _cfg.get("w_gp", 0.30)
    w_pm             = _cfg.get("w_plus_minus", 0.15)
    tier_percentiles = tuple(_cfg.get("tier_percentiles", (50, 80)))
    nan_floor        = float(_cfg.get("nan_floor", -3.0))

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
    df["composite_score"] = _compute_composite_score(df, w_min=w_min, w_gp=w_gp, w_pm=w_pm, nan_floor=nan_floor)

    # NBA position group for position-relative mode
    df["nba_pos_group"] = df["nba_position"].apply(_map_pos_group)

    df["nba_role_zscore"] = _compute_nba_role_score(
        df, mode=_role_mode, weights=_role_weights,
        winsor_clip=_role_winsor, nan_floor=_role_nan_floor,
    )
    df["prospect_tier"] = _assign_tier_thresholded(df["nba_role_zscore"], thresholds=_role_thresholds)

    # Feature engineering
    df["height_in"] = df[HEIGHT_FEATURE].apply(parse_height)
    df[CLASS_FEATURE]    = df[CLASS_FEATURE].str.strip().replace({"Fr": "Fr.", "So": "So.", "Jr": "Jr.", "Sr": "Sr."})
    df[POSITION_FEATURE] = df[POSITION_FEATURE].str.strip()

    # Broad position group (Guard / Forward / Center) derived from the written-out
    # position column. Used for position-relative z-score computation in Phase 3;
    # not added to the model feature matrix.
    df["pos_group"] = df["position"].apply(_map_pos_group)

    pos_avg_ht = df.groupby(POSITION_FEATURE)["height_in"].transform("mean")
    df[HEIGHT_DEV_FEATURE] = (df["height_in"] - pos_avg_ht).abs()
    # Prospect context: parse MPG column, fall back to total seconds (MP) / G / 60
    _mpg_from_col = df["MPG"].apply(_parse_mpg_to_minutes)
    _mp_total     = pd.to_numeric(df["MP"], errors="coerce")
    _g_nonzero    = df["G"].replace(0, np.nan)
    _mpg_fallback = _mp_total / _g_nonzero / 60.0
    _mpg_minutes  = _mpg_from_col.fillna(_mpg_fallback)

    df[TEAM_DIFFICULTY_FEATURE]  = df["Team"].map(_team_difficulty)
    df[MPG_MINUTES_FEATURE]      = _mpg_minutes.values  # stored for analysis only; never used in training
    df[PROSPECT_CONTEXT_FEATURE] = [
        compute_prospect_context_score(team, cl)
        for team, cl in zip(df["Team"], df[CLASS_FEATURE])
    ]

    g = df["G"].replace(0, np.nan)

    # Given per-game columns are primary; backfill from totals/G only if missing
    df["pts_pg"] = df["PPG"].fillna(df["PTS"]  / g)
    df["reb_pg"] = df["RPG"].fillna(df["REB"]  / g)
    df["ast_pg"] = df["APG"].fillna(df["AST"]  / g)
    df["blk_pg"] = df["BKPG"].fillna(df["BLKS"] / g)
    df["stl_pg"] = df["STPG"].fillna(df["ST"]   / g)

    # Computed per-game (no given equivalents in source data)
    df["fgm_pg"] = df["FGM"]  / g
    df["fga_pg"] = df["FGA"]  / g
    df["ft_pg"]  = df["FT"]   / g
    df["fta_pg"] = df["FTA"]  / g
    df["fg3_pg"] = df["3FG"]  / g

    # Given percentage columns are primary; backfill from totals if missing
    df["fg_pct"] = df["FG%"].fillna((df["FGM"] / df["FGA"].replace(0, np.nan)) * 100)
    df["ft_pct"] = df["FT%"].fillna((df["FT"]  / df["FTA"].replace(0, np.nan)) * 100)

    df["pts_per_fga"] = df["PTS"] / df["FGA"].replace(0, np.nan)
    df["ft_rate"]     = df["FTA"] / df["FGA"].replace(0, np.nan)
    df["fg3_share"]   = df["3FG"].fillna(0) / df["FGM"].replace(0, np.nan)

    # ── Engineered features (classification-specific) ──────────────────────────
    _to = pd.to_numeric(df["TO"], errors="coerce")
    df["to_pg"]       = _to / g
    df["oreb_pg"]     = pd.to_numeric(df["ORebs"], errors="coerce") / g
    df["dreb_pg"]     = pd.to_numeric(df["DRebs"], errors="coerce") / g
    df["fg3a_pg"]     = pd.to_numeric(df["3FGA"], errors="coerce") / g
    df["ast_to"]      = df["AST"] / _to.replace(0, np.nan)
    df["stocks_pg"]   = (df["ST"].fillna(0) + df["BLKS"].fillna(0)) / g
    df["usage_proxy"] = (df["FGA"].fillna(0) + 0.44 * df["FTA"].fillna(0) + _to.fillna(0)) / g
    _fga_safe         = df["FGA"].replace(0, np.nan)
    df["efg_pct"]     = (df["FGM"].fillna(0) + 0.5 * df["3FG"].fillna(0)) / _fga_safe
    _denom_ts         = 2.0 * (df["FGA"].fillna(0) + 0.44 * df["FTA"].fillna(0))
    df["ts_pct"]      = df["PTS"] / _denom_ts.replace(0, np.nan)

    all_feature_cols = NUMERIC_FEATURES + CLASSIFICATION_ENGINEERED_NUMERIC
    print(f"\n{'='*50}")
    print(f"  Data Quality Report  ({len(df)} players)")
    print(f"{'='*50}")
    for col in all_feature_cols:
        n_null = df[col].isna().sum()
        print(f"  {col:<16}: {n_null:4d} null ({100*n_null/len(df):5.1f}%)")
    total_null = df[all_feature_cols].isna().sum(axis=1)
    print(f"\n  Players with 0 missing features: {(total_null == 0).sum()}")
    print(f"  Players with >50% missing:       {(total_null > len(all_feature_cols)//2).sum()}")
    print(f"{'='*50}\n")

    return df


# ── Position-relative input normalizer ─────────────────────────────────────────

class PositionGroupNormalizer(BaseEstimator, TransformerMixin):
    """Normalize numeric features relative to position group (Guard/Forward/Center).

    Expects DataFrame input containing `pos_col`.  Replaces each feature with
    its within-group z-score.  NaN values are ignored when computing stats and
    remain NaN so the downstream SimpleImputer can fill them.  Rows whose
    position is not seen during fit fall back to global stats.
    """

    def __init__(self, feature_cols, pos_col: str = "pos_group"):
        # sklearn's clone() requires __init__ to store params with the exact same
        # identity; do not wrap in list() here.
        self.feature_cols = feature_cols
        self.pos_col = pos_col

    def fit(self, X, y=None):
        cols = list(self.feature_cols)
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        feat = df[cols].astype(float)
        self.global_mean_ = feat.mean()
        self.global_std_  = feat.std().clip(lower=1e-8)
        self.group_stats_: dict = {}
        for group, subset in df.groupby(self.pos_col):
            s = subset[cols].astype(float)
            mean = s.mean()
            std  = s.std().clip(lower=1e-8).fillna(self.global_std_)
            self.group_stats_[group] = (mean, std)
        return self

    def transform(self, X):
        cols = list(self.feature_cols)
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        df[cols] = df[cols].astype(float)
        for group in df[self.pos_col].unique():
            mask = df[self.pos_col] == group
            mean, std = self.group_stats_.get(group, (self.global_mean_, self.global_std_))
            df.loc[mask, cols] = (df.loc[mask, cols] - mean) / std
        return df


# ── Feature matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(
    df,
    use_draft_pick: bool = False,
    exclude_features=None,
    prospect_context_mode: str = PROSPECT_CONTEXT_MODE,
    use_engineered_features: bool = False,
    use_pos_categorical: bool = False,
    input_normalization_mode: str = "global",
):
    """Build a sklearn preprocessor and column lists for model training.

    Returns
    -------
    preprocessor, numeric_cols, categorical_cols, ordinal_cols, passthrough_cols

    ``passthrough_cols`` contains any columns the pipeline needs that are not
    in the three named lists (e.g. ``pos_group`` for position-relative mode).
    Callers should include it in ``feature_cols`` so the columns are present
    in X when ``pipeline.fit`` / ``pipeline.predict`` is called.
    """
    _exclude     = frozenset(exclude_features or [])
    numeric_cols = [f for f in NUMERIC_FEATURES + [HEIGHT_DEV_FEATURE, TEAM_DIFFICULTY_FEATURE] if f not in _exclude]
    if use_engineered_features:
        numeric_cols += [f for f in CLASSIFICATION_ENGINEERED_NUMERIC if f not in _exclude]
    if use_draft_pick:
        numeric_cols = numeric_cols + [DRAFT_PICK_FEATURE]
    if prospect_context_mode in ("composite", "both"):
        numeric_cols.append(PROSPECT_CONTEXT_FEATURE)
    categorical_cols = [POSITION_FEATURE] if use_pos_categorical else []
    # Cl (ordinal class standing) is included only in individual/both modes
    ordinal_cols = [CLASS_FEATURE] if prospect_context_mode in ("individual", "both") else []

    use_pos_relative = input_normalization_mode == "position_relative"

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        *([("scaler", StandardScaler())] if not use_pos_relative else []),
    ])

    transformers = [("num", num_pipeline, numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
    if ordinal_cols:
        transformers.append(("ord", Pipeline([
            ("imputer",  SimpleImputer(strategy="most_frequent")),
            ("ordinal",  OrdinalEncoder(categories=CLASS_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler",   StandardScaler()),
        ]), ordinal_cols))
    column_transformer = ColumnTransformer(transformers)

    if use_pos_relative:
        pos_norm = PositionGroupNormalizer(feature_cols=numeric_cols, pos_col="pos_group")
        preprocessor = Pipeline([("pos_norm", pos_norm), ("ct", column_transformer)])
        passthrough_cols = ["pos_group"]
    else:
        preprocessor = column_transformer
        passthrough_cols = []

    return preprocessor, numeric_cols, categorical_cols, ordinal_cols, passthrough_cols
