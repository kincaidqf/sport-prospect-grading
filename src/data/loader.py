"""
Shared data loading and feature engineering for NBA draft prospect models.

Loads NCAA and NBA master CSVs, merges them, engineers features, and builds
sklearn ColumnTransformer preprocessors.
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NCAA_PATH    = os.path.join(PROJECT_ROOT, "data", "ncaa", "ncaa_master.csv")
NBA_PATH     = os.path.join(PROJECT_ROOT, "data", "nba", "nba_master.csv")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "data", "nba", "season_cache")

# ── Feature config ─────────────────────────────────────────────────────────────

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Maps TARGET_MODE strings to DataFrame column names
TARGET_COL = {
    "plus_minus":      "PLUS_MINUS",
    "became_starter":  "became_starter",
    "prospect_tier":   "prospect_tier",
    "composite_score": "composite_score",
}

NUMERIC_FEATURES = [
    # core per-game (computed from totals)
    "pts_pg", "reb_pg", "ast_pg", "blk_pg", "stl_pg",
    "fgm_pg", "fga_pg", "ft_pg", "fta_pg", "fg3_pg",
    # percentages (computed from totals)
    "fg_pct", "ft_pct",
    # derived efficiency
    "pts_per_fga", "ft_rate", "fg3_share",
    # other
    "G", "shoots_3s",
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
    "height_in",    # height in inches
]

# Raw counting-stat equivalents excluded from classification (per user spec:
# PTS, FGA, 3PG, REB, AST, BLKS, ST — Ratio and MP are not in the model).
CLASSIFICATION_EXCLUDED_NUMERIC: frozenset[str] = frozenset({
    "G"
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
PROSPECT_CONTEXT_MODE     = "composite"  # "composite" | "individual" | "both" | "none"

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


def _assign_tier(composite, percentiles=(40, 80)):
    """Bin composite scores into 0=Bust, 1=Contributor, 2=Star (40/40/20 split).

    Cut points are derived from the given percentiles of the composite
    distribution, making it easy to shift tier boundaries via config.
    """
    cuts = np.percentile(composite, list(percentiles))
    return pd.Series(np.digitize(composite.values, cuts), index=composite.index, dtype=int)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(composite_cfg=None):
    ncaa = pd.read_csv(NCAA_PATH)
    nba  = pd.read_csv(NBA_PATH)

    df = ncaa.merge(
        nba[["player_name", "draft_year", "PLUS_MINUS", "MIN", "GP", "player_id"]],
        left_on=["Name", "draft_year"],
        right_on=["player_name", "draft_year"],
        how="inner",
    )

    _cfg             = composite_cfg or {}
    w_min            = _cfg.get("w_min", 0.35)
    w_gp             = _cfg.get("w_gp", 0.25)
    w_pm             = _cfg.get("w_plus_minus", 0.40)
    tier_percentiles = tuple(_cfg.get("tier_percentiles", (40, 80)))
    nan_floor        = float(_cfg.get("nan_floor", -3.0))

    # Derived targets
    df["became_starter"]  = (df["MIN"] >= 25).astype(int)
    df["composite_score"] = _compute_composite_score(df, w_min=w_min, w_gp=w_gp, w_pm=w_pm, nan_floor=nan_floor)
    df["prospect_tier"]   = _assign_tier(df["composite_score"], percentiles=tier_percentiles)

    # Feature engineering
    df["height_in"] = df[HEIGHT_FEATURE].apply(parse_height)
    df[CLASS_FEATURE]    = df[CLASS_FEATURE].str.strip().replace({"Fr": "Fr.", "So": "So.", "Jr": "Jr.", "Sr": "Sr."})
    df[POSITION_FEATURE] = df[POSITION_FEATURE].str.strip()

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
    df["pts_pg"] = df["PTS"]  / g
    df["reb_pg"] = df["REB"]  / g
    df["ast_pg"] = df["AST"]  / g
    df["blk_pg"] = df["BLKS"] / g
    df["stl_pg"] = df["ST"]   / g
    df["fgm_pg"] = df["FGM"]  / g
    df["fga_pg"] = df["FGA"]  / g
    df["ft_pg"]  = df["FT"]   / g
    df["fta_pg"] = df["FTA"]  / g
    df["fg3_pg"] = df["3FG"]  / g

    _backfill = [
        ("pts_pg", "PPG"), ("reb_pg", "RPG"), ("ast_pg", "APG"),
        ("blk_pg", "BKPG"), ("stl_pg", "STPG"),
    ]
    for computed, original in _backfill:
        df[computed] = df[computed].fillna(df[original])

    df["fg_pct"] = (df["FGM"] / df["FGA"].replace(0, np.nan)) * 100
    df["fg_pct"] = df["fg_pct"].fillna(df["FG%"])

    df["ft_pct"] = (df["FT"] / df["FTA"].replace(0, np.nan)) * 100
    df["ft_pct"] = df["ft_pct"].fillna(df["FT%"])

    df["pts_per_fga"] = df["PTS"] / df["FGA"].replace(0, np.nan)
    df["ft_rate"]     = df["FTA"] / df["FGA"].replace(0, np.nan)
    df["fg3_share"]   = df["3FG"].fillna(0) / df["FGM"].replace(0, np.nan)
    df["shoots_3s"]   = (df["3FG"].fillna(0) > 0).astype(float)

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


# ── Feature matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(
    df,
    use_draft_pick: bool = False,
    exclude_features=None,
    prospect_context_mode: str = PROSPECT_CONTEXT_MODE,
    use_engineered_features: bool = False,
    use_pos_categorical: bool = False,
):
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

    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_cols),
    ]
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
    preprocessor = ColumnTransformer(transformers)

    return preprocessor, numeric_cols, categorical_cols, ordinal_cols
