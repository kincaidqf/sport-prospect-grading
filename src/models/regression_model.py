"""
NBA Draft Prospect Model
Predicts NBA outcomes from final-year NCAA college stats.

Target modes (set TARGET_MODE below):
  "plus_minus"      — best-season PLUS_MINUS (regression)
  "became_starter"  — averaged 25+ MIN/game in any season (classification)
  "survived_3yrs"   — appeared in NBA rosters for 3+ seasons (classification)

Models:
  Regression:     Lasso, Ridge, XGBoost
  Classification: LogisticCV L1, LogisticCV L2, XGBoost

Run from project root:
    python src/models/regression_model.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, roc_auc_score, classification_report,
)
from xgboost import XGBRegressor, XGBClassifier

MLFLOW_EXPERIMENT = "nba-draft-prospect-regression"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NCAA_PATH = os.path.join(PROJECT_ROOT, "data", "ncaa", "ncaa_master.csv")
NBA_PATH  = os.path.join(PROJECT_ROOT, "data", "nba", "nba_master.csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "nba", "season_cache")

# ── Toggles ────────────────────────────────────────────────────────────────────

# Target variable:
#   "plus_minus"     → regression on best-season PLUS_MINUS
#   "became_starter" → classification: averaged ≥25 MIN/game in any NBA season
#   "survived_3yrs"  → classification: appeared in 3+ distinct NBA seasons
TARGET_MODE = "survived_3yrs"

# Maps TARGET_MODE to the actual DataFrame column name
TARGET_COL = {
    "plus_minus":     "PLUS_MINUS",
    "became_starter": "became_starter",
    "survived_3yrs":  "survived_3yrs",
}

# Include draft pick as a feature? (leaks scout judgment — off by default)
USE_DRAFT_PICK = False

# ── Feature config ─────────────────────────────────────────────────────────────

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Per-game stats computed from raw totals for better coverage, plus derived
# efficiency metrics.  Original leaderboard per-game columns (PPG, RPG, …)
# are only present when a player appeared on that stat's NCAA leaderboard,
# so we recompute them from the season totals (PTS/G, REB/G, …).
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
DRAFT_PICK_FEATURE = "draft_pick"
HEIGHT_FEATURE     = "Ht"
CLASS_FEATURE      = "Cl"
CLASS_ORDER        = [["Fr.", "So.", "Jr.", "Sr."]]
POSITION_FEATURE   = "Pos"
CONF_FEATURE       = "conf_tier"

POWER_5_TEAMS = {
    # ACC
    "Duke", "North Carolina", "Syracuse", "Louisville", "Virginia", "Florida St.",
    "Georgia Tech", "Boston College", "Clemson", "Notre Dame", "Miami (FL)",
    "Pittsburgh", "Wake Forest", "NC State", "Virginia Tech",
    # Big Ten
    "Michigan", "Michigan St.", "Ohio St.", "Indiana", "Illinois", "Iowa",
    "Minnesota", "Northwestern", "Wisconsin", "Purdue", "Maryland", "Nebraska",
    "Rutgers", "Penn St.",
    # Big 12
    "Kansas", "Texas", "Oklahoma", "Baylor", "Texas Tech", "Oklahoma St.",
    "Iowa St.", "Kansas St.", "West Virginia", "TCU",
    # SEC
    "Kentucky", "Florida", "Alabama", "Arkansas", "Auburn", "Georgia", "LSU",
    "Ole Miss", "Mississippi", "Mississippi St.", "Missouri", "South Carolina",
    "Tennessee", "Texas A&M", "Vanderbilt",
    # Pac-12
    "Arizona", "Arizona St.", "California", "Colorado", "Oregon", "Oregon St.",
    "Stanford", "UCLA", "USC", "Utah", "Washington", "Washington St.",
}

HIGH_MID_MAJOR_TEAMS = {
    "Georgetown", "St. John's", "Seton Hall", "Villanova", "Connecticut",
    "Providence", "Marquette", "DePaul", "Creighton", "Butler", "Xavier",
    "Gonzaga", "Memphis", "Houston", "Cincinnati", "SMU", "Wichita St.",
    "BYU", "San Diego St.", "Nevada", "UNLV", "Utah St.",
    "Dayton", "Davidson", "Rhode Island", "VCU", "Saint Mary's",
}


# ── Data Loading ───────────────────────────────────────────────────────────────

def parse_height(ht_str):
    try:
        parts = str(ht_str).strip().split("-")
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return int(parts[0]) * 12 + int(parts[1])
    except Exception:
        pass
    return np.nan


def assign_conf_tier(team):
    if team in POWER_5_TEAMS:
        return 2
    if team in HIGH_MID_MAJOR_TEAMS:
        return 1
    return 0


def _seasons_played(nba_master):
    """Count distinct NBA seasons per player_id from the season cache."""
    files = glob.glob(os.path.join(CACHE_DIR, "*.csv"))
    if not files:
        return pd.Series(dtype=int)
    all_seasons = pd.concat(
        [pd.read_csv(f, usecols=["PLAYER_ID", "SEASON_ID"]) for f in files],
        ignore_index=True,
    )
    counts = (
        all_seasons.drop_duplicates()
        .groupby("PLAYER_ID")
        .size()
        .rename("seasons_played")
    )
    return counts


def load_data():
    ncaa = pd.read_csv(NCAA_PATH)
    nba  = pd.read_csv(NBA_PATH)

    # Always pull PLUS_MINUS and MIN (needed to derive targets)
    df = ncaa.merge(
        nba[["player_name", "draft_year", "PLUS_MINUS", "MIN", "player_id"]],
        left_on=["Name", "draft_year"],
        right_on=["player_name", "draft_year"],
        how="inner",
    )

    # Derived targets
    df["became_starter"] = (df["MIN"] >= 25).astype(int)

    seasons = _seasons_played(nba)
    df["survived_3yrs"] = df["player_id"].map(seasons).fillna(0).ge(3).astype(int)

    # ── Feature engineering ───────────────────────────────────────────────
    df["height_in"] = df[HEIGHT_FEATURE].apply(parse_height)
    df[CLASS_FEATURE]    = df[CLASS_FEATURE].str.strip().replace({"Fr": "Fr.", "So": "So.", "Jr": "Jr.", "Sr": "Sr."})
    df[POSITION_FEATURE] = df[POSITION_FEATURE].str.strip()
    df[CONF_FEATURE]     = df["Team"].apply(assign_conf_tier)

    # Compute per-game stats from raw totals — these have better coverage
    # than the pre-computed leaderboard columns (PPG, RPG, etc.).
    g = df["G"].replace(0, np.nan)  # guard against division by zero
    df["pts_pg"] = df["PTS"] / g
    df["reb_pg"] = df["REB"] / g
    df["ast_pg"] = df["AST"] / g
    df["blk_pg"] = df["BLKS"] / g
    df["stl_pg"] = df["ST"]  / g
    df["fgm_pg"] = df["FGM"] / g
    df["fga_pg"] = df["FGA"] / g
    df["ft_pg"]  = df["FT"]  / g
    df["fta_pg"] = df["FTA"] / g
    df["fg3_pg"] = df["3FG"] / g

    # Fill computed per-game with original leaderboard values where totals
    # were missing but the per-game column existed.
    _backfill = [
        ("pts_pg", "PPG"), ("reb_pg", "RPG"), ("ast_pg", "APG"),
        ("blk_pg", "BKPG"), ("stl_pg", "STPG"),
    ]
    for computed, original in _backfill:
        df[computed] = df[computed].fillna(df[original])

    # Percentages — compute from totals, fall back to leaderboard columns
    df["fg_pct"] = (df["FGM"] / df["FGA"].replace(0, np.nan)) * 100
    df["fg_pct"] = df["fg_pct"].fillna(df["FG%"])

    df["ft_pct"] = (df["FT"] / df["FTA"].replace(0, np.nan)) * 100
    df["ft_pct"] = df["ft_pct"].fillna(df["FT%"])

    # Derived efficiency features
    df["pts_per_fga"] = df["PTS"] / df["FGA"].replace(0, np.nan)
    df["ft_rate"]     = df["FTA"] / df["FGA"].replace(0, np.nan)
    df["fg3_share"]   = df["3FG"].fillna(0) / df["FGM"].replace(0, np.nan)

    # Binary flag
    df["shoots_3s"] = (df["3FG"].fillna(0) > 0).astype(float)

    # ── Data quality report ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Data Quality Report  ({len(df)} players)")
    print(f"{'='*50}")
    for col in NUMERIC_FEATURES:
        n_null = df[col].isna().sum()
        print(f"  {col:<14}: {n_null:4d} null ({100*n_null/len(df):5.1f}%)")
    total_null = df[NUMERIC_FEATURES].isna().sum(axis=1)
    print(f"\n  Players with 0 missing features: {(total_null == 0).sum()}")
    print(f"  Players with >50% missing:       {(total_null > len(NUMERIC_FEATURES)//2).sum()}")
    print(f"{'='*50}\n")

    return df


# ── Feature Matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(df, use_draft_pick=USE_DRAFT_PICK):
    numeric_cols = NUMERIC_FEATURES + ["height_in", CONF_FEATURE]
    if use_draft_pick:
        numeric_cols = numeric_cols + [DRAFT_PICK_FEATURE]
    categorical_cols = [POSITION_FEATURE]
    ordinal_cols     = [CLASS_FEATURE]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols),
        ("ord", Pipeline([
            ("imputer",  SimpleImputer(strategy="most_frequent")),
            ("ordinal",  OrdinalEncoder(categories=CLASS_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler",   StandardScaler()),
        ]), ordinal_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols, ordinal_cols


# ── Train / Evaluate ───────────────────────────────────────────────────────────

def _is_classification():
    return TARGET_MODE in ("became_starter", "survived_3yrs")


def train_and_evaluate(df, use_draft_pick=USE_DRAFT_PICK):
    preprocessor, numeric_cols, categorical_cols, ordinal_cols = build_feature_matrix(df, use_draft_pick)
    feature_cols = numeric_cols + categorical_cols + ordinal_cols

    col = TARGET_COL[TARGET_MODE]
    y = df[col]
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train, y_train = train[feature_cols], train[col]
    X_test,  y_test  = test[feature_cols],  test[col]

    print(f"\nTarget:  {TARGET_MODE}")
    if _is_classification():
        print(f"Classes: {dict(y.value_counts().sort_index())}")
    print(f"Dataset: {len(df)} total players")
    print(f"Train:   {len(train)} | Test: {len(test)}")
    print(f"Features: {len(feature_cols)} raw columns → expanded after one-hot\n")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    results = {}

    # Store column info for plotting
    col_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
    }

    if _is_classification():
        _run_classification(
            preprocessor, feature_cols, numeric_cols, categorical_cols, ordinal_cols,
            X_train, y_train, X_test, y_test, train, test, use_draft_pick, results,
        )
    else:
        _run_regression(
            preprocessor, feature_cols, numeric_cols, categorical_cols, ordinal_cols,
            X_train, y_train, X_test, y_test, train, test, use_draft_pick, results,
        )

    return results, y_test, col_info


def _run_regression(
    preprocessor, feature_cols, numeric_cols, categorical_cols, ordinal_cols,
    X_train, y_train, X_test, y_test, train, test, use_draft_pick, results,
):
    alphas = np.logspace(-3, 2, 100)

    for name, model in [
        ("Lasso", LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)),
        ("Ridge", RidgeCV(alphas=alphas, cv=5)),
    ]:
        with mlflow.start_run(run_name=f"{name}_{TARGET_MODE}"):
            pipe = Pipeline([("preprocessor", preprocessor), (name.lower(), model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            r2   = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)

            mlflow.log_params({
                "model": name, "target": TARGET_MODE, "alpha": model.alpha_,
                "n_train": len(train), "n_test": len(test), "use_draft_pick": use_draft_pick,
            })
            mlflow.log_metrics({"r2": r2, "rmse": rmse, "mae": mae})

            if name == "Lasso":
                coef_df = _get_lasso_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols)
                mlflow.log_metric("n_nonzero_features", len(coef_df))

            mlflow.sklearn.log_model(pipe, artifact_path=name.lower())
            results[name] = {"pipe": pipe, "model": model, "y_pred": y_pred,
                             "r2": r2, "rmse": rmse, "mae": mae, "alpha": model.alpha_}

            print(f"{'='*40}")
            print(f"  {name} (alpha={model.alpha_:.4f})")
            print(f"  R²   = {r2:.4f}")
            print(f"  RMSE = {rmse:.4f}")
            print(f"  MAE  = {mae:.4f}")

    # XGBoost
    xgb_preprocessor, _, _, _ = build_feature_matrix(X_train.assign(**{c: X_train[c] for c in X_train.columns}))
    xgb_base = Pipeline([
        ("preprocessor", xgb_preprocessor),
        ("xgb", XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, min_child_weight=5)),
    ])
    gs = GridSearchCV(xgb_base, {
        "xgb__n_estimators": [100, 200],
        "xgb__max_depth": [2, 3],
        "xgb__learning_rate": [0.05, 0.1],
        "xgb__subsample": [0.7, 0.9],
    }, cv=5, scoring="r2", n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred_xgb = best.predict(X_test)
    r2_xgb   = r2_score(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)

    with mlflow.start_run(run_name=f"XGBoost_{TARGET_MODE}"):
        mlflow.log_params({"model": "XGBoost", "target": TARGET_MODE,
                           **{k.replace("xgb__", ""): v for k, v in gs.best_params_.items()},
                           "n_train": len(X_train), "n_test": len(X_test), "use_draft_pick": use_draft_pick})
        mlflow.log_metrics({"r2": r2_xgb, "rmse": rmse_xgb, "mae": mae_xgb})
        _log_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)
        mlflow.sklearn.log_model(best, artifact_path="xgboost")

    results["XGBoost"] = {"pipe": best, "model": best.named_steps["xgb"], "y_pred": y_pred_xgb,
                          "r2": r2_xgb, "rmse": rmse_xgb, "mae": mae_xgb, "alpha": None}

    print(f"{'='*40}")
    print(f"  XGBoost  (best: {gs.best_params_})")
    print(f"  R²   = {r2_xgb:.4f}")
    print(f"  RMSE = {rmse_xgb:.4f}")
    print(f"  MAE  = {mae_xgb:.4f}")

    print(f"\n{'='*40}\n  Lasso Feature Importances (non-zero only)\n{'='*40}")
    _print_lasso_coefficients(results["Lasso"]["pipe"], numeric_cols, categorical_cols, ordinal_cols)
    print(f"\n{'='*40}\n  XGBoost Feature Importances (top 10)\n{'='*40}")
    _print_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)


def _run_classification(
    preprocessor, feature_cols, numeric_cols, categorical_cols, ordinal_cols,
    X_train, y_train, X_test, y_test, train, test, use_draft_pick, results,
):
    Cs = np.logspace(-3, 2, 20)

    for name, penalty in [("LogisticL1", "l1"), ("LogisticL2", "l2")]:
        with mlflow.start_run(run_name=f"{name}_{TARGET_MODE}"):
            model = LogisticRegressionCV(
                Cs=Cs, cv=5, penalty=penalty, solver="saga",
                max_iter=5000, random_state=42, n_jobs=-1,
            )
            pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            best_C = model.C_[0]

            mlflow.log_params({
                "model": name, "target": TARGET_MODE, "penalty": penalty,
                "C": best_C, "n_train": len(train), "n_test": len(test),
                "use_draft_pick": use_draft_pick,
            })
            mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})
            mlflow.sklearn.log_model(pipe, artifact_path=name.lower())

            results[name] = {"pipe": pipe, "model": model, "y_pred": y_pred,
                             "y_prob": y_prob, "accuracy": acc, "auc": auc, "C": best_C}

            print(f"{'='*40}")
            print(f"  {name} (C={best_C:.4f}, penalty={penalty})")
            print(f"  Accuracy = {acc:.4f}")
            print(f"  ROC-AUC  = {auc:.4f}")
            print(classification_report(y_test, y_pred, target_names=["No", "Yes"], zero_division=0))

    # XGBoost classifier
    xgb_preprocessor, _, _, _ = build_feature_matrix(X_train)
    xgb_base = Pipeline([
        ("preprocessor", xgb_preprocessor),
        ("xgb", XGBClassifier(random_state=42, n_jobs=-1, verbosity=0,
                              min_child_weight=5, eval_metric="logloss")),
    ])
    gs = GridSearchCV(xgb_base, {
        "xgb__n_estimators": [100, 200],
        "xgb__max_depth": [2, 3],
        "xgb__learning_rate": [0.05, 0.1],
        "xgb__subsample": [0.7, 0.9],
    }, cv=5, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred_xgb = best.predict(X_test)
    y_prob_xgb = best.predict_proba(X_test)[:, 1]
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)

    with mlflow.start_run(run_name=f"XGBoost_{TARGET_MODE}"):
        mlflow.log_params({"model": "XGBoost", "target": TARGET_MODE,
                           **{k.replace("xgb__", ""): v for k, v in gs.best_params_.items()},
                           "n_train": len(X_train), "n_test": len(X_test)})
        mlflow.log_metrics({"accuracy": acc_xgb, "roc_auc": auc_xgb})
        _log_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)
        mlflow.sklearn.log_model(best, artifact_path="xgboost")

    results["XGBoost"] = {"pipe": best, "model": best.named_steps["xgb"],
                          "y_pred": y_pred_xgb, "y_prob": y_prob_xgb,
                          "accuracy": acc_xgb, "auc": auc_xgb, "C": None}

    print(f"{'='*40}")
    print(f"  XGBoost  (best: {gs.best_params_})")
    print(f"  Accuracy = {acc_xgb:.4f}")
    print(f"  ROC-AUC  = {auc_xgb:.4f}")
    print(classification_report(y_test, y_pred_xgb, target_names=["No", "Yes"], zero_division=0))

    print(f"\n{'='*40}\n  XGBoost Feature Importances (top 10)\n{'='*40}")
    _print_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)


# ── Feature importance helpers ─────────────────────────────────────────────────

def _get_lasso_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols):
    preprocessor = pipe.named_steps["preprocessor"]
    lasso = pipe.named_steps["lasso"]
    cat_names = list(preprocessor.named_transformers_["cat"]
                     .named_steps["onehot"].get_feature_names_out(categorical_cols))
    all_names = list(numeric_cols) + cat_names + list(ordinal_cols)
    coef_df = pd.DataFrame({"feature": all_names, "coefficient": lasso.coef_})
    coef_df = coef_df[coef_df["coefficient"] != 0].copy()
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    return coef_df.sort_values("abs_coef", ascending=False)


def _print_lasso_coefficients(pipe, numeric_cols, categorical_cols, ordinal_cols):
    preprocessor = pipe.named_steps["preprocessor"]
    cat_names = list(preprocessor.named_transformers_["cat"]
                     .named_steps["onehot"].get_feature_names_out(categorical_cols))
    all_names = list(numeric_cols) + cat_names + list(ordinal_cols)
    coef_df = _get_lasso_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols)
    print(f"  {len(coef_df)} / {len(all_names)} features retained\n")
    for _, row in coef_df.iterrows():
        sign = "+" if row["coefficient"] > 0 else "-"
        print(f"  {sign} {row['feature']:<20}  coef = {row['coefficient']:+.4f}")


def _get_xgb_importance_df(pipe, numeric_cols, categorical_cols, ordinal_cols):
    preprocessor = pipe.named_steps["preprocessor"]
    xgb = pipe.named_steps["xgb"]
    cat_names = list(preprocessor.named_transformers_["cat"]
                     .named_steps["onehot"].get_feature_names_out(categorical_cols))
    all_names = list(numeric_cols) + cat_names + list(ordinal_cols)
    return pd.DataFrame({"feature": all_names, "importance": xgb.feature_importances_})\
             .sort_values("importance", ascending=False)


def _log_xgb_importances(pipe, numeric_cols, categorical_cols, ordinal_cols):
    for _, row in _get_xgb_importance_df(pipe, numeric_cols, categorical_cols, ordinal_cols).head(15).iterrows():
        safe = row["feature"].replace(" ", "_").replace("%", "pct").replace("-", "_")
        mlflow.log_metric(f"imp_{safe}", row["importance"])


def _print_xgb_importances(pipe, numeric_cols, categorical_cols, ordinal_cols, top_n=10):
    for _, row in _get_xgb_importance_df(pipe, numeric_cols, categorical_cols, ordinal_cols).head(top_n).iterrows():
        bar = "#" * int(row["importance"] * 50)
        print(f"  {row['feature']:<20}  {row['importance']:.4f}  {bar}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def _get_all_feature_names(pipe, numeric_cols, categorical_cols, ordinal_cols):
    """Resolve expanded feature names (after one-hot encoding)."""
    preprocessor = pipe.named_steps["preprocessor"]
    cat_names = list(preprocessor.named_transformers_["cat"]
                     .named_steps["onehot"].get_feature_names_out(categorical_cols))
    return list(numeric_cols) + cat_names + list(ordinal_cols)


def _get_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols, step_name):
    """Get coefficient DataFrame from a linear model pipeline step."""
    model = pipe.named_steps[step_name]
    all_names = _get_all_feature_names(pipe, numeric_cols, categorical_cols, ordinal_cols)
    coefs = model.coef_.ravel()
    df = pd.DataFrame({"feature": all_names, "coefficient": coefs})
    df["abs_coef"] = df["coefficient"].abs()
    return df.sort_values("abs_coef", ascending=False)


def plot_results(results, y_test, col_info):
    if _is_classification():
        _plot_classification(results, y_test)
    else:
        _plot_regression(results, y_test)

    _plot_feature_importance(results, col_info)
    _plot_model_summary(results, y_test)


def _plot_regression(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    fig.suptitle(f"NBA {TARGET_MODE} Prediction from College Stats", fontsize=14, fontweight="bold")

    for col_idx, (name, res) in enumerate(results.items()):
        y_pred = res["y_pred"]

        # Row 1: actual vs predicted scatter
        ax = axes[0][col_idx]
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors="none")
        lim = max(abs(y_test.min()), abs(y_test.max()), abs(np.min(y_pred)), abs(np.max(y_pred))) + 1
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Perfect")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}  (R²={res['r2']:.3f})")
        ax.legend(fontsize=8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # Row 2: residual plot
        ax2 = axes[1][col_idx]
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors="none")
        ax2.axhline(0, color="r", linestyle="--", linewidth=1)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residual (Actual - Predicted)")
        ax2.set_title(f"{name} Residuals")

    plt.tight_layout()
    _save_and_log(fig, "regression_results.png", results,
                  {n: {"r2": r["r2"], "rmse": r["rmse"], "mae": r["mae"]} for n, r in results.items()})


def _plot_classification(results, y_test):
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    fig.suptitle(f"NBA {TARGET_MODE} Classification from College Stats", fontsize=13, fontweight="bold")

    for col, (name, res) in enumerate(results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, res["y_pred"], display_labels=["No", "Yes"], ax=axes[0][col],
        )
        axes[0][col].set_title(f"{name}\nAcc={res['accuracy']:.3f}")

        RocCurveDisplay.from_predictions(y_test, res["y_prob"], ax=axes[1][col], name=name)
        axes[1][col].set_title(f"ROC  AUC={res['auc']:.3f}")

    plt.tight_layout()
    _save_and_log(fig, "classification_results.png", results,
                  {n: {"accuracy": r["accuracy"], "roc_auc": r["auc"]} for n, r in results.items()})


def _plot_feature_importance(results, col_info):
    """Plot feature importance / coefficients for all models side by side."""
    numeric_cols = col_info["numeric_cols"]
    categorical_cols = col_info["categorical_cols"]
    ordinal_cols = col_info["ordinal_cols"]

    # Collect importance data per model
    importance_data = {}

    for name, res in results.items():
        pipe = res["pipe"]
        if name in ("Lasso", "Ridge"):
            coef_df = _get_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols, name.lower())
            importance_data[name] = coef_df.rename(columns={"coefficient": "importance"})
        elif name in ("LogisticL1", "LogisticL2"):
            coef_df = _get_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols, "clf")
            importance_data[name] = coef_df.rename(columns={"coefficient": "importance"})
        elif name == "XGBoost":
            imp_df = _get_xgb_importance_df(pipe, numeric_cols, categorical_cols, ordinal_cols)
            importance_data[name] = imp_df.rename(columns={"importance": "importance"})

    n_models = len(importance_data)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8))
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Feature Importance — {TARGET_MODE}", fontsize=14, fontweight="bold")

    for ax, (name, imp_df) in zip(axes, importance_data.items()):
        top = imp_df.head(15).copy()
        is_coef = name in ("Lasso", "Ridge", "LogisticL1", "LogisticL2")

        if is_coef:
            # Show signed coefficients — sort by absolute value
            top = top.sort_values("abs_coef", ascending=True)
            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top["importance"]]
            ax.barh(top["feature"], top["importance"], color=colors)
            ax.set_xlabel("Coefficient (standardized)")
            ax.axvline(0, color="black", linewidth=0.5)
        else:
            # XGBoost importance — always positive
            top = top.sort_values("importance", ascending=True)
            ax.barh(top["feature"], top["importance"], color="#3498db")
            ax.set_xlabel("Feature Importance (gain)")

        ax.set_title(name)
        ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "src", "models", "feature_importance.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()

    # ── Coefficient / importance table (all models combined) ──────────────
    _plot_importance_heatmap(importance_data)


def _plot_importance_heatmap(importance_data):
    """Heatmap comparing feature importance across all models."""
    # Build a unified dataframe
    all_features = set()
    for imp_df in importance_data.values():
        all_features.update(imp_df["feature"].tolist())
    all_features = sorted(all_features)

    matrix = pd.DataFrame(index=all_features)
    for name, imp_df in importance_data.items():
        col_name = name
        vals = imp_df.set_index("feature")
        if "abs_coef" in vals.columns:
            matrix[col_name] = vals["abs_coef"]
        else:
            matrix[col_name] = vals["importance"]
    matrix = matrix.fillna(0)

    # Normalize each column to 0-1 for comparison
    matrix_norm = matrix.copy()
    for c in matrix_norm.columns:
        mx = matrix_norm[c].max()
        if mx > 0:
            matrix_norm[c] = matrix_norm[c] / mx

    # Sort by mean importance across models
    matrix_norm["_mean"] = matrix_norm.mean(axis=1)
    matrix_norm = matrix_norm.sort_values("_mean", ascending=True)
    matrix_norm = matrix_norm.drop(columns=["_mean"])

    # Take top 20
    matrix_norm = matrix_norm.tail(20)

    fig, ax = plt.subplots(figsize=(8, max(6, len(matrix_norm) * 0.4)))
    im = ax.imshow(matrix_norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(matrix_norm.columns)))
    ax.set_xticklabels(matrix_norm.columns, fontsize=10)
    ax.set_yticks(range(len(matrix_norm.index)))
    ax.set_yticklabels(matrix_norm.index, fontsize=9)

    # Annotate cells
    for i in range(len(matrix_norm.index)):
        for j in range(len(matrix_norm.columns)):
            val = matrix_norm.values[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"Normalized Feature Importance Comparison — {TARGET_MODE}",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Relative importance (0–1)", shrink=0.8)
    plt.tight_layout()

    out_path = os.path.join(PROJECT_ROOT, "src", "models", "importance_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


def _plot_model_summary(results, y_test):
    """Summary table showing all model metrics side by side."""
    is_clf = _is_classification()

    rows = []
    for name, res in results.items():
        row = {"Model": name}
        if is_clf:
            row["Accuracy"] = res["accuracy"]
            row["ROC-AUC"] = res["auc"]
            row["C / alpha"] = f"{res.get('C', res.get('alpha', ''))}"
        else:
            row["R²"] = res["r2"]
            row["RMSE"] = res["rmse"]
            row["MAE"] = res["mae"]
            row["alpha"] = res.get("alpha", "")
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(summary_df.columns)), 1.2 + 0.5 * len(rows)))
    ax.axis("off")

    # Format numeric values
    cell_text = []
    for _, row in summary_df.iterrows():
        formatted = []
        for v in row:
            if isinstance(v, float):
                formatted.append(f"{v:.4f}")
            else:
                formatted.append(str(v))
        cell_text.append(formatted)

    table = ax.table(
        cellText=cell_text,
        colLabels=summary_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(summary_df.columns)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(rows)):
        color = "#ecf0f1" if i % 2 == 0 else "white"
        for j in range(len(summary_df.columns)):
            table[i + 1, j].set_facecolor(color)

    title = "Classification" if is_clf else "Regression"
    ax.set_title(f"Model Comparison — {title} ({TARGET_MODE})",
                 fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "src", "models", "model_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


def _save_and_log(fig, filename, results, summary_metrics):
    out_path = os.path.join(PROJECT_ROOT, "src", "models", filename)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"summary_{TARGET_MODE}"):
        mlflow.log_artifact(out_path, artifact_path="plots")
        for name, metrics in summary_metrics.items():
            mlflow.log_metrics({f"{name.lower()}_{k}": v for k, v in metrics.items()})

    plt.show()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    results, y_test, col_info = train_and_evaluate(df)
    plot_results(results, y_test, col_info)
