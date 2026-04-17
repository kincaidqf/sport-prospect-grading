"""
NBA Draft Prospect Regression Model
Predicts NBA PLUS_MINUS (best season) from final-year NCAA college stats.

Why Lasso over Ridge:
- High multicollinearity in college stats (PPG, FGM, FGA all measure scoring)
- Lasso zeros out redundant features, surfacing the true predictors
- Interpretability: scouts benefit from knowing *which* college stats matter
- Ridge kept as comparison baseline

Run from project root:
    python src/models/regression_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NCAA_PATH = os.path.join(PROJECT_ROOT, "data", "ncaa", "ncaa_master.csv")
NBA_PATH = os.path.join(PROJECT_ROOT, "data", "nba", "nba_master.csv")

# Constants
TARGET = "PLUS_MINUS"

# College stats with low missingness — used directly
NUMERIC_FEATURES = ["PPG", "RPG", "FG%", "FT%", "APG", "3FG%", "BKPG", "STPG", "G", "draft_pick"]

# Height needs parsing "6-10" → inches
HEIGHT_FEATURE = "Ht"

# Class year: Fr.=1, So.=2, Jr.=3, Sr.=4
CLASS_FEATURE = "Cl"
CLASS_ORDER = [["Fr.", "So.", "Jr.", "Sr."]]

# Position: G, F, C and combos like G-F
POSITION_FEATURE = "Pos"

# Train on draft years 2009–2018, test on 2019–2026
TRAIN_YEARS = range(2009, 2019)
TEST_YEARS = range(2019, 2027)


# Data Loading & Feature Engineering

def parse_height(ht_str):
    """Convert '6-10' → 82 (inches). Returns NaN on parse failure."""
    try:
        parts = str(ht_str).strip().split("-")
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return int(parts[0]) * 12 + int(parts[1])
    except Exception:
        pass
    return np.nan


def load_data():
    ncaa = pd.read_csv(NCAA_PATH)
    nba = pd.read_csv(NBA_PATH)

    # Merge on player name + draft year (both files have 691 matched rows)
    df = ncaa.merge(
        nba[["player_name", "draft_year", TARGET]],
        left_on=["Name", "draft_year"],
        right_on=["player_name", "draft_year"],
        how="inner",
    )

    # Parse height to numeric inches
    df["height_in"] = df[HEIGHT_FEATURE].apply(parse_height)

    # Normalize class labels — strip trailing period variations
    df[CLASS_FEATURE] = df[CLASS_FEATURE].str.strip()
    # Map abbreviated forms to canonical labels
    class_map = {"Fr": "Fr.", "So": "So.", "Jr": "Jr.", "Sr": "Sr."}
    df[CLASS_FEATURE] = df[CLASS_FEATURE].replace(class_map)

    # Broad position: keep only first character group (G-F → G-F stays; fine for one-hot)
    df[POSITION_FEATURE] = df[POSITION_FEATURE].str.strip()

    return df


def build_feature_matrix(df):
    numeric_cols = NUMERIC_FEATURES + ["height_in"]
    categorical_cols = [POSITION_FEATURE]
    ordinal_cols = [CLASS_FEATURE]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=CLASS_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("ord", ordinal_transformer, ordinal_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols, ordinal_cols


# Train / Evaluate

def train_and_evaluate(df):
    preprocessor, numeric_cols, categorical_cols, ordinal_cols = build_feature_matrix(df)

    feature_cols = numeric_cols + categorical_cols + ordinal_cols

    train = df[df["draft_year"].isin(TRAIN_YEARS)]
    test = df[df["draft_year"].isin(TEST_YEARS)]

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET]

    print(f"\nDataset: {len(df)} total players")
    print(f"Train: {len(train)} players (draft years 2009–2018)")
    print(f"Test:  {len(test)} players (draft years 2019–2026)")
    print(f"Features: {len(feature_cols)} raw columns → expanded after one-hot\n")

    alphas = np.logspace(-3, 2, 100)
    results = {}

    for name, model in [
        ("Lasso", LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)),
        ("Ridge", RidgeCV(alphas=alphas, cv=5)),
    ]:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            (name.lower(), model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        best_alpha = model.alpha_

        results[name] = {
            "pipe": pipe,
            "model": model,
            "y_pred": y_pred,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "alpha": best_alpha,
        }

        print(f"{'='*40}")
        print(f"  {name} (alpha={best_alpha:.4f})")
        print(f"  R²   = {r2:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE  = {mae:.4f}")

    # Feature importance: Lasso non-zero coefficients
    print(f"\n{'='*40}")
    print("  Lasso Feature Importances (non-zero only)")
    print(f"{'='*40}")
    _print_lasso_coefficients(results["Lasso"]["pipe"], numeric_cols, categorical_cols, ordinal_cols)

    return results, y_test


def _print_lasso_coefficients(pipe, numeric_cols, categorical_cols, ordinal_cols):
    preprocessor = pipe.named_steps["preprocessor"]
    lasso = pipe.named_steps["lasso"]
    coefs = lasso.coef_

    # Reconstruct feature names after ColumnTransformer
    num_names = list(numeric_cols)
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_cols)
    )
    ord_names = ordinal_cols

    all_names = num_names + cat_names + ord_names

    coef_df = pd.DataFrame({"feature": all_names, "coefficient": coefs})
    coef_df = coef_df[coef_df["coefficient"] != 0].copy()
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    print(f"  {len(coef_df)} / {len(all_names)} features retained\n")
    for _, row in coef_df.iterrows():
        bar = "+" if row["coefficient"] > 0 else "-"
        print(f"  {bar} {row['feature']:<20}  coef = {row['coefficient']:+.4f}")


# Plots

def plot_results(results, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("NBA PLUS_MINUS Prediction from College Stats", fontsize=14, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors="none")
        lim = max(abs(y_test.min()), abs(y_test.max()), abs(np.min(y_pred)), abs(np.max(y_pred))) + 1
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual PLUS_MINUS")
        ax.set_ylabel("Predicted PLUS_MINUS")
        ax.set_title(f"{name}  (R²={res['r2']:.3f}, RMSE={res['rmse']:.2f})")
        ax.legend(fontsize=8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    plt.tight_layout()

    out_path = os.path.join(PROJECT_ROOT, "src", "models", "regression_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


# Entry Point

if __name__ == "__main__":
    df = load_data()
    results, y_test = train_and_evaluate(df)
    plot_results(results, y_test)
