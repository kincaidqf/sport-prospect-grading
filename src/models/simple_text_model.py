"""Shallow lexicon + sentiment features on scouting text → Ridge on ``nba_role_zscore`` → Gaussian tier probs for PSM."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mlflow
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data.loader import RANDOM_STATE, TARGET_COL
from src.models.probability import PROBA_COLUMNS, normalize_proba
from src.models.text_model import (
    UniformTextTierBundle,
    compute_tier_residual_std,
    load_text_data,
    predict_tier_proba_from_role_z,
)
from src.utils.mlflow_utils import build_mlflow_context, log_common_params, log_config_dict, managed_run

TARGET_Z = TARGET_COL["nba_role_zscore"]

_DEFAULT_SUCCESS_WORDS: tuple[str, ...] = (
    "elite",
    "versatile",
    "explosive",
    "high-iq",
    "high iq",
    "polished",
    "nba-ready",
    "nba ready",
    "length",
    "motor",
    "vertical",
    "creates",
    "efficient",
    "vision",
    "shot-maker",
    "shot maker",
    "translatable",
    "two-way",
    "two way",
)
_DEFAULT_RED_FLAG_WORDS: tuple[str, ...] = (
    "limited",
    "raw",
    "streaky",
    "undersized",
    "slow",
    "stiff",
    "struggles",
    "inconsistent",
    "concerns",
    "issues",
    "weakness",
    "liability",
)
_DEFAULT_RED_FLAG_PHRASES: tuple[str, ...] = (
    "lack of effort",
    "injury prone",
    "injury-prone",
    "offensive liability",
    "below average athlete",
)


def ensure_vader_lexicon() -> None:
    """Download VADER lexicon if missing (safe for CI / first local run)."""
    try:
        import nltk
        from nltk.data import find

        find("sentiment/vader_lexicon.zip")
    except LookupError:
        import nltk

        nltk.download("vader_lexicon", quiet=True)


def _get_vader_analyzer():
    ensure_vader_lexicon()
    from nltk.sentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def _word_pattern(term: str) -> re.Pattern[str]:
    t = re.escape(term.strip().lower())
    return re.compile(rf"(?<!\w){t}(?!\w)", re.IGNORECASE)


def _count_phrase(hay: str, phrase: str) -> int:
    if not phrase.strip():
        return 0
    return len(re.findall(re.escape(phrase.lower()), hay.lower()))


@dataclass
class ShallowLexiconConfig:
    sentiment_enabled: bool = True
    sentiment_use_pos_neg_neu: bool = True
    success_words: tuple[str, ...] = _DEFAULT_SUCCESS_WORDS
    red_flag_words: tuple[str, ...] = _DEFAULT_RED_FLAG_WORDS
    red_flag_phrases: tuple[str, ...] = _DEFAULT_RED_FLAG_PHRASES
    tfidf_enabled: bool = False
    tfidf_max_features: int = 1000
    tfidf_min_df: int | float = 2
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    ridge_alpha: float = 1.0
    random_state: int = RANDOM_STATE

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ShallowLexiconConfig:
        if not d:
            return cls()
        sent = d.get("sentiment") or {}
        lex = d.get("lexicons") or {}
        tfidf = d.get("tfidf") or {}
        success = lex.get("success_words")
        red_w = lex.get("red_flag_words")
        red_p = lex.get("red_flag_phrases")
        return cls(
            sentiment_enabled=bool(sent.get("enabled", True)),
            sentiment_use_pos_neg_neu=bool(sent.get("use_pos_neg_neu", True)),
            success_words=tuple(success) if success is not None else _DEFAULT_SUCCESS_WORDS,
            red_flag_words=tuple(red_w) if red_w is not None else _DEFAULT_RED_FLAG_WORDS,
            red_flag_phrases=tuple(red_p) if red_p is not None else _DEFAULT_RED_FLAG_PHRASES,
            tfidf_enabled=bool(tfidf.get("enabled", False)),
            tfidf_max_features=int(tfidf.get("max_features", 1000)),
            tfidf_min_df=tfidf.get("min_df", 2),
            tfidf_ngram_min=int(tfidf.get("ngram_range", [1, 2])[0]),
            tfidf_ngram_max=int(tfidf.get("ngram_range", [1, 2])[-1]),
            ridge_alpha=float(d.get("ridge_alpha", 1.0)),
            random_state=int(d.get("random_state", RANDOM_STATE)),
        )


def merge_shallow_cfg(global_d: dict[str, Any] | None, tower_d: dict[str, Any] | None) -> ShallowLexiconConfig:
    """Deep-merge ``model.text_shallow`` with optional per-tower ``shallow`` overrides."""
    g = dict(global_d or {})
    t = dict(tower_d or {})
    merged: dict[str, Any] = {**g, **t}
    for key in ("sentiment", "lexicons", "tfidf"):
        if key in g or key in t:
            inner: dict[str, Any] = {**(g.get(key) or {}), **(t.get(key) or {})}
            merged[key] = inner
    return ShallowLexiconConfig.from_dict(merged)


def handcrafted_feature_matrix(texts: Sequence[str], cfg: ShallowLexiconConfig) -> np.ndarray:
    """Dense design block: VADER + lexicon counts/rates (no fitting)."""
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    word_re_success = [_word_pattern(w) for w in cfg.success_words]
    word_re_red = [_word_pattern(w) for w in cfg.red_flag_words]

    n_sent = 4 if cfg.sentiment_enabled and cfg.sentiment_use_pos_neg_neu else (1 if cfg.sentiment_enabled else 0)
    n_lex = (
        1
        + len(word_re_success) * 2
        + len(word_re_red) * 2
        + len(cfg.red_flag_phrases) * 2
    )
    n_cols = n_sent + n_lex
    if n_cols == 0:
        return np.zeros((n, 0), dtype=np.float64)

    X = np.zeros((n, n_cols), dtype=np.float64)
    analyzer = _get_vader_analyzer() if cfg.sentiment_enabled else None

    for i, raw in enumerate(texts):
        s = str(raw) if pd.notna(raw) else ""
        low = s.lower()
        wc = max(1, len(s.split()))
        col = 0

        if analyzer is not None:
            scores = analyzer.polarity_scores(s)
            if cfg.sentiment_use_pos_neg_neu:
                X[i, col : col + 4] = (
                    scores["compound"],
                    scores["pos"],
                    scores["neg"],
                    scores["neu"],
                )
                col += 4
            else:
                X[i, col] = scores["compound"]
                col += 1

        X[i, col] = np.log1p(wc)
        col += 1
        for rx in word_re_success:
            cnt = len(rx.findall(low))
            X[i, col] = cnt
            X[i, col + 1] = cnt / wc
            col += 2
        for rx in word_re_red:
            cnt = len(rx.findall(low))
            X[i, col] = cnt
            X[i, col + 1] = cnt / wc
            col += 2
        for phrase in cfg.red_flag_phrases:
            cnt = _count_phrase(s, phrase)
            X[i, col] = cnt
            X[i, col + 1] = cnt / wc
            col += 2

    return X


def _stack_horizontal_handcrafted_tfidf(
    texts: Sequence[str],
    hc: np.ndarray,
    vectorizer: TfidfVectorizer | None,
) -> sparse.csr_matrix:
    if vectorizer is None:
        return sparse.csr_matrix(hc.astype(np.float64))
    tfidf_X = vectorizer.transform(list(texts))
    hc_sp = sparse.csr_matrix(hc.astype(np.float64))
    return sparse.hstack([hc_sp, tfidf_X], format="csr")


@dataclass
class ShallowTextTierBundle:
    """Fitted shallow text model; ``predict_tier_proba`` matches stacker contract."""

    ridge: Ridge
    tier_residual_std: float
    feat_cfg: ShallowLexiconConfig
    tfidf: TfidfVectorizer | None = None
    handcrafted_dim: int = 0

    def predict_role_z(self, df: pd.DataFrame) -> np.ndarray:
        if "text" not in df.columns:
            raise ValueError("Shallow text bundle requires a 'text' column.")
        mask = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
        out = np.zeros(len(df), dtype=np.float64)
        if mask.any():
            sub = df.loc[mask, "text"].astype(str).tolist()
            hc = handcrafted_feature_matrix(sub, self.feat_cfg)
            X = _stack_horizontal_handcrafted_tfidf(sub, hc, self.tfidf)
            pred = np.asarray(self.ridge.predict(X), dtype=np.float64)
            out[np.flatnonzero(mask.to_numpy(dtype=bool))] = pred
        return out

    def predict_tier_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            raise ValueError("Shallow text bundle requires a 'text' column.")
        mask = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
        out = np.full((len(df), 4), 0.25, dtype=np.float64)
        if mask.any():
            texts = df.loc[mask, "text"].astype(str).tolist()
            hc = handcrafted_feature_matrix(texts, self.feat_cfg)
            X = _stack_horizontal_handcrafted_tfidf(texts, hc, self.tfidf)
            z_hat = np.asarray(self.ridge.predict(X), dtype=np.float64)
            if self.tier_residual_std > 0.0:
                part = predict_tier_proba_from_role_z(z_hat, float(self.tier_residual_std)).to_numpy(
                    dtype=np.float64, copy=False,
                )
            else:
                part = np.full((len(z_hat), 4), 0.25, dtype=np.float64)
            out[np.flatnonzero(mask.to_numpy(dtype=bool))] = part
        out = normalize_proba(out)
        return pd.DataFrame(out, columns=PROBA_COLUMNS, index=df.index)


def fit_shallow_text_tier_bundle_for_multimodal(
    fold_core: pd.DataFrame,
    cfg: dict,
    *,
    shallow_cfg: ShallowLexiconConfig | dict[str, Any] | None = None,
    silent: bool = False,
) -> ShallowTextTierBundle | UniformTextTierBundle:
    """Train Ridge on handcrafted (+ optional TF–IDF) features; Gaussian tier calibration like deep text."""
    mm_section: dict = (cfg.get("model") or {}).get("multimodal") or {}
    global_shallow = (cfg.get("model") or {}).get("text_shallow") or {}

    if isinstance(shallow_cfg, ShallowLexiconConfig):
        sc = shallow_cfg
    elif shallow_cfg is not None:
        sc = merge_shallow_cfg(global_shallow, shallow_cfg)
    else:
        sc = ShallowLexiconConfig.from_dict(global_shallow)

    min_rows = int(mm_section.get("text_min_train_rows", 8))
    regression_target_col = TARGET_Z

    need = {"text", regression_target_col}
    missing = need - set(fold_core.columns)
    if missing:
        raise KeyError(f"fold_core missing columns for shallow text multimodal bundle: {sorted(missing)}")

    tc = fold_core.copy()
    tc = tc[tc["text"].notna() & (tc["text"].astype(str).str.strip() != "")]
    tc = tc.dropna(subset=[regression_target_col])
    tc[regression_target_col] = tc[regression_target_col].astype(float)

    if len(tc) < min_rows:
        return UniformTextTierBundle()

    texts = tc["text"].astype(str).tolist()
    y = tc[regression_target_col].to_numpy(dtype=np.float64)

    hc_train = handcrafted_feature_matrix(texts, sc)
    tfidf: TfidfVectorizer | None = None
    if sc.tfidf_enabled and sc.tfidf_max_features > 0:
        min_df_eff: int | float = sc.tfidf_min_df
        if isinstance(min_df_eff, int) and min_df_eff > 1 and len(texts) < min_df_eff * 3:
            min_df_eff = 1
        tfidf = TfidfVectorizer(
            max_features=sc.tfidf_max_features,
            min_df=min_df_eff,
            ngram_range=(sc.tfidf_ngram_min, sc.tfidf_ngram_max),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        tfidf.fit(texts)
        X_train = _stack_horizontal_handcrafted_tfidf(texts, hc_train, tfidf)
    else:
        X_train = sparse.csr_matrix(hc_train.astype(np.float64))

    ridge = Ridge(alpha=sc.ridge_alpha, random_state=sc.random_state)
    ridge.fit(X_train, y)

    y_pred_train = np.asarray(ridge.predict(X_train), dtype=np.float64)
    tier_residual_std = max(compute_tier_residual_std(y, y_pred_train), 1e-8)

    if not silent and tfidf is not None:
        print(f"[multimodal:text:shallow] fitted Ridge + TF-IDF (max_features={sc.tfidf_max_features}), n_train={len(tc)}")
    elif not silent:
        print(f"[multimodal:text:shallow] fitted Ridge (handcrafted only), n_train={len(tc)}")

    return ShallowTextTierBundle(
        ridge=ridge,
        tier_residual_std=float(tier_residual_std),
        feat_cfg=sc,
        tfidf=tfidf,
        handcrafted_dim=int(hc_train.shape[1]),
    )


def train_and_evaluate_shallow_text_model(
    *,
    cfg: dict | None = None,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_seed: int = RANDOM_STATE,
    tier_proba_csv_path: str | None = None,
) -> tuple[ShallowTextTierBundle, dict[str, float]]:
    """Standalone shallow text run: scouting report text -> role z -> Gaussian tier probabilities."""
    if train_frac <= 0 or val_frac <= 0 or (train_frac + val_frac) >= 1:
        raise ValueError("train_frac and val_frac must be > 0 and sum to < 1.")

    model_cfg = (cfg or {}).get("model") or {}
    composite_cfg = model_cfg.get("composite_score")
    shallow_cfg_raw = model_cfg.get("text_shallow") or {}
    shallow_cfg = ShallowLexiconConfig.from_dict(shallow_cfg_raw)

    df = load_text_data(composite_cfg=composite_cfg)
    if "text" not in df.columns or TARGET_Z not in df.columns:
        raise KeyError(f"load_text_data must provide columns 'text' and {TARGET_Z!r}.")
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")].copy()
    df = df.dropna(subset=[TARGET_Z]).copy()
    df[TARGET_Z] = df[TARGET_Z].astype(float)
    if df.empty:
        raise ValueError("No valid rows for shallow text training after filtering text/target.")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=random_seed,
        shuffle=True,
    )
    val_ratio_of_temp = val_frac / (1.0 - train_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_of_temp,
        random_state=random_seed,
        shuffle=True,
    )

    texts_train = train_df["text"].astype(str).tolist()
    y_train = train_df[TARGET_Z].to_numpy(dtype=np.float64)
    hc_train = handcrafted_feature_matrix(texts_train, shallow_cfg)

    tfidf: TfidfVectorizer | None = None
    if shallow_cfg.tfidf_enabled and shallow_cfg.tfidf_max_features > 0:
        min_df_eff: int | float = shallow_cfg.tfidf_min_df
        if isinstance(min_df_eff, int) and min_df_eff > 1 and len(texts_train) < min_df_eff * 3:
            min_df_eff = 1
        tfidf = TfidfVectorizer(
            max_features=shallow_cfg.tfidf_max_features,
            min_df=min_df_eff,
            ngram_range=(shallow_cfg.tfidf_ngram_min, shallow_cfg.tfidf_ngram_max),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        tfidf.fit(texts_train)
    X_train = _stack_horizontal_handcrafted_tfidf(texts_train, hc_train, tfidf)

    ridge = Ridge(alpha=shallow_cfg.ridge_alpha, random_state=shallow_cfg.random_state)
    ridge.fit(X_train, y_train)
    y_train_pred = np.asarray(ridge.predict(X_train), dtype=np.float64)
    tier_residual_std = max(compute_tier_residual_std(y_train, y_train_pred), 1e-8)
    bundle = ShallowTextTierBundle(
        ridge=ridge,
        tier_residual_std=float(tier_residual_std),
        feat_cfg=shallow_cfg,
        tfidf=tfidf,
        handcrafted_dim=int(hc_train.shape[1]),
    )

    test_pred = bundle.predict_role_z(test_df)
    metrics = {
        "r2": float(r2_score(test_df[TARGET_Z], test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(test_df[TARGET_Z], test_pred))),
        "mae": float(mean_absolute_error(test_df[TARGET_Z], test_pred)),
        "tier_residual_std": float(tier_residual_std),
    }

    mlflow_ctx = build_mlflow_context(
        cfg=cfg,
        model_type="text_shallow",
        target_name=TARGET_Z,
        fallback_experiment_name="nba-draft-prospect-text-shallow",
        tracking_uri=tracking_uri,
        run_name=run_name,
    )
    with managed_run(mlflow_ctx):
        if cfg is not None:
            log_config_dict(cfg)
        log_common_params(
            {
                "model_family": "text_shallow",
                "target": TARGET_Z,
                "n_total": len(df),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "ridge_alpha": shallow_cfg.ridge_alpha,
                "sentiment_enabled": shallow_cfg.sentiment_enabled,
                "sentiment_use_pos_neg_neu": shallow_cfg.sentiment_use_pos_neg_neu,
                "tfidf_enabled": shallow_cfg.tfidf_enabled,
                "tfidf_max_features": shallow_cfg.tfidf_max_features,
                "tfidf_min_df": shallow_cfg.tfidf_min_df,
                "tfidf_ngram_min": shallow_cfg.tfidf_ngram_min,
                "tfidf_ngram_max": shallow_cfg.tfidf_ngram_max,
                "handcrafted_dim": int(hc_train.shape[1]),
            }
        )
        mlflow.log_metrics(metrics)
        if tier_proba_csv_path:
            proba_full = bundle.predict_tier_proba(df[["text"]])
            id_cols = [c for c in ("Name", "draft_year") if c in df.columns]
            out_csv = pd.concat(
                [df[id_cols].reset_index(drop=True), proba_full.reset_index(drop=True)],
                axis=1,
            )
            out_p = Path(tier_proba_csv_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_csv.to_csv(out_p, index=False)
            if mlflow.active_run() is not None:
                mlflow.log_artifact(str(out_p), artifact_path="tier_proba")

    print("\n" + "=" * 40)
    print("Shallow Text Model Test Metrics")
    print("=" * 40)
    print(f"R2   = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.4f}")
    print(f"MAE  = {metrics['mae']:.4f}")
    print(f"Tier residual std = {metrics['tier_residual_std']:.4f}")
    return bundle, metrics
