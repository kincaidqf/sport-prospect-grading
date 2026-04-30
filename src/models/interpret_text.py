"""Interpretability for TextProspectPredictor: probes, occlusion, log-odds, sentiment, REPORT.md.

Run from repo root (after training with ``save_path``):

.. code-block:: bash

    uv run python -m src.models.interpret_text --checkpoint outputs/checkpoints/text_model.pt

Or train then interpret:

.. code-block:: bash

    uv run python -m src.models.interpret_text --retrain --checkpoint-out outputs/checkpoints/text_model.pt
"""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.models.text_model import (
    PROJECT_ROOT,
    ScoutingReportEncoder,
    TextProspectPredictor,
    load_text_data,
    train_and_evaluate_text_model,
)

HEAD_KEYS = ("vorp", "star", "survived", "starter")
HEAD_LABELS = {
    "vorp": "VORP (regression)",
    "star": "is_star",
    "survived": "survived_3yrs",
    "starter": "became_starter",
}

DEFAULT_OUT = os.path.join(PROJECT_ROOT, "outputs", "interpretability")

# Small NBA team / league tokens to optionally strip from log-odds (reduce artifacts)
NBA_TEAM_TOKENS = frozenset(
    {
        "lakers", "celtics", "warriors", "heat", "bucks", "nuggets", "suns", "mavericks",
        "nets", "sixers", "76ers", "knicks", "bulls", "cavs", "cavaliers", "hawks", "hornets",
        "pistons", "pacers", "magic", "raptors", "wizards", "rockets", "grizzlies", "timberwolves",
        "pelicans", "thunder", "trail", "blazers", "kings", "spurs", "jazz", "clippers", "nba",
    }
)


def _ensure_nltk_vader() -> None:
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def load_checkpoint(path: str, device: torch.device) -> tuple[TextProspectPredictor, dict]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    encoder = ScoutingReportEncoder(
        pretrained=ckpt["pretrained"],
        output_dim=int(ckpt["output_dim"]),
        freeze_base=bool(ckpt["freeze_base"]),
    )
    model = TextProspectPredictor(
        text_encoder=encoder,
        hidden_dim=int(ckpt.get("hidden_dim", 64)),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    meta = {
        "target_mean": float(ckpt["target_mean"]),
        "target_std": float(ckpt["target_std"]),
        "star_threshold": float(ckpt["star_threshold"]),
        "max_length": int(ckpt.get("max_length", 256)),
    }
    model.eval()
    return model, meta


def score_texts(
    model: TextProspectPredictor,
    texts: list[str],
    device: torch.device,
    target_mean: float,
    target_std: float,
    max_length: int = 256,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Return per-head scores: VORP in original units; classifiers as sigmoid probabilities."""
    if not texts:
        empty = np.array([], dtype=np.float64)
        return {h: empty.copy() for h in HEAD_KEYS}

    tokenizer = model.text_encoder.tokenizer
    n = len(texts)
    all_reg: list[np.ndarray] = []
    all_star: list[np.ndarray] = []
    all_surv: list[np.ndarray] = []
    all_start: list[np.ndarray] = []

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = texts[start : start + batch_size]
            batch = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            reg, star_logit, survived_logit, starter_logit = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            reg_np = reg.detach().cpu().numpy().astype(np.float64)
            reg_np = reg_np * target_std + target_mean
            all_reg.append(reg_np)
            all_star.append(torch.sigmoid(star_logit).detach().cpu().numpy().astype(np.float64))
            all_surv.append(torch.sigmoid(survived_logit).detach().cpu().numpy().astype(np.float64))
            all_start.append(torch.sigmoid(starter_logit).detach().cpu().numpy().astype(np.float64))

    if was_training:
        model.train()

    return {
        "vorp": np.concatenate(all_reg),
        "star": np.concatenate(all_star),
        "survived": np.concatenate(all_surv),
        "starter": np.concatenate(all_start),
    }


def probe_phrase_bank() -> list[tuple[str, str]]:
    """(category, phrase) curated probes."""
    return [
        ("Shooting", "elite shooter"),
        ("Shooting", "knockdown shooter from deep"),
        ("Shooting", "inconsistent jumper"),
        ("Shooting", "poor free throw shooter"),
        ("Shooting", "limited shooting range"),
        ("Athleticism", "explosive athlete"),
        ("Athleticism", "high level athlete"),
        ("Athleticism", "average athlete"),
        ("Athleticism", "below average athleticism"),
        ("Athleticism", "limited burst"),
        ("Size / Frame", "great length"),
        ("Size / Frame", "NBA-ready frame"),
        ("Size / Frame", "undersized"),
        ("Size / Frame", "slight build"),
        ("Size / Frame", "needs to add strength"),
        ("Defense", "lockdown defender"),
        ("Defense", "switchable defender"),
        ("Defense", "rim protector"),
        ("Defense", "weak defender"),
        ("Defense", "poor defensive instincts"),
        ("Motor / Character", "high motor"),
        ("Motor / Character", "relentless competitor"),
        ("Motor / Character", "blue collar"),
        ("Motor / Character", "low motor"),
        ("Motor / Character", "questionable work ethic"),
        ("Motor / Character", "high basketball IQ"),
        ("Motor / Character", "low basketball IQ"),
        ("Skills", "elite passer"),
        ("Skills", "advanced ball handler"),
        ("Skills", "poor handle"),
        ("Skills", "elite finisher at the rim"),
        ("Skills", "turnover prone"),
        ("Projection", "lottery pick"),
        ("Projection", "All-Star potential"),
        ("Projection", "future starter"),
        ("Projection", "rotation player"),
        ("Projection", "G-League prospect"),
        ("Projection", "two-way player"),
        ("Projection", "fringe NBA player"),
    ]


def expand_probe_texts(phrase: str) -> list[str]:
    bare = phrase.strip()
    if not bare.endswith("."):
        bare = bare + "."
    return [
        bare,
        f"Strengths: {bare}",
        f"Outlook: {bare}",
    ]


NEUTRAL_BASELINE = "Plays basketball at the college level."


def run_probes(
    model: TextProspectPredictor,
    device: torch.device,
    meta: dict,
    out_dir: str,
) -> dict[str, pd.DataFrame]:
    """Write probes_{head}.csv and probes_{head}.png per head."""
    bank = probe_phrase_bank()
    baseline_scores = score_texts(
        model,
        [NEUTRAL_BASELINE],
        device,
        meta["target_mean"],
        meta["target_std"],
        max_length=meta["max_length"],
    )
    rows: list[dict] = []
    for category, phrase in bank:
        variants = expand_probe_texts(phrase)
        per_variant = score_texts(
            model,
            variants,
            device,
            meta["target_mean"],
            meta["target_std"],
            max_length=meta["max_length"],
        )
        for h in HEAD_KEYS:
            mean_score = float(np.mean(per_variant[h]))
            base = float(baseline_scores[h][0])
            rows.append(
                {
                    "category": category,
                    "phrase": phrase,
                    "head": h,
                    "raw_score": mean_score,
                    "delta_vs_baseline": mean_score - base,
                }
            )

    df_all = pd.DataFrame(rows)
    by_head: dict[str, pd.DataFrame] = {}
    for h in HEAD_KEYS:
        sub = df_all[df_all["head"] == h].copy()
        sub = sub.sort_values("delta_vs_baseline", ascending=False)
        sub[["category", "phrase", "raw_score", "delta_vs_baseline"]].to_csv(
            os.path.join(out_dir, f"probes_{h}.csv"), index=False
        )
        by_head[h] = sub
        _plot_probe_bars(sub, h, out_dir)
    return by_head


def _plot_probe_bars(df: pd.DataFrame, head: str, out_dir: str) -> None:
    top = df.nlargest(15, "delta_vs_baseline")
    bottom = df.nsmallest(15, "delta_vs_baseline")
    plot_df = pd.concat([top, bottom])
    plot_df = plot_df.sort_values("delta_vs_baseline", ascending=True)
    categories = plot_df["category"].tolist()
    uniq = sorted(set(categories))
    cat_to_color = {c: f"C{i % 10}" for i, c in enumerate(uniq)}
    colors = [cat_to_color[c] for c in categories]
    fig, ax = plt.subplots(figsize=(8, 10))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["delta_vs_baseline"].values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["phrase"].tolist(), fontsize=8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Δ vs neutral baseline")
    ax.set_title(f"Synthetic probes — {HEAD_LABELS[head]}")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=cat_to_color[c], label=c)
        for c in uniq
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"probes_{head}.png"), dpi=150)
    plt.close()


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _mask_words(words: list[str], start: int, end_exclusive: int, mask: str) -> str:
    """Replace words[start:end_exclusive] with single [MASK] token joined by spaces."""
    out = words[:start] + [mask] + words[end_exclusive:]
    return " ".join(out)


def run_occlusion(
    model: TextProspectPredictor,
    device: torch.device,
    meta: dict,
    texts: list[str],
    out_dir: str,
    min_reports: int = 10,
    batch_size: int = 16,
) -> dict[str, pd.DataFrame]:
    """Aggregate occlusion deltas per n-gram per head."""
    tokenizer = model.text_encoder.tokenizer
    mask_tok = tokenizer.mask_token or "[MASK]"
    max_length = meta["max_length"]

    # Collect deltas per (head, ngram_key) -> list of deltas
    deltas_store: dict[tuple[str, str], list[float]] = defaultdict(list)

    for text in texts:
        words = _word_tokens(text)
        if len(words) < 2:
            continue
        variants: list[str] = []
        keys: list[str] = []

        seen_uni: set[str] = set()
        seen_bi: set[str] = set()

        for i in range(len(words)):
            key = words[i]
            if key not in seen_uni:
                seen_uni.add(key)
                variants.append(_mask_words(words, i, i + 1, mask_tok))
                keys.append(key)

        for i in range(len(words) - 1):
            key = f"{words[i]} {words[i + 1]}"
            if key not in seen_bi:
                seen_bi.add(key)
                variants.append(_mask_words(words, i, i + 2, mask_tok))
                keys.append(key)

        if not variants:
            continue

        orig_scores = score_texts(
            model,
            [text],
            device,
            meta["target_mean"],
            meta["target_std"],
            max_length=max_length,
            batch_size=batch_size,
        )

        # Batch masked variants
        for start in range(0, len(variants), batch_size):
            chunk = variants[start : start + batch_size]
            chunk_keys = keys[start : start + batch_size]
            masked_scores = score_texts(
                model,
                chunk,
                device,
                meta["target_mean"],
                meta["target_std"],
                max_length=max_length,
                batch_size=batch_size,
            )
            for j, k in enumerate(chunk_keys):
                for h in HEAD_KEYS:
                    d = float(orig_scores[h][0] - masked_scores[h][j])
                    deltas_store[(h, k)].append(d)

    rows = []
    for (h, ngram), ds in deltas_store.items():
        if len(ds) < min_reports:
            continue
        arr = np.asarray(ds, dtype=np.float64)
        mean_d = float(arr.mean())
        std_d = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        se = std_d / np.sqrt(len(arr)) if len(arr) > 1 else 1.0
        t_stat = mean_d / se if se > 0 else 0.0
        rows.append(
            {
                "ngram": ngram,
                "n_reports": len(ds),
                "mean_delta": mean_d,
                "t_stat": t_stat,
                "head": h,
            }
        )

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        df_all = pd.DataFrame(columns=["ngram", "n_reports", "mean_delta", "t_stat", "head"])
    by_head: dict[str, pd.DataFrame] = {}
    for h in HEAD_KEYS:
        sub = df_all[df_all["head"] == h].copy() if not df_all.empty else pd.DataFrame()
        if sub.empty:
            sub = pd.DataFrame(columns=["ngram", "n_reports", "mean_delta", "t_stat", "head"])
        else:
            sub = sub.sort_values("mean_delta", ascending=False)
        cols = [c for c in ["ngram", "n_reports", "mean_delta", "t_stat"] if c in sub.columns]
        sub[cols].to_csv(os.path.join(out_dir, f"occlusion_{h}.csv"), index=False)
        by_head[h] = sub
        if not sub.empty:
            _plot_occlusion_bars(sub, h, out_dir)
    return by_head


def _plot_occlusion_bars(df: pd.DataFrame, head: str, out_dir: str) -> None:
    top = df.nlargest(20, "mean_delta")
    bottom = df.nsmallest(20, "mean_delta")
    plot_df = pd.concat([bottom, top])
    plot_df = plot_df.sort_values("mean_delta", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["mean_delta"].values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["ngram"].tolist(), fontsize=8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean Δ prediction (original − masked)")
    ax.set_title(f"Occlusion — {HEAD_LABELS[head]}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"occlusion_{head}.png"), dpi=150)
    plt.close()


def _load_stopwords() -> set[str]:
    from nltk.corpus import stopwords

    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk

        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def _strip_names_teams(text: str, name_tokens: set[str]) -> str:
    words = _word_tokens(text)
    kept = [w for w in words if w not in name_tokens and w not in NBA_TEAM_TOKENS]
    return " ".join(kept)


def _document_ngrams(text: str, max_n: int = 2) -> list[str]:
    words = _word_tokens(text)
    out: list[str] = []
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            out.append(" ".join(words[i : i + n]))
    return out


def monroe_log_odds(
    counts_a: Counter[str],
    counts_b: Counter[str],
    total_a: int,
    total_b: int,
    prior: float = 0.01,
) -> dict[str, float]:
    """Bayesian log-odds ratio between two corpora (informative Dirichlet prior)."""
    vocab = set(counts_a.keys()) | set(counts_b.keys())
    scores: dict[str, float] = {}
    for w in vocab:
        ca = counts_a.get(w, 0)
        cb = counts_b.get(w, 0)
        # log((ca + prior)/(total_a - ca + prior)) - log((cb + prior)/(total_b - cb + prior))
        num_a = ca + prior
        den_a = total_a - ca + prior
        num_b = cb + prior
        den_b = total_b - cb + prior
        if den_a <= 0 or den_b <= 0:
            continue
        scores[w] = float(np.log2(num_a / den_a) - np.log2(num_b / den_b))
    return scores


def run_log_odds(
    texts: list[str],
    preds_by_head: dict[str, np.ndarray],
    name_tokens: set[str],
    stop: set[str],
    out_dir: str,
) -> dict[str, pd.DataFrame]:
    processed = []
    for t in texts:
        stripped = _strip_names_teams(t, name_tokens)
        processed.append(stripped)

    by_head: dict[str, pd.DataFrame] = {}
    for h in HEAD_KEYS:
        preds = preds_by_head[h]
        q_hi = np.quantile(preds, 0.75)
        q_lo = np.quantile(preds, 0.25)
        top_mask = preds >= q_hi
        bot_mask = preds <= q_lo

        counts_top = Counter()
        counts_bot = Counter()
        total_top = 0
        total_bot = 0

        for doc, is_top, is_bot in zip(processed, top_mask, bot_mask):
            grams = [
                g
                for g in _document_ngrams(doc, max_n=2)
                if all(tok not in stop and tok not in name_tokens for tok in g.split())
            ]
            if is_top:
                for g in grams:
                    counts_top[g] += 1
                total_top += len(grams)
            if is_bot:
                for g in grams:
                    counts_bot[g] += 1
                total_bot += len(grams)

        scores = monroe_log_odds(counts_top, counts_bot, total_top, total_bot, prior=0.01)
        rows = [
            {"ngram": w, "log_odds": s, "count_top": counts_top.get(w, 0), "count_bot": counts_bot.get(w, 0)}
            for w, s in scores.items()
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[(df["count_top"] + df["count_bot"]) >= 5]
            df = df.sort_values("log_odds", ascending=False)
        df.to_csv(os.path.join(out_dir, f"logodds_{h}.csv"), index=False)
        by_head[h] = df
        if not df.empty:
            _plot_logodds_bars(df, h, out_dir)
    return by_head


def _plot_logodds_bars(df: pd.DataFrame, head: str, out_dir: str) -> None:
    top = df.nlargest(15, "log_odds")
    bottom = df.nsmallest(15, "log_odds")
    plot_df = pd.concat([bottom, top])
    plot_df = plot_df.sort_values("log_odds", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["log_odds"].values, color="darkgreen")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["ngram"].tolist(), fontsize=8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Log-odds (top quartile vs bottom quartile predictions)")
    ax.set_title(f"Corpus log-odds — {HEAD_LABELS[head]}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"logodds_{head}.png"), dpi=150)
    plt.close()


SECTION_MARKERS = (
    ("strengths", re.compile(r"strengths\s*:", re.IGNORECASE)),
    ("weaknesses", re.compile(r"weaknesses\s*:", re.IGNORECASE)),
    ("outlook", re.compile(r"outlook\s*:", re.IGNORECASE)),
)


def split_report_sections(text: str) -> dict[str, str]:
    """Rough split on Strengths / Weaknesses / Outlook markers."""
    lower = text
    indices = []
    for name, pat in SECTION_MARKERS:
        m = pat.search(lower)
        if m:
            indices.append((m.start(), name, m.end()))
    indices.sort(key=lambda x: x[0])
    sections: dict[str, str] = {"full": text}
    for i, (start, name, end) in enumerate(indices):
        next_start = indices[i + 1][0] if i + 1 < len(indices) else len(lower)
        sections[name] = lower[end:next_start].strip()
    return sections


def run_sentiment_correlation(
    texts: list[str],
    preds_by_head: dict[str, np.ndarray],
    out_dir: str,
) -> pd.DataFrame:
    _ensure_nltk_vader()
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    rows = []
    section_names = ["strengths", "weaknesses", "outlook"]

    for sec in section_names:
        compounds = []
        for t in texts:
            sect = split_report_sections(t).get(sec, "")
            if not sect:
                compounds.append(np.nan)
            else:
                compounds.append(sia.polarity_scores(sect)["compound"])
        compounds_arr = np.asarray(compounds, dtype=np.float64)

        for h in HEAD_KEYS:
            y = preds_by_head[h].astype(np.float64)
            mask = ~np.isnan(compounds_arr)
            if mask.sum() < 5:
                rows.append(
                    {
                        "section": sec,
                        "head": h,
                        "pearson_r": np.nan,
                        "spearman_r": np.nan,
                        "n": int(mask.sum()),
                    }
                )
                continue
            x = compounds_arr[mask]
            yy = y[mask]

            def pearson(a: np.ndarray, b: np.ndarray) -> float:
                if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                    return float("nan")
                return float(np.corrcoef(a, b)[0, 1])

            def spearman(a: np.ndarray, b: np.ndarray) -> float:
                ra = pd.Series(a).rank().to_numpy()
                rb = pd.Series(b).rank().to_numpy()
                return pearson(ra, rb)

            rows.append(
                {
                    "section": sec,
                    "head": h,
                    "pearson_r": pearson(x, yy),
                    "spearman_r": spearman(x, yy),
                    "n": int(mask.sum()),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "sentiment_correlation.csv"), index=False)
    return df


def _normalize_term(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def agreement_terms(
    probes_df: pd.DataFrame,
    occlusion_df: pd.DataFrame,
    logodds_df: pd.DataFrame,
    head: str,
    top_k: int = 15,
) -> tuple[list[str], list[str]]:
    """Terms appearing in >=2 of top/bottom lists."""
    pos_sets: list[set[str]] = []
    neg_sets: list[set[str]] = []

    if not probes_df.empty:
        p = probes_df[probes_df["head"] == head] if "head" in probes_df.columns else probes_df
        pos_sets.append(set(_normalize_term(x) for x in p.nlargest(top_k, "delta_vs_baseline")["phrase"]))
        neg_sets.append(set(_normalize_term(x) for x in p.nsmallest(top_k, "delta_vs_baseline")["phrase"]))

    if not occlusion_df.empty:
        o = occlusion_df[occlusion_df["head"] == head] if "head" in occlusion_df.columns else occlusion_df
        pos_sets.append(set(_normalize_term(x) for x in o.nlargest(top_k, "mean_delta")["ngram"]))
        neg_sets.append(set(_normalize_term(x) for x in o.nsmallest(top_k, "mean_delta")["ngram"]))

    if not logodds_df.empty:
        l = logodds_df  # no head column
        pos_sets.append(set(_normalize_term(x) for x in l.nlargest(top_k, "log_odds")["ngram"]))
        neg_sets.append(set(_normalize_term(x) for x in l.nsmallest(top_k, "log_odds")["ngram"]))

    def pair_agreement(sets: list[set[str]]) -> list[str]:
        if len(sets) < 2:
            return []
        freq = Counter()
        for s in sets:
            for t in s:
                freq[t] += 1
        return sorted([t for t, c in freq.items() if c >= 2])

    return pair_agreement(pos_sets), pair_agreement(neg_sets)


def write_report(
    out_dir: str,
    probe_tables: dict[str, pd.DataFrame],
    occlusion_tables: dict[str, pd.DataFrame],
    logodds_tables: dict[str, pd.DataFrame],
    sentiment_df: pd.DataFrame,
) -> None:
    lines = [
        "# Text model interpretability",
        "",
        "Methods: **synthetic phrase probes** (Δ vs neutral baseline), **aggregated occlusion** (original − masked prediction), **corpus log-odds** (top vs bottom predicted quartiles).",
        "",
        "## Caveats",
        "",
        "- Probes are out-of-distribution; prefer **relative** ordering and Δ vs baseline.",
        "- Occlusion and log-odds use **word-level** n-grams; subword effects are not shown.",
        "- Player names and common NBA team tokens are stripped for log-odds; occlusion uses raw reports.",
        "",
    ]

    for h in HEAD_KEYS:
        lines.append(f"## {HEAD_LABELS[h]}")
        lines.append("")

        p = probe_tables.get(h, pd.DataFrame())
        if not p.empty:
            lines.append("### Probes (top / bottom Δ vs baseline)")
            top_p = p.nlargest(15, "delta_vs_baseline")[["phrase", "category", "delta_vs_baseline"]]
            bot_p = p.nsmallest(15, "delta_vs_baseline")[["phrase", "category", "delta_vs_baseline"]]
            lines.append("| phrase | category | Δ |")
            lines.append("| --- | --- | --- |")
            for _, row in top_p.iterrows():
                lines.append(f"| {row['phrase']} | {row['category']} | {row['delta_vs_baseline']:.4f} |")
            lines.append("")
            lines.append("| phrase | category | Δ |")
            lines.append("| --- | --- | --- |")
            for _, row in bot_p.iterrows():
                lines.append(f"| {row['phrase']} | {row['category']} | {row['delta_vs_baseline']:.4f} |")
            lines.append("")

        o = occlusion_tables.get(h, pd.DataFrame())
        if not o.empty:
            lines.append("### Occlusion (mean Δ over reports)")
            top_o = o.nlargest(15, "mean_delta")
            bot_o = o.nsmallest(15, "mean_delta")
            lines.append("| ngram | n_reports | mean Δ |")
            lines.append("| --- | --- | --- |")
            for _, row in top_o.iterrows():
                lines.append(f"| {row['ngram']} | {row['n_reports']} | {row['mean_delta']:.4f} |")
            lines.append("")
            lines.append("| ngram | n_reports | mean Δ |")
            lines.append("| --- | --- | --- |")
            for _, row in bot_o.iterrows():
                lines.append(f"| {row['ngram']} | {row['n_reports']} | {row['mean_delta']:.4f} |")
            lines.append("")

        lg = logodds_tables.get(h, pd.DataFrame())
        if not lg.empty:
            lines.append("### Log-odds (predicted top vs bottom quartile)")
            top_l = lg.nlargest(15, "log_odds")
            bot_l = lg.nsmallest(15, "log_odds")
            lines.append("| ngram | log_odds |")
            lines.append("| --- | --- |")
            for _, row in top_l.iterrows():
                lines.append(f"| {row['ngram']} | {row['log_odds']:.4f} |")
            lines.append("")
            lines.append("| ngram | log_odds |")
            lines.append("| --- | --- |")
            for _, row in bot_l.iterrows():
                lines.append(f"| {row['ngram']} | {row['log_odds']:.4f} |")
            lines.append("")

        pos_agree, neg_agree = agreement_terms(p, o, lg, h)
        lines.append("### Cross-method agreement (term in ≥2 of probe / occlusion / log-odds top-15 lists)")
        lines.append("")
        lines.append("**Positive-associated:** " + (", ".join(pos_agree) if pos_agree else "_none_"))
        lines.append("")
        lines.append("**Negative-associated:** " + (", ".join(neg_agree) if neg_agree else "_none_"))
        lines.append("")

        sub = sentiment_df[sentiment_df["head"] == h]
        if not sub.empty:
            lines.append("### VADER sentiment vs predictions (Pearson / Spearman)")
            lines.append("| section | Pearson r | Spearman r | n |")
            lines.append("| --- | --- | --- | --- |")
            for _, row in sub.iterrows():
                pr = row["pearson_r"]
                sr = row["spearman_r"]
                pr_s = f"{float(pr):.4f}" if pd.notna(pr) else "nan"
                sr_s = f"{float(sr):.4f}" if pd.notna(sr) else "nan"
                lines.append(f"| {row['section']} | {pr_s} | {sr_s} | {row['n']} |")
            lines.append("")

    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))


def build_name_tokens(df: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    if "player_name" in df.columns:
        names = df["player_name"].dropna().astype(str)
    elif "name" in df.columns:
        names = df["name"].dropna().astype(str)
    else:
        return tokens
    for name in names:
        name = re.sub(r"^\d+\s*-\s*", "", name.strip())
        for part in re.split(r"[\s.'-]+", name.lower()):
            if len(part) >= 2:
                tokens.add(part)
    return tokens


def pick_occlusion_sample(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    if len(df) <= n:
        return df["text"].tolist()
    return df.sample(n=n, random_state=seed)["text"].tolist()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interpret TextProspectPredictor (probes, occlusion, log-odds).")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt from train_and_evaluate_text_model(save_path=...)",
    )
    p.add_argument(
        "--retrain",
        action="store_true",
        help="Train a fresh model and save to --checkpoint-out, then interpret.",
    )
    p.add_argument(
        "--checkpoint-out",
        default=os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "text_model.pt"),
        help="When --retrain, save checkpoint here.",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--n-occlusion", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.retrain:
        ckpt_path = os.path.abspath(args.checkpoint_out)
        _, _ = train_and_evaluate_text_model(save_path=ckpt_path)
        args.checkpoint = ckpt_path

    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise SystemExit("Provide --checkpoint to a saved .pt file, or use --retrain.")

    model, meta = load_checkpoint(args.checkpoint, device)
    df = load_text_data()
    texts_full = df["text"].tolist()

    name_tokens = build_name_tokens(df)
    stop = _load_stopwords()

    # Score full corpus for log-odds + sentiment
    preds_by_head = score_texts(
        model,
        texts_full,
        device,
        meta["target_mean"],
        meta["target_std"],
        max_length=meta["max_length"],
    )

    probe_tables = run_probes(model, device, meta, out_dir)

    occ_sample = pick_occlusion_sample(df, args.n_occlusion, args.seed)
    occlusion_frames = run_occlusion(model, device, meta, occ_sample, out_dir)

    logodds_tables = run_log_odds(texts_full, preds_by_head, name_tokens, stop, out_dir)

    sentiment_df = run_sentiment_correlation(texts_full, preds_by_head, out_dir)

    # Occlusion/DataFrames don't have redundant head column in probe_tables
    write_report(out_dir, probe_tables, occlusion_frames, logodds_tables, sentiment_df)
    print(f"Interpretability artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
