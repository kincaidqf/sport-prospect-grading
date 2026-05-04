"""NLP encoder for scouting report texts."""
from __future__ import annotations

import os
import re
import glob
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer

from src.data.loader import CACHE_DIR, RANDOM_STATE, TARGET_COL, load_data
from src.models.probability import (
    PROBA_COLUMNS,
    TIER_THRESHOLDS,
    normalize_proba,
    proba_to_dataframe,
    zscore_to_tier_proba,
)
from src.utils.mlflow_utils import (
    build_mlflow_context,
    log_common_params,
    log_config_dict,
    log_epoch_metrics,
    managed_run,
)

# Default regression target; same column used by stats regression for Gaussian tier probabilities.
TARGET = TARGET_COL["nba_role_zscore"]


class ScoutingReportEncoder(nn.Module):
    """Fine-tuned transformer encoder that produces a prospect embedding from text."""

    def __init__(
        self,
        pretrained: str = "distilbert-base-uncased",
        output_dim: int = 128,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.backbone = AutoModel.from_pretrained(pretrained)
        if freeze_base:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.output_dim = output_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized scouting text into dense embeddings.

        Expects `input_ids` and `attention_mask` with shape [batch_size, seq_len].
        Returns tensor of shape [batch_size, output_dim].
        """
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if hidden_states.last_hidden_state is None:
            raise ValueError("Transformer backbone did not return last_hidden_state.")

        cls_embedding = hidden_states.last_hidden_state[:, 0, :]
        return self.head(cls_embedding)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        max_length: int = 512,
        batch_size: int = 32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode raw scouting report strings into deep embeddings.

        Returns a tensor with shape [num_texts, output_dim].
        """
        if not texts:
            return torch.empty((0, self.output_dim), dtype=torch.float32)

        device = device or next(self.parameters()).device
        encoded_batches: list[torch.Tensor] = []
        was_training = self.training
        self.eval()

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = self.tokenizer(
                list(batch_texts),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            embeddings = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            encoded_batches.append(embeddings.detach().cpu())

        if was_training:
            self.train()

        return torch.cat(encoded_batches, dim=0)


class TextProspectPredictor(nn.Module):
    """Text-only predictor with regression + star-classification heads."""

    def __init__(
        self,
        text_encoder: ScoutingReportEncoder,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.shared = nn.Sequential(
            nn.Linear(text_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.star_head = nn.Linear(hidden_dim, 1)
        self.survived_head = nn.Linear(hidden_dim, 1)
        self.starter_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.shared(embeddings)
        reg = self.reg_head(hidden).squeeze(-1)
        star_logit = self.star_head(hidden).squeeze(-1)
        survived_logit = self.survived_head(hidden).squeeze(-1)
        starter_logit = self.starter_head(hidden).squeeze(-1)
        return reg, star_logit, survived_logit, starter_logit

    @torch.no_grad()
    def predict_from_texts(
        self,
        texts: Sequence[str],
        max_length: int = 512,
        batch_size: int = 32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Run inference directly from raw report text strings."""
        embeddings = self.text_encoder.encode_texts(
            texts=texts,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        if embeddings.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)

        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        hidden = self.shared(embeddings.to(device))
        preds = self.reg_head(hidden).squeeze(-1).detach().cpu()
        if was_training:
            self.train()
        return preds

    @torch.no_grad()
    def predict_tier_proba_from_texts(
        self,
        texts: Sequence[str],
        *,
        target_mean: float,
        target_std: float,
        tier_residual_std: float,
        max_length: int = 512,
        batch_size: int = 32,
        device: torch.device | None = None,
    ) -> pd.DataFrame:
        """Return ``PROBA_COLUMNS`` tier probabilities from text (multimodal / PSM).

        Denormalizes the regression head with ``target_mean`` / ``target_std`` then
        applies the same Gaussian CDF mapping as stats regression bundles. Use
        ``tier_residual_std`` from the training checkpoint (or recompute on a
        calibration split).
        """
        raw = self.predict_from_texts(
            texts,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        if raw.numel() == 0:
            return proba_to_dataframe(np.zeros((0, 4), dtype=np.float64))
        z = raw.numpy().astype(float) * float(target_std) + float(target_mean)
        return predict_tier_proba_from_role_z(z, tier_residual_std)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCOUTING_PATH = os.path.join(PROJECT_ROOT, "data", "scouting", "players.csv")


def _clean_player_name(name: str) -> str:
    """Normalize scouting CSV names like '2 - KJ Simpson' to plain player names."""
    name = str(name).strip()
    name = re.sub(r"^\d+\s*-\s*", "", name)
    return re.sub(r"\s+", " ", name).strip()


def load_text_data(composite_cfg: dict | None = None) -> pd.DataFrame:
    """Merge scouting report text onto the canonical NCAA+NBA frame from ``load_data``.

    Carries ``prospect_tier`` and ``nba_role_zscore`` (and related columns) so labels
    match the stats-based classification/regression pipelines.
    """
    canonical = load_data(composite_cfg=composite_cfg)
    scouting = pd.read_csv(SCOUTING_PATH)
    scouting["draft_year"] = pd.to_numeric(scouting["draft_year"], errors="coerce")
    scouting["Name"] = scouting["name"].map(_clean_player_name)
    scouting = scouting.dropna(subset=["Name", "draft_year", "full_scouting_report"])
    scouting["text"] = scouting["full_scouting_report"].astype(str).str.strip()
    scouting = scouting[scouting["text"] != ""].copy()
    scouting = scouting.drop_duplicates(subset=["Name", "draft_year"], keep="first")

    merged = canonical.merge(
        scouting[["Name", "draft_year", "text"]],
        on=["Name", "draft_year"],
        how="inner",
    )
    merged["draft_year"] = merged["draft_year"].astype(int)
    merged["MIN"] = pd.to_numeric(merged["MIN"], errors="coerce").fillna(0.0)

    # Derive survival labels from cached season participation.
    season_files = glob.glob(os.path.join(CACHE_DIR, "*.csv"))
    if season_files:
        all_seasons = pd.concat(
            [pd.read_csv(f, usecols=["PLAYER_ID", "SEASON_ID"]) for f in season_files],
            ignore_index=True,
        )
        seasons_played = (
            all_seasons.drop_duplicates()
            .groupby("PLAYER_ID")
            .size()
            .rename("seasons_played")
        )
        merged["survived_3yrs"] = (
            pd.to_numeric(merged["player_id"], errors="coerce")
            .map(seasons_played)
            .fillna(0)
            .ge(3)
            .astype(int)
        )
    else:
        # Fallback if cache is missing: at least one known NBA season in merged table.
        merged["survived_3yrs"] = (merged["MIN"] > 0).astype(int)
    return merged


def attach_scouting_text_columns(df: pd.DataFrame, composite_cfg: dict | None = None) -> pd.DataFrame:
    """Left-merge scouting ``text`` and text-aux labels onto a stats frame (``Name`` + ``draft_year``).

    ``survived_3yrs`` is defined in ``load_text_data`` (season-cache logic) and is absent from
    ``load_data()`` alone; multimodal text training needs it alongside ``became_starter``.
    """
    d = df.copy()
    for col in ("text", "survived_3yrs"):
        if col in d.columns:
            d = d.drop(columns=[col])
    side = load_text_data(composite_cfg=composite_cfg)[["Name", "draft_year", "text", "survived_3yrs"]]
    return d.merge(side, on=["Name", "draft_year"], how="left")


def compute_tier_residual_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Training residual std in the regression target's native units (e.g. ``nba_role_zscore``)."""
    return float(np.std(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float), ddof=0))


def predict_tier_proba_from_role_z(
    role_z_pred: np.ndarray,
    tier_residual_std: float,
    *,
    thresholds: tuple = TIER_THRESHOLDS,
) -> pd.DataFrame:
    """Map predicted NBA role z-scores to calibrated 4-class tier probabilities.

    Uses the same Gaussian CDF layer as stats regression bundles
    (:func:`src.models.probability.zscore_to_tier_proba`) so columns match
    ``PROBA_COLUMNS`` for multimodal stacking.
    """
    if tier_residual_std <= 0.0:
        raise ValueError("tier_residual_std must be positive for tier probability export.")
    proba = zscore_to_tier_proba(
        np.asarray(role_z_pred, dtype=float),
        float(tier_residual_std),
        thresholds,
    )
    return proba_to_dataframe(proba)


class UniformTextTierBundle:
    """Fallback tier probabilities when too few text rows exist to fine-tune."""

    def predict_tier_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        return proba_to_dataframe(np.full((len(df), 4), 0.25, dtype=float), index=df.index)


class TextTierPredictorBundle:
    """Fitted text head + calibration stats; exposes ``predict_tier_proba`` like stats bundles."""

    def __init__(
        self,
        model: TextProspectPredictor,
        *,
        target_mean: float,
        target_std: float,
        tier_residual_std: float,
        max_length: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.tier_residual_std = float(tier_residual_std)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = device

    def predict_tier_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            raise ValueError("Text tier bundle requires a 'text' column (merge scouting text onto the frame).")
        mask = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
        out = np.full((len(df), 4), 0.25, dtype=float)
        if mask.any() and self.tier_residual_std > 0.0:
            texts = df.loc[mask, "text"].astype(str).tolist()
            part = self.model.predict_tier_proba_from_texts(
                texts,
                target_mean=self.target_mean,
                target_std=self.target_std,
                tier_residual_std=self.tier_residual_std,
                max_length=self.max_length,
                batch_size=self.batch_size,
                device=self.device,
            )
            out[np.flatnonzero(mask.to_numpy(dtype=bool))] = part.to_numpy(dtype=float, copy=False)
        out = normalize_proba(out)
        return pd.DataFrame(out, columns=PROBA_COLUMNS, index=df.index)


def fit_text_tier_bundle_for_multimodal(
    fold_core: pd.DataFrame,
    cfg: dict,
    *,
    epochs: int | None = None,
    silent: bool = False,
) -> TextTierPredictorBundle | UniformTextTierBundle:
    """Train the scouting text tower on ``fold_core`` (same role as tabular cores in multimodal OOF/final).

    Uses rows with non-empty ``text`` and a valid regression target. Returns a uniform bundle if
    too few examples exist or if ``regression_target_col`` is not ``nba_role_zscore`` (Gaussian tier map).
    """
    text_section: dict = (cfg.get("model") or {}).get("text") or {}
    train_section: dict = cfg.get("training") or {}
    mm_section: dict = (cfg.get("model") or {}).get("multimodal") or {}

    pretrained = str(text_section.get("pretrained", "distilbert-base-uncased"))
    output_dim = int(text_section.get("output_dim", 128))
    freeze_base = bool(text_section.get("freeze_base", True))
    max_length = int(text_section.get("max_length", 256))
    batch_size = int(train_section.get("batch_size", 32))
    lr = float(text_section.get("lr", train_section.get("lr", 2e-5)))
    n_epochs = int(epochs if epochs is not None else train_section.get("epochs", 3))

    tail_weight = float(text_section.get("tail_weight", 0.8))
    variance_penalty = float(text_section.get("variance_penalty", 0.15))
    edge_oversample_weight = float(text_section.get("edge_oversample_weight", 4.0))
    overpredict_discount = float(text_section.get("overpredict_discount", 0.35))
    high_pred_relief = float(text_section.get("high_pred_relief", 0.20))
    underpredict_high_target_boost = float(text_section.get("underpredict_high_target_boost", 0.60))
    classification_weight = float(text_section.get("classification_weight", 1.25))
    star_pos_weight = float(text_section.get("star_pos_weight", 4.0))
    survived_pos_weight = float(text_section.get("survived_pos_weight", 2.5))
    starter_pos_weight = float(text_section.get("starter_pos_weight", 2.5))
    auxiliary_regression_weight = float(text_section.get("auxiliary_regression_weight", 0.25))

    regression_target_col = str(
        text_section.get("regression_target_col") or TARGET_COL["nba_role_zscore"],
    )
    if regression_target_col != TARGET_COL["nba_role_zscore"]:
        return UniformTextTierBundle()

    need = {"text", "prospect_tier", "survived_3yrs", "became_starter", regression_target_col}
    missing = need - set(fold_core.columns)
    if missing:
        raise KeyError(f"fold_core missing columns for text multimodal bundle: {sorted(missing)}")

    tc = fold_core.copy()
    tc = tc[tc["text"].notna() & (tc["text"].astype(str).str.strip() != "")]
    tc = tc.dropna(subset=[regression_target_col])
    tc[regression_target_col] = tc[regression_target_col].astype(float)

    min_rows = int(mm_section.get("text_min_train_rows", 8))
    if len(tc) < min_rows:
        return UniformTextTierBundle()

    tier_col = TARGET_COL["prospect_tier"]
    val_fraction = float(mm_section.get("text_val_fraction", 0.15))
    try_strat = len(tc) >= 12 and tc[tier_col].nunique() > 1
    if try_strat:
        try:
            tc_train, tc_val = train_test_split(
                tc,
                test_size=val_fraction,
                stratify=tc[tier_col],
                random_state=RANDOM_STATE,
            )
        except ValueError:
            tc_train, tc_val = train_test_split(
                tc, test_size=val_fraction, random_state=RANDOM_STATE,
            )
    else:
        tc_train, tc_val = train_test_split(tc, test_size=val_fraction, random_state=RANDOM_STATE)
    has_val = len(tc_val) > 0

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    encoder = ScoutingReportEncoder(
        pretrained=pretrained,
        output_dim=output_dim,
        freeze_base=freeze_base,
    )
    model = TextProspectPredictor(text_encoder=encoder, hidden_dim=64, dropout=0.2).to(device)

    target_mean = float(tc_train[regression_target_col].mean())
    target_std = float(tc_train[regression_target_col].std(ddof=0) + 1e-8)

    star_train = (tc_train["prospect_tier"] == 3).astype(np.float32)
    star_val = (tc_val["prospect_tier"] == 3).astype(np.float32) if has_val else np.array([], dtype=np.float32)

    train_ds = _TokenizedTextDataset(
        tc_train["text"].tolist(),
        tc_train[regression_target_col].tolist(),
        tokenizer=encoder.tokenizer,
        max_length=max_length,
        target_mean=target_mean,
        target_std=target_std,
        star_labels=star_train.tolist(),
        survived_labels=tc_train["survived_3yrs"].tolist(),
        starter_labels=tc_train["became_starter"].tolist(),
    )
    val_ds = (
        _TokenizedTextDataset(
            tc_val["text"].tolist(),
            tc_val[regression_target_col].tolist(),
            tokenizer=encoder.tokenizer,
            max_length=max_length,
            target_mean=target_mean,
            target_std=target_std,
            star_labels=star_val.tolist(),
            survived_labels=tc_val["survived_3yrs"].tolist(),
            starter_labels=tc_val["became_starter"].tolist(),
        )
        if has_val
        else None
    )

    train_sampler = _build_extreme_sampler(
        tc_train[regression_target_col].to_numpy(dtype=np.float32),
        edge_weight=edge_oversample_weight,
    )
    num_workers = 0
    if text_section.get("num_workers") is not None:
        num_workers = max(0, int(float(text_section["num_workers"])))
    dl_kw: dict[str, Any] = {"num_workers": num_workers}
    if num_workers > 0:
        dl_kw["persistent_workers"] = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw) if val_ds is not None else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = WeightedAntiCollapseLoss(
        tail_weight=tail_weight,
        variance_penalty=variance_penalty,
        use_huber=True,
        huber_beta=1.0,
        overpredict_discount=overpredict_discount,
        high_pred_relief=high_pred_relief,
        underpredict_high_target_boost=underpredict_high_target_boost,
        classification_weight=classification_weight,
        star_pos_weight=star_pos_weight,
        survived_pos_weight=survived_pos_weight,
        starter_pos_weight=starter_pos_weight,
        auxiliary_regression_weight=auxiliary_regression_weight,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    for epoch in range(1, n_epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, optimizer=optimizer, device=device, criterion=criterion,
        )
        if val_loader is not None:
            val_loss = _run_epoch(
                model, val_loader, optimizer=None, device=device, criterion=criterion,
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not silent:
            msg = f"[multimodal:text] epoch {epoch:02d}/{n_epochs} train_loss={train_loss:.4f}"
            if val_loader is not None:
                msg += f" val_loss={val_loss:.4f}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    (
        y_tr_resid,
        y_tr_pred,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _predict(
        model,
        train_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    tier_residual_std = max(compute_tier_residual_std(y_tr_resid, y_tr_pred), 1e-8)

    return TextTierPredictorBundle(
        model,
        target_mean=target_mean,
        target_std=target_std,
        tier_residual_std=tier_residual_std,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )


class _TokenizedTextDataset(Dataset):
    """Tokenized text dataset for multi-task outcomes."""

    def __init__(
        self,
        texts: Sequence[str],
        targets: Sequence[float],
        tokenizer,
        max_length: int = 256,
        target_mean: float | None = None,
        target_std: float | None = None,
        star_labels: Sequence[float] | None = None,
        survived_labels: Sequence[float] | None = None,
        starter_labels: Sequence[float] | None = None,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        targets_arr = np.asarray(targets, dtype=np.float32)
        if target_mean is not None and target_std is not None and target_std > 0:
            targets_arr = (targets_arr - target_mean) / target_std
        self.targets = torch.tensor(targets_arr, dtype=torch.float32)
        if star_labels is None or survived_labels is None or starter_labels is None:
            raise ValueError("star_labels, survived_labels, and starter_labels are required.")
        self.star_labels = torch.tensor(np.asarray(star_labels, dtype=np.float32), dtype=torch.float32)
        self.survived_labels = torch.tensor(
            np.asarray(survived_labels, dtype=np.float32),
            dtype=torch.float32,
        )
        self.starter_labels = torch.tensor(
            np.asarray(starter_labels, dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "target": self.targets[idx],
            "star_label": self.star_labels[idx],
            "survived_label": self.survived_labels[idx],
            "starter_label": self.starter_labels[idx],
        }


class WeightedAntiCollapseLoss(nn.Module):
    """
    Regression loss that:
      1) upweights higher-|target| examples (tail focus),
      2) penalizes low prediction variance (anti-collapse).
    """

    def __init__(
        self,
        tail_weight: float = 0.8,
        variance_penalty: float = 0.15,
        use_huber: bool = True,
        huber_beta: float = 1.0,
        overpredict_discount: float = 0.35,
        high_pred_relief: float = 0.20,
        underpredict_high_target_boost: float = 0.60,
        classification_weight: float = 1.0,
        star_pos_weight: float = 4.0,
        survived_pos_weight: float = 2.5,
        starter_pos_weight: float = 2.5,
        auxiliary_regression_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.tail_weight = tail_weight
        self.variance_penalty = variance_penalty
        self.use_huber = use_huber
        self.huber_beta = huber_beta
        self.overpredict_discount = overpredict_discount
        self.high_pred_relief = high_pred_relief
        self.underpredict_high_target_boost = underpredict_high_target_boost
        self.classification_weight = classification_weight
        self.star_pos_weight = star_pos_weight
        self.survived_pos_weight = survived_pos_weight
        self.starter_pos_weight = starter_pos_weight
        self.auxiliary_regression_weight = auxiliary_regression_weight

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        star_logits: torch.Tensor,
        star_labels: torch.Tensor,
        survived_logits: torch.Tensor,
        survived_labels: torch.Tensor,
        starter_logits: torch.Tensor,
        starter_labels: torch.Tensor,
    ) -> torch.Tensor:
        # Weight tails so the model doesn't optimize only around near-zero targets.
        target_scale = target.abs().mean().clamp_min(1e-6)
        sample_weight = 1.0 + self.tail_weight * (target.abs() / target_scale)

        # Asymmetry to encourage bolder positive predictions:
        # - reduce penalty when model overpredicts,
        # - reduce penalty as prediction value increases,
        # - increase penalty when model underpredicts high-target examples.
        over_mask = preds > target
        under_mask = ~over_mask
        asym_weight = torch.ones_like(target)
        asym_weight = torch.where(
            over_mask,
            asym_weight * (1.0 - self.overpredict_discount),
            asym_weight,
        )
        pred_scale = preds.detach().abs().mean().clamp_min(1e-6)
        pred_relief = (preds.clamp(min=0.0) / pred_scale).clamp(min=0.0)
        asym_weight = asym_weight / (1.0 + self.high_pred_relief * pred_relief)
        high_target_boost = (target.clamp(min=0.0) / target_scale).clamp(min=0.0)
        asym_weight = torch.where(
            under_mask,
            asym_weight * (1.0 + self.underpredict_high_target_boost * high_target_boost),
            asym_weight,
        )

        if self.use_huber:
            per_sample = nn.functional.smooth_l1_loss(
                preds, target, beta=self.huber_beta, reduction="none"
            )
        else:
            per_sample = (preds - target) ** 2
        weighted_data_loss = (per_sample * sample_weight * asym_weight).mean()

        # Penalize collapsed predictions (near-constant output variance).
        target_std = target.std(unbiased=False).detach()
        pred_std = preds.std(unbiased=False)
        collapse_penalty = torch.relu(target_std - pred_std) ** 2
        regression_loss = weighted_data_loss + self.variance_penalty * collapse_penalty

        star_weight = torch.tensor(
            self.star_pos_weight,
            dtype=star_logits.dtype,
            device=star_logits.device,
        )
        star_loss = nn.functional.binary_cross_entropy_with_logits(
            star_logits,
            star_labels,
            pos_weight=star_weight,
        )
        survived_weight = torch.tensor(
            self.survived_pos_weight,
            dtype=survived_logits.dtype,
            device=survived_logits.device,
        )
        survived_loss = nn.functional.binary_cross_entropy_with_logits(
            survived_logits,
            survived_labels,
            pos_weight=survived_weight,
        )
        starter_weight = torch.tensor(
            self.starter_pos_weight,
            dtype=starter_logits.dtype,
            device=starter_logits.device,
        )
        starter_loss = nn.functional.binary_cross_entropy_with_logits(
            starter_logits,
            starter_labels,
            pos_weight=starter_weight,
        )
        cls_loss = star_loss + survived_loss + starter_loss
        return (self.auxiliary_regression_weight * regression_loss) + (self.classification_weight * cls_loss)


def _run_epoch(
    model: TextProspectPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """Run one epoch and return mean regression loss."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].to(device)
        star_label = batch["star_label"].to(device)
        survived_label = batch["survived_label"].to(device)
        starter_label = batch["starter_label"].to(device)

        preds, star_logits, survived_logits, starter_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        loss = criterion(
            preds,
            target,
            star_logits,
            star_label,
            survived_logits,
            survived_label,
            starter_logits,
            starter_label,
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _predict(
    model: TextProspectPredictor,
    loader: DataLoader,
    device: torch.device,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    star_probs: list[np.ndarray] = []
    star_true: list[np.ndarray] = []
    survived_probs: list[np.ndarray] = []
    survived_true: list[np.ndarray] = []
    starter_probs: list[np.ndarray] = []
    starter_true: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].cpu().numpy()
        pred, star_logit, survived_logit, starter_logit = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pred = pred.cpu().numpy()
        star_prob = torch.sigmoid(star_logit).cpu().numpy()
        survived_prob = torch.sigmoid(survived_logit).cpu().numpy()
        starter_prob = torch.sigmoid(starter_logit).cpu().numpy()
        ys.append(target)
        preds.append(pred)
        star_probs.append(star_prob)
        star_true.append(batch["star_label"].cpu().numpy())
        survived_probs.append(survived_prob)
        survived_true.append(batch["survived_label"].cpu().numpy())
        starter_probs.append(starter_prob)
        starter_true.append(batch["starter_label"].cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    if target_mean is not None and target_std is not None and target_std > 0:
        y_true = y_true * target_std + target_mean
        y_pred = y_pred * target_std + target_mean
    return (
        y_true,
        y_pred,
        np.concatenate(star_probs),
        np.concatenate(star_true),
        np.concatenate(survived_probs),
        np.concatenate(survived_true),
        np.concatenate(starter_probs),
        np.concatenate(starter_true),
    )


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    star_prob: np.ndarray,
    star_true: np.ndarray,
    survived_prob: np.ndarray,
    survived_true: np.ndarray,
    starter_prob: np.ndarray,
    starter_true: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    if len(np.unique(star_true)) > 1:
        metrics["star_auc"] = float(roc_auc_score(star_true, star_prob))
        metrics["star_ap"] = float(average_precision_score(star_true, star_prob))
        metrics["star_acc"] = float(accuracy_score(star_true, (star_prob >= 0.5).astype(int)))
    else:
        metrics["star_auc"] = float("nan")
        metrics["star_ap"] = float("nan")
        metrics["star_acc"] = float("nan")
    if len(np.unique(survived_true)) > 1:
        metrics["survived_auc"] = float(roc_auc_score(survived_true, survived_prob))
        metrics["survived_ap"] = float(average_precision_score(survived_true, survived_prob))
        metrics["survived_acc"] = float(accuracy_score(survived_true, (survived_prob >= 0.5).astype(int)))
    else:
        metrics["survived_auc"] = float("nan")
        metrics["survived_ap"] = float("nan")
        metrics["survived_acc"] = float("nan")
    if len(np.unique(starter_true)) > 1:
        metrics["starter_auc"] = float(roc_auc_score(starter_true, starter_prob))
        metrics["starter_ap"] = float(average_precision_score(starter_true, starter_prob))
        metrics["starter_acc"] = float(accuracy_score(starter_true, (starter_prob >= 0.5).astype(int)))
    else:
        metrics["starter_auc"] = float("nan")
        metrics["starter_ap"] = float("nan")
        metrics["starter_acc"] = float("nan")
    return metrics


def _build_extreme_sampler(
    targets: np.ndarray,
    edge_weight: float = 4.0,
) -> WeightedRandomSampler:
    """
    Oversample top/bottom decile targets in the training split.
    """
    low_q = float(np.quantile(targets, 0.10))
    high_q = float(np.quantile(targets, 0.90))
    is_extreme = (targets <= low_q) | (targets >= high_q)
    weights = np.where(is_extreme, edge_weight, 1.0).astype(np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )


def train_and_evaluate_text_model(
    pretrained: str = "distilbert-base-uncased",
    output_dim: int = 128,
    freeze_base: bool = True,
    max_length: int = 256,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 2e-5,
    cfg: dict | None = None,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    tail_weight: float = 0.8,
    variance_penalty: float = 0.15,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_seed: int = 42,
    edge_oversample_weight: float = 4.0,
    overpredict_discount: float = 0.35,
    high_pred_relief: float = 0.20,
    underpredict_high_target_boost: float = 0.60,
    classification_weight: float = 1.25,
    star_pos_weight: float = 4.0,
    survived_pos_weight: float = 2.5,
    starter_pos_weight: float = 2.5,
    auxiliary_regression_weight: float = 0.25,
    regression_target_col: str | None = None,
    tier_proba_csv_path: str | None = None,
    save_path: str | None = None,
) -> tuple[TextProspectPredictor, dict[str, float]]:
    """Train/evaluate text-only predictor using scouting report text.

    When ``regression_target_col`` is ``nba_role_zscore`` (default), training
    residuals yield ``tier_residual_std`` for exporting 4-class probabilities
    compatible with multimodal meta-features (``tier_proba_csv_path`` writes
    every row in the text merge frame: ``Name``, ``draft_year``, ``p_bust``, …).
    """
    # Callers often pass YAML-loaded values: PyYAML 1.1 can leave scientific notation
    # (e.g. ``1e-3``) as str, which breaks torch optimizers.
    lr = float(lr)
    batch_size = int(batch_size)
    epochs = int(epochs)
    max_length = int(max_length)
    output_dim = int(output_dim)
    freeze_base = bool(freeze_base)

    if train_frac <= 0 or val_frac <= 0 or (train_frac + val_frac) >= 1:
        raise ValueError("train_frac and val_frac must be > 0 and sum to < 1.")

    composite_cfg = None
    text_section: dict = {}
    if cfg is not None:
        composite_cfg = (cfg.get("model") or {}).get("composite_score")
        text_section = (cfg.get("model") or {}).get("text") or {}

    if text_section.get("edge_oversample_weight") is not None:
        edge_oversample_weight = float(text_section["edge_oversample_weight"])

    regression_target_col = regression_target_col or TARGET_COL["nba_role_zscore"]
    df = load_text_data(composite_cfg=composite_cfg)
    if regression_target_col not in df.columns:
        raise KeyError(
            f"regression_target_col={regression_target_col!r} not in merged text frame; "
            f"choose one of the numeric targets present in load_data() (e.g. nba_role_zscore)."
        )
    df = df.dropna(subset=[regression_target_col]).copy()
    df[regression_target_col] = df[regression_target_col].astype(float)
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

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more random splits are empty; adjust split fractions.")

    print(f"\nText dataset: {len(df)} total players")
    print(f"Train: {len(train_df)} players (random all-years split)")
    print(f"Val:   {len(val_df)} players (random all-years split)")
    print(f"Test:  {len(test_df)} players (random all-years split)")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder = ScoutingReportEncoder(
        pretrained=pretrained,
        output_dim=output_dim,
        freeze_base=freeze_base,
    )
    model = TextProspectPredictor(text_encoder=encoder, hidden_dim=64, dropout=0.2).to(device)

    target_mean = float(train_df[regression_target_col].mean())
    target_std = float(train_df[regression_target_col].std(ddof=0) + 1e-8)
    star_train = (train_df["prospect_tier"] == 3).astype(np.float32)
    star_val = (val_df["prospect_tier"] == 3).astype(np.float32)
    star_test = (test_df["prospect_tier"] == 3).astype(np.float32)

    train_ds = _TokenizedTextDataset(
        train_df["text"].tolist(),
        train_df[regression_target_col].tolist(),
        tokenizer=encoder.tokenizer,
        max_length=max_length,
        target_mean=target_mean,
        target_std=target_std,
        star_labels=star_train.tolist(),
        survived_labels=train_df["survived_3yrs"].tolist(),
        starter_labels=train_df["became_starter"].tolist(),
    )
    val_ds = _TokenizedTextDataset(
        val_df["text"].tolist(),
        val_df[regression_target_col].tolist(),
        tokenizer=encoder.tokenizer,
        max_length=max_length,
        target_mean=target_mean,
        target_std=target_std,
        star_labels=star_val.tolist(),
        survived_labels=val_df["survived_3yrs"].tolist(),
        starter_labels=val_df["became_starter"].tolist(),
    )
    test_ds = _TokenizedTextDataset(
        test_df["text"].tolist(),
        test_df[regression_target_col].tolist(),
        tokenizer=encoder.tokenizer,
        max_length=max_length,
        target_mean=target_mean,
        target_std=target_std,
        star_labels=star_test.tolist(),
        survived_labels=test_df["survived_3yrs"].tolist(),
        starter_labels=test_df["became_starter"].tolist(),
    )

    train_sampler = _build_extreme_sampler(
        train_df[regression_target_col].to_numpy(dtype=np.float32),
        edge_weight=edge_oversample_weight,
    )
    num_workers = 0
    if text_section.get("num_workers") is not None:
        num_workers = max(0, int(float(text_section["num_workers"])))
    dl_kw: dict = {"num_workers": num_workers}
    if num_workers > 0:
        dl_kw["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, **dl_kw
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}
    mlflow_ctx = build_mlflow_context(
        cfg=cfg,
        model_type="text",
        target_name=regression_target_col,
        fallback_experiment_name="nba-draft-prospect-text",
        tracking_uri=tracking_uri,
        run_name=run_name,
    )

    with managed_run(mlflow_ctx):
        if cfg is not None:
            log_config_dict(cfg)
        log_common_params(
            {
                "model_family": "text",
                "target": regression_target_col,
                "pretrained": pretrained,
                "output_dim": output_dim,
                "freeze_base": freeze_base,
                "max_length": max_length,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "n_total": len(df),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
            }
        )
        criterion = WeightedAntiCollapseLoss(
            tail_weight=tail_weight,
            variance_penalty=variance_penalty,
            use_huber=True,
            huber_beta=1.0,
            overpredict_discount=overpredict_discount,
            high_pred_relief=high_pred_relief,
            underpredict_high_target_boost=underpredict_high_target_boost,
            classification_weight=classification_weight,
            star_pos_weight=star_pos_weight,
            survived_pos_weight=survived_pos_weight,
            starter_pos_weight=starter_pos_weight,
            auxiliary_regression_weight=auxiliary_regression_weight,
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        for epoch in range(1, epochs + 1):
            train_loss = _run_epoch(
            model, train_loader, optimizer=optimizer, device=device, criterion=criterion
        )
            val_loss = _run_epoch(
            model, val_loader, optimizer=None, device=device, criterion=criterion
        )
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            log_epoch_metrics({"train_loss": train_loss, "val_loss": val_loss}, epoch)
            print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        (
            y_tr_resid,
            y_tr_pred,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = _predict(
            model,
            train_loader,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )
        tier_residual_std = 0.0
        if regression_target_col == TARGET_COL["nba_role_zscore"]:
            tier_residual_std = max(compute_tier_residual_std(y_tr_resid, y_tr_pred), 1e-8)

        if save_path:
            ckpt_parent = os.path.dirname(os.path.abspath(save_path))
            if ckpt_parent:
                os.makedirs(ckpt_parent, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "regression_target_col": regression_target_col,
                    "tier_residual_std": tier_residual_std,
                    "star_threshold": float("nan"),
                    "pretrained": pretrained,
                    "output_dim": output_dim,
                    "freeze_base": freeze_base,
                    "max_length": max_length,
                    "hidden_dim": 64,
                    "dropout": 0.2,
                },
                save_path,
            )
            print(f"\nCheckpoint saved to {save_path}")

    (
        y_test,
        y_pred,
        star_prob,
        star_true,
        survived_prob,
        survived_true,
        starter_prob,
        starter_true,
    ) = _predict(
        model,
        test_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    metrics = _metrics(
        y_test,
        y_pred,
        star_prob,
        star_true=star_true,
        survived_prob=survived_prob,
        survived_true=survived_true,
        starter_prob=starter_prob,
        starter_true=starter_true,
    )
    if tier_residual_std > 0.0:
        metrics["tier_residual_std"] = float(tier_residual_std)
    mlflow.log_metrics(metrics)
    mlflow.log_metric("best_val_loss", float(best_val))
    if tier_residual_std > 0.0:
        mlflow.log_metric("tier_residual_std", float(tier_residual_std))
    mlflow.pytorch.log_model(model, name="model")

    print("\n" + "=" * 40)
    print("Text Model Test Metrics")
    print("=" * 40)
    print(f"R2   = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.4f}")
    print(f"MAE  = {metrics['mae']:.4f}")
    print(f"is_star         | AUC={metrics['star_auc']:.4f} AP={metrics['star_ap']:.4f} ACC={metrics['star_acc']:.4f}")
    print(f"survived_3yrs   | AUC={metrics['survived_auc']:.4f} AP={metrics['survived_ap']:.4f} ACC={metrics['survived_acc']:.4f}")
    print(f"became_starter  | AUC={metrics['starter_auc']:.4f} AP={metrics['starter_ap']:.4f} ACC={metrics['starter_acc']:.4f}")

    if tier_proba_csv_path and regression_target_col != TARGET_COL["nba_role_zscore"]:
        print(
            "[tier_proba] Skipped CSV export: Gaussian tier probabilities require "
            "regression_target_col='nba_role_zscore' (stats regression alignment)."
        )
    if tier_proba_csv_path and regression_target_col == TARGET_COL["nba_role_zscore"] and tier_residual_std > 0.0:
        # Full-frame export so multimodal can join on every (Name, draft_year) with scouting text,
        # not only the random held-out test split.
        proba_full = model.predict_tier_proba_from_texts(
            df["text"].tolist(),
            target_mean=target_mean,
            target_std=target_std,
            tier_residual_std=tier_residual_std,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        id_cols = [c for c in ("Name", "draft_year") if c in df.columns]
        out_csv = pd.concat(
            [df[id_cols].reset_index(drop=True), proba_full.reset_index(drop=True)],
            axis=1,
        )
        out_p = Path(tier_proba_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_csv.to_csv(out_p, index=False)
        print(f"\n[tier_proba] Wrote {len(out_csv)} rows (full text frame) to {out_p}")
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(out_p), name="tier_proba")

    _plot_text_results(
        y_test,
        y_pred,
        history,
        plot_dir=mlflow_ctx.plot_dir,
        regression_target_col=regression_target_col,
    )
    return model, metrics


def _plot_text_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: dict[str, list[float]],
    plot_dir: str,
    regression_target_col: str,
) -> None:
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors="none")
    upperLim = max(abs(float(np.max(y_true))), abs(float(np.max(y_pred)))) + 1
    bottomLim = min(float(np.min(y_true)), float(np.min(y_pred))) - 1
    plt.plot([bottomLim, upperLim], [bottomLim, upperLim], "r--", linewidth=1, label="Perfect prediction")
    plt.xlabel(f"Actual {regression_target_col}")
    plt.ylabel(f"Predicted {regression_target_col}")
    plt.title(f"Text model: scouting report → {regression_target_col}")
    plt.legend(fontsize=8)
    plt.xlim(bottomLim, upperLim)
    plt.ylim(bottomLim, upperLim)
    plt.tight_layout()
    out_path = plot_path / "text_model_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(out_path), name="plots")

    plt.figure(figsize=(7, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="train_loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="val_loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Text Model Loss Curves")
    plt.legend()
    plt.tight_layout()
    loss_out_path = plot_path / "text_model_loss_curves.png"
    plt.savefig(loss_out_path, dpi=150)
    print(f"Plot saved to {loss_out_path}")
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(loss_out_path), name="plots")


if __name__ == "__main__":
    train_and_evaluate_text_model()
