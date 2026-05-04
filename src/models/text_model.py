"""Scouting-report text encoder with regression (role z, VORP, DPM, …) or 4-class tier classification (softmax for PSM)."""
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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
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

# Gaussian tier CSV / legacy regression→tier calibration (thresholds are for ``nba_role_zscore`` scale).
_GAUSSIAN_TIER_REGRESSION_COLS = frozenset({TARGET_COL["nba_role_zscore"]})

# Interpretability / reporting short names (see ``interpret_head_key_for_target``).
INTERPRET_HEAD_BY_TARGET: dict[str, str] = {
    TARGET_COL["nba_role_zscore"]: "role_z",
    "VORP": "vorp",
    "DPM": "darko",
    TARGET_COL["prospect_tier"]: "tier",
}


def interpret_head_key_for_target(column: str) -> str:
    """Stable slug for plots/CSVs (e.g. ``VORP`` → ``vorp``, ``DPM`` → ``darko``)."""
    return INTERPRET_HEAD_BY_TARGET.get(column, column.replace(" ", "_").lower())


def _resolve_text_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ScoutingReportEncoder(nn.Module):
    """Fine-tuned transformer encoder that produces a prospect embedding from text."""

    def __init__(
        self,
        pretrained: str = "distilbert-base-uncased",
        output_dim: int = 64,
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
    """Text → normalized regression score, or ``num_classes`` tier logits (e.g. 4 for ``prospect_tier``)."""

    def __init__(
        self,
        text_encoder: ScoutingReportEncoder,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.num_classes = num_classes
        self.shared = nn.Sequential(
            nn.Linear(text_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if num_classes is not None:
            self.cls_head = nn.Linear(hidden_dim, num_classes)
            self.reg_head = None
        else:
            self.reg_head = nn.Linear(hidden_dim, 1)
            self.cls_head = None

    @property
    def head_in_features(self) -> int:
        if self.cls_head is not None:
            return int(self.cls_head.in_features)
        assert self.reg_head is not None
        return int(self.reg_head.in_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.shared(embeddings)
        if self.cls_head is not None:
            return self.cls_head(hidden)
        assert self.reg_head is not None
        return self.reg_head(hidden).squeeze(-1)

    @torch.no_grad()
    def predict_from_texts(
        self,
        texts: Sequence[str],
        max_length: int = 512,
        batch_size: int = 32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Regression only: predicted target in normalized z-space."""
        if self.cls_head is not None:
            raise TypeError("predict_from_texts applies to regression models only; use predict_tier_proba_from_texts.")
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
        assert self.reg_head is not None
        hidden = self.shared(embeddings.to(device))
        preds = self.reg_head(hidden).squeeze(-1).detach().cpu()
        if was_training:
            self.train()
        return preds

    @torch.no_grad()
    def _predict_logits_batched(
        self,
        texts: Sequence[str],
        max_length: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Classification: logits [N, num_classes] on CPU."""
        if not texts:
            n_cls = self.num_classes or 4
            return torch.empty((0, n_cls), dtype=torch.float32)
        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        tokenizer = self.text_encoder.tokenizer
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(
                    list(texts[start : start + batch_size]),
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = self(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                chunks.append(logits.detach().cpu())
        if was_training:
            self.train()
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def predict_tier_proba_from_texts(
        self,
        texts: Sequence[str],
        *,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        tier_residual_std: float = 1.0,
        max_length: int = 512,
        batch_size: int = 32,
        device: torch.device | None = None,
    ) -> pd.DataFrame:
        """``PROBA_COLUMNS`` for PSM: softmax (classification) or Gaussian-on-z (regression on role score)."""
        device = device or _resolve_text_device()
        if self.cls_head is not None:
            logits = self._predict_logits_batched(texts, max_length, batch_size, device)
            if logits.numel() == 0:
                return proba_to_dataframe(np.zeros((0, 4), dtype=np.float64))
            proba = torch.softmax(logits, dim=-1).numpy().astype(np.float64)
            return proba_to_dataframe(proba)
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
    """Left-merge scouting ``text`` and ``survived_3yrs`` onto a stats frame (``Name`` + ``draft_year``).

    ``survived_3yrs`` comes from ``load_text_data`` (season-cache logic); kept for downstream tabular flows.
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
        if mask.any():
            texts = df.loc[mask, "text"].astype(str).tolist()
            if self.model.cls_head is not None:
                part = self.model.predict_tier_proba_from_texts(
                    texts,
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                    device=self.device,
                )
            elif self.tier_residual_std > 0.0:
                part = self.model.predict_tier_proba_from_texts(
                    texts,
                    target_mean=self.target_mean,
                    target_std=self.target_std,
                    tier_residual_std=self.tier_residual_std,
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                    device=self.device,
                )
            else:
                part = None
            if part is not None:
                out[np.flatnonzero(mask.to_numpy(dtype=bool))] = part.to_numpy(dtype=float, copy=False)
        out = normalize_proba(out)
        return pd.DataFrame(out, columns=PROBA_COLUMNS, index=df.index)


class _TokenizedTextClassificationDataset(Dataset):
    """Tokenized text + integer tier labels (0 .. num_classes - 1)."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 128,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
        }


def _run_classification_epoch(
    model: TextProspectPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.functional.cross_entropy(logits, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _predict_classification(
    model: TextProspectPredictor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].cpu().numpy()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = logits.argmax(dim=-1).cpu().numpy()
        ys.append(labels)
        preds.append(pred)
    return np.concatenate(ys), np.concatenate(preds)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _train_text_classifier_arrays(
    *,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str] | None,
    val_labels: list[int] | None,
    num_classes: int,
    pretrained: str,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
    freeze_base: bool,
    max_length: int,
    batch_size: int,
    lr: float,
    n_epochs: int,
    device: torch.device,
    num_workers: int,
    silent: bool,
    log_epoch: Any | None = None,
    log_prefix: str = "",
) -> tuple[TextProspectPredictor, dict[str, list[float]]]:
    encoder = ScoutingReportEncoder(
        pretrained=pretrained,
        output_dim=output_dim,
        freeze_base=freeze_base,
    )
    model = TextProspectPredictor(
        text_encoder=encoder,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)

    train_ds = _TokenizedTextClassificationDataset(
        train_texts,
        train_labels,
        tokenizer=encoder.tokenizer,
        max_length=max_length,
    )
    dl_kw: dict[str, Any] = {"num_workers": num_workers}
    if num_workers > 0:
        dl_kw["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)

    val_loader: DataLoader | None = None
    if val_texts is not None and val_labels is not None and len(val_texts) > 0:
        val_ds = _TokenizedTextClassificationDataset(
            val_texts,
            val_labels,
            tokenizer=encoder.tokenizer,
            max_length=max_length,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss = _run_classification_epoch(
            model, train_loader, optimizer=optimizer, device=device,
        )
        history["train_loss"].append(train_loss)
        if val_loader is not None:
            val_loss = _run_classification_epoch(
                model, val_loader, optimizer=None, device=device,
            )
            history["val_loss"].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if log_epoch is not None:
                log_epoch(epoch, train_loss, val_loss)
            elif not silent:
                pf = log_prefix or ""
                print(
                    f"{pf}epoch {epoch:02d}/{n_epochs} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f}"
                )
        else:
            history["val_loss"].append(float("nan"))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if log_epoch is not None:
                log_epoch(epoch, train_loss, float("nan"))
            elif not silent:
                pf = log_prefix or ""
                print(f"{pf}epoch {epoch:02d}/{n_epochs} train_loss={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def fit_text_tier_bundle_for_multimodal(
    fold_core: pd.DataFrame,
    cfg: dict,
    *,
    epochs: int | None = None,
    silent: bool = False,
) -> TextTierPredictorBundle | UniformTextTierBundle:
    """Train the scouting text tower on ``fold_core`` (same role as tabular cores in multimodal OOF/final).

    - ``task: classification`` — cross-entropy on ``classification_target_col`` (default ``prospect_tier``);
      stacker gets softmax tier probabilities.
    - ``task: regression`` (default) — only ``nba_role_zscore`` produces a non-uniform Gaussian tier bundle;
      other numeric targets (e.g. ``VORP``, ``DPM``) fall back to ``UniformTextTierBundle`` for stacking.
    """
    text_section: dict = (cfg.get("model") or {}).get("text") or {}
    train_section: dict = cfg.get("training") or {}
    mm_section: dict = (cfg.get("model") or {}).get("multimodal") or {}

    pretrained = str(text_section.get("pretrained", "distilbert-base-uncased"))
    output_dim = int(text_section.get("output_dim", 64))
    hidden_dim = int(text_section.get("hidden_dim", 32))
    freeze_base = bool(text_section.get("freeze_base", True))
    dropout = float(text_section.get("dropout", 0.2))
    max_length = int(text_section.get("max_length", 128))
    batch_size = int(train_section.get("batch_size", 32))
    lr = float(text_section.get("lr", train_section.get("lr", 2e-5)))
    n_epochs = int(epochs if epochs is not None else train_section.get("epochs", 3))
    huber_beta = float(text_section.get("huber_beta", 1.0))
    task = str(text_section.get("task", "regression")).lower().strip()
    num_classes = int(text_section.get("num_classes", 4))

    min_rows = int(mm_section.get("text_min_train_rows", 8))
    val_fraction = float(mm_section.get("text_val_fraction", 0.15))
    num_workers = 0
    if text_section.get("num_workers") is not None:
        num_workers = max(0, int(float(text_section["num_workers"])))
    device = _resolve_text_device()

    if task == "classification":
        label_col = str(text_section.get("classification_target_col", TARGET_COL["prospect_tier"]))
        need = {"text", label_col}
        missing = need - set(fold_core.columns)
        if missing:
            raise KeyError(f"fold_core missing columns for text multimodal bundle: {sorted(missing)}")

        tc = fold_core.copy()
        tc = tc[tc["text"].notna() & (tc["text"].astype(str).str.strip() != "")]
        tc[label_col] = pd.to_numeric(tc[label_col], errors="coerce")
        tc = tc.dropna(subset=[label_col])
        tc[label_col] = tc[label_col].astype(int)
        if len(tc) < min_rows:
            return UniformTextTierBundle()
        if tc[label_col].min() < 0 or tc[label_col].max() >= num_classes:
            raise ValueError(
                f"{label_col} must be integer class labels in 0..{num_classes - 1} for text classification.",
            )

        strat_col = label_col
        try_strat = len(tc) >= 12 and tc[strat_col].nunique() > 1
        if try_strat:
            try:
                tc_train, tc_val = train_test_split(
                    tc,
                    test_size=val_fraction,
                    stratify=tc[strat_col],
                    random_state=RANDOM_STATE,
                )
            except ValueError:
                tc_train, tc_val = train_test_split(
                    tc, test_size=val_fraction, random_state=RANDOM_STATE,
                )
        else:
            tc_train, tc_val = train_test_split(tc, test_size=val_fraction, random_state=RANDOM_STATE)
        has_val = len(tc_val) > 0

        model, _history = _train_text_classifier_arrays(
            train_texts=tc_train["text"].tolist(),
            train_labels=tc_train[label_col].tolist(),
            val_texts=tc_val["text"].tolist() if has_val else None,
            val_labels=tc_val[label_col].tolist() if has_val else None,
            num_classes=num_classes,
            pretrained=pretrained,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_base=freeze_base,
            max_length=max_length,
            batch_size=batch_size,
            lr=lr,
            n_epochs=n_epochs,
            device=device,
            num_workers=num_workers,
            silent=silent,
            log_epoch=None,
            log_prefix="[multimodal:text] ",
        )
        return TextTierPredictorBundle(
            model,
            target_mean=0.0,
            target_std=1.0,
            tier_residual_std=1.0,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )

    regression_target_col = str(
        text_section.get("regression_target_col") or TARGET_COL["nba_role_zscore"],
    )
    if regression_target_col not in _GAUSSIAN_TIER_REGRESSION_COLS:
        if not silent:
            print(
                f"[multimodal:text] regression_target_col={regression_target_col!r} is not "
                f"nba_role_zscore; Gaussian tier bundle unavailable — using uniform text meta-features.",
            )
        return UniformTextTierBundle()

    need = {"text", regression_target_col}
    missing = need - set(fold_core.columns)
    if missing:
        raise KeyError(f"fold_core missing columns for text multimodal bundle: {sorted(missing)}")

    tc = fold_core.copy()
    tc = tc[tc["text"].notna() & (tc["text"].astype(str).str.strip() != "")]
    tc = tc.dropna(subset=[regression_target_col])
    tc[regression_target_col] = tc[regression_target_col].astype(float)

    if len(tc) < min_rows:
        return UniformTextTierBundle()

    tier_col = TARGET_COL["prospect_tier"]
    try_strat = (
        tier_col in tc.columns
        and len(tc) >= 12
        and tc[tier_col].nunique() > 1
    )
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

    model, target_mean, target_std, tier_residual_std, _ = _train_text_predictor_arrays(
        train_texts=tc_train["text"].tolist(),
        train_targets=tc_train[regression_target_col].tolist(),
        val_texts=tc_val["text"].tolist() if has_val else None,
        val_targets=tc_val[regression_target_col].tolist() if has_val else None,
        pretrained=pretrained,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_base=freeze_base,
        max_length=max_length,
        batch_size=batch_size,
        lr=lr,
        n_epochs=n_epochs,
        device=device,
        num_workers=num_workers,
        silent=silent,
        huber_beta=huber_beta,
        log_epoch=None,
        log_prefix="[multimodal:text] ",
    )

    return TextTierPredictorBundle(
        model,
        target_mean=target_mean,
        target_std=target_std,
        tier_residual_std=tier_residual_std,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )


def _train_text_predictor_arrays(
    *,
    train_texts: list[str],
    train_targets: list[float],
    val_texts: list[str] | None,
    val_targets: list[float] | None,
    pretrained: str,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
    freeze_base: bool,
    max_length: int,
    batch_size: int,
    lr: float,
    n_epochs: int,
    device: torch.device,
    num_workers: int,
    silent: bool,
    huber_beta: float = 1.0,
    log_epoch: Any | None = None,
    log_prefix: str = "",
) -> tuple[TextProspectPredictor, float, float, float, dict[str, list[float]]]:
    """Train a single-task text regressor; return model, normalization stats, tier residual std, loss history."""
    target_mean = float(np.mean(train_targets))
    target_std = float(np.std(np.asarray(train_targets, dtype=np.float64), ddof=0) + 1e-8)

    encoder = ScoutingReportEncoder(
        pretrained=pretrained,
        output_dim=output_dim,
        freeze_base=freeze_base,
    )
    model = TextProspectPredictor(
        text_encoder=encoder,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=None,
    ).to(device)

    train_ds = _TokenizedTextDataset(
        train_texts,
        train_targets,
        tokenizer=encoder.tokenizer,
        max_length=max_length,
        target_mean=target_mean,
        target_std=target_std,
    )
    dl_kw: dict[str, Any] = {"num_workers": num_workers}
    if num_workers > 0:
        dl_kw["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)

    val_loader: DataLoader | None = None
    if val_texts is not None and val_targets is not None and len(val_texts) > 0:
        val_ds = _TokenizedTextDataset(
            val_texts,
            val_targets,
            tokenizer=encoder.tokenizer,
            max_length=max_length,
            target_mean=target_mean,
            target_std=target_std,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            huber_beta=huber_beta,
        )
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = _run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                huber_beta=huber_beta,
            )
            history["val_loss"].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if log_epoch is not None:
                log_epoch(epoch, train_loss, val_loss)
            elif not silent:
                pf = log_prefix or ""
                print(
                    f"{pf}epoch {epoch:02d}/{n_epochs} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f}"
                )
        else:
            history["val_loss"].append(float("nan"))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if log_epoch is not None:
                log_epoch(epoch, train_loss, float("nan"))
            elif not silent:
                pf = log_prefix or ""
                print(f"{pf}epoch {epoch:02d}/{n_epochs} train_loss={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    y_tr_resid, y_tr_pred = _predict(
        model,
        train_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )
    tier_residual_std = max(compute_tier_residual_std(y_tr_resid, y_tr_pred), 1e-8)
    return model, target_mean, target_std, tier_residual_std, history


class _TokenizedTextDataset(Dataset):
    """Tokenized text + regression target (normalized z-score)."""

    def __init__(
        self,
        texts: Sequence[str],
        targets: Sequence[float],
        tokenizer,
        max_length: int = 128,
        target_mean: float | None = None,
        target_std: float | None = None,
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

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "target": self.targets[idx],
        }


def _run_epoch(
    model: TextProspectPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    huber_beta: float = 1.0,
) -> float:
    """Run one epoch; mean Smooth L1 loss on normalized targets."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.functional.smooth_l1_loss(preds, target, beta=huber_beta)

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
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].cpu().numpy()
        pred = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = pred.cpu().numpy()
        ys.append(target)
        preds.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    if target_mean is not None and target_std is not None and target_std > 0:
        y_true = y_true * target_std + target_mean
        y_pred = y_pred * target_std + target_mean
    return y_true, y_pred


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_and_evaluate_text_model(
    pretrained: str = "distilbert-base-uncased",
    output_dim: int = 64,
    hidden_dim: int = 32,
    dropout: float = 0.2,
    freeze_base: bool = True,
    max_length: int = 128,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 2e-5,
    huber_beta: float = 1.0,
    cfg: dict | None = None,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_seed: int = 42,
    task: str | None = None,
    classification_target_col: str | None = None,
    num_classes: int = 4,
    regression_target_col: str | None = None,
    tier_proba_csv_path: str | None = None,
    save_path: str | None = None,
) -> tuple[TextProspectPredictor, dict[str, float]]:
    """Train/evaluate text-only predictor using scouting report text.

    - ``task=regression`` (default): predict a numeric column (``nba_role_zscore``,
      ``VORP``, ``DPM``, …). Gaussian tier CSV export applies only to ``nba_role_zscore``.
    - ``task=classification``: cross-entropy on ``classification_target_col`` (default
      ``prospect_tier``); tier probabilities for stacking come from softmax.
    """
    lr = float(lr)
    batch_size = int(batch_size)
    epochs = int(epochs)
    max_length = int(max_length)
    output_dim = int(output_dim)
    hidden_dim = int(hidden_dim)
    dropout = float(dropout)
    freeze_base = bool(freeze_base)
    huber_beta = float(huber_beta)
    num_classes = int(num_classes)

    if train_frac <= 0 or val_frac <= 0 or (train_frac + val_frac) >= 1:
        raise ValueError("train_frac and val_frac must be > 0 and sum to < 1.")

    composite_cfg = None
    text_section: dict = {}
    if cfg is not None:
        composite_cfg = (cfg.get("model") or {}).get("composite_score")
        text_section = (cfg.get("model") or {}).get("text") or {}

    if text_section.get("pretrained") is not None:
        pretrained = str(text_section["pretrained"])
    if text_section.get("output_dim") is not None:
        output_dim = int(text_section["output_dim"])
    if text_section.get("hidden_dim") is not None:
        hidden_dim = int(text_section["hidden_dim"])
    if text_section.get("dropout") is not None:
        dropout = float(text_section["dropout"])
    if text_section.get("freeze_base") is not None:
        freeze_base = bool(text_section["freeze_base"])
    if text_section.get("max_length") is not None:
        max_length = int(text_section["max_length"])
    if text_section.get("huber_beta") is not None:
        huber_beta = float(text_section["huber_beta"])
    if text_section.get("num_classes") is not None:
        num_classes = int(text_section["num_classes"])

    text_task = str(task or text_section.get("task") or "regression").lower().strip()
    label_col = str(
        classification_target_col
        or text_section.get("classification_target_col")
        or TARGET_COL["prospect_tier"],
    )

    df = load_text_data(composite_cfg=composite_cfg)

    if text_task == "classification":
        if label_col not in df.columns:
            raise KeyError(
                f"classification_target_col={label_col!r} not in merged text frame.",
            )
        df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")].copy()
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.dropna(subset=[label_col]).copy()
        df[label_col] = df[label_col].astype(int)
        if df[label_col].min() < 0 or df[label_col].max() >= num_classes:
            raise ValueError(
                f"{label_col} must be integer labels in 0..{num_classes - 1} for classification.",
            )
        mlflow_target = f"classification:{label_col}"
    else:
        regression_target_col = regression_target_col or TARGET_COL["nba_role_zscore"]
        if regression_target_col not in df.columns:
            raise KeyError(
                f"regression_target_col={regression_target_col!r} not in merged text frame; "
                f"try nba_role_zscore, VORP, DPM, etc. (VORP/DPM require columns in nba master).",
            )
        df = df.dropna(subset=[regression_target_col]).copy()
        df[regression_target_col] = df[regression_target_col].astype(float)
        mlflow_target = regression_target_col

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

    print(f"\nText dataset: {len(df)} total players (task={text_task})")
    print(f"Train: {len(train_df)} players (random all-years split)")
    print(f"Val:   {len(val_df)} players (random all-years split)")
    print(f"Test:  {len(test_df)} players (random all-years split)")

    device = _resolve_text_device()
    num_workers = 0
    if text_section.get("num_workers") is not None:
        num_workers = max(0, int(float(text_section["num_workers"])))

    mlflow_ctx = build_mlflow_context(
        cfg=cfg,
        model_type="text",
        target_name=mlflow_target,
        fallback_experiment_name="nba-draft-prospect-text",
        tracking_uri=tracking_uri,
        run_name=run_name,
    )

    def log_epoch_cb(epoch: int, train_loss: float, val_loss: float) -> None:
        log_epoch_metrics({"train_loss": train_loss, "val_loss": val_loss}, epoch)
        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    with managed_run(mlflow_ctx):
        if cfg is not None:
            log_config_dict(cfg)
        log_common_params(
            {
                "model_family": "text",
                "task": text_task,
                "target": mlflow_target,
                "pretrained": pretrained,
                "output_dim": output_dim,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "freeze_base": freeze_base,
                "max_length": max_length,
                "huber_beta": huber_beta,
                "num_classes": num_classes,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "n_total": len(df),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
            }
        )

        dl_kw: dict[str, Any] = {"num_workers": num_workers}
        if num_workers > 0:
            dl_kw["persistent_workers"] = True

        if text_task == "classification":
            model, history = _train_text_classifier_arrays(
                train_texts=train_df["text"].tolist(),
                train_labels=train_df[label_col].tolist(),
                val_texts=val_df["text"].tolist(),
                val_labels=val_df[label_col].tolist(),
                num_classes=num_classes,
                pretrained=pretrained,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                freeze_base=freeze_base,
                max_length=max_length,
                batch_size=batch_size,
                lr=lr,
                n_epochs=epochs,
                device=device,
                num_workers=num_workers,
                silent=True,
                log_epoch=log_epoch_cb,
                log_prefix="",
            )
            target_mean = 0.0
            target_std = 1.0
            tier_residual_std = 0.0
            best_val_loss = float(min(history["val_loss"]))

            test_ds = _TokenizedTextClassificationDataset(
                test_df["text"].tolist(),
                test_df[label_col].tolist(),
                tokenizer=model.text_encoder.tokenizer,
                max_length=max_length,
            )
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kw)
            ckpt_hidden = model.head_in_features

            if save_path:
                ckpt_parent = os.path.dirname(os.path.abspath(save_path))
                if ckpt_parent:
                    os.makedirs(ckpt_parent, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "task": "classification",
                        "num_classes": num_classes,
                        "classification_target_col": label_col,
                        "target_mean": 0.0,
                        "target_std": 1.0,
                        "regression_target_col": "",
                        "tier_residual_std": 0.0,
                        "star_threshold": float("nan"),
                        "pretrained": pretrained,
                        "output_dim": output_dim,
                        "freeze_base": freeze_base,
                        "max_length": max_length,
                        "hidden_dim": ckpt_hidden,
                        "dropout": dropout,
                        "huber_beta": huber_beta,
                    },
                    save_path,
                )
                print(f"\nCheckpoint saved to {save_path}")

            y_test, y_pred = _predict_classification(model, test_loader, device=device)
            metrics = _classification_metrics(y_test, y_pred)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.pytorch.log_model(model, name="model")

            print("\n" + "=" * 40)
            print("Text Model Test Metrics (classification)")
            print("=" * 40)
            print(f"Accuracy  = {metrics['accuracy']:.4f}")
            print(f"F1 macro  = {metrics['f1_macro']:.4f}")

            if tier_proba_csv_path:
                proba_full = model.predict_tier_proba_from_texts(
                    df["text"].tolist(),
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
                print(f"\n[tier_proba] Wrote {len(out_csv)} rows (softmax) to {out_p}")
                if mlflow.active_run() is not None:
                    mlflow.log_artifact(str(out_p), artifact_path="tier_proba")

            _plot_text_classification_results(
                y_test,
                y_pred,
                history,
                plot_dir=mlflow_ctx.plot_dir,
                label_name=label_col,
                num_classes=num_classes,
            )

        else:
            model, target_mean, target_std, tier_std_raw, history = _train_text_predictor_arrays(
                train_texts=train_df["text"].tolist(),
                train_targets=train_df[regression_target_col].tolist(),
                val_texts=val_df["text"].tolist(),
                val_targets=val_df[regression_target_col].tolist(),
                pretrained=pretrained,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                freeze_base=freeze_base,
                max_length=max_length,
                batch_size=batch_size,
                lr=lr,
                n_epochs=epochs,
                device=device,
                num_workers=num_workers,
                silent=True,
                huber_beta=huber_beta,
                log_epoch=log_epoch_cb,
                log_prefix="",
            )

            tier_residual_std = (
                float(tier_std_raw)
                if regression_target_col in _GAUSSIAN_TIER_REGRESSION_COLS
                else 0.0
            )
            best_val_loss = float(min(history["val_loss"]))

            test_ds = _TokenizedTextDataset(
                test_df["text"].tolist(),
                test_df[regression_target_col].tolist(),
                tokenizer=model.text_encoder.tokenizer,
                max_length=max_length,
                target_mean=target_mean,
                target_std=target_std,
            )
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kw)

            ckpt_hidden = model.head_in_features

            if save_path:
                ckpt_parent = os.path.dirname(os.path.abspath(save_path))
                if ckpt_parent:
                    os.makedirs(ckpt_parent, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "task": "regression",
                        "num_classes": None,
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "regression_target_col": regression_target_col,
                        "tier_residual_std": tier_residual_std,
                        "star_threshold": float("nan"),
                        "pretrained": pretrained,
                        "output_dim": output_dim,
                        "freeze_base": freeze_base,
                        "max_length": max_length,
                        "hidden_dim": ckpt_hidden,
                        "dropout": dropout,
                        "huber_beta": huber_beta,
                    },
                    save_path,
                )
                print(f"\nCheckpoint saved to {save_path}")

            y_test, y_pred = _predict(
                model,
                test_loader,
                device=device,
                target_mean=target_mean,
                target_std=target_std,
            )
            metrics = _regression_metrics(y_test, y_pred)
            if tier_residual_std > 0.0:
                metrics["tier_residual_std"] = float(tier_residual_std)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("best_val_loss", best_val_loss)
            if tier_residual_std > 0.0:
                mlflow.log_metric("tier_residual_std", float(tier_residual_std))
            mlflow.pytorch.log_model(model, name="model")

            print("\n" + "=" * 40)
            print("Text Model Test Metrics (regression)")
            print("=" * 40)
            print(f"R2   = {metrics['r2']:.4f}")
            print(f"RMSE = {metrics['rmse']:.4f}")
            print(f"MAE  = {metrics['mae']:.4f}")

            if tier_proba_csv_path:
                if (
                    regression_target_col in _GAUSSIAN_TIER_REGRESSION_COLS
                    and tier_residual_std > 0.0
                ):
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
                    print(f"\n[tier_proba] Wrote {len(out_csv)} rows (Gaussian) to {out_p}")
                    if mlflow.active_run() is not None:
                        mlflow.log_artifact(str(out_p), artifact_path="tier_proba")
                else:
                    print(
                        "[tier_proba] Skipped: Gaussian tier export applies only to "
                        "regression_target_col=nba_role_zscore; use task=classification for softmax tier CSV, "
                        "or change target.",
                    )

            _plot_text_results(
                y_test,
                y_pred,
                history,
                plot_dir=mlflow_ctx.plot_dir,
                regression_target_col=regression_target_col,
            )

    return model, metrics


def _plot_text_classification_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: dict[str, list[float]],
    plot_dir: str,
    label_name: str,
    num_classes: int,
) -> None:
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Text model (CE) — {label_name}")
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel("True tier")
    plt.xlabel("Predicted tier")
    plt.tight_layout()
    cm_path = plot_path / "text_model_confusion.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nPlot saved to {cm_path}")
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

    plt.figure(figsize=(7, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="train_loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="val_loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (CE)")
    plt.title("Text Model Loss Curves (classification)")
    plt.legend()
    plt.tight_layout()
    loss_out_path = plot_path / "text_model_loss_curves.png"
    plt.savefig(loss_out_path, dpi=150)
    print(f"Plot saved to {loss_out_path}")
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(loss_out_path), artifact_path="plots")


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
        mlflow.log_artifact(str(out_path), artifact_path="plots")

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
        mlflow.log_artifact(str(loss_out_path), artifact_path="plots")


if __name__ == "__main__":
    train_and_evaluate_text_model()
