"""NLP encoder for scouting report texts."""
from __future__ import annotations

import os
import re
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


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
    """Text-only predictor that maps scouting reports to a scalar target."""

    def __init__(
        self,
        text_encoder: ScoutingReportEncoder,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.head = nn.Sequential(
            nn.Linear(text_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(embeddings).squeeze(-1)

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
        preds = self.head(embeddings.to(device)).squeeze(-1).detach().cpu()
        if was_training:
            self.train()
        return preds


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCOUTING_PATH = os.path.join(PROJECT_ROOT, "data", "scouting", "players.csv")
NBA_PATH = os.path.join(PROJECT_ROOT, "data", "nba", "nba_master.csv")
TARGET = "PLUS_MINUS"

TRAIN_YEARS = range(2009, 2019)
VAL_YEARS = [2019, 2020]
TEST_YEARS = range(2021, 2027)


def _clean_player_name(name: str) -> str:
    """Normalize scouting CSV names like '2 - KJ Simpson' to plain player names."""
    name = str(name).strip()
    name = re.sub(r"^\d+\s*-\s*", "", name)
    return re.sub(r"\s+", " ", name).strip()


def load_text_data() -> pd.DataFrame:
    """Merge scouting reports with NBA outcomes."""
    scouting = pd.read_csv(SCOUTING_PATH)
    nba = pd.read_csv(NBA_PATH)

    scouting["player_name"] = scouting["name"].map(_clean_player_name)
    scouting["draft_year"] = pd.to_numeric(scouting["draft_year"], errors="coerce")
    nba["draft_year"] = pd.to_numeric(nba["draft_year"], errors="coerce")

    scouting = scouting.dropna(subset=["player_name", "draft_year", "full_scouting_report"])
    nba = nba.dropna(subset=["player_name", "draft_year", TARGET])

    merged = scouting.merge(
        nba[["player_name", "draft_year", TARGET]],
        on=["player_name", "draft_year"],
        how="inner",
    )

    merged["text"] = merged["full_scouting_report"].astype(str).str.strip()
    merged = merged[merged["text"] != ""].copy()
    merged = merged.drop_duplicates(subset=["player_name", "draft_year"], keep="first")
    merged[TARGET] = merged[TARGET].astype(float)
    merged["draft_year"] = merged["draft_year"].astype(int)
    return merged


class _TokenizedTextDataset(Dataset):
    """Simple tokenized text dataset for regression targets."""

    def __init__(
        self,
        texts: Sequence[str],
        targets: Sequence[float],
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.targets = torch.tensor(targets, dtype=torch.float32)

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
) -> float:
    """Run one epoch and return mean MSE loss."""
    criterion = nn.MSELoss()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(preds, target)

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
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target"].cpu().numpy()
        pred = model(input_ids=input_ids, attention_mask=attention_mask).cpu().numpy()
        ys.append(target)
        preds.append(pred)
    return np.concatenate(ys), np.concatenate(preds)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_and_evaluate_text_model(
    pretrained: str = "distilbert-base-uncased",
    output_dim: int = 128,
    freeze_base: bool = False,
    max_length: int = 256,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 2e-5,
) -> tuple[TextProspectPredictor, dict[str, float]]:
    """Train/evaluate text-only predictor using scouting report text."""
    df = load_text_data()
    train_df = df[df["draft_year"].isin(TRAIN_YEARS)].copy()
    val_df = df[df["draft_year"].isin(VAL_YEARS)].copy()
    test_df = df[df["draft_year"].isin(TEST_YEARS)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more splits are empty; check year split ranges and data coverage.")

    print(f"\nText dataset: {len(df)} total players")
    print(f"Train: {len(train_df)} players ({min(TRAIN_YEARS)}-{max(TRAIN_YEARS)})")
    print(f"Val:   {len(val_df)} players ({min(VAL_YEARS)}-{max(VAL_YEARS)})")
    print(f"Test:  {len(test_df)} players ({min(TEST_YEARS)}-{max(TEST_YEARS)})")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder = ScoutingReportEncoder(
        pretrained=pretrained,
        output_dim=output_dim,
        freeze_base=freeze_base,
    )
    model = TextProspectPredictor(text_encoder=encoder, hidden_dim=64, dropout=0.2).to(device)

    train_ds = _TokenizedTextDataset(
        train_df["text"].tolist(),
        train_df[TARGET].tolist(),
        tokenizer=encoder.tokenizer,
        max_length=max_length,
    )
    val_ds = _TokenizedTextDataset(
        val_df["text"].tolist(),
        val_df[TARGET].tolist(),
        tokenizer=encoder.tokenizer,
        max_length=max_length,
    )
    test_ds = _TokenizedTextDataset(
        test_df["text"].tolist(),
        test_df[TARGET].tolist(),
        tokenizer=encoder.tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss = _run_epoch(model, val_loader, optimizer=None, device=device)
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, y_pred = _predict(model, test_loader, device=device)
    metrics = _metrics(y_test, y_pred)
    print("\n" + "=" * 40)
    print("Text Model Test Metrics")
    print("=" * 40)
    print(f"R2   = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.4f}")
    print(f"MAE  = {metrics['mae']:.4f}")

    _plot_text_results(y_test, y_pred)
    return model, metrics


def _plot_text_results(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors="none")
    lim = max(abs(float(np.min(y_true))), abs(float(np.max(y_true))), abs(float(np.min(y_pred))), abs(float(np.max(y_pred)))) + 1
    plt.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Perfect prediction")
    plt.xlabel("Actual PLUS_MINUS")
    plt.ylabel("Predicted PLUS_MINUS")
    plt.title("Text Model: Scouting Report to NBA PLUS_MINUS")
    plt.legend(fontsize=8)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "src", "models", "text_model_results.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    train_and_evaluate_text_model()
