"""Dataset classes for NCAA stats + scouting report data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class ProspectStatsDataset(Dataset):
    """Numerical NCAA statistics dataset for lasso regression."""

    def __init__(
        self,
        stats_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> None:
        self.features = torch.tensor(
            stats_df[feature_cols].values, dtype=torch.float32
        )
        self.targets = torch.tensor(
            stats_df[target_col].values, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class ScoutingReportDataset(Dataset):
    """Tokenized scouting report dataset for the NLP layer."""

    def __init__(
        self,
        texts: list[str],
        targets: list[float],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "target": self.targets[idx],
        }


class MultimodalProspectDataset(Dataset):
    """Combined stats + scouting report dataset for the multimodal model."""

    def __init__(
        self,
        stats_df: pd.DataFrame,
        texts: list[str],
        targets: list[float],
        feature_cols: list[str],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.features = torch.tensor(
            stats_df[feature_cols].values, dtype=torch.float32
        )
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict:
        return {
            "stats": self.features[idx],
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "target": self.targets[idx],
        }
