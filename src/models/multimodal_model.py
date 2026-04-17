"""Multimodal model fusing NCAA stats (lasso features) + scouting text embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn


class MultimodalProspectModel(nn.Module):
    """Fuses numerical stat features with scouting report embeddings to predict NBA performance."""

    def __init__(
        self,
        stats_input_dim: int,
        text_encoder: nn.Module,
        fusion: str = "concat",   # concat | attention | gate
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.fusion = fusion

        # TODO: define fusion layers based on self.fusion strategy
        # concat:    combined_dim = stats_input_dim + text_encoder.output_dim
        # attention: cross-attention between stat and text representations
        # gate:      learned gating scalar per modality

        # self.stats_proj = nn.Linear(stats_input_dim, hidden_dim)
        # self.fusion_layer = ...
        # self.head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim // 2, 1),
        # )

    def forward(
        self,
        stats: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: implement forward pass
        raise NotImplementedError
