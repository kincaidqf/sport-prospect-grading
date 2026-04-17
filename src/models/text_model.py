"""NLP encoder for scouting report texts."""
from __future__ import annotations

import torch
import torch.nn as nn


class ScoutingReportEncoder(nn.Module):
    """Fine-tuned transformer encoder that produces a prospect embedding from text."""

    def __init__(
        self,
        pretrained: str = "distilbert-base-uncased",
        output_dim: int = 128,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        # TODO: load pretrained transformer backbone
        # self.backbone = AutoModel.from_pretrained(pretrained)
        # if freeze_base:
        #     for p in self.backbone.parameters():
        #         p.requires_grad = False
        # hidden = self.backbone.config.hidden_size
        # self.head = nn.Sequential(
        #     nn.Linear(hidden, output_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        # )
        self.output_dim = output_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: implement forward pass
        # hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # cls = hidden.last_hidden_state[:, 0, :]   # [CLS] token
        # return self.head(cls)
        raise NotImplementedError
