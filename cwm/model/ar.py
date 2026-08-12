"""Autoregressive (Chess-GPT) arm.

Causal backbone + a linear head over the move vocabulary, trained with next-move
cross-entropy. This is the control and the harness validator: before trusting any JEPA
result, its hidden states must linearly probe to the board at ~99%.

The head returns full-vocab logits, and a legal-move mask is available in ``cwm.moves``,
so a future play module drops in without touching this class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cwm import moves
from cwm.model.gpt import GPTBackbone, GPTConfig


class ARModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = GPTBackbone(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying (standard, saves params and tends to help).
        self.lm_head.weight = self.backbone.tok_emb.weight

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False):
        out = self.backbone(input_ids, is_causal=True, return_hidden=return_hidden)
        if return_hidden:
            x, hiddens = out
            return self.lm_head(x), hiddens
        return self.lm_head(out)

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Next-move cross-entropy. Targets are inputs shifted by one; padding is ignored."""
        logits = self(input_ids)
        # Predict token t+1 from token t.
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:].clone()
        targets[targets == moves.PAD_ID] = -100  # ignore padding positions
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """Per-position hidden states at a given layer (default: final), for probing."""
        _, hiddens = self(input_ids, return_hidden=True)
        return hiddens[layer]
