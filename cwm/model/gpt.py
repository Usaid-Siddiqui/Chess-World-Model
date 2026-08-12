"""Shared transformer backbone (nanoGPT-style).

One backbone serves both arms:
  * AR baseline runs it **causal** (`is_causal=True`);
  * the JEPA student/teacher run it **bidirectionally** over a visible move prefix.

``forward(..., return_hidden=True)`` returns the per-layer hidden states so the probe
harness can read out the board from any layer. Attention uses
``F.scaled_dot_product_attention`` (flash on CUDA, works on MPS/CPU too).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    ctx: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True


class SelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, is_causal: bool, key_padding_mask: torch.Tensor | None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # True = keep. Build an additive/boolean mask broadcast over heads and queries.
            attn_mask = key_padding_mask[:, None, None, :].expand(B, self.n_head, T, T)
            if is_causal:
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                attn_mask = attn_mask & causal
                is_causal = False  # folded into attn_mask

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, is_causal, key_padding_mask):
        x = x + self.attn(self.ln1(x), is_causal, key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPTBackbone(nn.Module):
    """Token+position embeddings, transformer blocks, final norm. No output head."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.ctx, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        is_causal: bool = True,
        key_padding_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        B, T = input_ids.shape
        assert T <= self.cfg.ctx, f"sequence length {T} exceeds ctx {self.cfg.ctx}"
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :])

        hiddens = []
        for block in self.blocks:
            x = block(x, is_causal, key_padding_mask)
            if return_hidden:
                hiddens.append(x)
        x = self.ln_f(x)
        if return_hidden:
            hiddens.append(x)
            return x, hiddens
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
