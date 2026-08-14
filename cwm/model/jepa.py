"""JEPA arm: an action-conditioned latent world model.

I-JEPA with a **temporal** mask, conditioned on the action:

  * **student** encoder ``f_theta`` (causal GPT backbone) maps ``m_1..m_t -> s_t`` — one
    causal pass yields ``s_t`` for every position, and causal masking enforces the
    "action withheld from the student" invariant (position t never sees token t+1);
  * **EMA teacher** ``f_xi`` (stop-grad copy, kept in eval so no dropout) maps
    ``m_1..m_{t+1} -> s~_{t+1}``, the target;
  * **action-conditioned predictor** ``g(s_t, a_t) -> s^_{t+1}`` advances the latent by the
    withheld next move ``a_t = m_{t+1}``, unrolled ``K`` steps (predictions fed back);
  * **loss** is cosine distance to the stop-grad teacher latent — computed in representation
    space, nothing decoded back to moves (pure JEPA, so a probe result is attributable to
    the latent-predictive objective, not a move-prediction head).

Collapse is resisted by the BYOL recipe (predictor asymmetry + EMA teacher + stop-grad),
with a logged ``last_latent_std`` guard and an optional VICReg term as backup.

The student is probed exactly like the AR arm (``hidden_states``), so the two arms are
scored on the same board-reconstruction ruler.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cwm import moves
from cwm.model.gpt import GPTBackbone, GPTConfig


class ActionPredictor(nn.Module):
    """Residual-MLP latent dynamics: (latent, action) -> next latent, in the encoder's
    output space. Action is injected by projection + add; output is LayerNorm'd to match the
    teacher's (post-``ln_f``) target distribution."""

    def __init__(self, dim: int, n_layers: int, mlp_ratio: int):
        super().__init__()
        self.action_proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_ratio * dim),
                nn.GELU(),
                nn.Linear(mlp_ratio * dim, dim),
            )
            for _ in range(n_layers)
        )
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = s + self.action_proj(a)
        for blk in self.blocks:
            h = h + blk(h)
        return self.ln_out(h)


class JEPAModel(nn.Module):
    def __init__(self, model_cfg: GPTConfig, jepa_cfg: dict):
        super().__init__()
        self.cfg = model_cfg
        self.jepa_cfg = dict(jepa_cfg)
        self.ema_decay = float(jepa_cfg.get("ema_decay", 0.999))
        self.K = int(jepa_cfg.get("rollout_steps", 4))
        self.loss_kind = jepa_cfg.get("loss", "cosine")
        self.vicreg_weight = float(jepa_cfg.get("vicreg_weight", 0.0))

        self.backbone = GPTBackbone(model_cfg)  # student f_theta
        self.teacher = GPTBackbone(model_cfg)  # EMA teacher f_xi
        self.teacher.load_state_dict(self.backbone.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.action_emb = nn.Embedding(model_cfg.vocab_size, model_cfg.n_embd)
        self.predictor = ActionPredictor(
            model_cfg.n_embd,
            n_layers=int(jepa_cfg.get("predictor_layers", 2)),
            mlp_ratio=int(jepa_cfg.get("predictor_mlp_ratio", 4)),
        )
        self.last_latent_std = float("nan")

    def train(self, mode: bool = True):
        """Keep the teacher in eval() always — deterministic targets, no dropout."""
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self) -> None:
        d = self.ema_decay
        for pt, ps in zip(self.teacher.parameters(), self.backbone.parameters()):
            pt.mul_(d).add_(ps.detach(), alpha=1.0 - d)

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        K = min(self.K, T - 1)
        s = self.backbone(input_ids, is_causal=True)  # (B,T,C) student, grad
        with torch.no_grad():
            s_tgt = self.teacher(input_ids, is_causal=True)  # (B,T,C) teacher, stop-grad
        a = self.action_emb(input_ids)  # (B,T,C)
        real = input_ids != moves.PAD_ID  # (B,T) True for BOS+moves, False for padding

        n = T - K
        pred = s[:, :n, :]  # start latents s_t, t = 0..n-1
        total, count = 0.0, 0
        for k in range(1, K + 1):
            pred = self.predictor(pred, a[:, k:k + n, :])  # -> s^_{t+k}
            tgt = s_tgt[:, k:k + n, :]                     # teacher s~_{t+k}
            mask = real[:, k:k + n]                        # target position real?
            total = total + self._latent_loss(pred, tgt, mask)
            count += 1
        loss = total / max(count, 1)

        with torch.no_grad():
            flat = s[real]
            self.last_latent_std = flat.std(dim=0).mean().item() if flat.numel() else float("nan")
        if self.vicreg_weight > 0:
            loss = loss + self.vicreg_weight * self._vicreg(s[real])
        return loss

    def _latent_loss(self, pred, tgt, mask) -> torch.Tensor:
        if self.loss_kind == "smooth_l1":
            per = F.smooth_l1_loss(pred, tgt, reduction="none").mean(-1)
        else:  # cosine
            per = 1.0 - F.cosine_similarity(pred, tgt, dim=-1)
        m = mask.float()
        return (per * m).sum() / m.sum().clamp_min(1.0)

    def _vicreg(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(x.var(dim=0) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        xc = x - x.mean(dim=0)
        cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return var_loss + off_diag / x.shape[1]

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """Student hidden states at a layer — the probing interface, identical to ARModel."""
        _, hiddens = self.backbone(input_ids, is_causal=True, return_hidden=True)
        return hiddens[layer]
