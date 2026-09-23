"""EscalationNet: relational, causal model for assault anticipation.

Each timestep, people attend to each other with pair features (distance,
closing speed, facing, contact, ...) added as attention biases, so the model
can represent situations like "three people facing one person who is backing
away". A causal GRU carries that scene state forward in time. For online use,
feed one step at a time and pass the returned state back in.

Heads:
    hazard_logits (B, T, K)     discrete-time hazard; bin k covers an onset in
                                (k * bin_s, (k + 1) * bin_s] seconds from now
    phase_logits  (B, T, 4)     calm / precursor / build-up / assault
    assault_logit (B, T)        assault in progress now (auxiliary)
    role_logits   (B, T, N, 2)  per-person target / aggressor scores

Inputs come from social_features.to_model_inputs(). An optional per-person
pose embedding (for example the AGCN encoder output) can be concatenated onto
the person features.

Keep this model only if it beats the Stage 0 baseline in leave-one-dataset-out
evaluation (design doc, section 6). The detection side has already shown that a
learned model can be worth only ~5 points over one hand-made feature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EdgeBiasedAttention(nn.Module):
    """Multi-head self-attention over people with a learned bias per pair."""

    def __init__(self, d: int, heads: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.edge = nn.Sequential(nn.Linear(edge_dim, d), nn.GELU(), nn.Linear(d, heads))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x (B, N, d), e (B, N, N, E), mask (B, N) with True = person present
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dk).unbind(2)
        att = torch.einsum("bihd,bjhd->bhij", q, k) / self.dk**0.5
        att = att + self.edge(e).permute(0, 3, 1, 2)
        # Large finite negative instead of -inf: rows with nobody present stay
        # finite (uniform) instead of producing NaN gradients; they are masked later.
        att = att.masked_fill(~mask[:, None, None, :], torch.finfo(att.dtype).min)
        att = self.drop(torch.softmax(att, dim=-1))
        y = torch.einsum("bhij,bjhd->bihd", att, v).reshape(B, N, -1)
        return self.out(y)


class RelationalBlock(nn.Module):
    def __init__(self, d: int, heads: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = EdgeBiasedAttention(d, heads, edge_dim, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.att(self.n1(x), e, mask)
        x = x + self.ff(self.n2(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


class EscalationNet(nn.Module):
    def __init__(
        self,
        person_dim: int,
        edge_dim: int,
        scene_dim: int,
        d: int = 64,
        heads: int = 4,
        layers: int = 2,
        n_bins: int = 10,
        n_phases: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.person_in = nn.Sequential(nn.Linear(person_dim, d), nn.GELU())
        self.blocks = nn.ModuleList(
            RelationalBlock(d, heads, edge_dim, dropout) for _ in range(layers)
        )
        self.pool_q = nn.Parameter(torch.randn(d) / d**0.5)
        self.scene_in = nn.Linear(scene_dim, d)
        self.gru = nn.GRU(3 * d, d, batch_first=True)  # forward in time only: causal
        self.hazard = nn.Linear(d, n_bins)
        self.phase = nn.Linear(d, n_phases)
        self.assault = nn.Linear(d, 1)
        self.role = nn.Linear(2 * d, 2)

    def forward(
        self,
        person: torch.Tensor,  # (B, T, N, P)
        edge: torch.Tensor,  # (B, T, N, N, E)
        scene: torch.Tensor,  # (B, T, S)
        mask: torch.Tensor,  # (B, T, N) bool
        state: torch.Tensor | None = None,
    ) -> dict:
        B, T, N, _ = person.shape
        m = mask.reshape(B * T, N).bool()
        x = self.person_in(person).reshape(B * T, N, -1)
        e = edge.reshape(B * T, N, N, -1)
        for blk in self.blocks:
            x = blk(x, e, m)

        neg = torch.finfo(x.dtype).min
        w = torch.softmax((x @ self.pool_q).masked_fill(~m, neg), dim=-1)
        att_pool = (w.unsqueeze(-1) * x).sum(1)
        max_pool = x.masked_fill(~m.unsqueeze(-1), neg).amax(1)
        max_pool = torch.where(m.any(1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        z = torch.cat([att_pool, max_pool, self.scene_in(scene.reshape(B * T, -1))], -1).reshape(
            B, T, -1
        )

        ctx, state = self.gru(z, state)  # (B, T, d)
        per = torch.cat([x.reshape(B, T, N, -1), ctx.unsqueeze(2).expand(-1, -1, N, -1)], -1)
        return {
            "hazard_logits": self.hazard(ctx),
            "phase_logits": self.phase(ctx),
            "assault_logit": self.assault(ctx).squeeze(-1),
            "role_logits": self.role(per),
            "state": state,
        }


# ------------------------------------------------------------------ losses


def within_horizon_prob(hazard_logits: torch.Tensor, bins: int) -> torch.Tensor:
    """P(onset within the first `bins` hazard bins) = 1 - prod_k (1 - h_k)."""
    return 1.0 - torch.exp(F.logsigmoid(-hazard_logits[..., :bins]).sum(-1))


def hazard_nll(
    hazard_logits: torch.Tensor,  # (n, K)
    event: torch.Tensor,  # (n,) 1 = onset inside the horizon
    bin_idx: torch.Tensor,  # (n,) onset bin for events (ignored when censored)
    weight: torch.Tensor,  # (n,)
) -> torch.Tensor:
    """Discrete-time survival NLL with right-censoring at the horizon."""
    K = hazard_logits.shape[-1]
    log_h = F.logsigmoid(hazard_logits)  # log h_k
    log_s = F.logsigmoid(-hazard_logits)  # log (1 - h_k)
    ar = torch.arange(K, device=hazard_logits.device)
    b = bin_idx.long().clamp(max=K - 1).unsqueeze(-1)
    before = (ar < b).to(hazard_logits.dtype)
    at = (ar == b).to(hazard_logits.dtype)
    ll_event = (log_s * before).sum(-1) + (log_h * at).sum(-1)
    ll_censored = log_s.sum(-1)
    ll = torch.where(event.bool(), ll_event, ll_censored)
    weight = weight.to(ll.dtype)
    return -(ll * weight).sum() / weight.sum().clamp(min=1e-6)


def exp_anticipation_loss(
    logit: torch.Tensor,  # (n,) logit of P(onset within the horizon)
    tau_s: torch.Tensor,  # (n,) seconds to onset (inf if none)
    event: torch.Tensor,
    weight: torch.Tensor,
    decay_s: float = 2.0,
) -> torch.Tensor:
    """Accident-anticipation style exponential loss, for comparison with hazard_nll.

    Positive windows are weighted by exp(-tau / decay_s): misses close to the
    onset cost most, very early windows (where evidence may not be visible yet)
    cost less. Censored windows get plain cross-entropy.
    """
    pos_w = torch.exp(-tau_s.clamp(min=0.0) / decay_s)
    loss = torch.where(event.bool(), -pos_w * F.logsigmoid(logit), -F.logsigmoid(-logit))
    weight = weight.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp(min=1e-6)


def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0
) -> torch.Tensor:
    """Match per-bin hazards of a teacher that saw future context.

    Train the teacher first (e.g. same network with a bidirectional GRU and a
    few seconds of future frames); the causal student then learns to predict
    what the teacher can already see.
    """
    target = torch.sigmoid(teacher_logits.detach() / temperature)
    return F.binary_cross_entropy_with_logits(student_logits / temperature, target)


def escalation_loss(
    out: dict,
    batch: dict,
    w_hazard: float = 1.0,
    w_assault: float = 0.5,
    w_phase: float = 0.3,
) -> dict:
    """Combined training loss.

    batch tensors, all (B, T): event (long), bin (long), weight (float),
    in_assault (bool), phase (long, -1 = unannotated), valid (bool, a label exists).
    """
    valid = batch["valid"]
    zero = out["hazard_logits"].sum() * 0.0
    haz = valid & ~batch["in_assault"] & (batch["weight"] > 0)
    labelled = valid & (batch["phase"] >= 0)
    parts = {
        "hazard": hazard_nll(
            out["hazard_logits"][haz],
            batch["event"][haz],
            batch["bin"][haz],
            batch["weight"][haz],
        )
        if haz.any()
        else zero,
        "assault": F.binary_cross_entropy_with_logits(
            out["assault_logit"][valid], batch["in_assault"][valid].float()
        )
        if valid.any()
        else zero,
        "phase": F.cross_entropy(out["phase_logits"][labelled], batch["phase"][labelled])
        if labelled.any()
        else zero,
    }
    parts["total"] = (
        w_hazard * parts["hazard"] + w_assault * parts["assault"] + w_phase * parts["phase"]
    )
    return parts
