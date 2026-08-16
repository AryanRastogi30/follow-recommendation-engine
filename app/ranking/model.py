"""
Stage B: Neural Ranking (Scoring)

Two-tower architecture: one MLP tower encodes the target user, a second
(weight-sharing) tower encodes each candidate, and affinity is the dot
product of the two resulting embeddings, squashed to [0, 1] with a sigmoid.

Why two-tower over a single concatenated MLP:
- Candidate embeddings can be precomputed once per (re)training cycle and
  reused across every request -- at serving time we only run the target
  user through the tower once and do a batched dot-product against
  precomputed candidate embeddings, which is much cheaper than re-running
  a full MLP over every (target, candidate) pair on every request.
- It mirrors how this would actually scale (see README Stage 4): the
  candidate tower's output is exactly what you'd put behind an ANN index.

Weights are shared between the two towers since "user" and "candidate" are
drawn from the same feature space/distribution here -- there's no inherent
asymmetry (unlike, say, query vs. document towers in search).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.tower = Tower(input_dim, hidden_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(5.0))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(x)

    def forward(self, user_x: torch.Tensor, cand_x: torch.Tensor) -> torch.Tensor:
        """user_x: (B, D), cand_x: (B, D) -> affinity logits (B,)"""
        u = self.embed(user_x)
        c = self.embed(cand_x)
        sim = (u * c).sum(dim=-1) * self.logit_scale
        return sim  # raw logits; apply sigmoid/BCEWithLogits outside
