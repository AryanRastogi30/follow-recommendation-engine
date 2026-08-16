"""
The raw dataset has no real "did user A follow user B" labels -- it's just
a static user directory. To train the ranker at all we construct a proxy
affinity label from interest overlap + geographic proximity (the same two
signals retrieval uses, but combined into a single continuous target
instead of a hand-tuned weighted sum). This is a documented limitation,
not a hidden assumption -- see README "Trade-offs & Constraints".

For each anchor user we sample:
  - `positives_per_user` candidates weighted toward high interest overlap
    (proxy for "would plausibly follow")
  - `negatives_per_user` uniformly random candidates (proxy for "unrelated")
and label them with a continuous proxy score in [0, 1], trained with BCE.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from app.data.schema import UserRecord
from app.ranking.features import FeatureVocab, encode_user


def _jaccard(a: UserRecord, b: UserRecord) -> float:
    sa, sb = {i.lower() for i in a.interests}, {i.lower() for i in b.interests}
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _proxy_label(a: UserRecord, b: UserRecord) -> float:
    from app.retrieval.candidate_generator import _haversine_km
    jac = _jaccard(a, b)
    dist = _haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)
    geo = 1.0 / (1.0 + dist / 500.0)
    return 0.7 * jac + 0.3 * geo


class PairDataset(Dataset):
    def __init__(self, users: List[UserRecord], vocab: FeatureVocab,
                 pairs_per_user: int = 8, seed: int = 42):
        self.vocab = vocab
        rng = random.Random(seed)
        self.pairs: List[Tuple[UserRecord, UserRecord, float]] = []

        for anchor in users:
            others = rng.sample(users, min(len(users), pairs_per_user * 4))
            others = [o for o in others if o.user_id != anchor.user_id][:pairs_per_user]
            for other in others:
                label = _proxy_label(anchor, other)
                self.pairs.append((anchor, other, label))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        anchor, other, label = self.pairs[idx]
        return (
            encode_user(anchor, self.vocab),
            encode_user(other, self.vocab),
            torch.tensor(label, dtype=torch.float32),
        )
