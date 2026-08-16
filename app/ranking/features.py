"""
Turns a UserRecord into fixed-size tensors the two-tower model can consume.

Interests -> multi-hot bag over a fixed vocabulary (small, ~20-40 categories
in this dataset, so multi-hot is cheap and avoids needing a tokenizer).
Age -> min-max normalized.
Gender -> one-hot.
Location -> lat/lon normalized to [-1, 1] (lat/90, lon/180). This is a
simple continuous signal; it does not need to be a true "similarity"
feature since the tower learns the relationship during training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from app.data.schema import UserRecord


class FeatureVocab:
    def __init__(self, interests: List[str], genders: List[str]):
        self.interest_to_idx: Dict[str, int] = {v.lower(): i for i, v in enumerate(sorted(set(interests)))}
        self.gender_to_idx: Dict[str, int] = {v: i for i, v in enumerate(sorted(set(genders)))}

    @property
    def num_interests(self) -> int:
        return len(self.interest_to_idx)

    @property
    def num_genders(self) -> int:
        return max(len(self.gender_to_idx), 1)

    @classmethod
    def from_records(cls, records: List[UserRecord]) -> "FeatureVocab":
        interests = [i for r in records for i in r.interests]
        genders = [r.gender for r in records]
        return cls(interests, genders)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({
            "interest_to_idx": self.interest_to_idx,
            "gender_to_idx": self.gender_to_idx,
        }))

    @classmethod
    def load(cls, path: str) -> "FeatureVocab":
        obj = json.loads(Path(path).read_text())
        v = cls.__new__(cls)
        v.interest_to_idx = obj["interest_to_idx"]
        v.gender_to_idx = obj["gender_to_idx"]
        return v


AGE_MIN, AGE_MAX = 13.0, 90.0


def encode_user(user: UserRecord, vocab: FeatureVocab) -> torch.Tensor:
    interest_vec = torch.zeros(vocab.num_interests)
    for interest in user.interests:
        idx = vocab.interest_to_idx.get(interest.lower())
        if idx is not None:
            interest_vec[idx] = 1.0

    gender_vec = torch.zeros(vocab.num_genders)
    idx = vocab.gender_to_idx.get(user.gender)
    if idx is not None:
        gender_vec[idx] = 1.0

    age_norm = torch.tensor([(min(max(user.age, AGE_MIN), AGE_MAX) - AGE_MIN) / (AGE_MAX - AGE_MIN)])
    geo = torch.tensor([user.latitude / 90.0, user.longitude / 180.0])

    return torch.cat([interest_vec, gender_vec, age_norm, geo])


def feature_dim(vocab: FeatureVocab) -> int:
    return vocab.num_interests + vocab.num_genders + 1 + 2
