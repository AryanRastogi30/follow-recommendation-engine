"""
Loads all artifacts once (user directory, retrieval index, trained ranker)
and exposes a single `recommend(target_user)` call used by both the API
and the offline verification script, so the two never drift apart.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import torch

from app.data.schema import UserRecord
from app.retrieval.candidate_generator import CandidateGenerator
from app.ranking.features import FeatureVocab, encode_user, feature_dim
from app.ranking.model import TwoTowerModel


def _load_all_users(processed_dir: str) -> List[UserRecord]:
    users = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(Path(processed_dir) / f"{split}.parquet")
        for row in df.itertuples(index=False):
            users.append(UserRecord(
                user_id=row.user_id, name=row.name, gender=row.gender, age=row.age,
                interests=list(row.interests), city=row.city, country=row.country,
                latitude=row.latitude, longitude=row.longitude,
            ))
    return users


class RecommendationService:
    def __init__(self, artifacts_dir: str = "artifacts"):
        artifacts = Path(artifacts_dir)
        self.users = _load_all_users(str(artifacts / "processed"))
        self.generator = CandidateGenerator(self.users)

        self.vocab = FeatureVocab.load(str(artifacts / "vocab.json"))
        ckpt = torch.load(artifacts / "two_tower.pt", map_location="cpu")
        self.model = TwoTowerModel(input_dim=ckpt["input_dim"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def recommend(self, target: UserRecord, top_k: int = 10, candidate_pool: int = 100) -> List[dict]:
        candidates = self.generator.generate(target, k=candidate_pool)
        if not candidates:
            return []

        with torch.no_grad():
            user_x = encode_user(target, self.vocab).unsqueeze(0).repeat(len(candidates), 1)
            cand_x = torch.stack([encode_user(c, self.vocab) for c in candidates])
            logits = self.model(user_x, cand_x)
            scores = torch.sigmoid(logits).tolist()

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "user_id": c.user_id,
                "name": c.name,
                "city": c.city,
                "country": c.country,
                "shared_interests": sorted(set(i.lower() for i in c.interests) & set(i.lower() for i in target.interests)),
                "affinity_score": round(score, 4),
            }
            for c, score in ranked
        ]
