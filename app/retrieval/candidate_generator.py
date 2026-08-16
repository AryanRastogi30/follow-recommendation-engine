"""
Stage A: Candidate Generation (Retrieval)

Goal: given a target user, cut the full user base down to ~100 candidates
in-memory, fast, without touching the neural ranker.

Approach
--------
1. Inverted index: interest -> set of user_ids. For the target user we union
   the posting lists of their interests. This turns "scan everyone" into
   "scan only people who share at least one interest", which is the single
   biggest cost reducer since interest overlap is sparse in this dataset
   (most users have 1-5 of ~20 interest categories).
2. Geo bucketing: users are additionally bucketed into ~5-degree lat/lon
   cells. We only keep candidates within the target's cell or its 8
   neighbours, which bounds candidates to roughly the same region of the
   world (a user in Delhi has near-zero chance of being a *good* candidate
   for someone in Lima, and calculating exact haversine distance for the
   full base is unnecessary work).
3. Score survivors of both filters by a weighted blend of interest Jaccard
   similarity and geo proximity (haversine, decayed), take the top N (100).

This is intentionally a plain Python/NumPy structure rather than a vector
DB (FAISS/Annoy) -- at ~25k users the linear scan over the *filtered* set
is a few milliseconds, so a dedicated ANN index would be over-engineering
for this dataset size. The README's scaling section covers what changes at
millions of users.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from app.data.schema import UserRecord

GEO_CELL_SIZE_DEG = 5.0


def _geo_cell(lat: float, lon: float) -> Tuple[int, int]:
    return (int(lat // GEO_CELL_SIZE_DEG), int(lon // GEO_CELL_SIZE_DEG))


def _neighbour_cells(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    cx, cy = cell
    return [(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class CandidateGenerator:
    def __init__(self, users: List[UserRecord]):
        self.users_by_id: Dict[int, UserRecord] = {u.user_id: u for u in users}

        self.interest_index: Dict[str, set] = defaultdict(set)
        self.geo_index: Dict[Tuple[int, int], set] = defaultdict(set)

        for u in users:
            for interest in u.interests:
                self.interest_index[interest.lower()].add(u.user_id)
            self.geo_index[_geo_cell(u.latitude, u.longitude)].add(u.user_id)

    def _interest_candidates(self, user: UserRecord) -> set:
        ids = set()
        for interest in user.interests:
            ids |= self.interest_index.get(interest.lower(), set())
        ids.discard(user.user_id)
        return ids

    def _geo_candidates(self, user: UserRecord) -> set:
        ids = set()
        for cell in _neighbour_cells(_geo_cell(user.latitude, user.longitude)):
            ids |= self.geo_index.get(cell, set())
        ids.discard(user.user_id)
        return ids

    def generate(self, target: UserRecord, k: int = 100,
                 interest_weight: float = 0.6, geo_weight: float = 0.4) -> List[UserRecord]:
        interest_pool = self._interest_candidates(target)
        geo_pool = self._geo_candidates(target)

        # Prefer users satisfying both filters; if that's too small (sparse
        # interests + sparse region), fall back to the union so we still
        # reach k candidates.
        pool = interest_pool & geo_pool
        if len(pool) < k:
            pool = interest_pool | geo_pool
        if len(pool) < k:
            # last resort: everyone (only triggers for tiny/edge-case inputs)
            pool = set(self.users_by_id.keys()) - {target.user_id}

        target_interests = {i.lower() for i in target.interests}
        scored: List[Tuple[float, UserRecord]] = []

        for uid in pool:
            cand = self.users_by_id[uid]
            cand_interests = {i.lower() for i in cand.interests}

            union = target_interests | cand_interests
            jaccard = len(target_interests & cand_interests) / len(union) if union else 0.0

            dist_km = _haversine_km(target.latitude, target.longitude, cand.latitude, cand.longitude)
            geo_score = 1.0 / (1.0 + dist_km / 500.0)  # decays smoothly, ~0.5 at 500km

            score = interest_weight * jaccard + geo_weight * geo_score
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [u for _, u in scored[:k]]
