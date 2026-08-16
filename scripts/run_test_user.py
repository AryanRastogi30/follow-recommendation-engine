"""
Stage 2: Offline Verification & Pipeline Integration Test.

Builds one dummy "test" user, runs them through the exact same
RecommendationService used by the API (no separate code path), and writes
the ranked output to sample_results.csv. Test user details are also
printed so they can be copied into the README.
"""
from __future__ import annotations

import csv
from pathlib import Path

from app.data.schema import UserRecord
from app.pipeline_service import RecommendationService

TEST_USER = UserRecord(
    user_id=-1,
    name="Rohan Verma",
    gender="Male",
    age=23.0,
    interests=["Technology", "Gaming", "Music", "Fitness"],
    city="Delhi",
    country="India",
    latitude=28.6139,
    longitude=77.2090,
)


def main():
    service = RecommendationService(artifacts_dir="artifacts")
    results = service.recommend(TEST_USER, top_k=10, candidate_pool=100)

    out_path = Path("sample_results.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "user_id", "name", "city", "country",
                                                 "shared_interests", "affinity_score"])
        writer.writeheader()
        for rank, r in enumerate(results, start=1):
            writer.writerow({
                "rank": rank,
                "user_id": r["user_id"],
                "name": r["name"],
                "city": r["city"],
                "country": r["country"],
                "shared_interests": "; ".join(r["shared_interests"]),
                "affinity_score": r["affinity_score"],
            })

    print(f"Test user: {TEST_USER.to_dict()}")
    print(f"Wrote {len(results)} ranked recommendations to {out_path.resolve()}")


if __name__ == "__main__":
    main()
