"""
Loads Assessment_TwitterDataset.csv, cleans it, geocodes it, and produces a
deterministic 80/10/10 train/val/test split.

Design notes (see README for the full "why"):
- Interests arrive as a stringified Python-list-like column
  ("'Gaming', 'Travel'"). We parse them into a real list and dedupe, since a
  few rows repeat the same interest twice.
- Age is derived from DOB relative to a fixed reference date so the dataset
  is reproducible regardless of when the pipeline is run.
- Split is done by hashing user_id (not random.sample) so the split is
  stable across runs/machines without needing to persist an index file.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from app.data.geocode import resolve_coordinates
from app.data.schema import UserRecord

REFERENCE_DATE = date(2026, 8, 16)  # fixed so age is reproducible, not "today"


def _parse_interests(raw: str) -> List[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    # values look like: 'Gaming', 'Finance and investments', 'Travel'
    items = re.findall(r"'([^']+)'", raw)
    if not items:
        items = [p.strip() for p in raw.split(",") if p.strip()]
    # dedupe, preserve order
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _parse_age(dob: str) -> float:
    try:
        d = pd.to_datetime(dob).date()
    except Exception:
        return float("nan")
    years = REFERENCE_DATE.year - d.year - ((REFERENCE_DATE.month, REFERENCE_DATE.day) < (d.month, d.day))
    return float(years)


def load_and_clean(csv_path: str) -> List[UserRecord]:
    df = pd.read_csv(csv_path)
    records: List[UserRecord] = []

    for _, row in df.iterrows():
        interests = _parse_interests(row.get("Interests", ""))
        age = _parse_age(row.get("DOB", ""))
        city = str(row.get("City", "") or "")
        country = str(row.get("Country", "") or "")

        coords = resolve_coordinates(city, country)
        if coords is None or not interests or pd.isna(age):
            # Drop rows we can't place on a map or that have no interest
            # signal at all -- both fields are load-bearing for retrieval.
            continue

        lat, lon = coords
        records.append(
            UserRecord(
                user_id=int(row["UserID"]),
                name=str(row.get("Name", "")),
                gender=str(row.get("Gender", "")),
                age=age,
                interests=interests,
                city=city,
                country=country,
                latitude=lat,
                longitude=lon,
            )
        )

    return records


def _split_bucket(user_id: int, train_pct: float = 0.8, val_pct: float = 0.1) -> str:
    """Deterministic 80/10/10 split via hashing -- stable across runs."""
    h = hashlib.md5(str(user_id).encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    if frac < train_pct:
        return "train"
    if frac < train_pct + val_pct:
        return "val"
    return "test"


def split_records(records: List[UserRecord]) -> Tuple[List[UserRecord], List[UserRecord], List[UserRecord]]:
    train, val, test = [], [], []
    for r in records:
        bucket = _split_bucket(r.user_id)
        if bucket == "train":
            train.append(r)
        elif bucket == "val":
            val.append(r)
        else:
            test.append(r)
    return train, val, test


def build_and_cache(csv_path: str, cache_dir: str) -> Tuple[List[UserRecord], List[UserRecord], List[UserRecord]]:
    records = load_and_clean(csv_path)
    train, val, test = split_records(records)

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("val", val), ("test", test)):
        pd.DataFrame([asdict(r) for r in subset]).to_parquet(cache / f"{name}.parquet", index=False)

    return train, val, test


if __name__ == "__main__":
    tr, va, te = build_and_cache("data/Assessment_TwitterDataset.csv", "artifacts/processed")
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
