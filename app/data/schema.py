"""
Typed user record used throughout the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class UserRecord:
    user_id: int
    name: str
    gender: str
    age: float
    interests: List[str]
    city: str
    country: str
    latitude: float
    longitude: float

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "gender": self.gender,
            "age": self.age,
            "interests": self.interests,
            "city": self.city,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }
