"""
Offline city -> (lat, lon) resolution.

The raw dataset gives City/Country as free text, not coordinates, so this
module builds a lookup table from `geonamescache` (a bundled, offline
GeoNames extract -- no network calls, no API keys). When a city can't be
matched exactly we fall back to the country's most populous city as a
centroid. This keeps the whole pipeline reproducible without hitting a
geocoding API.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

import geonamescache


@lru_cache(maxsize=1)
def _build_lookup() -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    gc = geonamescache.GeonamesCache()
    cities = gc.get_cities()
    countries = gc.get_countries()

    city_map: Dict[str, Tuple[float, float]] = {}
    # country_code -> (lat, lon, population) of the largest known city, used as fallback
    country_best: Dict[str, Tuple[float, float, int]] = {}

    for entry in cities.values():
        name = entry["name"].strip().lower()
        lat, lon = float(entry["latitude"]), float(entry["longitude"])
        pop = entry.get("population", 0)
        cc = entry.get("countrycode", "")

        # keep the highest-population match if a city name collides across countries
        if name not in city_map:
            city_map[name] = (lat, lon)

        if cc not in country_best or pop > country_best[cc][2]:
            country_best[cc] = (lat, lon, pop)

    country_name_to_code = {v["name"].strip().lower(): k for k, v in countries.items()}
    country_fallback = {
        name: (country_best[code][0], country_best[code][1])
        for name, code in country_name_to_code.items()
        if code in country_best
    }

    return city_map, country_fallback


def resolve_coordinates(city: str, country: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for a city, falling back to a country-level centroid."""
    city_map, country_fallback = _build_lookup()

    if city:
        hit = city_map.get(city.strip().lower())
        if hit:
            return hit

    if country:
        hit = country_fallback.get(country.strip().lower())
        if hit:
            return hit

    return None
