"""Which vehicles the panel grades, and why those.

Deterministic given a seed: the same seed returns the same 60 vehicles and the
same pilot, so a re-run is a re-run and not a new sample. Selection reads only
`data/metadata/listings.csv` and `data/metadata/images.csv` - never a photo -
so nothing about a vehicle's appearance can leak into whether it was chosen.

The strata, and the reason for each:

  TR km deciles    Distance is the strongest thing the rubric conditions on -
                   `EXPECTED_WEAR` is a function of it - so the reference has
                   to cover the whole range or it only tests one band.
  all 6 TR MAN     78 of 84 Turkish vehicles are Ford. A rubric validated only
                   on F-MAX is a rubric validated on one cab. The MAN are a
                   census, not a sample: there are six and all six are in.
  US conventionals The corpus's other body shape, its other market, and the
                   path an unseen brand takes at judging. Sampled across make
                   and km so it is not twenty Cascadias.

Photos-on-disk is an eligibility condition, not a preference: a vehicle whose
frames are missing cannot be graded, and one with very few frames grades a
different question than the rest.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from app import config
from app.evidence import prompts as P
from panel.protocol import RUBRIC_SHA

DATA = Path(__file__).resolve().parent.parent / "data"
LISTINGS = DATA / "metadata" / "listings.csv"
IMAGES = DATA / "metadata" / "images.csv"

TARGET_N = 60
TARGET_TR = 40
TARGET_US = 20
MIN_PHOTOS = 12
SEED = 20260912


@dataclass
class Vehicle:
    listing_id: str
    source_key: str
    market: str
    make: str
    model: str
    year: int | None
    km: int | None
    wear_band: str | None
    n_photos: int
    photos: list[str]
    stratum: str = ""

    def vehicle_line(self) -> str:
        return " ".join(str(x) for x in (self.year, self.make, self.model) if x)


def _int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_vehicles(*, listings: Path = LISTINGS, images: Path = IMAGES,
                  root: Path = DATA, check_disk: bool = True) -> list[Vehicle]:
    """Every corpus vehicle with its ORIGINAL frames, in listing order."""
    photos: dict[tuple[str, str], list[tuple[int, str]]] = {}
    with images.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("variant") != "original":
                continue
            key = (row["source_key"], row["listing_id"])
            photos.setdefault(key, []).append(
                (_int(row["image_index"]) or 0, row["path"]))

    out: list[Vehicle] = []
    with listings.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["source_key"], row["listing_id"])
            paths = [p for _, p in sorted(photos.get(key, []))]
            if check_disk:
                paths = [p for p in paths if (root / p).exists()]
            km = _int(row["km"])
            out.append(Vehicle(
                listing_id=row["listing_id"], source_key=row["source_key"],
                market=row["market"], make=(row["make"] or "").upper(),
                model=row["model"] or "", year=_int(row["year"]), km=km,
                wear_band=P.wear_band(km), n_photos=len(paths),
                photos=[str(root / p) for p in paths]))
    return out


def eligible(vehicles: list[Vehicle]) -> list[Vehicle]:
    return [v for v in vehicles if v.n_photos >= MIN_PHOTOS]


def _decile(values: list[int], value: int) -> int:
    """0-9, by rank within the sorted list. Ties share the lower decile."""
    rank = sorted(values).index(value)
    return min(9, rank * 10 // max(1, len(values)))


def _spread(pool: list[Vehicle], n: int, *, primary, secondary=None,
            rng: random.Random) -> list[Vehicle]:
    """Round-robin over the primary stratum, filling the least-used secondary.

    The bucket order is shuffled with the seeded rng rather than sorted, so
    that when the quota runs out before the buckets do it is the seed that
    decides which strata miss out and not the alphabet. Sorting by name once
    cost the US sample every make after "K".
    """
    buckets: dict[object, list[Vehicle]] = {}
    for v in sorted(pool, key=lambda v: (v.source_key, v.listing_id)):
        buckets.setdefault(primary(v), []).append(v)
    order = sorted(buckets, key=str)
    rng.shuffle(order)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked: list[Vehicle] = []
    used: Counter = Counter()
    while len(picked) < n and any(buckets.values()):
        for key in order:
            if len(picked) >= n:
                break
            bucket = buckets[key]
            if not bucket:
                continue
            if secondary is not None:
                bucket.sort(key=lambda v: used[secondary(v)])
                used[secondary(bucket[0])] += 1
            picked.append(bucket.pop(0))
    return picked


def select_target(vehicles: list[Vehicle] | None = None, *, seed: int = SEED
                  ) -> list[Vehicle]:
    """The 60. All 6 TR MAN, TR Ford across km deciles, US across make x decile."""
    pool = eligible(vehicles if vehicles is not None else load_vehicles())
    rng = random.Random(seed)

    tr = [v for v in pool if v.market == "TR"]
    tr_km = [v.km for v in tr if v.km is not None]
    man = sorted([v for v in tr if v.make == "MAN"], key=lambda v: v.listing_id)
    for v in man:
        v.stratum = "tr_man"
    rest = [v for v in tr if v.make != "MAN"]
    fords = _spread(rest, TARGET_TR - len(man), rng=rng,
                    primary=lambda v: _decile(tr_km, v.km) if v.km is not None else -1,
                    secondary=lambda v: v.year)
    for v in fords:
        v.stratum = f"tr_km_decile_{_decile(tr_km, v.km)}" if v.km is not None else "tr_km_unknown"

    us = [v for v in pool if v.market == "US"]
    us_km = [v.km for v in us if v.km is not None]
    picked_us = _spread(us, TARGET_US, rng=rng,
                        primary=lambda v: v.make or v.model.split()[0].upper(),
                        secondary=lambda v: _decile(us_km, v.km) if v.km is not None else -1)
    for v in picked_us:
        v.stratum = f"us_{v.make.lower()}"

    chosen = man + fords + picked_us
    return sorted(chosen, key=lambda v: (v.market, v.source_key, v.listing_id))


# --- the pilot -------------------------------------------------------------
# Six vehicles, chosen by stated rule rather than by eye, because a pilot
# picked by looking at the photographs is a pilot that already knows its
# answer. "Visibly rough" has no label in this corpus - `damaged` is null in
# every row - so it is approximated by the oldest registration year, which at
# this corpus's distances is also a truck near a million kilometres.

PILOT_RULES = [
    ("tr_lowest_km", "the lowest-km Turkish vehicle"),
    ("tr_low_km_other_year", "the next-lowest-km Turkish vehicle of a different year"),
    ("tr_highest_km", "the highest-km Turkish vehicle"),
    ("tr_high_km_other_make", "the highest-km Turkish vehicle of a different make"),
    ("tr_oldest_proxy_rough", "the oldest Turkish vehicle - the rough-truck proxy, "
                              "since the corpus carries no condition label to pick one by"),
    ("us_highest_km", "the highest-km US conventional"),
]


def select_pilot(target: list[Vehicle] | None = None, *, seed: int = SEED
                 ) -> list[Vehicle]:
    chosen = target if target is not None else select_target(seed=seed)
    tr = [v for v in chosen if v.market == "TR" and v.km is not None]
    us = [v for v in chosen if v.market == "US" and v.km is not None]
    by_km = sorted(tr, key=lambda v: (v.km, v.listing_id))
    picks: list[Vehicle] = []

    def take(v: Vehicle | None, stratum: str) -> None:
        if v is not None and v.listing_id not in {p.listing_id for p in picks}:
            v.stratum = stratum
            picks.append(v)

    take(by_km[0], "tr_lowest_km")
    take(next((v for v in by_km[1:] if v.year != by_km[0].year), None),
         "tr_low_km_other_year")
    high = by_km[-1]
    take(high, "tr_highest_km")
    take(next((v for v in reversed(by_km) if v.make != high.make), None),
         "tr_high_km_other_make")
    take(min(tr, key=lambda v: (v.year or 9999, -(v.km or 0))), "tr_oldest_proxy_rough")
    take(max(us, key=lambda v: (v.km, v.listing_id)) if us else None, "us_highest_km")
    return picks


def manifest(vehicles: list[Vehicle], *, name: str) -> dict:
    return {
        "what": name,
        "seed": SEED,
        "rubric_sha": RUBRIC_SHA,
        "n": len(vehicles),
        "vehicles": [asdict(v) for v in vehicles],
    }


if __name__ == "__main__":
    target = select_target()
    pilot = select_pilot(target)
    for label, rows in (("TARGET 60", target), ("PILOT", pilot)):
        print(f"\n=== {label} ({len(rows)}) ===")
        from collections import Counter
        print(Counter(f"{v.market}/{v.make}" for v in rows))
        print(Counter(v.wear_band for v in rows))
        if len(rows) <= 10:
            for v in rows:
                print(f"  {v.stratum:<24} {v.source_key}/{v.listing_id} "
                      f"{v.vehicle_line():<28} {v.km:>9,} km  {v.n_photos} photos")
