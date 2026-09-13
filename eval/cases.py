"""Which trucks and which photographs a run looks at, decided by a seed.

Two jobs. The first is vehicle-level stratified sampling over the corpus -
market, capture-quality bucket and kilometre tercile - so a 40-vehicle standard
tier is not 40 Ford F-MAXes off one dealer's lot. The second is twin pairing:
`data/metadata/images.csv` carries the original and its degraded twin as two
rows sharing `(listing_id, image_index)`, so the pair needs no join and no
model, only a seed and a rule.

Sampling is grouped by vehicle everywhere, never by image, for the reason the
dataset pipeline already encodes: one tractor contributes 15-40 near-identical
frames, and treating those as independent observations inflates n by an order
of magnitude and shrinks every interval to match.

Determinism is not a nicety here - a scorecard whose sample moved is not
comparable to the one before it, and `--diff` would report the resampling as a
result. Every draw goes through `random.Random(seed)` over a list that was
sorted first, so the same seed on the same corpus gives the same trucks on any
machine.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import DATA, IMAGES_CSV, LISTINGS_CSV

# Asked for on every vehicle that has them, before any diversity fill. These
# three carry most of the money: tread depth is the single most expensive
# consumable on a tractor unit, the frame decides whether it is worth buying at
# all, and the dashboard is the only checkable number in the set.
FORCE_VIEWS = ("tire_wheel", "chassis_undercarriage", "dashboard_odometer")

# Vehicles below this many distinct views cannot support a view-stratified
# sample, and a vehicle photographed from one angle is not an appraisal example.
MIN_VIEWS = 6


@lru_cache(maxsize=2)
def load_images(path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(Path(path or IMAGES_CSV), low_memory=False)
    df["listing_id"] = df["listing_id"].astype(str)
    return df


@lru_cache(maxsize=2)
def load_listings(path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(Path(path or LISTINGS_CSV), low_memory=False)
    df["listing_id"] = df["listing_id"].astype(str)
    return df


def full_path(rel: str) -> Path:
    """`images/<source>/<listing>/NNN.jpg` as it sits under data/."""
    return DATA / str(rel)


def _quantile_band(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
    """Rank-based bands that survive ties and missing values.

    `pd.qcut` raises on duplicate edges, and km in this corpus has them: the
    Mascus rows repeat round numbers. Ranking first makes the bands equal-count
    by construction instead of equal-width-with-exceptions.
    """
    ranked = series.rank(method="first", na_option="keep")
    valid = ranked.notna()
    out = pd.Series(["unknown"] * len(series), index=series.index, dtype=object)
    if valid.sum() >= q:
        binned = pd.qcut(ranked[valid], q=q, labels=labels)
        out.loc[valid] = binned.astype(str)
    return out


@lru_cache(maxsize=4)
def vehicles(images_csv: str | None = None) -> pd.DataFrame:
    """One row per vehicle, with the columns the strata are built from.

    Derived from images.csv rather than listings.csv because the view counts,
    the capture-quality bucket and the twin severities only exist per photo -
    and `n_images_kept` in listings.csv predates `filter_dataset.py`.
    """
    df = load_images(images_csv)
    originals = df[df["variant"] == "original"]
    rows = []
    for listing_id, group in originals.groupby("listing_id", sort=True):
        first = group.iloc[0]
        views = sorted(set(group["view"].dropna().astype(str)))
        buckets = group["quality_bucket"].dropna().astype(str)
        rows.append({
            "listing_id": str(listing_id),
            "source_key": str(first.get("source_key", "")),
            "market": str(first.get("market", "")),
            "make": str(first.get("make", "")),
            "model": str(first.get("model", "")),
            "year": first.get("year"),
            "km": first.get("km"),
            "price": first.get("price"),
            "currency": str(first.get("currency", "")),
            "split": str(first.get("split", "")),
            "n_photos": int(len(group)),
            "n_views": len(views),
            "views": "|".join(views),
            # Modal bucket: the label gallery.py uses for the whole set, so a
            # stratum here means the same thing it means on screen.
            "quality_bucket": (buckets.mode().iloc[0] if not buckets.empty else "unknown"),
            "mean_capture_quality": float(
                pd.to_numeric(group["capture_quality"], errors="coerce").mean()),
        })
    out = pd.DataFrame(rows).sort_values("listing_id").reset_index(drop=True)
    # Deciles and terciles are computed WITHIN market: 116 US trucks in miles-
    # derived km and 84 Turkish ones do not share a distribution, and a pooled
    # tercile would just re-encode the market dummy.
    out["km_decile"] = "unknown"
    out["km_tercile"] = "unknown"
    for market, idx in out.groupby("market").groups.items():
        km = pd.to_numeric(out.loc[idx, "km"], errors="coerce")
        out.loc[idx, "km_decile"] = _quantile_band(km, 10, [f"d{i}" for i in range(1, 11)])
        out.loc[idx, "km_tercile"] = _quantile_band(km, 3, ["low", "mid", "high"])
    return out


# --- allocation ------------------------------------------------------------

def allocate(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Largest-remainder apportionment. Deterministic, and it sums to `total`.

    Proportional rounding by `round()` does not sum to the target and drifts
    with dict order; this does neither, and ties break on the stratum name so
    two machines agree.
    """
    pool = sum(weights.values())
    if pool <= 0 or total <= 0:
        return {k: 0 for k in weights}
    exact = {k: total * v / pool for k, v in weights.items()}
    floor = {k: int(v) for k, v in exact.items()}
    left = total - sum(floor.values())
    order = sorted(weights, key=lambda k: (-(exact[k] - floor[k]), k))
    for k in order[:left]:
        floor[k] += 1
    # Never allocate more of a stratum than it holds; spill the excess to
    # whichever stratum still has room, in name order.
    for k in sorted(floor):
        over = floor[k] - weights[k]
        if over > 0:
            floor[k] = weights[k]
            for other in sorted(floor):
                room = weights[other] - floor[other]
                if room > 0:
                    take = min(room, over)
                    floor[other] += take
                    over -= take
                if over <= 0:
                    break
    return floor


def sample_vehicles(n: int, *, seed: int = 7, min_views: int = MIN_VIEWS,
                    strata: tuple[str, ...] = ("market",),
                    images_csv: str | None = None) -> list[str]:
    """`n` listing_ids, proportionally stratified, deterministic given the seed.

    `strata` names columns of `vehicles()`. The default is market alone because
    at n = 40 a market x bucket x tercile cross has 18 cells and most of them
    hold one truck; the distribution suite asks for the full cross and gets it
    by passing the wider tuple, with the understanding that the per-cell n is
    then too small for a per-cell interval.
    """
    df = vehicles(images_csv)
    pool = df[df["n_views"] >= min_views]
    if pool.empty:
        pool = df
    pool = pool.sort_values("listing_id")
    groups: dict[str, list[str]] = {}
    for _, row in pool.iterrows():
        key = "|".join(str(row[s]) for s in strata)
        groups.setdefault(key, []).append(str(row["listing_id"]))

    quota = allocate(min(n, len(pool)), {k: len(v) for k, v in groups.items()})
    rng = random.Random(seed)
    picked: list[str] = []
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        picked.extend(ids[:quota.get(key, 0)])
    return sorted(picked)


# --- twin pairs ------------------------------------------------------------

@dataclass
class TwinPair:
    """One photograph and its degraded twin. The unit of the twin_fp suite.

    `view` is the ORIGINAL's tag, carried on both halves deliberately: the
    zero-shot view classifier moves under degradation, and a prompt that
    differs between the two members turns the measurement into a comparison of
    prompts. Every other confound is killed the same way, in `twin_fp`.
    """
    listing_id: str
    image_index: int
    view: str
    market: str
    make: str = ""
    model_name: str = ""      # not `model`: every other module in this repo
                              # means a fitted estimator by that name
    year: int | None = None
    km: float | None = None
    quality_bucket: str = ""
    severity: float = 0.0
    severity_band: str = ""
    degradations: list[str] = field(default_factory=list)
    original_rel: str = ""
    twin_rel: str = ""

    @property
    def unit(self) -> str:
        return f"{self.listing_id}:{self.image_index:03d}"

    def path(self, variant: str) -> Path:
        return full_path(self.original_rel if variant == "original" else self.twin_rel)

    def exists(self) -> bool:
        return self.path("original").exists() and self.path("degraded").exists()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unit"] = self.unit
        return d


def _severity_bands(df: pd.DataFrame) -> pd.Series:
    return _quantile_band(pd.to_numeric(df["severity"], errors="coerce"), 3,
                          ["light", "medium", "heavy"])


def twin_pairs(listing_ids: list[str] | None = None,
               images_csv: str | None = None) -> list[TwinPair]:
    """Every (original, twin) pair in the corpus, or in the named vehicles.

    Pairing is on `(listing_id, image_index)` and nothing else. `clean_path` on
    the degraded row equals the original's `path` in 3,729 of 3,729 rows, which
    is a second, independent check on the same pairing - but it is a check, not
    the key, because a key that reads a path string breaks the first time
    anything is moved on disk.
    """
    df = load_images(images_csv)
    if listing_ids is not None:
        keep = {str(x) for x in listing_ids}
        df = df[df["listing_id"].isin(keep)]
    originals = df[df["variant"] == "original"].set_index(["listing_id", "image_index"])
    degraded = df[df["variant"] == "degraded"].copy()
    degraded["severity_band"] = _severity_bands(degraded)
    out: list[TwinPair] = []
    for _, twin in degraded.sort_values(["listing_id", "image_index"]).iterrows():
        key = (twin["listing_id"], twin["image_index"])
        if key not in originals.index:
            continue
        original = originals.loc[key]
        if isinstance(original, pd.DataFrame):      # duplicate index, take the first
            original = original.iloc[0]
        km = pd.to_numeric(pd.Series([original.get("km")]), errors="coerce").iloc[0]
        year = pd.to_numeric(pd.Series([original.get("year")]), errors="coerce").iloc[0]
        out.append(TwinPair(
            listing_id=str(twin["listing_id"]),
            image_index=int(twin["image_index"]),
            view=str(original.get("view") or "unknown"),
            market=str(original.get("market") or ""),
            make=str(original.get("make") or ""),
            model_name=str(original.get("model") or ""),
            year=None if pd.isna(year) else int(year),
            km=None if pd.isna(km) else float(km),
            quality_bucket=str(original.get("quality_bucket") or "unknown"),
            severity=float(twin.get("severity") or 0.0),
            severity_band=str(twin.get("severity_band") or "unknown"),
            degradations=[a for a in str(twin.get("degradations") or "").split("|") if a],
            original_rel=str(original.get("path") or ""),
            twin_rel=str(twin.get("path") or "")))
    return out


def sample_twin_pairs(n_vehicles: int, per_vehicle: int, *, seed: int = 7,
                      min_views: int = MIN_VIEWS,
                      images_csv: str | None = None,
                      require_files: bool = True) -> list[TwinPair]:
    """The twin_fp sample: stratified vehicles, then view-forced, severity-balanced.

    Per vehicle: take `FORCE_VIEWS` where present, fill the rest by view
    diversity (one slot per view before any view gets a second), and among
    equally-diverse candidates prefer the severity band that is currently
    thinnest across the whole sample. The last clause is what keeps the
    dose-response curve inside the suite from being three points on one tercile
    - which is the only part of the twin suite that can speak to monotonicity
    with a stimulus whose ordering is known by construction.
    """
    chosen = sample_vehicles(n_vehicles, seed=seed, min_views=min_views,
                             images_csv=images_csv)
    by_vehicle: dict[str, list[TwinPair]] = {}
    for pair in twin_pairs(chosen, images_csv=images_csv):
        if require_files and not pair.exists():
            continue
        by_vehicle.setdefault(pair.listing_id, []).append(pair)

    rng = random.Random(seed + 1)
    band_count: dict[str, int] = {}
    picked: list[TwinPair] = []
    for listing_id in sorted(by_vehicle):
        candidates = sorted(by_vehicle[listing_id], key=lambda p: p.image_index)
        rng.shuffle(candidates)
        by_view: dict[str, list[TwinPair]] = {}
        for pair in candidates:
            by_view.setdefault(pair.view, []).append(pair)
        # Thinnest band first inside a view, so the fill balances severity.
        for bucket in by_view.values():
            bucket.sort(key=lambda p: (band_count.get(p.severity_band, 0), p.image_index))

        order = [v for v in FORCE_VIEWS if v in by_view] + \
                sorted(v for v in by_view if v not in FORCE_VIEWS)
        taken: list[TwinPair] = []
        while len(taken) < per_vehicle:
            progressed = False
            for view in order:
                bucket = by_view.get(view) or []
                if not bucket:
                    continue
                pair = bucket.pop(0)
                taken.append(pair)
                band_count[pair.severity_band] = band_count.get(pair.severity_band, 0) + 1
                progressed = True
                if len(taken) >= per_vehicle:
                    break
            if not progressed:
                break
        picked.extend(taken)
    return sorted(picked, key=lambda p: (p.listing_id, p.image_index))


def describe(pairs: list[TwinPair]) -> dict:
    """The sampling block that goes in the scorecard - what was actually drawn."""
    def tally(key):
        out: dict[str, int] = {}
        for p in pairs:
            out[str(key(p))] = out.get(str(key(p)), 0) + 1
        return dict(sorted(out.items()))
    return {"pairs": len(pairs),
            "vehicles": len({p.listing_id for p in pairs}),
            "by_market": tally(lambda p: p.market),
            "by_view": tally(lambda p: p.view),
            "by_severity_band": tally(lambda p: p.severity_band)}
