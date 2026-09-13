"""Feature construction for the hedonic price model.

The design matrix is deliberately small. 155 priced listings does not support
a wide model, and the brief is judged on whether the approach is defensible -
so every column has to earn its place and be explainable in one sentence to
someone who buys trucks for a living.

    log1p(age)   depreciation is convex in age, not linear: the first year
                 costs far more than the tenth
    log(km)      Turkish heavy-truck listings routinely show 1M+ km and a
                 linear term lets that tail drag the fit
    brand        one column per brand with enough listings to estimate, and a
                 shared `other` column that an unseen brand falls into
    market       TR, US and EU price differently in absolute terms; an
                 intercept shift per market lets them all inform the age and
                 km slopes without pretending they are one market
    euro6        Euro 5 -> Euro 6 is a step change in this segment. Stated by
                 the dealer where the source provides it, otherwise derived
                 from year and flagged as inferred.

The target is log price in each listing's OWN currency, with no FX conversion.
That is not laziness: a currency conversion is multiplication by a constant, so
in log space it is an additive constant, which the market dummy absorbs exactly.
Converting first would put a stamped exchange rate inside the fitted model and
make it go stale; this way FX appears only where it belongs, in the display of
a price.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ..config import euro_norm_for_year

REF_YEAR = 2026
MIN_BRAND_N = 5           # listings needed before a brand gets its own column
KM_FLOOR = 1000.0         # guards log(0) on a listing that claims no mileage


def normalise_brand(make: str | None) -> str:
    if not make:
        return "other"
    m = str(make).strip().upper()
    aliases = {
        "FORD TRUCKS": "FORD", "FORD TRUCK": "FORD",
        "MERCEDES": "MERCEDES-BENZ", "MERCEDES BENZ": "MERCEDES-BENZ",
        "WESTERN STAR": "WESTERNSTAR", "INTERNATIONAL HARVESTER": "INTERNATIONAL",
    }
    return aliases.get(m, m)


def row_features(year, km, make, market, euro_norm=None) -> dict:
    age = max(0.0, REF_YEAR - float(year))
    km = max(KM_FLOOR, float(km))
    norm = euro_norm or euro_norm_for_year(int(year) if year else None)
    market = str(market).upper()
    return {
        "log1p_age": math.log1p(age),
        "log_km": math.log(km),
        "brand": normalise_brand(make),
        **{f"market_{m}": 1.0 if market == m.upper() else 0.0 for m in MARKETS},
        "euro6": 1.0 if norm == "Euro 6" else 0.0,
    }


MARKETS = ("tr", "eu")          # US is the reference level


def design_columns(brands: list[str]) -> list[str]:
    return (["log1p_age", "log_km"] + [f"market_{m}" for m in MARKETS] + ["euro6"]
            + [f"brand_{b}" for b in brands])


def to_vector(feats: dict, brands: list[str]) -> np.ndarray:
    brand = feats["brand"] if feats["brand"] in brands else "other"
    row = [feats["log1p_age"], feats["log_km"]]
    row += [feats.get(f"market_{m}", 0.0) for m in MARKETS]
    row.append(feats["euro6"])
    row += [1.0 if brand == b else 0.0 for b in brands]
    return np.array(row, dtype=float)


def load_eu_listings() -> pd.DataFrame:
    """TruckStore's European tractor units, if they have been harvested.

    Kept out of listings.csv on purpose: that file is the photo corpus, and
    these rows carry one thumbnail each, no usable imagery. They exist only as
    price comparables.
    """
    from ..config import META
    path = META / "sources" / "truckstore_eu.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    df = pd.DataFrame(rows)
    df["source_key"] = "truckstore_eu"
    df["price_usd"] = np.nan
    return df


def build_frame(listings: pd.DataFrame, include_eu: bool = False) -> pd.DataFrame:
    """Priced listings -> a tidy frame with features, target and a group key."""
    if include_eu:
        eu = load_eu_listings()
        if not eu.empty:
            listings = pd.concat([listings, eu], ignore_index=True)
    df = listings[listings.price.notna() & listings.year.notna() & listings.km.notna()].copy()
    df["brand"] = df.make.map(normalise_brand)
    df["log1p_age"] = np.log1p((REF_YEAR - df.year).clip(lower=0))
    df["log_km"] = np.log(df.km.clip(lower=KM_FLOOR))
    market = df.market.str.upper()
    for m in MARKETS:
        df[f"market_{m}"] = (market == m.upper()).astype(float)
    # Dealer-stated where the source carries it (TruckStore does), inferred
    # from year where it does not (TruckMarket, SelecTrucks).
    stated = df["euro_norm"] if "euro_norm" in df.columns else pd.Series(index=df.index, dtype=object)
    df["euro_norm"] = stated.where(stated.notna() & (stated != ""),
                                   df.year.map(lambda y: euro_norm_for_year(int(y))))
    df["euro6"] = (df.euro_norm.astype(str).str.strip() == "Euro 6").astype(float)
    # Native currency, no FX: a conversion is an additive constant in log space
    # and the market dummy absorbs it exactly.
    df["y"] = np.log(df.price)
    # Dealer stock is listed in batches: 26 of the Turkish listings are the
    # same 2022 F-MAX at the same asking price. Folds are grouped on this key
    # so an identical spec can never sit on both sides of a split - the same
    # reason the image splits are grouped by vehicle.
    df["group"] = (df.market.astype(str) + "|" + df.brand + "|" + df.model.astype(str)
                   + "|" + df.year.astype(int).astype(str) + "|" + df.price.astype(str))
    return df


def brand_vocabulary(df: pd.DataFrame, min_n: int = MIN_BRAND_N) -> list[str]:
    counts = df.brand.value_counts()
    brands = sorted(b for b, n in counts.items() if n >= min_n)
    return brands + ["other"]


def matrix(df: pd.DataFrame, brands: list[str]) -> np.ndarray:
    cols = ["log1p_age", "log_km"] + [f"market_{m}" for m in MARKETS] + ["euro6"]
    base = df[cols].to_numpy(dtype=float)
    brand = df.brand.where(df.brand.isin(brands), "other")
    dummies = np.stack([(brand == b).to_numpy(dtype=float) for b in brands], axis=1)
    return np.concatenate([base, dummies], axis=1)
