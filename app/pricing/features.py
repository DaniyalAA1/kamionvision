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
    market       TR and US price differently in absolute terms; a shared
                 intercept shift lets both markets inform the age and km
                 slopes without pretending they are one market
    euro6        Euro 5 -> Euro 6 is a step change in this segment. Derived
                 from year when not stated, and always flagged as inferred.
"""
from __future__ import annotations

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
    return {
        "log1p_age": math.log1p(age),
        "log_km": math.log(km),
        "brand": normalise_brand(make),
        "market_tr": 1.0 if str(market).upper() == "TR" else 0.0,
        "euro6": 1.0 if norm == "Euro 6" else 0.0,
    }


def design_columns(brands: list[str]) -> list[str]:
    return ["log1p_age", "log_km", "market_tr", "euro6"] + [f"brand_{b}" for b in brands]


def to_vector(feats: dict, brands: list[str]) -> np.ndarray:
    brand = feats["brand"] if feats["brand"] in brands else "other"
    row = [feats["log1p_age"], feats["log_km"], feats["market_tr"], feats["euro6"]]
    row += [1.0 if brand == b else 0.0 for b in brands]
    return np.array(row, dtype=float)


def build_frame(listings: pd.DataFrame) -> pd.DataFrame:
    """Priced listings -> a tidy frame with features, target and a group key."""
    df = listings[listings.price.notna() & listings.year.notna() & listings.km.notna()].copy()
    df["brand"] = df.make.map(normalise_brand)
    df["log1p_age"] = np.log1p((REF_YEAR - df.year).clip(lower=0))
    df["log_km"] = np.log(df.km.clip(lower=KM_FLOOR))
    df["market_tr"] = (df.market.str.upper() == "TR").astype(float)
    df["euro_norm"] = df.year.map(lambda y: euro_norm_for_year(int(y)))
    df["euro6"] = (df.euro_norm == "Euro 6").astype(float)
    df["y"] = np.log(df.price_usd.where(df.price_usd.notna(), df.price))
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
    rows = []
    for r in df.itertuples():
        brand = r.brand if r.brand in brands else "other"
        row = [r.log1p_age, r.log_km, r.market_tr, r.euro6]
        row += [1.0 if brand == b else 0.0 for b in brands]
        rows.append(row)
    return np.array(rows, dtype=float)
