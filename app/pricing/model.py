"""The fitted price model: predict, interval, comparables, condition adjustment.

Two things this deliberately is not:

  * It is not the vision model's opinion. The VLM never sees a price and never
    emits one. It supplies year/km/brand evidence and a condition read; the
    number comes from a regression over harvested asking prices.
  * It is not a point estimate dressed up with a made-up +/-. The band is the
    empirical 10th-90th percentile of out-of-fold residuals, and the fraction
    of held-out trucks it actually contained is stored in the model file and
    printed on the report.

The condition adjustment is capped at one residual standard deviation, which
is a measured quantity rather than a chosen one: it is the total variation in
asking price that year, kilometres, brand and market do NOT explain. Condition
cannot be worth more than everything we cannot see, so that is the ceiling.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (FX_AS_OF, LISTINGS_CSV, PRICE_MODEL, USD_TRY,
                      euro_norm_for_year)
from ..schema import Comparable, ConditionAdjustment, EvidenceReport, PriceEstimate
from . import features as F

# Severity and stated price impact combine multiplicatively, then scale by the
# model's own confidence in the finding. A low-confidence major defect should
# not move the price like a certain one.
SEVERITY_WEIGHT = {"cosmetic": 0.15, "minor": 0.40, "moderate": 1.00, "major": 2.20}
IMPACT_WEIGHT = {"none": 0.0, "low": 0.40, "medium": 1.00, "high": 2.00}
# Issue-score at which the adjustment reaches ~76% of its cap. Six is roughly
# four confident moderate/medium findings - a visibly tired but working truck.
ADJUSTMENT_SCALE = 6.0


@dataclass
class PriceModel:
    brands: list[str]
    columns: list[str]
    coef: list[float]
    intercept: float
    mean: list[float]
    scale: list[float]
    residual_std: float
    offsets: dict          # interval offsets in log space, per level
    calibration: dict      # measured coverage, MAE, R2
    widening: dict         # measured/assumed multipliers for missing inputs
    meta: dict = field(default_factory=dict)

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("brands", "columns", "coef", "intercept", "mean", "scale",
                 "residual_std", "offsets", "calibration", "widening", "meta")}

    def save(self, path: Path = PRICE_MODEL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = PRICE_MODEL) -> "PriceModel":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run: .venv/bin/python -m app.pricing.train")
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    # --- prediction -------------------------------------------------------
    def predict_log(self, vector: np.ndarray) -> float:
        z = (vector - np.array(self.mean)) / np.array(self.scale)
        return float(np.dot(z, np.array(self.coef)) + self.intercept)

    def contributions(self, vector: np.ndarray) -> list[dict]:
        """Per-feature contribution to the log prediction, for the report.

        Shown so a buyer can see *why* the number is what it is - which is the
        brief's optional "reasoning you can check", made non-optional.
        """
        z = (vector - np.array(self.mean)) / np.array(self.scale)
        parts = z * np.array(self.coef)
        out = []
        for name, value, part in zip(self.columns, vector, parts):
            if name.startswith("brand_") and value == 0.0:
                continue
            if abs(part) < 1e-6:
                continue
            out.append({"feature": name, "value": round(float(value), 4),
                        "log_contribution": round(float(part), 4),
                        "pct_effect": round((math.exp(float(part)) - 1) * 100, 1)})
        return sorted(out, key=lambda d: -abs(d["log_contribution"]))


def load_model(path: Path = PRICE_MODEL) -> PriceModel:
    return PriceModel.load(path)


# --- condition adjustment -------------------------------------------------

def condition_adjustment(evidence: EvidenceReport | None, cap_log: float) -> ConditionAdjustment:
    cap_pct = round((1 - math.exp(-cap_log)) * 100, 1)
    adj = ConditionAdjustment(
        cap_pct=cap_pct,
        basis=(f"capped at one residual standard deviation of the price model "
               f"(+/-{cap_pct}%) - the variation in asking price that year, "
               f"kilometres, brand and market do not explain"),
    )
    if evidence is None or not evidence.issues:
        adj.drivers = ["no visible defects were reported - no adjustment applied"]
        return adj

    score = 0.0
    drivers = []
    for issue in evidence.issues:
        weight = (SEVERITY_WEIGHT.get(issue.severity, 0.4)
                  * IMPACT_WEIGHT.get(issue.price_impact, 0.4)
                  * max(0.1, issue.confidence))
        score += weight
        if weight > 0.25:
            drivers.append(f"{issue.component.replace('_', ' ')} "
                           f"({issue.severity}, {issue.price_impact} impact)")

    adj_log = -cap_log * math.tanh(score / ADJUSTMENT_SCALE)
    adj.multiplier = round(math.exp(adj_log), 4)
    adj.pct = round((adj.multiplier - 1) * 100, 1)
    adj.drivers = drivers[:6] or ["only cosmetic findings - negligible effect"]
    return adj


# --- comparables ----------------------------------------------------------

def find_comparables(df: pd.DataFrame, feats: dict, market: str, k: int = 5) -> list[Comparable]:
    """Nearest priced listings in (age, km) space, same market, brand first.

    The brief asks "what is it comparing against". This is the literal answer:
    the rows the regression was fit on that sit closest to this truck.
    """
    pool = df[df.market.str.upper() == str(market).upper()]
    if pool.empty:
        pool = df
    brand = feats["brand"]
    same_brand = pool[pool.brand == brand]
    scope = "same brand" if len(same_brand) >= k else "all brands in market"
    if len(same_brand) >= k:
        pool = same_brand

    age_sd = float(df.log1p_age.std()) or 1.0
    km_sd = float(df.log_km.std()) or 1.0
    d = np.sqrt(((pool.log1p_age - feats["log1p_age"]) / age_sd) ** 2
                + ((pool.log_km - feats["log_km"]) / km_sd) ** 2)
    pool = pool.assign(comp_dist=d).nsmallest(k, "comp_dist")

    out = []
    for r in pool.itertuples():
        price_try = float(r.price) if str(r.currency).upper() == "TRY" else None
        price_usd = (float(r.price_usd) if not pd.isna(getattr(r, "price_usd", np.nan))
                     else (float(r.price) if str(r.currency).upper() == "USD" else None))
        out.append(Comparable(
            listing_id=str(r.listing_id), source=str(r.source),
            make=str(r.make), model=str(r.model), year=int(r.year), km=float(r.km),
            price=float(r.price), currency=str(r.currency),
            price_try=price_try, price_usd=price_usd,
            url=str(getattr(r, "url", "") or ""),
            distance=round(float(r.comp_dist), 3),
            why=f"{int(r.year)}, {float(r.km) / 1000:.0f}k km, {r.make} ({scope})",
        ))
    return out


# --- the estimate ---------------------------------------------------------

def estimate(model: PriceModel, *, year, km, make, market: str = "TR",
             evidence: EvidenceReport | None = None,
             provenance: dict | None = None,
             extra_widening: list[tuple[str, float]] | None = None,
             level: float = 0.8,
             listings: pd.DataFrame | None = None) -> PriceEstimate:
    est = PriceEstimate(interval_level=level)
    est.currency = "TRY" if str(market).upper() == "TR" else "USD"

    if year is None or km is None:
        est.ok = False
        est.reason = ("no usable year or kilometre reading - neither the seller nor "
                      "the photos supplied one, and age and mileage are most of the price")
        return est

    feats = F.row_features(year, km, make, market)
    vector = F.to_vector(feats, model.brands)
    mu = model.predict_log(vector)

    adj = condition_adjustment(evidence, model.residual_std)
    adj_log = math.log(adj.multiplier) if adj.multiplier > 0 else 0.0

    lo_off, hi_off = model.offsets[str(level)]
    widened: list[str] = []
    factor = 1.0
    if feats["brand"] not in model.brands:
        f = model.widening["unknown_brand"]
        factor *= f
        widened.append(f"{make} is not in the fitted comparables, so the brand term "
                       f"falls back to the market average and the band widens {f:.2f}x "
                       f"(measured by refitting with the brand column removed)")
    for label, f in (extra_widening or []):
        factor *= f
        widened.append(label)

    usd = math.exp(mu + adj_log)
    lo_usd = math.exp(mu + adj_log + lo_off * factor)
    hi_usd = math.exp(mu + adj_log + hi_off * factor)

    rate = USD_TRY if est.currency == "TRY" else 1.0
    est.point, est.low, est.high = round(usd * rate, -3), round(lo_usd * rate, -3), round(hi_usd * rate, -3)
    est.baseline_point = round(math.exp(mu) * rate, -3)
    est.baseline_low = round(math.exp(mu + lo_off * factor) * rate, -3)
    est.baseline_high = round(math.exp(mu + hi_off * factor) * rate, -3)
    est.point_usd, est.low_usd, est.high_usd = round(usd, -2), round(lo_usd, -2), round(hi_usd, -2)
    est.adjustment = adj
    est.widened = widened
    est.drivers = model.contributions(vector)
    est.inputs = {"year": int(year), "km": int(km), "make": make, "market": market,
                  "brand_used": feats["brand"] if feats["brand"] in model.brands else "other",
                  "euro_norm": euro_norm_for_year(int(year))}
    est.inputs_provenance = provenance or {}
    est.model_card = {
        "n_listings": model.meta.get("n_listings"),
        "n_groups": model.meta.get("n_groups"),
        "r2": model.calibration.get("r2_oof"),
        "mae_pct": model.calibration.get("mae_pct_oof"),
        "coverage": model.calibration.get(f"coverage_{level}"),
        "coverage_n": model.calibration.get("coverage_n"),
        "fx": {"usd_try": USD_TRY, "as_of": FX_AS_OF},
        "fitted_at": model.meta.get("fitted_at"),
    }
    est.caveats = [
        "Trained on ASKING prices, not realised sale prices - the honest caveat, "
        "not a hidden one. Expect realised prices to sit below the band.",
        "The measured interval coverage belongs to the comparable-asking band. "
        "The condition-adjusted band is that estimate moved by what the photos "
        "show, which is a deliberate departure from what sellers ask and is not "
        "covered by the same measurement.",
        f"Turkish prices converted at {USD_TRY} TRY/USD as of {FX_AS_OF}; at ~30% "
        "annual inflation this rate goes stale fast - refit rather than quote it later.",
    ]
    if str(market).upper() == "TR":
        est.caveats.append(
            "The Turkish comparables are brand-concentrated ("
            f"{model.meta.get('tr_brand_note', 'one make dominates')}), so an unfamiliar "
            "make is priced off the market average rather than off its own curve.")

    if listings is not None:
        est.comparables = find_comparables(listings, feats, market)
    return est


def price_from_evidence(model: PriceModel, evidence: EvidenceReport | None,
                        declared: dict | None, *, market: str = "TR",
                        listings: pd.DataFrame | None = None,
                        extra_widening: list[tuple[str, float]] | None = None) -> PriceEstimate:
    """Resolve year / km / make from what the seller typed and what the photos show.

    The brief's point is that "sellers leave things out and sometimes lie", so
    the photos have to be able to carry the inputs on their own. Declared
    values win when present because they are checkable against documents, but
    every field records where it came from, and a disagreement between the
    odometer in the photo and the kilometres in the form is surfaced rather
    than quietly resolved.
    """
    declared = declared or {}
    prov: dict = {}
    widening = list(extra_widening or [])

    year = declared.get("year")
    if year:
        prov["year"] = "stated by seller"
    elif evidence and evidence.vehicle.approx_year_range:
        years = [int(y) for y in __import__("re").findall(r"\d{4}",
                 evidence.vehicle.approx_year_range)]
        if years:
            year = int(round(sum(years) / len(years)))
            prov["year"] = (f"estimated from the photos as "
                            f"{evidence.vehicle.approx_year_range} - midpoint used")
            widening.append(("year was read off the truck's generation rather than stated, "
                             "so the band widens 1.35x", 1.35))

    km = declared.get("km")
    if km:
        prov["km"] = "stated by seller"
    elif evidence and evidence.vehicle.odometer_km:
        km = evidence.vehicle.odometer_km
        prov["km"] = f"read from the odometer in photo {evidence.vehicle.odometer_photo_id}"
        widening.append(("kilometres were read off the dashboard rather than stated, "
                         "so the band widens 1.15x", 1.15))

    make = declared.get("make") or (evidence.vehicle.make if evidence else None)
    prov["make"] = "stated by seller" if declared.get("make") else "identified from the photos"

    # The seller's form and the truck's dashboard disagreeing is a finding, not
    # a tie to break silently.
    if (declared.get("km") and evidence and evidence.vehicle.odometer_km
            and declared["km"] > 0):
        delta = abs(evidence.vehicle.odometer_km - declared["km"]) / declared["km"]
        if delta > 0.10:
            prov["km_conflict"] = (
                f"the seller states {int(declared['km']):,} km but the odometer in "
                f"photo {evidence.vehicle.odometer_photo_id} reads "
                f"{evidence.vehicle.odometer_km:,} km "
                f"({delta * 100:.0f}% apart) - priced on the stated figure, but this "
                f"needs documentary support before anyone pays")
            widening.append(("the stated kilometres and the odometer in the photos "
                             "disagree, so the band widens 1.25x", 1.25))

    return estimate(model, year=year, km=km, make=make, market=market,
                    evidence=evidence, provenance=prov, listings=listings,
                    extra_widening=widening)
