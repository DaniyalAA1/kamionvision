"""The fitted price model: predict, interval, comparables, condition adjustment.

Two things this deliberately is not:

  * It is not the vision model's opinion. The VLM never sees a price and never
    emits one. It supplies year/km/brand evidence and a condition read; the
    number comes from a regression over harvested asking prices.
  * It is not a point estimate dressed up with a made-up +/-. The band is the
    empirical 10th-90th percentile of out-of-fold residuals, and the fraction
    of held-out trucks it actually contained is stored in the model file and
    printed on the report.

The condition adjustment is bounded by one out-of-fold residual standard
deviation, and that split is three-way rather than two:

  MEASURED   sigma itself, 0.0988 in log space, from 958 out-of-fold
             evaluations over 84 listings collapsing to 24 distinct specs. It
             is the variation in asking price that year, kilometres, brand and
             market do NOT explain.
  ASSUMED    the decision to cap condition at EXACTLY one sigma. The reasoning
             - condition cannot be worth more than everything we cannot see -
             is sound and no experiment picks 1 sigma over 0.5 or 2.
  ASSUMED    every weight that decides where inside the cap a truck lands. See
             `app.condition`, which owns them and labels them.

The adjustment is two-sided. The comparable baseline is average-condition
asking prices, so a demonstrably clean, well-photographed truck belongs above
it - `strengths` were collected on all sixteen calls, rendered on three
surfaces and worth exactly nothing until now. Two guards keep that honest, and
both live in `app.condition`: merit is bounded by coverage in the arithmetic,
so absence of evidence cannot become evidence of excellence, and one finding
the model itself calls real cancels the premium outright.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import condition as C
from ..condition import IMPACT_WEIGHT, SEVERITY_WEIGHT  # noqa: F401  (re-exported)
from ..config import (FX_AS_OF, LISTINGS_CSV, PRICE_MODEL, USD_TRY,
                      euro_norm_for_year)
from ..schema import (AskingVerdict, Comparable, ConditionAdjustment, EvidenceReport,
                       PriceEstimate)
from . import anchor
from . import features as F

# The weight tables live in `app.condition` with the rest of the rollup, so the
# grade and the price cannot be computed from two different sets of numbers.
SEVERITY_WEIGHT_BASIS = C.WEIGHTS_BASIS

# --- how much wider the condition band is drawn than the comparable one -----
# "Two bands, and they are not interchangeable" was asserted but the two were
# drawn the SAME WIDTH, which quietly implied the condition band inherited the
# measured 80.3%. Drawing it wider when the condition read is uncertain puts
# the disclaimer in the geometry.
#
# ASSUMED: thin coverage widens up to 1.35x. The photographs not showing a
# subsystem is uncertainty about the truck, not a defect on it.
COVERAGE_WIDENING = 0.35
# ASSUMED: the deterministic rollup and the synthesis pass reaching different
# grades is a measurable disagreement between two graders of the same evidence.
GRADE_DISAGREEMENT_WIDENING = {0: 1.00, 1: 1.08, 2: 1.20, 3: 1.20}
# The condition band may never be narrower than the comparable band - the
# mirror of the anchor's floor, and for the same reason: the measured 80.3%
# belongs to the unwidened hedonic interval and nothing here has earned the
# right to claim a tighter one.
CONDITION_WIDENING_FLOOR = 1.0
CONDITION_WIDENING_CAP = 1.50


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
    # The second pricing route: coefficients of log(price / new price) on age
    # and distance, plus its own out-of-fold error. Empty on a model fitted
    # before the reference table existed, which anchor.estimate handles.
    anchor: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("brands", "columns", "coef", "intercept", "mean", "scale",
                 "residual_std", "offsets", "calibration", "widening", "anchor", "meta")}

    def save(self, path: Path = PRICE_MODEL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = PRICE_MODEL) -> "PriceModel":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run: .venv/bin/python -m app.pricing.train")
        d = json.loads(path.read_text(encoding="utf-8"))
        # Tolerate an artifact written before a field existed rather than
        # crashing the demo on a stale models/ directory.
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

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

def rollup_for(evidence: EvidenceReport | None):
    """The condition rollup the grade was taken from, or one computed to match.

    Preferring the stored object is what makes "the grade and the price agree"
    a fact rather than a hope: the evidence stage froze one rollup, graded from
    it, and the price reads the same one.
    """
    if evidence is None:
        return None
    if evidence.condition is not None:
        return evidence.condition
    return C.rollup(evidence.issues, evidence.photo_findings)


def condition_adjustment(evidence: EvidenceReport | None, cap_log: float) -> ConditionAdjustment:
    """Move the comparable baseline by what the photographs actually showed.

    Two-sided, and bounded symmetrically IN PERCENT: the discount floor is
    `exp(-sigma)`, and the premium ceiling is the same magnitude the other way
    rather than `exp(+sigma)`, so the single cap figure printed on every
    surface is true in both directions and the newer, less defensible direction
    is the more conservative one.
    """
    cap_frac = 1 - math.exp(-cap_log)
    cap_pct = round(cap_frac * 100, 1)
    adj = ConditionAdjustment(
        cap_pct=cap_pct,
        cap_basis=(f"measured: one out-of-fold residual standard deviation of the "
                   f"price model (+/-{cap_pct}%) - the variation in asking price "
                   f"that year, kilometres, brand and market do not explain"),
        weights_basis=SEVERITY_WEIGHT_BASIS,
        basis=(f"capped at one residual standard deviation of the price model "
               f"(+/-{cap_pct}%) - the variation in asking price that year, "
               f"kilometres, brand and market do not explain"),
    )
    roll = rollup_for(evidence)
    if roll is None:
        adj.drivers = ["no photographs were read - no adjustment applied"]
        return adj

    issues = list(evidence.issues or [])
    adj.coverage_pct = round(roll.coverage * 100, 1)
    adj.merit_pct = round(roll.merit * 100, 1)
    net = C.net_score(roll, issues)
    adj.multiplier = round(1.0 + cap_frac * math.tanh(net), 4)
    adj.pct = round((adj.multiplier - 1) * 100, 1)
    adj.direction = "premium" if adj.pct > 0 else "discount" if adj.pct < 0 else "none"

    worst = sorted((f for f in roll.families if f.demerit > 0),
                   key=lambda f: -f.demerit)
    drivers = [f"{C.FAMILY_LABEL.get(f.family, f.family)} - {f.note.split(': ', 1)[-1]}"
               for f in worst[:5]]
    if adj.direction == "premium":
        drivers.insert(0, f"{adj.merit_pct:.0f}% of the truck by value was photographed "
                           f"legibly AND positively called sound")
    elif not drivers:
        drivers = ["nothing was found that carries a price - no adjustment applied"]
    adj.drivers = drivers[:6]

    if roll.ungraded_findings:
        adj.notes.append(
            f"{roll.ungraded_findings} finding(s) came back with a severity or price "
            f"impact outside the enum; they are shown with their photo and weighted "
            f"at zero rather than rounded up")
    if roll.grade != "excellent" and roll.merit > 0:
        if C.premium_block(roll, issues) <= 0.0:
            adj.notes.append(
                f"{adj.merit_pct:.0f}% of the truck was affirmed sound, but a finding "
                f"the model itself rates real cancels the premium - a truck with one "
                f"thing badly wrong is not paid a premium for the rest of it being "
                f"clean")
        elif C.coverage_gate(roll.coverage) <= 0.0:
            adj.notes.append(
                f"only {adj.coverage_pct:.0f}% of the truck by value was photographed "
                f"legibly, which is too little to be paid for - absence of evidence is "
                f"not evidence of excellence")
        else:
            adj.notes.append(
                f"{adj.merit_pct:.0f}% of the truck was affirmed sound, but the "
                f"comparable band is what an AVERAGE-condition truck is asked for, and "
                f"only a truck this system will call excellent is priced above it")
    return adj


def condition_widening(model: "PriceModel", evidence: EvidenceReport | None,
                       adj: ConditionAdjustment) -> tuple[float, list[str]]:
    """How much wider than the comparable band the condition band is drawn.

    Floored at 1.0, always: the condition band can never be narrower than the
    comparable-asking band it is derived from.
    """
    roll = rollup_for(evidence)
    if roll is None or (not evidence.issues and roll.coverage <= 0.0):
        return CONDITION_WIDENING_FLOOR, []

    reasons: list[str] = []
    f_cov = 1.0 + COVERAGE_WIDENING * (1.0 - min(1.0, max(0.0, roll.coverage)))
    if f_cov > 1.005:
        reasons.append(f"only {adj.coverage_pct:.0f}% of the truck by value was "
                       f"photographed legibly, so the condition band widens "
                       f"{f_cov:.2f}x (an assumption, not a measurement)")

    f_dis = 1.0
    model_grade = (evidence.condition_grade_model or "").strip()
    if model_grade in C.GRADE_RANK and roll.grade in C.GRADE_RANK:
        steps = abs(C.GRADE_RANK[model_grade] - C.GRADE_RANK[roll.grade])
        f_dis = GRADE_DISAGREEMENT_WIDENING.get(steps, 1.20)
        if f_dis > 1.0:
            reasons.append(f"the synthesis pass graded this {model_grade} and the "
                           f"deterministic rollup grades it {roll.grade}, so the "
                           f"condition band widens {f_dis:.2f}x (an assumption)")

    f_ret = float(model.widening.get("condition_read", 1.0) or 1.0)
    if f_ret > 1.0:
        reasons.append(f"repeat runs of the same photo set move the condition "
                       f"multiplier enough to widen the band {f_ret:.2f}x "
                       f"({model.widening.get('condition_read_basis', 'measured')})")

    factor = min(CONDITION_WIDENING_CAP,
                 max(CONDITION_WIDENING_FLOOR, f_cov * f_dis * f_ret))
    return factor, (reasons if factor > 1.0 else [])


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


def judge_asking_price(asking: float, est: PriceEstimate) -> AskingVerdict:
    """Compare the seller's number to the two bands.

    A listing can be priced exactly like its comparables and still be poor value
    once the photos are read, so both comparisons are reported. The headline
    label is the comparables one, because that is the measured band.
    """
    comp_mid = (est.baseline_low + est.baseline_high) / 2 or est.baseline_point
    v = AskingVerdict(
        asking=round(asking, -3), currency=est.currency,
        vs_comparables_pct=round((asking / comp_mid - 1) * 100, 1) if comp_mid else 0.0,
        vs_estimate_pct=round((asking / est.point - 1) * 100, 1) if est.point else 0.0,
        inside_comparable_band=bool(est.baseline_low <= asking <= est.baseline_high),
    )
    if v.inside_comparable_band:
        v.label = "in line with the market"
        v.summary = ("The asking price sits inside the band that comparable trucks "
                     "of this age and mileage are listed at.")
    elif asking > est.baseline_high:
        v.label = "above the market"
        v.summary = (f"The asking price is {v.vs_comparables_pct:+.0f}% against the middle "
                     f"of the comparable band and sits above its top end.")
    else:
        v.label = "below the market"
        v.summary = (f"The asking price is {v.vs_comparables_pct:+.0f}% against the middle "
                     f"of the comparable band and sits below its bottom end. Worth asking "
                     f"why before treating it as a bargain.")
    if est.point and abs(v.vs_estimate_pct) >= 5:
        direction = "more" if v.vs_estimate_pct > 0 else "less"
        v.summary += (f" Against the condition-adjusted estimate it is "
                      f"{abs(v.vs_estimate_pct):.0f}% {direction}.")
    return v


# --- the estimate ---------------------------------------------------------

def estimate(model: PriceModel, *, year, km, make, market: str = "TR",
             vehicle_model: str | None = None,
             evidence: EvidenceReport | None = None,
             provenance: dict | None = None,
             extra_widening: list[tuple[str, float]] | None = None,
             level: float = 0.8,
             asking_price: float | None = None,
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
    unknown_brand = feats["brand"] not in model.brands
    if unknown_brand:
        f = model.widening["unknown_brand"]
        factor *= f
        widened.append(f"{make} is not in the fitted comparables, so the brand term "
                       f"falls back to the market average and the band widens {f:.2f}x "
                       f"(measured by refitting with the brand column removed)")

    # --- second route: what it cost new, depreciated ----------------------
    # Combined by inverse variance, so neither route is privileged. The whole
    # point is the unseen-brand case: inflating the hedonic variance by the
    # measured 1.85x is what hands the decision to the anchor, and the band
    # tightens back towards 1.0x because two routes now agree where before
    # there was only one that had never seen the brand.
    anchor_est = anchor.estimate(model.anchor, year=year, km=km, make=make,
                                 model=vehicle_model, market=market)
    if anchor_est.ok:
        sd_r = model.residual_std * (model.widening["unknown_brand"] if unknown_brand else 1.0)
        sd_a = anchor.sigma(model.anchor, anchor_est)
        mu, sd_blend, w_anchor = anchor.blend(mu, sd_r, math.log(anchor_est.point), sd_a)
        anchor_est.weight = round(w_anchor, 3)
        # Never narrower than the band whose coverage was actually measured.
        # The 80.3% belongs to the unwidened hedonic interval; claiming a
        # tighter one on the strength of an unmeasured blend would be exactly
        # the overconfidence the two separate bands exist to prevent.
        blended_factor = max(1.0, sd_blend / model.residual_std)
        if blended_factor < factor:
            widened.append(f"the new-price anchor ({anchor_est.matched}) agrees closely "
                           f"enough to bring that back to {blended_factor:.2f}x, and "
                           f"contributes {w_anchor:.0%} of the estimate")
        factor = blended_factor
    est.anchor = anchor_est

    for label, f in (extra_widening or []):
        factor *= f
        widened.append(label)

    # Two factors from here on. `factor` is the SPEC band - unknown brand,
    # anchor blend, missing inputs - and the measured 80.3% coverage stays
    # welded to exactly the interval it was measured on. `factor_cond` is that
    # band widened by how uncertain the condition READ is, and only the
    # condition band uses it. The widening multiplies the offsets; it never
    # touches how they were derived.
    cond_factor, cond_reasons = condition_widening(model, evidence, adj)
    factor_cond = factor * cond_factor
    widened.extend(cond_reasons)

    # The model predicts log price in the NATIVE currency of the market it was
    # fit on - no FX inside the fit, because a conversion is an additive
    # constant in log space that the market dummy absorbs. So exp(mu) is TRY
    # for a Turkish query and USD for an American one, and the only conversion
    # happens here, for display.
    native = math.exp(mu + adj_log)
    lo_native = math.exp(mu + adj_log + lo_off * factor_cond)
    hi_native = math.exp(mu + adj_log + hi_off * factor_cond)
    to_usd = (1.0 / USD_TRY) if est.currency == "TRY" else 1.0

    est.point, est.low, est.high = (round(native, -3), round(lo_native, -3),
                                    round(hi_native, -3))
    # baseline_* is deliberately the estimate BEFORE condition, and it is the
    # band the measured coverage belongs to. It now also includes the anchor,
    # because the anchor is a statement about the vehicle's specification, not
    # about what the photos show - the split this pair protects is
    # spec-vs-condition, not comparables-vs-anchor.
    est.baseline_point = round(math.exp(mu), -3)
    est.baseline_low = round(math.exp(mu + lo_off * factor), -3)
    est.baseline_high = round(math.exp(mu + hi_off * factor), -3)
    est.point_usd = round(native * to_usd, -2)
    est.low_usd = round(lo_native * to_usd, -2)
    est.high_usd = round(hi_native * to_usd, -2)
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

    if asking_price:
        est.asking = judge_asking_price(float(asking_price), est)
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

    # The model name only feeds the new-price lookup, never the hedonic fit.
    # The seller's word wins when given; otherwise the badge the VLM read.
    vehicle_model = declared.get("model") or (evidence.vehicle.model if evidence else None)

    return estimate(model, year=year, km=km, make=make, market=market,
                    vehicle_model=vehicle_model,
                    evidence=evidence, provenance=prov, listings=listings,
                    extra_widening=widening,
                    asking_price=declared.get("asking_price"))
