"""gate -> evidence -> price -> report, with a trace the judges can see.

The ordering is the whole argument. The cheap deterministic check runs first
and can stop the pipeline before a token is spent; the vision model runs
second and is allowed to describe but never to price; the regression runs
third on evidence it did not author. A single VLM call that returned a number
would be faster and would be the thing the brief says will not win.

Every stage records its own elapsed time into `Appraisal.trace`, which is
printed under the report and rendered in the web UI - that is the cheap,
legible version of an observability stack, and it is what makes "more than a
thin wrapper" a claim you can check rather than assert.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import evidence as evidence_stage
from . import gate as gate_stage
from .config import IMAGE_SUFFIXES, LISTINGS_CSV
from .pricing import model as pricing
from .schema import Appraisal, GateDecision, PriceEstimate, TraceStep

# Stated assumption, not a measurement: each missing required view widens the
# band by this factor. Unlike the interval coverage and the unseen-brand
# factor, there is no held-out experiment behind it - a listing in the corpus
# either has the view or the vehicle was dropped, so there is nothing to
# measure it against. Labelled as an assumption everywhere it is surfaced.
MISSING_VIEW_WIDENING = 1.12

# Every comparable in the corpus is a tractor unit (çekici): vehicle type came
# from listing metadata, never from a photo. So a rigid, a tipper or a box truck
# has no comparables here at all, and pricing one off this fit would be a
# confident wrong answer of exactly the kind the brief warns about. The gate
# cannot catch it - COCO calls all of them "truck" - so it is caught here, from
# the body type the vision model read off the vehicle.
PRICEABLE_BODY_TYPES = {"tractor_unit", None, "", "unknown"}
BODY_TYPE_CONFIDENCE = 0.55


def pricing_blocker(ev) -> tuple[str, str] | None:
    """Reasons the comparables cannot honestly price what the photos show.

    Returns (headline, reason) or None. Kept pure and separate from `appraise`
    so both conditions can be tested without spending a vision call.
    """
    if ev is None:
        return None

    if not ev.same_vehicle:
        detail = (ev.vehicle_mismatch or "").strip()
        return (("These photos are not all of the same truck. " + detail).strip(),
                ("these photos are not all the same vehicle, so there is nothing "
                 "coherent to price. " + detail).strip())

    body = (ev.vehicle.body_type or "").strip().lower().replace(" ", "_")
    if body and body not in PRICEABLE_BODY_TYPES and ev.vehicle.confidence >= BODY_TYPE_CONFIDENCE:
        pretty = body.replace("_", " ")
        return (f"This looks like a {pretty}. I can describe its condition, but I have "
                f"no comparable {pretty}s to price it against.",
                f"the photos show a {pretty}, not a tractor unit. Every comparable this "
                f"model was fit on is a tractor unit, so it has nothing honest to price "
                f"a {pretty} against. The condition notes below still stand.")
    return None

_LISTINGS: pd.DataFrame | None = None
_MODEL = None


def listings() -> pd.DataFrame:
    global _LISTINGS
    if _LISTINGS is None:
        from .pricing import features as F
        _LISTINGS = F.build_frame(pd.read_csv(LISTINGS_CSV))
    return _LISTINGS


def price_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = pricing.load_model()
    return _MODEL


def collect_photos(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_dir():
        return sorted(p for p in path.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return [path] if path.is_file() else []


def appraise(photos: list[Path], declared: dict | None = None, *,
             market: str = "TR", backend: str | None = None,
             on_step=None, on_gate=None) -> Appraisal:
    from . import __version__

    t0 = time.time()
    result = Appraisal(declared=declared or {}, version=__version__)
    note = on_step or (lambda *_: None)

    # --- 1. gate ----------------------------------------------------------
    note("gate", "checking the photos")
    gate = gate_stage.run(photos)
    result.gate = gate
    result.trace.append(TraceStep("gate", gate.headline, gate.elapsed_s))
    result.requests = list(gate.requests)

    # The gate is complete ~1 s in and the vision call that follows takes half a
    # minute. Handing the finished report over now lets a caller show the real
    # per-photo detections and view coverage during that wait instead of hiding
    # data it already has. Fired before the refusal return below, so a refusal
    # gets the same evidence - that frame is the whole explanation.
    if on_gate:
        on_gate(gate)

    if gate.decision in (GateDecision.REFUSE_NOT_A_TRUCK, GateDecision.REFUSE_QUALITY,
                         GateDecision.REFUSE_NO_PHOTOS):
        result.status = "refused"
        result.headline = gate.headline
        result.elapsed_s = round(time.time() - t0, 2)
        return result

    # --- 2. evidence ------------------------------------------------------
    note("evidence", f"reading {len(gate.usable_photo_ids)} photos")
    ev = evidence_stage.run(gate, declared, backend=backend)
    result.evidence = ev
    detail = (f"{len(ev.issues)} findings, {len(ev.coverage_gaps)} gaps "
              f"({ev.backend}/{ev.model})")
    if ev.fell_back_from:
        detail += f" — fell back from {len(ev.fell_back_from)} failed backend(s)"
    result.trace.append(TraceStep("evidence", detail, ev.elapsed_s))
    # Deliberately NOT merged into result.requests: the gate's requests are
    # canonical views the seller can go and shoot right now, the evidence's
    # coverage gaps are things that could not be assessed at all. Collapsing
    # them produced a re-ask list with every item in it twice.

    if gate.blocks_pricing:
        result.status = "need_more_photos"
        result.headline = gate.headline
        result.elapsed_s = round(time.time() - t0, 2)
        return result

    # --- 3. price ---------------------------------------------------------
    note("price", "looking up comparables")
    t = time.time()

    blocked = pricing_blocker(ev)
    if blocked:
        headline, reason = blocked
        price = PriceEstimate(ok=False, reason=reason,
                              currency="TRY" if market.upper() == "TR" else "USD")
        price.elapsed_s = round(time.time() - t, 2)
        result.price = price
        result.trace.append(TraceStep("price", f"declined: {reason[:60]}", price.elapsed_s))
        result.status = "need_more_photos"
        result.headline = headline
        if not ev.same_vehicle:
            result.requests.insert(0, "one set of photos of the single truck you are selling")
        result.elapsed_s = round(time.time() - t0, 2)
        return result

    widening = []
    if gate.missing_views:
        pretty = ", ".join(v.replace("_", " ") for v in gate.missing_views)
        widening.append((f"{pretty} {'was' if len(gate.missing_views) == 1 else 'were'} "
                         f"not photographed, so the band widens "
                         f"{MISSING_VIEW_WIDENING ** len(gate.missing_views):.2f}x "
                         f"(a stated assumption, not a measured one)",
                         MISSING_VIEW_WIDENING ** len(gate.missing_views)))
    price = pricing.price_from_evidence(
        price_model(), ev, declared, market=market,
        listings=listings(), extra_widening=widening)
    price.elapsed_s = round(time.time() - t, 2)
    result.price = price
    result.trace.append(TraceStep(
        "price",
        (f"{price.low:,.0f}-{price.high:,.0f} {price.currency}" if price.ok
         else f"declined: {price.reason}"),
        price.elapsed_s))

    # --- headline ---------------------------------------------------------
    if not price.ok:
        result.status = "need_more_photos"
        result.headline = price.reason
    else:
        # The headline deliberately does not call this "the 80% band". The
        # measured coverage belongs to the comparable-asking interval; this
        # number is that interval moved by what the photos show, and labelling
        # it with someone else's measurement is the exact conflation the two
        # separate bands exist to prevent.
        grade = ev.condition_grade if ev else "unknown"
        band = (f"{price.low:,.0f}–{price.high:,.0f} {price.currency}")
        if abs(price.point - price.baseline_point) > 1:
            band += (f" after condition, against "
                     f"{price.baseline_low:,.0f}–{price.baseline_high:,.0f} asked "
                     f"for comparable trucks")
        if gate.decision == GateDecision.ASK_MORE:
            result.status = "ok_with_requests"
            result.headline = f"{band}. {gate.headline}"
        else:
            result.status = "ok"
            result.headline = (f"{band}. Condition {grade}, from "
                               f"{len(gate.usable_photo_ids)} photos.")

    result.elapsed_s = round(time.time() - t0, 2)
    return result
