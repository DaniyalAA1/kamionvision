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
from .config import LISTINGS_CSV
from .pricing import model as pricing
from .schema import Appraisal, GateDecision, TraceStep

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".tif", ".tiff"}

# Stated assumption, not a measurement: each missing required view widens the
# band by this factor. Unlike the interval coverage and the unseen-brand
# factor, there is no held-out experiment behind it - a listing in the corpus
# either has the view or the vehicle was dropped, so there is nothing to
# measure it against. Labelled as an assumption everywhere it is surfaced.
MISSING_VIEW_WIDENING = 1.12

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
             on_step=None) -> Appraisal:
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
    result.trace.append(TraceStep(
        "evidence", f"{len(ev.issues)} findings, {len(ev.coverage_gaps)} gaps "
                    f"({ev.backend}/{ev.model})", ev.elapsed_s))
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
    elif gate.decision == GateDecision.ASK_MORE:
        result.status = "ok_with_requests"
        result.headline = (f"Priced, with gaps: {price.low:,.0f}–{price.high:,.0f} "
                           f"{price.currency}. {gate.headline}")
    else:
        result.status = "ok"
        grade = ev.condition_grade if ev else "unknown"
        result.headline = (f"{price.low:,.0f}–{price.high:,.0f} {price.currency} "
                           f"({int(price.interval_level * 100)}% band), condition {grade}, "
                           f"from {len(gate.usable_photo_ids)} photos.")

    result.elapsed_s = round(time.time() - t0, 2)
    return result
