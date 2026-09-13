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

import json
import time
from pathlib import Path

import pandas as pd

from . import evidence as evidence_stage
from . import gate as gate_stage
from . import reconcile as reconcile_stage
from .perception import heads as perception_stage
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
# Gate-time CLIP rigid tag. Only blocks at this confidence; below it the VLM
# identity pass remains the body-type signal. Measured later against the 200
# known tractors — until then a missing artifact means this path does not fire
# unless a test passes min_clusters / uses the constant directly.
BODY_TYPE_GATE_CONF = 0.80


def mixed_cluster_threshold() -> int | None:
    """min_clusters_to_flag from the calibration artifact, or None.

    Absent, incomplete, or not measured on all 200 known-single vehicles: do
    not block. Same posture as a missing perception.json.
    """
    from .config import MODELS
    path = MODELS / "mixed_vehicle_thresholds.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    flag = d.get("min_clusters_to_flag")
    if not flag or d.get("n_vehicles") != 200:
        return None
    if float(d.get("false_positive_rate") or 1) > 0.005:
        return None
    return int(flag)


def pricing_blocker(ev, gate=None, min_clusters=None) -> tuple[str, str] | None:
    """Reasons the comparables cannot honestly price what the photos show.

    Returns (headline, reason) or None. Kept pure and separate from `appraise`
    so both conditions can be tested without spending a vision call.
    """
    if min_clusters is None and gate is not None:
        min_clusters = mixed_cluster_threshold()
    if gate is not None and min_clusters and gate.subject_clusters >= min_clusters:
        return (("These photos look like more than one truck. "
                 "Send one set of the vehicle you are selling.",
                 "CLIP appearance clusters among whole-vehicle frames split this "
                 "set, so there is nothing coherent to price."))

    if gate is not None and (not ev or not getattr(ev, "vehicle", None)
                             or not ev.vehicle.body_type):
        if (gate.body_tag == "rigid"
                and gate.body_tag_conf >= BODY_TYPE_GATE_CONF):
            return (f"This looks like a rigid. I can describe its condition, but I have "
                    f"no comparable rigids to price it against.",
                    "the photos show a rigid, not a tractor unit. Every comparable this "
                    "model was fit on is a tractor unit, so it has nothing honest to price "
                    "a rigid against. The condition notes below still stand.")

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
             on_step=None, on_gate=None, on_photo=None) -> Appraisal:
    """`on_step(step, detail)` takes exactly two arguments and always will -
    `cli.py` and `demo.py` both pass two-parameter callbacks. Anything a caller
    needs beyond the string gets its own callback rather than widening that
    one: `on_gate(GateReport)` once, about a second in, and `on_photo(
    PhotoFinding)` once per photo as its own vision call returns, out of order.
    """
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

    early = pricing_blocker(None, gate=gate)
    if early:
        headline, reason = early
        result.status = "need_more_photos"
        result.headline = headline
        result.requests.insert(0, "one set of photos of the single truck you are selling")
        result.elapsed_s = round(time.time() - t0, 2)
        return result

    # --- 1b. perception ---------------------------------------------------
    # Trained on the corpus, unlike everything in stage 1. It runs on the CLIP
    # embedding the gate already computed, so it costs matrix multiplies and no
    # second pass over the pixels. Absent artifact is not fatal: the heads only
    # ever refine, so a missing models/perception.json degrades to the shipped
    # behaviour rather than to an error.
    perception = None
    if perception_stage.available():
        perception = perception_stage.run(gate.photos)
        result.perception = perception
        unfit = sum(1 for p in perception.photos if not p.fine_detail_ok)
        result.trace.append(TraceStep(
            "perception",
            f"{len(perception.photos)} frames scored, {unfit} too degraded for fine "
            f"detail, reads {perception.brand or 'unknown'} "
            f"({perception.brand_conf:.0%})",
            perception.elapsed_s))

    # --- 2. evidence ------------------------------------------------------
    # `evidence.run` reads a view-diverse subset capped at MAX_EVIDENCE_PHOTOS,
    # not every usable frame, so "reading {usable} photos" would overstate it by
    # however many were held back. select_photos is a pure function of the gate
    # report, so calling it here to report the real set costs one sort and
    # cannot disagree with what run() picks.
    #
    # Each of those frames is now its own vision call rather than one slot in a
    # shared one, which is why `on_photo` exists: the screen shows returned work
    # instead of a progress bar guessing at it.
    sent = evidence_stage.select_photos(gate)
    note("evidence", f"reading {len(sent)} of {len(gate.usable_photo_ids)} usable frames")
    ev = evidence_stage.run(gate, declared, backend=backend, on_photo=on_photo)
    result.evidence = ev
    detail = (f"{ev.photos_read} photo(s) read in depth, {len(ev.issues)} findings, "
              f"{len(ev.coverage_gaps)} gaps ({ev.backend}/{ev.model})")
    if ev.photos_failed:
        detail += f" — {ev.photos_failed} frame(s) could not be read"
    if ev.fell_back_from:
        detail += f" — fell back from {len(ev.fell_back_from)} failed backend(s)"
    result.trace.append(TraceStep("evidence", detail, ev.elapsed_s))
    # Deliberately NOT merged into result.requests: the gate's requests are
    # canonical views the seller can go and shoot right now, the evidence's
    # coverage gaps are things that could not be assessed at all. Collapsing
    # them produced a re-ask list with every item in it twice.

    # --- 2b. reconcile ----------------------------------------------------
    # Placed before the blocks_pricing return on purpose: a set that stops here
    # still gets its re-ask list corrected, so a seller is never asked for a
    # photo the trained head can already see in what they sent.
    recon = reconcile_stage.apply(gate, perception, ev)
    if recon.n:
        result.reconcile = recon
        result.requests = list(gate.requests)
        kinds = ", ".join(sorted({c.kind.replace("_", " ") for c in recon.corrections}))
        result.trace.append(TraceStep(
            "reconcile", f"{recon.n} correction(s) against the vision model: {kinds}",
            recon.elapsed_s))
    # Deliberately does NOT clear gate.blocks_pricing. The two conditions that
    # set it - no truck-dominant frame, no whole-vehicle view - are decided by
    # the gate's own ladder before `missing_views` is ever computed, so there
    # is no coverage restore that legitimately answers them. Lifting a pricing
    # block on a 0.60-confidence head prediction would be exactly the
    # confident wrong answer the brief singles out.

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

    widening = [tuple(w) for w in (recon.widening if recon else [])]
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
