"""Cross-check the vision model against the trained heads, in the open.

The vision model is the only stage that can describe a truck, and it is also
the only stage whose confidence is self-reported. `Issue.confidence` is a
number GPT-5.6 writes about its own claim, and it is multiplied straight into
the price. Nothing else in the pipeline is in a position to disagree with it.

The perception heads are. They were trained on 3,729 exactly-paired
clean/degraded images, so they can say - with a held-out AUC behind it - that a
particular frame is too corrupted to read tread depth off. When the vision
model reads tread depth off that frame anyway, this stage lowers the confidence
of that claim and writes down that it did.

Four rules, all of them one-directional:

  unsupported_detail  a fine-detail claim resting on a photo the degradation
                      head scores past its calibrated cutoff
  identity_conflict   the identity head disagrees with the badge the VLM read
  coverage_restored   the gate asked for a view the robust head can already see
  odometer_*          an OCR model reads the dashboard independently of the VLM
                      and either supplies a mileage it missed (odometer_recovered)
                      or disputes the one it read (odometer_conflict)

The odometer rule is the one that most changes the number. The condition
adjustment is capped at one residual sigma by design, so a damage signal can
only nudge the price - but kilometres are a first-class term in the hedonic
model, so a second, checkable reading of the odometer is worth more than any
cosmetic finding. It is a pretrained scene-text model (RapidOCR), no training,
fully offline, and a soft dependency: absent it, the rule simply no-ops.

Nothing here deletes a finding. A downgraded claim is still shown, still cites
its photo, and carries the reason it was downgraded - the point is to make the
buyer's trust track the evidence, not to hide the disagreement.
"""
from __future__ import annotations

import re
import time

from .schema import Correction, ReconcileReport

# Vocabulary of claims that need pixels a degraded frame does not have. Tread
# depth and hairline cracks are exactly what a rain-and-motion-blur twin
# destroys; "the bumper is missing" survives any amount of corruption.
FINE_DETAIL_TERMS = re.compile(
    r"\b(tread|depth|mm\b|hairline|fine crack|crazing|pitting|pinhole|"
    r"seep|weep|hazing|casting|stamp|serial|legible|digits?|reading|"
    r"surface rust|light scoring|scuff|feathering|cupping|sidewall crack)\b",
    re.IGNORECASE)

# Stated assumption, not a measurement: a fine-detail claim on a frame the
# degradation head scores at severity s keeps (1 - s) of its confidence, with a
# floor so a downgrade never becomes a silent deletion. There is no held-out
# experiment behind the floor - the corpus has no condition ground truth to
# calibrate one against - so it is labelled an assumption wherever it surfaces.
CONFIDENCE_FLOOR = 0.2

# The identity head only earns an opinion above this, and only about brands it
# was actually trained on. A judge's Scania is not in the corpus; a head that
# has never seen one will still name a class, confidently, and that guess must
# never be allowed to contradict a badge the VLM can plainly read.
IDENTITY_CONFIDENCE = 0.60
IDENTITY_WIDENING = 1.25
VIEW_CONFIDENCE = 0.60

# Odometer rule. A digit slip on a six-figure reading is a few km; a genuine
# odometer/vision disagreement is a different number entirely. The tolerance is
# generous enough to absorb the former and still catch the latter.
ODOMETER_VIEW = "dashboard_odometer"
ODOMETER_TOLERANCE_FRAC = 0.02
ODOMETER_TOLERANCE_FLOOR = 1_000        # km
ODOMETER_CONFLICT_WIDENING = 1.25
ODOMETER_MAX_FRAMES = 4                  # cap OCR cost on a 40-photo set


def apply(gate, perception, evidence) -> ReconcileReport:
    """Reconcile the tracks. Mutates issue confidences; records every change."""
    t0 = time.time()
    report = ReconcileReport()
    if evidence is None:
        report.elapsed_s = round(time.time() - t0, 3)
        return report

    # Runs whether or not the perception artifact is present: it needs only the
    # dashboard frames and the VLM's own reading, not the trained heads.
    _check_odometer(gate, evidence, report)

    if perception is not None and perception.photos:
        _downgrade_unsupported(perception, evidence, report)
        _check_identity(perception, evidence, report)
        _restore_coverage(gate, perception, report)

    report.elapsed_s = round(time.time() - t0, 3)
    return report


def _check_odometer(gate, evidence, report) -> None:
    """Fourth rule: read the odometer with OCR, reconcile it against the VLM.

    Two outcomes produce a correction; agreement is deliberately silent, exactly
    like the identity rule - this speaks only when it has something to say.

      odometer_recovered  the VLM read no odometer, OCR does. The reading is
                          written onto the vehicle so the price model gets a
                          mileage it would otherwise lack; that in turn triggers
                          the existing "read off the dashboard" widening in
                          price_from_evidence, so nothing is double-counted here.
      odometer_conflict   both read a number and they disagree past tolerance.
                          The band widens and both figures are shown; the VLM's
                          value stays the priced one, unchanged - same posture as
                          identity_conflict, which keeps the badge and widens.
    """
    if gate is None or not gate.photos:
        return
    try:
        from . import odometer
    except Exception:                                    # noqa: BLE001
        return                                           # module import failed

    by_id = {c.photo_id: c for c in gate.photos}
    cands = [c for c in gate.photos if c.usable and c.view == ODOMETER_VIEW]
    # Also try the exact frame the VLM said it read the odometer from, even if
    # the zero-shot view tag called it something else.
    vlm_pid = evidence.vehicle.odometer_photo_id
    if vlm_pid in by_id and by_id[vlm_pid].usable and by_id[vlm_pid] not in cands:
        cands.append(by_id[vlm_pid])
    if not cands:
        return
    cands = sorted(cands, key=lambda c: -c.capture_quality)[:ODOMETER_MAX_FRAMES]

    try:                                                 # RapidOCR loads here
        best_pid, best = None, None
        for c in cands:
            r = odometer.read(c.path)
            if r.km is None:
                continue
            if best is None or r.confidence > best.confidence:
                best_pid, best = c.photo_id, r
    except Exception:                                    # noqa: BLE001
        return                                           # OCR unavailable/failed
    if best is None:
        return

    ocr_km, vlm_km = best.km, evidence.vehicle.odometer_km

    if vlm_km is None:
        evidence.vehicle.odometer_km = ocr_km
        evidence.vehicle.odometer_photo_id = best_pid
        report.corrections.append(Correction(
            kind="odometer_recovered", photo_id=best_pid,
            detail=f"the vision model reported no odometer; OCR read {ocr_km:,} km "
                   f"off the dashboard (confidence {best.confidence:.2f})",
            before="odometer: not read", after=f"{ocr_km:,} km (OCR)"))
        return

    tol = max(ODOMETER_TOLERANCE_FLOOR, ODOMETER_TOLERANCE_FRAC * vlm_km)
    if abs(ocr_km - vlm_km) <= tol:
        return                                           # agreement: stay silent

    delta = abs(ocr_km - vlm_km) / vlm_km * 100
    report.corrections.append(Correction(
        kind="odometer_conflict", photo_id=best_pid,
        detail=f"the vision model read {vlm_km:,} km; OCR reads {ocr_km:,} km off "
               f"photo {best_pid} ({delta:.0f}% apart) - priced on the vision "
               f"figure, but the odometer needs documentary support before anyone pays",
        before=f"{vlm_km:,} km (vision)", after=f"{ocr_km:,} km (OCR)"))
    report.widening.append([
        f"the vision model and an OCR read of the odometer disagree "
        f"({vlm_km:,} vs {ocr_km:,} km), so the band widens "
        f"{ODOMETER_CONFLICT_WIDENING:.2f}x (a stated assumption)",
        ODOMETER_CONFLICT_WIDENING])


def _downgrade_unsupported(perception, evidence, report) -> None:
    for issue in evidence.issues:
        p = perception.by_id(issue.photo_id)
        if p is None or p.fine_detail_ok:
            continue
        if not FINE_DETAIL_TERMS.search(issue.observation or ""):
            continue
        before = issue.confidence
        issue.confidence = round(max(CONFIDENCE_FLOOR, before * (1.0 - p.severity)), 3)
        if issue.confidence >= before:
            continue
        why = ", ".join(p.degradations) or "degraded capture"
        report.corrections.append(Correction(
            kind="unsupported_detail", photo_id=issue.photo_id,
            detail=f"{issue.component}: fine detail claimed from a photo scored "
                   f"{p.severity:.2f} for degradation ({why})",
            before=f"confidence {before:.2f}", after=f"confidence {issue.confidence:.2f}"))


def _check_identity(perception, evidence, report) -> None:
    from .pricing.features import normalise_brand

    said = normalise_brand(getattr(evidence.vehicle, "make", None))
    seen = normalise_brand(perception.brand)
    if said == "other" or seen == "other":
        return
    if perception.brand_conf < IDENTITY_CONFIDENCE or said == seen:
        return
    # Only speak up about brands the head was trained to recognise. Outside
    # that set its top class is an artefact of the corpus, not an observation.
    classes = {normalise_brand(c) for c in _identity_classes()}
    if said not in classes:
        return
    report.identity_conflict = (
        f"the badge reads {said}, the identity head says {seen} "
        f"({perception.brand_conf:.0%})")
    report.corrections.append(Correction(
        kind="identity_conflict", detail=report.identity_conflict,
        before=said, after=seen))
    report.widening.append(["identity disputed between the badge read and the "
                            "trained head (stated assumption)", IDENTITY_WIDENING])


def _identity_classes() -> list[str]:
    from . import perception as perception_pkg
    try:
        return list(perception_pkg.model().identity["classes"])
    except Exception:                                    # noqa: BLE001
        return []


def _restore_coverage(gate, perception, report) -> None:
    """A muddy tire close-up the zero-shot tagger fumbled is still a tire shot.

    The gate's view labels are zero-shot; the head holds its answer under
    degradation measurably more often. Where the head confidently sees a view
    the gate called missing, the re-ask for it is withdrawn - asking a seller
    to send a photo they already sent is the cheapest way to look broken.
    """
    if not gate or not gate.missing_views:
        return
    confident = {p.view for p in perception.photos if p.view_conf >= VIEW_CONFIDENCE}
    restored = [v for v in gate.missing_views if v in confident]
    for view in restored:
        report.corrections.append(Correction(
            kind="coverage_restored", detail=f"{view} is present after all",
            before="listed as missing by the zero-shot view tag",
            after="found by the trained view head"))
    if restored:
        gate.missing_views = [v for v in gate.missing_views if v not in restored]
        gate.views_present = sorted(set(gate.views_present) | set(restored))
        from .gate import VIEW_REQUESTS
        wanted = {VIEW_REQUESTS[v] for v in restored if v in VIEW_REQUESTS}
        gate.requests = [r for r in gate.requests if r not in wanted]
