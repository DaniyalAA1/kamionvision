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

Three rules, all of them one-directional:

  unsupported_detail  a fine-detail claim resting on a photo the degradation
                      head scores past its calibrated cutoff
  identity_conflict   the identity head disagrees with the badge the VLM read
  coverage_restored   the gate asked for a view the robust head can already see

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


def apply(gate, perception, evidence) -> ReconcileReport:
    """Reconcile the two tracks. Mutates issue confidences; records every change."""
    t0 = time.time()
    report = ReconcileReport()
    if perception is None or evidence is None or not perception.photos:
        report.elapsed_s = round(time.time() - t0, 3)
        return report

    _downgrade_unsupported(perception, evidence, report)
    _check_identity(perception, evidence, report)
    _restore_coverage(gate, perception, report)

    report.elapsed_s = round(time.time() - t0, 3)
    return report


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
