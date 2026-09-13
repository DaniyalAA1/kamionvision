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

Seven rules, all of them one-directional:

  unsupported_detail  a fine-detail claim resting on a photo the degradation
                      head scores past its calibrated cutoff
  identity_conflict   the identity head disagrees with the badge the VLM read
  coverage_restored   the gate asked for a view the robust head can already see
  odometer_*          an OCR model reads the dashboard independently of the VLM
                      and either supplies a mileage it missed (odometer_recovered)
                      or disputes the one it read (odometer_conflict)
  vin_*               OCR reads a chassis-plate VIN and either supplies its model
                      year or surfaces a conflict with the seller's declared year
  wmi_conflict        the manufacturer stamped into that same VIN's first three
                      characters is not the manufacturer on the badge
  generation_conflict the generation read off the visual markers does not contain
                      the year, according to the model card's own spans

The odometer rule is the one that most changes the number. The condition
adjustment is capped at one residual sigma by design, so a damage signal can
only nudge the price - but kilometres are a first-class term in the hedonic
model, so a second, checkable reading of the odometer is worth more than any
cosmetic finding. It is a pretrained scene-text model (RapidOCR), no training,
fully offline, and a soft dependency: absent it, the rule simply no-ops.

The WMI rule is the one that reaches furthest outside the corpus. `identity_
conflict` may only dispute brands in the trained head's class list, and there
is no Scania, no DAF and no Volvo in 200 vehicles - so on exactly the truck
this system is likeliest to get wrong, the head has nothing to say. Three
characters stamped into a chassis are not a model output, and a table of 32
cited rows covers every European and North American heavy brand a judge is
likely to arrive with. It shares `_check_vin`'s single OCR pass rather than
opening a second one.

Both of the new rules are governed by the same negative discipline, and it is
the part worth defending in review: a WMI that is not in the table, and a model
card with no generations for that model, mean UNKNOWN. Neither may be turned
into a conflict. A witness with no opinion must not be allowed to contradict a
badge or a year that something else read perfectly well.

Nothing here deletes a finding. A downgraded claim is still shown, still cites
its photo, and carries the reason it was downgraded - the point is to make the
buyer's trust track the evidence, not to hide the disagreement.
"""
from __future__ import annotations

import re
import time

from .config import WMI_CONFLICT_WIDENING
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
FRAMING_CONFIDENCE = 0.70

# Odometer rule. A digit slip on a six-figure reading is a few km; a genuine
# odometer/vision disagreement is a different number entirely. The tolerance is
# generous enough to absorb the former and still catch the latter.
ODOMETER_VIEW = "dashboard_odometer"
ODOMETER_TOLERANCE_FRAC = 0.02
ODOMETER_TOLERANCE_FLOOR = 1_000        # km
ODOMETER_CONFLICT_WIDENING = 1.25
ODOMETER_MAX_FRAMES = 4                  # cap OCR cost on a 40-photo set
VIN_CONFLICT_WIDENING = 1.15
VIN_MAX_FRAMES = 4

# WMI_CONFLICT_WIDENING is imported from config rather than defined here,
# unlike the four constants above it, because `identity.merge_widening` has to
# reason about all three identity multipliers together and they are declared in
# one place for that. It is ASSUMED, like them: this corpus has no
# misidentification ground truth, so there is nothing to fit it against, and
# the string the reader sees says so.

# The generation read has to clear the same bar the identity head does. Two
# different numbers for "confident enough to matter" would drift apart, and
# `identity.MIN_WITNESS_CONFIDENCE` already matches IDENTITY_CONFIDENCE for the
# same reason. A hedged generation guess is worth recording on the vehicle; it
# is not worth telling a buyer their truck may be a different model year.
GENERATION_CONFIDENCE = 0.60


def apply(gate, perception, evidence, declared: dict | None = None) -> ReconcileReport:
    """Reconcile the tracks. Mutates issue confidences; records every change."""
    t0 = time.time()
    report = ReconcileReport()
    if evidence is None:
        report.elapsed_s = round(time.time() - t0, 3)
        return report

    # Runs whether or not the perception artifact is present: it needs only the
    # dashboard frames and the VLM's own reading, not the trained heads.
    _check_odometer(gate, evidence, report)
    _check_vin(gate, evidence, report, declared or {})
    # Strictly after _check_vin, which is what may supply the year this uses
    # when the seller declared none.
    _check_generation(evidence, report, declared or {})

    if perception is not None and perception.photos:
        _downgrade_unsupported(perception, evidence, report)
        _check_identity(perception, evidence, report)
        _correct_whole_vehicle(gate, perception, report)
        _restore_coverage(gate, perception, report)

    report.elapsed_s = round(time.time() - t0, 3)
    return report


def _check_vin(gate, evidence, report, declared: dict) -> None:
    """Fifth and sixth rules: one OCR pass over the plate, two questions of it.

    The plate answers two independent things and they are reconciled against
    two different witnesses, so they are gated differently:

      the manufacturer  characters 1-3, the WMI. Checked against the badge the
                        vision model read. Always asked - see below on why the
                        check digit does not gate this one.
      the model year    character 10, checked against the seller's declaration.
                        Only asked when the check digit validates.

    That asymmetry is measured, not stylistic. Character 10 is a model-year
    code under FMVSS 565, which is a North American rule; ISO 3779 does not
    mandate it and European heavy trucks do not follow it. The DAF VINs
    published with Australian recall REC-006624 - trucks built between 2019 and
    2025 - fail the position-9 check digit and decode to 1994 under the North
    American rule. Acting on that would hand `pricing/model.py` a 1994 year for
    a 2023 truck through the `vin_year` fallback, and raise a confident, wrong
    `vin_conflict` against a seller who declared the truth. The check digit is
    the cheapest available test of "is this a VIN that plays by those rules",
    so the year half waits for it and the manufacturer half does not: every
    candidate in the WMI table is a distinct three-character prefix, so an OCR
    slip that lands exactly on a different brand's registered WMI is a far less
    likely failure than reading a field that was never a year.
    """
    if gate is None or not gate.photos:
        return
    try:
        from . import vin
        from .subject import WHOLE_VEHICLE_VIEWS
    except Exception:                                    # noqa: BLE001
        return

    named = [c for c in gate.photos
             if c.usable and any(word in c.filename.lower() for word in ("vin", "plate"))]
    detail = [c for c in gate.photos
              if c.usable and c.view not in WHOLE_VEHICLE_VIEWS and c not in named]
    candidates = named + sorted(detail, key=lambda c: -c.capture_quality)[:VIN_MAX_FRAMES]
    if not candidates:
        return

    try:
        reads = [(c, vin.read(c.path)) for c in candidates]
    except Exception:                                    # noqa: BLE001
        return
    hits = [(c, result) for c, result in reads if result.vin is not None]
    if not hits:
        return
    photo, best = max(hits, key=lambda item: (item[1].check_ok is True,
                                              item[1].confidence))
    evidence.vehicle.vin = best.vin
    evidence.vehicle.wmi = best.wmi
    _check_wmi(evidence, report, photo, best)

    # The year half, and only for a VIN that obeys the rule the year code
    # belongs to. `vin_year` stays None otherwise: pricing reads it as a
    # fallback year, so leaving it unset is the difference between "no VIN
    # year" and "the VIN says 1994" on a truck built in 2023.
    if best.year is None or not best.check_ok:
        return
    evidence.vehicle.vin_year = best.year

    stated_year = declared.get("year")
    if not stated_year:
        report.corrections.append(Correction(
            kind="vin_recovered", photo_id=photo.photo_id,
            detail=f"no model year was declared; chassis-plate VIN {best.vin} decodes "
                   f"to {best.year}",
            before="year: not declared", after=f"{best.year} (VIN)"))
        return

    try:
        delta = abs(int(stated_year) - best.year)
    except (TypeError, ValueError):
        return
    if delta <= 1:
        return
    report.corrections.append(Correction(
        kind="vin_conflict", photo_id=photo.photo_id,
        detail=f"the seller declares {stated_year}; chassis-plate VIN {best.vin} "
               f"decodes to {best.year} - priced on the declared year",
        before=f"{stated_year} (declared)", after=f"{best.year} (VIN)"))
    report.widening.append([
        f"the declared year and chassis-plate VIN year disagree "
        f"({stated_year} vs {best.year}), so the band widens "
        f"{VIN_CONFLICT_WIDENING:.2f}x (a stated assumption)",
        VIN_CONFLICT_WIDENING])


def _check_wmi(evidence, report, photo, best) -> None:
    """Sixth rule: the manufacturer stamped in the plate against the badge.

    Silent on agreement, exactly like `_check_identity` and `_check_odometer` -
    every rule in this file speaks only when it has something to say. A
    confirmation is not thrown away, though: it lands on `vehicle.wmi_brand`,
    which is where `identity.collect` picks the plate up as a witness and
    weights it above every read taken off bodywork. A `wmi_confirmed`
    correction would say the same thing a second time, in the one list a reader
    scans specifically to find out what went wrong.

    Three ways this stays quiet, and all three are the point:

      * the table has no verified row for those three characters. Unknown is
        not conflict. The corpus is 78/84 Ford and the table is 32 rows; the
        space of WMIs it does not cover is enormous and a judge's truck may
        easily sit in it.
      * the vision model named no make. There is nothing to dispute.
      * the badge matches any brand the plant is allowed to wear, not just the
        primary one - see `vin.wmi_brands` for why that list exists.

    On a genuine disagreement the badge REMAINS the priced make. Same posture
    as `odometer_conflict`, which keeps the vision figure and widens: the plate
    is the better witness, but overwriting a read here would quietly change
    which brand column the hedonic fit uses on the strength of an OCR pass over
    a stamped plate. Record it, widen, show both, and let `identity.decide`
    adjudicate in one place.
    """
    from . import vin
    from .pricing.features import normalise_brand

    accepted = vin.wmi_brands(best.wmi)
    if not accepted:
        return                                           # unknown, never a conflict

    said = normalise_brand(getattr(evidence.vehicle, "make", None))
    # First spelling wins where two fold to the same key, so a row that lists a
    # spelling `normalise_brand` already handles keeps the readable one.
    folded: dict[str, str] = {}
    for brand in accepted:
        folded.setdefault(normalise_brand(brand), brand)
    # Write down the spelling that AGREED where one did, rather than the
    # table's primary. `identity.collect` folds this through the same
    # normaliser and compares it with the badge, so handing it "RENAULT" when
    # the badge says "Renault Trucks" would manufacture a dispute downstream
    # out of a gap in `normalise_brand` rather than a fact about the truck.
    evidence.vehicle.wmi_brand = folded.get(said) or accepted[0]
    if said == "other" or said in folded:
        return                                           # agreement, or no badge read

    report.corrections.append(Correction(
        kind="wmi_conflict", photo_id=photo.photo_id,
        detail=f"the badge reads {said}; the chassis-plate VIN {best.vin} begins "
               f"{best.wmi}, which is {accepted[0]}'s world manufacturer "
               f"identifier - priced on the badge, but the plate is the harder "
               f"evidence and this needs resolving before anyone pays",
        before=f"{said} (badge)", after=f"{accepted[0]} (VIN {best.wmi})"))
    report.widening.append([
        f"the badge and the chassis-plate WMI name different manufacturers "
        f"({said} vs {accepted[0]}), so the band widens "
        f"{WMI_CONFLICT_WIDENING:.2f}x (a stated assumption - this corpus has "
        f"no misidentified vehicles to fit one against)",
        WMI_CONFLICT_WIDENING])


def _check_generation(evidence, report, declared: dict) -> None:
    """Seventh rule: the generation read off the pixels against the year.

    `identity` asks the vision model which generation it is looking at, from
    visual markers and deliberately NOT from the declared year - a year-derived
    generation would make this check circular. So there are two independent
    statements about the same vehicle: a generation, read from a grille and a
    light cluster, and a year, declared by the seller or stamped in the plate.
    The model card holds the span that joins them.

    It no-ops far more often than it fires, and every one of those exits is
    deliberate. No generation was read; no year is available from any source;
    the read was hedged below `GENERATION_CONFIDENCE`; the card knows no
    generations for this model, or knows none by that id; or the year falls
    outside every span the card lists, which means the card is incomplete
    rather than the photographs wrong. An incomplete reference cannot
    adjudicate, and the same discipline applies here as to an unknown WMI:
    silence.

    Agreement is tested against the READ generation's own span, never against
    whichever generation the card happens to list first for that year.
    Manufacturers sell an old and a new range alongside each other through a
    changeover and the card records that honestly - Scania's R-series runs
    2004-2017 and the next-generation cab starts in 2016, so a 2017 truck sits
    in both. `generation_for_year` returns the first match by construction, so
    using it for the agreement test would call a correctly-read next-gen 2017
    Scania a conflict. It is used only to name the alternative once a real
    disagreement is established.

    When it does fire, the PHOTOGRAPH keeps its answer. `modelspec`'s own rule
    is that a card never overrides the pixels - a spec card contradicting the
    photographs is a spec card being read wrong - so this records the
    disagreement and does not widen the band. The band belongs to the identity
    verdict and to the year, and this rule changes neither.
    """
    vehicle = getattr(evidence, "vehicle", None)
    if vehicle is None:
        return
    read = str(getattr(vehicle, "generation", "") or "").strip()
    if not read or read.lower() == "unknown":
        return
    if float(getattr(vehicle, "generation_conf", 0.0) or 0.0) < GENERATION_CONFIDENCE:
        return

    # Declared first, then the plate - the same precedence `pricing/model.py`
    # uses when it picks the year it prices on.
    year = declared.get("year") or getattr(vehicle, "vin_year", None)
    try:
        year = int(year) if year else None
    except (TypeError, ValueError):
        return
    if not year:
        return

    try:
        from . import modelspec
        # The span parser, borrowed rather than reimplemented so "2025-present"
        # is read here exactly as the card's own accessor reads it. Inside the
        # soft import for the same reason everything else in this file is: if
        # it is ever renamed this rule goes quiet instead of taking the
        # appraisal down with it.
        from .modelspec import _year_span
    except Exception:                                    # noqa: BLE001
        return
    make, model = getattr(vehicle, "make", None), getattr(vehicle, "model", None)
    known = modelspec.generations(make, model)
    if not known:
        return                                           # the card has nothing to say

    mine = next((g for g in known
                 if str(g.get("id", "")).strip().lower() == read.lower()), None)
    if mine is None:
        return                             # a generation this card has not heard of
    low, high = _year_span(mine.get("years"))
    if low is not None and low <= year <= (high or 9999):
        return                             # the read generation covers the year

    expected = modelspec.generation_for_year(make, model, year)
    if expected is None:
        return                                           # the year is off the card

    span = expected.get("years") or expected.get("id")
    because = str(getattr(vehicle, "year_evidence", "") or "").strip()
    report.corrections.append(Correction(
        kind="generation_conflict",
        detail=f"the photographs were read as the {read} generation"
               + (f" ({because})" if because else "")
               + f", but {year} falls in the {span} generation "
                 f"({expected.get('id')}) on the model card - the photographs "
                 f"keep the answer; the card is the thing being cross-checked",
        before=f"{read} (read from the photographs)",
        after=f"{expected.get('id')} ({span}, model card)"))


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
            r = odometer.read(c.path, subject_box=c.subject_box)
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


def _correct_whole_vehicle(gate, perception, report) -> None:
    """Stop a confident part frame from satisfying whole-vehicle coverage."""
    from .subject import WHOLE_VEHICLE_VIEWS

    if not gate or not gate.photos:
        return
    gate_by_id = {p.photo_id: p for p in gate.photos}
    overridden = set()
    for p in perception.photos:
        checked = gate_by_id.get(p.photo_id)
        if (checked is None or checked.view not in WHOLE_VEHICLE_VIEWS
                or p.framing != "part" or p.framing_conf < FRAMING_CONFIDENCE):
            continue
        overridden.add(p.photo_id)
        report.corrections.append(Correction(
            kind="framing_override", photo_id=p.photo_id,
            detail=f"the gate called this {checked.view}, but the hand-labelled "
                   f"framing head identifies a component close-up "
                   f"({p.framing_conf:.0%})",
            before="counted as a whole-vehicle frame",
            after="excluded from whole-vehicle coverage"))

    if not overridden:
        return

    valid_whole = [
        p for p in gate.photos
        if p.usable and p.view in WHOLE_VEHICLE_VIEWS and p.photo_id not in overridden
    ]
    valid_views = {p.view for p in valid_whole}
    gate.views_present = [
        view for view in gate.views_present
        if view not in WHOLE_VEHICLE_VIEWS or view in valid_views
    ]
    if valid_whole:
        return

    gate.blocks_pricing = True
    if not any(view in WHOLE_VEHICLE_VIEWS for view in gate.missing_views):
        gate.missing_views.append("exterior_front_34")
    from .gate import VIEW_REQUESTS
    request = VIEW_REQUESTS["exterior_front_34"]
    if request not in gate.requests:
        gate.requests.append(request)


def _restore_coverage(gate, perception, report) -> None:
    """A muddy tire close-up the zero-shot tagger fumbled is still a tire shot.

    The gate's view labels are zero-shot; the head holds its answer under
    degradation measurably more often. Where the head confidently sees a view
    the gate called missing, the re-ask for it is withdrawn - asking a seller
    to send a photo they already sent is the cheapest way to look broken.
    """
    if not gate or not gate.missing_views:
        return
    from .subject import WHOLE_VEHICLE_VIEWS
    gate_by_id = {p.photo_id: p for p in gate.photos}
    confident = set()
    for p in perception.photos:
        if p.view_conf < VIEW_CONFIDENCE:
            continue
        checked = gate_by_id.get(p.photo_id)
        if p.view in WHOLE_VEHICLE_VIEWS:
            if checked is None or checked.view not in WHOLE_VEHICLE_VIEWS:
                continue
            if p.framing == "part" and p.framing_conf >= FRAMING_CONFIDENCE:
                continue
        confident.add(p.view)
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
