"""Stage 2 - the evidence fan-out, and the photo selection that feeds it.

Five passes now, described in `passes.py` and `calibration.py`. What this
module owns is the orchestration: which photos go to which pass, which backend
answers, how many calls run at once, what happens when one of them fails, and
how each finished photo reaches the screen while the rest are still in flight.

Two of those five are about identity rather than condition, and they no longer
read the same photographs as the rest. `select_photos` round-robins on a
condition-first view priority that leads with a tire close-up: correct for pass
B, wrong for pass A, where a tire contributes nothing to make, model or axle
count and costs a slot. `select_identity_photos` picks for pass A instead, and
the badge read gets one crop of one frame.

Pass D is sequenced where it is for a reason that is not negotiable: it runs
AFTER `merge_duplicates`. Severity cannot be decided set-aware until the
duplicates are folded, because `notes_block` numbers the pre-merge flat list
and three paraphrases of one worn drive tire are still three rows in it.

Failure posture, which is most of why this file exists:

  one sample of pass A fails      -> it retries on the next backend, and the
                                     identity is decided by the samples that
                                     did come back. Recorded.
  every sample of pass A fails    -> raise. There is nothing to describe.
  the badge read fails            -> the appraisal continues on the sampled
                                     identity pass alone, and says so. It is a
                                     corroborating witness, not the answer.
  one sample of a photo fails     -> it retries on the next backend, and if
                                     that fails too the photo is read by the
                                     samples that did come back. Recorded.
  every sample of one photo fails -> that photo carries an `error` and the
                                     appraisal continues on the other fifteen.
                                     Recorded, never swallowed.
  every close-up call fails       -> raise. A report with no observations in it
                                     is not an appraisal with thin coverage, it
                                     is a broken run wearing one.
  pass C fails                    -> deterministic rollup. Every finding already
                                     exists; the synthesis only groups them, so
                                     losing the call should not lose the work.
  pass D fails                    -> the provisional severities stand and a
                                     parse warning says so. Losing the
                                     calibration should cost the calibration.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import condition as condition_stage
from .. import vlm
from ..config import (BADGE_MAX_TOKENS, BADGE_READ, CALIBRATION_MAX_TOKENS,
                      CLOSEUP_MAX_TOKENS, EVIDENCE_CONCURRENCY,
                      IDENTITY_MAX_TOKENS, IDENTITY_PHOTOS,
                      MAX_EVIDENCE_PHOTOS, SYNTHESIS_MAX_TOKENS)
from ..schema import Correction, EvidenceReport, GateReport, PhotoFinding
from ..subject import WHOLE_VEHICLE_VIEWS
from . import calibration, passes, prompts, sampling

# View priority for photo selection: what a buyer needs, in order.
VIEW_PRIORITY = ["exterior_front_34", "tire_wheel", "dashboard_odometer", "exterior_side",
                 "interior_cab", "chassis_undercarriage", "fifth_wheel", "engine_bay",
                 "damage_detail", "exterior_front", "exterior_rear"]

# What pass A is looking for, in order, and it is a different list. Identity
# lives in the front three-quarter (badge and cab shape), the side profile -
# the only view that can honestly settle 4x2 against 6x2 - the front, and the
# rear. Everything else in VIEW_PRIORITY still follows, because `same_vehicle`
# is the other half of this pass's job: a padded listing whose odd frame is an
# interior shot of a tidier truck has to be able to reach it.
IDENTITY_VIEW_PRIORITY = ["exterior_front_34", "exterior_side", "exterior_front",
                          "exterior_rear"]


def _round_robin(usable: list, priority: list[str], limit: int) -> list:
    """One slot per view before any view gets a second, best capture first."""
    by_view: dict[str, list] = {}
    for check in sorted(usable, key=lambda c: -c.capture_quality):
        by_view.setdefault(check.view, []).append(check)

    order = [v for v in priority if v in by_view] + \
            [v for v in by_view if v not in priority]
    picked = []
    while len(picked) < limit:
        progressed = False
        for view in order:
            bucket = by_view[view]
            if not bucket:
                continue
            picked.append(bucket.pop(0))
            progressed = True
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked


def select_photos(gate: GateReport, limit: int = MAX_EVIDENCE_PHOTOS) -> list:
    """View-diverse subset of the usable photos, best capture quality first.

    A seller uploads 15-40 frames and most of them are the same three angles.
    Round-robin over views first, so one slot is spent per view before any
    view gets a second, then fill the remainder by capture quality.
    """
    return _round_robin([c for c in gate.photos if c.usable], VIEW_PRIORITY, limit)


def select_identity_photos(gate: GateReport, limit: int = IDENTITY_PHOTOS) -> list:
    """The subset pass A gets: whole-vehicle views first, then the rest.

    Same round-robin, a different priority, and fewer frames - which buys the
    resolution back. `IDENTITY_IMAGE_LONG_EDGE` is higher than the 1024 the
    close-ups use because a model badge is small in frame, and sending half as
    many photographs is what pays for it.

    A set with no whole-vehicle frame at all falls back to today's selection
    rather than sending nothing: a truck photographed only in close-up is still
    a truck being sold, and the identity pass reading tires and a dashboard is
    worth more than no identity pass.
    """
    usable = [c for c in gate.photos if c.usable]
    if not any(c.view in WHOLE_VEHICLE_VIEWS for c in usable):
        return select_photos(gate, limit)
    priority = IDENTITY_VIEW_PRIORITY + [v for v in VIEW_PRIORITY
                                         if v not in IDENTITY_VIEW_PRIORITY]
    return _round_robin(usable, priority, limit)


def _walk(chain, preferred, call, report: EvidenceReport, label: str):
    """Run `call` on the backend that answered pass A, then the rest of the chain.

    Until now only pass A walked the chain and B, C and D reused whichever
    client answered it, so a provider that died after the identity call took
    the appraisal with it. A fallback is recorded, as everywhere else here.
    """
    order = [preferred] + [c for c in chain if c is not preferred]
    failures: list[str] = []
    for candidate in order:
        try:
            return candidate, call(candidate)
        except vlm.VLMError as exc:
            failures.append(f"{label} on {candidate.name}: {exc}")
    report.fell_back_from.extend(failures)
    raise vlm.VLMError(f"every vision backend failed on the {label} pass:\n  "
                       + "\n  ".join(failures))


def _repair(client, response, exc, schema, max_tokens):
    """One text-only repair attempt.

    Cheaper than re-uploading the photos, and it fixes the common failure,
    which is a truncated or fenced object rather than a misunderstood task.
    """
    return client.complete(
        "That was not parseable as a single JSON object "
        f"({type(exc).__name__}: {exc}). Return the same content as ONE valid "
        "JSON object, no fence, no commentary:\n\n" + (response.text or "")[:6000],
        [], system=prompts.SYSTEM, max_tokens=max_tokens,
        json_schema=schema if client.supports_structured_output else None)


# --- pass A2: the badge read ----------------------------------------------

def _badge(chain, client, report: EvidenceReport, gate: GateReport,
           tmpdir: Path) -> None:
    """A second, independent witness to make and model. Never fatal.

    One full-resolution call on a crop of the badge band, rather than trusting
    a badge read off a downscaled montage of eight photographs. What it says is
    recorded on `VehicleRead.badge_*` and is never allowed to overwrite
    `make`/`model`: `app/identity.py` is the only thing that adjudicates
    between witnesses, and it needs both readings intact to do it.

    Every way this can fail costs the badge and not the appraisal, and each one
    says so - a witness that quietly did not turn up is worse than one that did
    not turn up, because the verdict would then read as if it had.
    """
    if not passes.badge_available():
        # Cannot happen in a built tree - `prompts.badge_prompt` ships - but a
        # witness that quietly did not turn up is worse than one that did not,
        # because the verdict would read as if it had.
        report.parse_warnings.append(
            "this build has no badge prompt, so the badge was not read separately")
        return
    check = passes.best_badge_frame(gate)
    crop = passes.write_badge_crop(check, tmpdir) if check is not None else None
    if crop is None:
        report.parse_warnings.append(
            "no whole-vehicle frame carried a subject box big enough to cut a "
            "badge crop from, so the badge was not read separately; identity "
            "rests on the sampled identity pass alone")
        return

    t = time.time()
    try:
        _, response = _walk(
            chain, client,
            lambda c: passes.badge(c, check, image=crop, max_tokens=BADGE_MAX_TOKENS),
            report, "badge")
        try:
            data = passes.parse_badge(response.text)
        except (ValueError, json.JSONDecodeError) as exc:
            # Text-only, and safe: the photo this read belongs to comes from
            # `check`, never out of the response, so a repair cannot re-bind it.
            response = _repair(client, response, exc,
                               getattr(prompts, "BADGE_SCHEMA", None), BADGE_MAX_TOKENS)
            data = passes.parse_badge(response.text)
            report.parse_warnings.append(
                f"the badge read was unparseable ({exc}); repaired")
    except (vlm.VLMError, ValueError, json.JSONDecodeError) as exc:
        report.parse_warnings.append(
            f"the badge read failed ({exc}); identity rests on the sampled "
            f"identity pass alone")
        report.calls.append(["badge (failed)", round(time.time() - t, 2)])
        return

    report.calls.append(["badge", response.elapsed_s])
    report.vehicle.badge_text = data["badge_text"]
    report.vehicle.badge_photo_id = check.photo_id
    if not data["legible"]:
        # "I looked at the badge and could not read it" is information, and it
        # is not the same as not having looked. `identity.collect` gets no
        # badge witness either way, which widens the band rather than narrowing
        # it on a guess.
        report.parse_warnings.append(
            f"the badge crop from photo {check.photo_id} was not legible; the make "
            f"and model rest on the sampled identity pass alone")
        return
    report.vehicle.badge_make = data["make"]
    report.vehicle.badge_model = data["model"]
    # The trim is literal text on the badge and there is nowhere else on
    # `VehicleRead` for it, so it joins the transcription rather than being
    # dropped: "F-MAX" and "F-MAX 500" are different rows to `anchor.lookup`.
    trim = data["trim_or_power"]
    if trim and not any(trim.lower() in line.lower()
                        for line in report.vehicle.badge_text):
        report.vehicle.badge_text.append(trim)


# --- pass B, concurrent ----------------------------------------------------

def _fan_out(chain, selected, vehicle_line: str, tmpdir: Path,
             on_photo, report: EvidenceReport, *, expectation: str = "",
             band: str | None = None, vehicle=None, on_activity=None) -> list[PhotoFinding]:
    """One photo per task, read `CLOSEUP_SAMPLES` times, `EVIDENCE_CONCURRENCY`
    tasks at a time.

    Results are handed to `on_photo` as they land, which is out of order - the
    screen is showing real returned work, and real work does not finish in the
    order it was started. `on_photo` still fires once per PHOTO, after its
    samples have been combined, never once per sample: the progress figure is a
    count of finished photos and it stays one.
    """
    findings: list[PhotoFinding] = []
    workers = max(1, min(EVIDENCE_CONCURRENCY, len(selected)))
    def inspect(check):
        if on_activity:
            on_activity({"phase": "photo", "photo_id": check.photo_id, "view": check.view})
        return sampling.closeup_consensus(chain, check, vehicle_line,
            tmpdir=tmpdir, max_tokens=CLOSEUP_MAX_TOKENS,
            expectation=expectation, band=band, repair=_repair,
            weak_points=passes.weak_points_for_view(vehicle, check.view))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(inspect, check): check
            for check in selected
        }
        for future in as_completed(futures):
            check = futures[future]
            try:
                finding, corrections, fallbacks = future.result()
                report.corrections.extend(corrections)
                report.fell_back_from.extend(fallbacks)
            except Exception as exc:
                # One lost frame is not a lost appraisal, but it is not nothing
                # either: the reader is told which photo went unread.
                finding = PhotoFinding(photo_id=check.photo_id, view=check.view,
                                       error=f"{type(exc).__name__}: {exc}")
                report.parse_warnings.append(
                    f"photo {check.photo_id} ({check.filename}) could not be read: {exc}")
            report.calls.append([f"photo {finding.photo_id}", finding.elapsed_s])
            findings.append(finding)
            if on_photo:
                on_photo(finding)
    return findings


# --- pass C, and what stands in for it ------------------------------------

def _fallback_synthesis(findings: list[PhotoFinding], flat: list) -> dict:
    """Roll the per-photo notes up without a model.

    Used when the synthesis call fails. Every observation already exists by
    this point - pass C only groups them - so losing that call should cost the
    prose, not the findings.

    It does not grade any more, and that is the point of the redesign. The old
    rollup graded on `max(severity)` with `default=-1` and no `-1` key, so
    sixteen spotless photos came back `unknown` while ONE cosmetic scuff
    upgraded the same truck to `good` - finding a fault improved the grade. The
    grade is now `app.condition.rollup`, on this path and on the happy path
    alike, so the two differ only in prose, which is what this docstring always
    claimed.
    """
    summary: dict[str, list[str]] = {}
    for issue in flat:
        key = prompts.COMPONENT_SUMMARY.get(issue.component)
        if key:
            summary.setdefault(key, []).append(issue.observation)
    gaps = []
    for finding in findings:
        gaps.extend(finding.cannot_tell)
    return {
        "condition_summary": {
            k: ("; ".join(summary[k][:3]) if k in summary else "not visible in these photos")
            for k in prompts.SUMMARY_KEYS},
        "condition_grade": "",
        "coverage_gaps": list(dict.fromkeys(gaps))[:10],
        "headline": "",
        "confidence": 0.0,
        "duplicates": [],
    }


def _parse_warnings(report: EvidenceReport, findings: list[PhotoFinding]) -> None:
    """Say out loud what the parser could not read, and what it read too much of.

    Neither of these truncates anything. An ungraded finding is still listed
    with its photograph and weighs zero; a photo that returned thirty findings
    keeps all thirty, because capping the list would delete evidence and
    per-family saturation has already removed the incentive to pad it.
    """
    ungraded = report.condition.ungraded_findings if report.condition else 0
    if ungraded:
        report.parse_warnings.append(
            f"{ungraded} finding(s) came back with a severity or price impact that "
            f"is not in the enum; they are shown with their photo and weighted at "
            f"zero rather than rounded up to minor")
    for finding in findings:
        if len(finding.issues) > condition_stage.FINDINGS_PER_PHOTO_WARN:
            report.parse_warnings.append(
                f"photo {finding.photo_id} returned {len(finding.issues)} findings; "
                f"that is a list, not an inspection - none were dropped, but the "
                f"subsystem they land on is capped")


# --- pass D ---------------------------------------------------------------

# Grade ranks, worst last. Read-only here: `GRADE_BY_WORST` above is what the
# deterministic rollup uses, and pass D only ever RELAXES a grade the model
# already gave, never tightens one.
_GRADE_RANK = {"excellent": 0, "good": 1, "fair": 2, "poor": 3}


def _calibrate(chain, client, report: EvidenceReport, findings, *,
               vehicle_line: str, expectation: str) -> None:
    """Re-decide every severity with the whole list in view. Never fatal."""
    if not report.issues:
        return
    t = time.time()
    coverage = ""
    if report.coverage_gaps:
        coverage = "Still unassessed across the whole set: " + \
                   "; ".join(g.rstrip(".") for g in report.coverage_gaps[:4]) + "."
    try:
        _, response = _walk(
            chain, client,
            lambda c: calibration.calibrate(
                c, report.issues, vehicle_line=vehicle_line,
                expectation=expectation, n_photos=report.photos_read,
                coverage=coverage, strengths=report.confirmed_sound,
                counts=passes.component_frame_counts(findings),
                samples={f.photo_id: f.samples for f in findings},
                max_tokens=CALIBRATION_MAX_TOKENS),
            report, "calibration")
        try:
            data = calibration.parse_calibration(response.text)
        except (ValueError, json.JSONDecodeError) as exc:
            response = _repair(client, response, exc, prompts.CALIBRATION_SCHEMA,
                               CALIBRATION_MAX_TOKENS)
            data = calibration.parse_calibration(response.text)
            report.parse_warnings.append(
                f"the calibration pass was unparseable ({exc}); repaired")
        corrections, warnings = calibration.apply_revisions(report.issues, data)
        report.corrections.extend(corrections)
        report.parse_warnings.extend(warnings)
        report.calibration_note = " ".join(x for x in (
            data["calibration_note"],
            calibration.worst_note(report.issues, data["worst_finding"])) if x)
        report.calls.append(["calibration", response.elapsed_s])
    except (vlm.VLMError, ValueError, json.JSONDecodeError) as exc:
        # The provisional severities stand. Every finding still exists, still
        # cites its photo and still says what it saw - what is lost is the one
        # pass that could see them all at once, and the reader is told that.
        report.parse_warnings.append(
            f"the calibration pass failed ({exc}); the severities below are the "
            f"per-photo ones, decided without the rest of the set in view")
        report.calls.append(["calibration (failed)", round(time.time() - t, 2)])


# --- the whole stage -------------------------------------------------------

def run(gate: GateReport, declared: dict | None = None, *,
        backend: str | None = None, limit: int = MAX_EVIDENCE_PHOTOS,
        on_photo=None, on_activity=None) -> EvidenceReport:
    t0 = time.time()
    selected = select_photos(gate, limit)
    report = EvidenceReport()
    if not selected:
        report.parse_warnings.append("no usable photos to send")
        report.elapsed_s = round(time.time() - t0, 2)
        return report

    chain = vlm.resolve_chain(backend)

    # --- pass A: what is this truck, and is it one truck ------------------
    # Its own photo selection, and read `IDENTITY_SAMPLES` times on one
    # backend. This is the most load-bearing call in the run - `make` picks the
    # brand column in the price model, `model` picks the anchor row,
    # `body_type` can stop the pricing stage and `same_vehicle` can stop the
    # valuation - and until now it was the only call in the pipeline still
    # decided by a single draw.
    # `limit` is pass B's budget and IDENTITY_PHOTOS is pass A's; the smaller
    # wins, so a caller asking for a cheap four-frame run gets one.
    identity_photos = select_identity_photos(gate, min(IDENTITY_PHOTOS, limit))
    if on_activity:
        on_activity({"phase": "identity", "detail": "Identifying the truck and checking that the photos show the same vehicle"})
    read = sampling.identity_consensus(chain, identity_photos, declared,
                                       max_tokens=IDENTITY_MAX_TOKENS, repair=_repair)
    client = read.client
    report.vehicle = read.vehicle
    report.same_vehicle, report.vehicle_mismatch = read.same_vehicle, read.vehicle_mismatch
    report.backend, report.model = read.backend, read.model
    report.fell_back_from = list(read.fallbacks)
    report.corrections.extend(read.corrections)
    report.parse_warnings.extend(read.warnings)
    report.calls.extend(read.calls)
    vehicle_line = passes.vehicle_line(report.vehicle)

    # The truck's own baseline, composed once. `declared` has been in hand
    # since the top of this function - what was missing was anywhere to put it:
    # a 2021 tractor at 400,000 km and the same one at 80,000 km used to get
    # byte-identical close-up prompts, so the only reference either had for
    # "worn" was a new truck. The distance goes in as a BAND, never the figure,
    # so it cannot be echoed back as an odometer reading.
    expectation = passes.expectation_line(declared)
    band = prompts.wear_band(passes._int((declared or {}).get("km")))

    # --- pass A2 and pass B, sharing one temp directory for their crops ----
    with passes.tempdir() as tmp:
        if BADGE_READ:
            if on_activity:
                on_activity({"phase": "badge", "detail": "Reading the make and model badges"})
            _badge(chain, client, report, gate, Path(tmp))
        findings = _fan_out(chain, selected, vehicle_line, Path(tmp), on_photo,
                            report, expectation=expectation, band=band,
                            vehicle=report.vehicle, **({"on_activity": on_activity} if on_activity else {}))
    findings.sort(key=lambda f: f.photo_id)
    report.photo_findings = findings
    report.photos_read = sum(1 for f in findings if not f.error)
    report.photos_failed = sum(1 for f in findings if f.error)
    if not report.photos_read:
        raise vlm.VLMError(
            f"every one of the {len(findings)} per-photo vision calls failed; "
            f"first was: {findings[0].error if findings else 'unknown'}")

    # The odometer is whatever a frame actually managed to read, preferring the
    # view that is looking straight at one. Never a guess: `_odometer` already
    # dropped anything that was not a plausible reading.
    read_it = [f for f in findings if f.odometer_km]
    read_it.sort(key=lambda f: (f.view != "dashboard_odometer", -f.confidence))
    if read_it:
        report.vehicle.odometer_km = read_it[0].odometer_km
        report.vehicle.odometer_photo_id = read_it[0].photo_id

    if on_activity:
        on_activity({"phase": "synthesis", "detail": "Cross-checking observations and combining the photo findings"})

    # --- pass C: roll it up -----------------------------------------------
    t = time.time()
    notes, flat = passes.notes_block(findings)
    try:
        client, (synth_response, flat) = _walk(
            chain, client,
            lambda c: passes.synthesize(c, findings, vehicle_line,
                                        max_tokens=SYNTHESIS_MAX_TOKENS),
            report, "synthesis")
        try:
            data = passes.parse_synthesis(synth_response.text)
        except (ValueError, json.JSONDecodeError) as exc:
            synth_response = _repair(client, synth_response, exc,
                                     prompts.synthesis_schema(), SYNTHESIS_MAX_TOKENS)
            data = passes.parse_synthesis(synth_response.text)
            report.parse_warnings.append(f"synthesis was unparseable ({exc}); repaired")
        report.calls.append(["synthesis", synth_response.elapsed_s])
        report.raw_text = synth_response.text
    except vlm.VLMError as exc:
        data = _fallback_synthesis(findings, flat)
        report.parse_warnings.append(
            f"the synthesis call failed ({exc}); the findings below are the "
            f"per-photo results grouped without it")
        report.calls.append(["synthesis (failed)", round(time.time() - t, 2)])

    report.issues = passes.merge_duplicates(flat, data["duplicates"])
    report.condition_summary = data["condition_summary"]

    # --- pass D: the severity the whole set decides -----------------------
    # Between the merge and the rollup, and neither side of that is negotiable.
    # After the merge, because the pre-merge list still has three paraphrases of
    # one worn drive tire in it and calibrating that list calibrates the wrong
    # one. Before the rollup, because the rollup is where severity becomes both
    # the grade and the price - run it first and pass D's work is computed and
    # then thrown away.
    report.confirmed_sound = calibration.confirmed_sound(findings)
    _calibrate(chain, client, report, findings, vehicle_line=vehicle_line,
               expectation=expectation)
    # One rollup, two consumers. The grade below and the price multiplier in
    # `pricing.condition_adjustment` are both functions of this object, so they
    # cannot disagree about which truck is worse - which they used to, and in
    # both directions.
    report.condition = condition_stage.rollup(report.issues, findings, gate=gate)
    report.condition_grade = report.condition.grade
    # The synthesis pass still grades, and its answer is kept as a second
    # opinion that is shown and not obeyed: the deterministic one wins because
    # it is reproducible and testable. Same posture as `fell_back_from`.
    report.condition_grade_model = data["condition_grade"]
    if (report.condition_grade_model in prompts.GRADES
            and report.condition_grade_model != report.condition_grade):
        report.grade_disagreement = (
            f"the synthesis pass graded this {report.condition_grade_model}; the "
            f"deterministic rollup grades it {report.condition_grade}")
    report.coverage_gaps = data["coverage_gaps"] or list(dict.fromkeys(
        g for f in findings for g in f.cannot_tell))[:10]
    report.confidence = data["confidence"]
    _parse_warnings(report, findings)
    if not report.raw_text:
        report.raw_text = notes
    report.elapsed_s = round(time.time() - t0, 2)
    return report
