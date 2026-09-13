"""Stage 2 - the evidence fan-out, and the photo selection that feeds it.

Four passes now, described in `passes.py` and `calibration.py`. What this
module owns is the orchestration: which photos go, which backend answers, how
many calls run at once, what happens when one of them fails, and how each
finished photo reaches the screen while the rest are still in flight.

Pass D is the newest and it is sequenced where it is for a reason that is not
negotiable: it runs AFTER `merge_duplicates`. Severity cannot be decided
set-aware until the duplicates are folded, because `notes_block` numbers the
pre-merge flat list and three paraphrases of one worn drive tire are still
three rows in it.

Failure posture, which is most of why this file exists:

  pass A fails on every backend   -> raise. There is nothing to describe.
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

from .. import vlm
from ..config import (CALIBRATION_MAX_TOKENS, CLOSEUP_MAX_TOKENS,
                      EVIDENCE_CONCURRENCY, IDENTITY_MAX_TOKENS,
                      MAX_EVIDENCE_PHOTOS, SYNTHESIS_MAX_TOKENS)
from ..schema import Correction, EvidenceReport, GateReport, PhotoFinding
from . import calibration, passes, prompts, sampling

# View priority for photo selection: what a buyer needs, in order.
VIEW_PRIORITY = ["exterior_front_34", "tire_wheel", "dashboard_odometer", "exterior_side",
                 "interior_cab", "chassis_undercarriage", "fifth_wheel", "engine_bay",
                 "damage_detail", "exterior_front", "exterior_rear"]

GRADE_BY_WORST = {"major": "poor", "moderate": "fair", "minor": "good", "cosmetic": "good"}
SEVERITY_RANK = {"major": 3, "moderate": 2, "minor": 1, "cosmetic": 0}


def select_photos(gate: GateReport, limit: int = MAX_EVIDENCE_PHOTOS) -> list:
    """View-diverse subset of the usable photos, best capture quality first.

    A seller uploads 15-40 frames and most of them are the same three angles.
    Round-robin over views first, so one slot is spent per view before any
    view gets a second, then fill the remainder by capture quality.
    """
    usable = [c for c in gate.photos if c.usable]
    by_view: dict[str, list] = {}
    for check in sorted(usable, key=lambda c: -c.capture_quality):
        by_view.setdefault(check.view, []).append(check)

    order = [v for v in VIEW_PRIORITY if v in by_view] + \
            [v for v in by_view if v not in VIEW_PRIORITY]
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


# --- pass A, with the backend chain ---------------------------------------

def _identity(chain, selected, declared, report: EvidenceReport):
    """Walk every usable backend rather than only the best one.

    A provider that rate-limits or 500s halfway through a live demo should cost
    one retry against the next provider, not the appraisal. The fallback is
    recorded on the report and shown on screen - falling back is allowed,
    doing it quietly is not.
    """
    failures: list[str] = []
    for candidate in chain:
        try:
            response = passes.identity(candidate, selected, declared,
                                       max_tokens=IDENTITY_MAX_TOKENS)
            return candidate, response, failures
        except vlm.VLMError as exc:
            failures.append(f"{candidate.name}: {exc}")
    raise vlm.VLMError("every vision backend failed:\n  " + "\n  ".join(failures))


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


# --- pass B, concurrent ----------------------------------------------------

def _fan_out(chain, selected, vehicle_line: str, tmpdir: Path,
             on_photo, report: EvidenceReport, *, expectation: str = "",
             band: str | None = None) -> list[PhotoFinding]:
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
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(sampling.closeup_consensus, chain, check, vehicle_line,
                        tmpdir=tmpdir, max_tokens=CLOSEUP_MAX_TOKENS,
                        expectation=expectation, band=band, repair=_repair): check
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
    this point - pass C only groups and grades them - so losing that call
    should cost the prose, not the findings.
    """
    summary: dict[str, list[str]] = {}
    for issue in flat:
        key = prompts.COMPONENT_SUMMARY.get(issue.component)
        if key:
            summary.setdefault(key, []).append(issue.observation)
    worst = max((SEVERITY_RANK.get(i.severity, 0) for i in flat), default=-1)
    grade = {3: "poor", 2: "fair", 1: "good", 0: "good"}.get(worst, "unknown")
    gaps = []
    for finding in findings:
        gaps.extend(finding.cannot_tell)
    return {
        "condition_summary": {
            k: ("; ".join(summary[k][:3]) if k in summary else "not visible in these photos")
            for k in prompts.SUMMARY_KEYS},
        "condition_grade": grade,
        "coverage_gaps": list(dict.fromkeys(gaps))[:10],
        "headline": "",
        "confidence": 0.0,
        "duplicates": [],
    }


# --- pass D ---------------------------------------------------------------

# Grade ranks, worst last. Read-only here: `GRADE_BY_WORST` above is what the
# deterministic rollup uses, and pass D only ever RELAXES a grade the model
# already gave, never tightens one.
_GRADE_RANK = {"excellent": 0, "good": 1, "fair": 2, "poor": 3}


def _relax_grade(report: EvidenceReport) -> None:
    """Stop the grade contradicting the list it is supposed to be a summary of.

    The synthesis pass grades before pass D re-decides the severities, so a
    truck whose worst finding has just been calibrated down from "major" to
    "moderate" could still be handed to a seller graded "poor" on the strength
    of a finding that no longer exists at that level. One-way: the calibrated
    list can only ever let a grade up, never push one down, because the grade
    also carries coverage reasoning that a severity list knows nothing about.
    """
    if not report.issues:
        return
    worst = max(report.issues, key=lambda i: SEVERITY_RANK.get(i.severity, 0))
    supported = GRADE_BY_WORST.get(worst.severity)
    current = _GRADE_RANK.get(report.condition_grade)
    if not supported or current is None:
        return
    if _GRADE_RANK[supported] < current:
        report.corrections.append(Correction(
            kind="severity_calibrated", photo_id=None,
            before=report.condition_grade, after=supported,
            detail=f"the grade was set before the severities were calibrated; the "
                   f"worst distinct finding left on this truck is "
                   f"\"{worst.severity}\" on the "
                   f"{worst.component.replace('_', ' ')}"))
        report.grade_disagreement = (
            f"graded {report.condition_grade} before calibration, "
            f"{supported} after")
        report.condition_grade = supported


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
        _relax_grade(report)
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
        on_photo=None) -> EvidenceReport:
    t0 = time.time()
    selected = select_photos(gate, limit)
    report = EvidenceReport()
    if not selected:
        report.parse_warnings.append("no usable photos to send")
        report.elapsed_s = round(time.time() - t0, 2)
        return report

    chain = vlm.resolve_chain(backend)

    # --- pass A: what is this truck, and is it one truck ------------------
    client, response, failures = _identity(chain, selected, declared, report)
    try:
        vehicle, same, mismatch = passes.parse_identity(response.text)
    except (ValueError, json.JSONDecodeError) as exc:
        response = _repair(client, response, exc, prompts.IDENTITY_SCHEMA,
                           IDENTITY_MAX_TOKENS)
        vehicle, same, mismatch = passes.parse_identity(response.text)
        report.parse_warnings.insert(0, f"identity pass was unparseable ({exc}); repaired")
    report.vehicle, report.same_vehicle, report.vehicle_mismatch = vehicle, same, mismatch
    report.backend, report.model = response.backend, response.model
    report.fell_back_from = failures
    report.calls.append(["identity", response.elapsed_s])
    vehicle_line = passes.vehicle_line(vehicle)

    # The truck's own baseline, composed once. `declared` has been in hand
    # since the top of this function - what was missing was anywhere to put it:
    # a 2021 tractor at 400,000 km and the same one at 80,000 km used to get
    # byte-identical close-up prompts, so the only reference either had for
    # "worn" was a new truck. The distance goes in as a BAND, never the figure,
    # so it cannot be echoed back as an odometer reading.
    expectation = passes.expectation_line(declared)
    band = prompts.wear_band(passes._int((declared or {}).get("km")))

    # --- pass B: every photo, on its own, CLOSEUP_SAMPLES times ------------
    with passes.tempdir() as tmp:
        findings = _fan_out(chain, selected, vehicle_line, Path(tmp), on_photo,
                            report, expectation=expectation, band=band)
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
    report.condition_grade = data["condition_grade"]
    report.coverage_gaps = data["coverage_gaps"] or list(dict.fromkeys(
        g for f in findings for g in f.cannot_tell))[:10]
    report.confidence = data["confidence"]

    # --- pass D: the severity the whole set decides -----------------------
    report.confirmed_sound = calibration.confirmed_sound(findings)
    _calibrate(chain, client, report, findings, vehicle_line=vehicle_line,
               expectation=expectation)
    if not report.raw_text:
        report.raw_text = notes
    report.elapsed_s = round(time.time() - t0, 2)
    return report
