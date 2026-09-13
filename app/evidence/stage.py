"""Stage 2 - the evidence fan-out, and the photo selection that feeds it.

Three passes, in order, described in `passes.py`. What this module owns is the
orchestration: which photos go, which backend answers, how many calls run at
once, what happens when one of them fails, and how each finished photo reaches
the screen while the rest are still in flight.

Failure posture, which is most of why this file exists:

  pass A fails on every backend   -> raise. There is nothing to describe.
  one close-up call fails         -> that photo carries an `error` and the
                                     appraisal continues on the other fifteen.
                                     Recorded, never swallowed.
  every close-up call fails       -> raise. A report with no observations in it
                                     is not an appraisal with thin coverage, it
                                     is a broken run wearing one.
  pass C fails                    -> deterministic rollup. Every finding already
                                     exists; the synthesis only groups them, so
                                     losing the call should not lose the work.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import condition as condition_stage
from .. import vlm
from ..config import (CLOSEUP_MAX_TOKENS, EVIDENCE_CONCURRENCY, IDENTITY_MAX_TOKENS,
                      MAX_EVIDENCE_PHOTOS, SYNTHESIS_MAX_TOKENS)
from ..schema import EvidenceReport, GateReport, PhotoFinding
from . import passes, prompts

# View priority for photo selection: what a buyer needs, in order.
VIEW_PRIORITY = ["exterior_front_34", "tire_wheel", "dashboard_odometer", "exterior_side",
                 "interior_cab", "chassis_undercarriage", "fifth_wheel", "engine_bay",
                 "damage_detail", "exterior_front", "exterior_rear"]



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

def _fan_out(client, selected, vehicle_line: str, tmpdir: Path,
             on_photo, report: EvidenceReport) -> list[PhotoFinding]:
    """One call per photo, `EVIDENCE_CONCURRENCY` at a time.

    Results are handed to `on_photo` as they land, which is out of order - the
    screen is showing real returned work, and real work does not finish in the
    order it was started.
    """
    findings: list[PhotoFinding] = []
    workers = max(1, min(EVIDENCE_CONCURRENCY, len(selected)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(passes.closeup, client, check, vehicle_line,
                        tmpdir=tmpdir, max_tokens=CLOSEUP_MAX_TOKENS): check
            for check in selected
        }
        for future in as_completed(futures):
            check = futures[future]
            try:
                finding = future.result()
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

    # --- pass B: every photo, on its own ----------------------------------
    with passes.tempdir() as tmp:
        findings = _fan_out(client, selected, vehicle_line, Path(tmp), on_photo, report)
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
        synth_response, flat = passes.synthesize(client, findings, vehicle_line,
                                                 max_tokens=SYNTHESIS_MAX_TOKENS)
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
