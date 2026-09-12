"""Stage 2 - one structured vision call, every claim bound to a photo.

What stops this from being "a thin wrapper that sends photos to a vision API":

  * The model never emits a price, or prose. It fills a fixed schema.
  * `component` is a closed enum of heavy-vehicle parts. A tool that offers
    dent/scratch/paint is a car tool pointed at a truck, and that is exactly
    what "sensible to someone who knows trucks" is scoring.
  * Every issue must name a `photo_id`. Issues that cite a photo that was
    never sent are dropped in `parse`, not trusted - an uncheckable claim is
    worse than a missing one.
  * Anything not visible goes to `coverage_gaps` and becomes a request for
    another photo, instead of becoming a confident guess.
  * Photos are selected view-first, so 30 frames of the same tire cost one
    slot, not thirty.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import vlm
from .config import EVIDENCE_MAX_TOKENS, MAX_EVIDENCE_PHOTOS
from .schema import EvidenceReport, GateReport, Issue, PhotoEvidence, VehicleRead

# Closed vocabulary. The model is told to use these ids and nothing else, so
# downstream severity weighting and the report's grouping are both stable.
COMPONENTS = [
    "steer_tires", "drive_tires", "wheels_rims", "brakes_hubs",
    "fifth_wheel", "coupling_airlines", "chassis_frame", "undercarriage",
    "air_suspension", "air_tanks_lines", "mudflaps_guards",
    "cab_exterior_panels", "front_bumper_valance", "fairings_skirts",
    "grille_headlights", "mirrors_visor", "roof_deflector", "windscreen_glass",
    "doors_handles", "cab_steps", "fuel_tank", "adblue_tank", "exhaust_dpf",
    "engine_bay", "fluid_leaks", "paint_finish", "corrosion",
    "cab_interior_seats", "steering_wheel_controls", "dashboard_instruments",
    "bunk_sleeper", "cab_floor_trim", "warning_lights",
]

SUMMARY_KEYS = ["tires", "wheels_brakes", "fifth_wheel_coupling", "chassis_corrosion",
                "body_paint", "cab_interior", "engine_driveline", "glass_lights"]

SEVERITIES = ["cosmetic", "minor", "moderate", "major"]
IMPACTS = ["none", "low", "medium", "high"]
GRADES = ["excellent", "good", "fair", "poor"]

SYSTEM = """\
You are a heavy-vehicle appraiser for a Turkish freight marketplace. You value \
second-hand tractor units (çekici) - Ford Trucks F-MAX and F-LINE, Mercedes-Benz \
Actros, MAN TGX/TGS, Scania R/S, DAF XF, Volvo FH, Iveco S-Way, Renault T, and \
North American conventionals such as Freightliner Cascadia, Kenworth, Peterbilt.

You speak about trucks the way a fleet buyer does. Tires are steer, drive or tag, \
and you talk about tread depth, shoulder and centre wear, cupping, feathering, \
sidewall cracking, mismatched brands and retreads - not "the wheels look old". \
You know what a worn fifth wheel plate, a dry coupling, corroded frame rails, a \
sagging air bag, a cracked AdBlue tank, a delaminating roof deflector and a \
collapsed seat bolster look like.

You are rigorous about what a photo can and cannot show. You never infer \
mechanical condition from a clean exterior, never report a component that is not \
visible in the photos you were given, and you say so plainly when a view is \
missing. Your notes go to a buyer who is deciding whether to drive six hours to \
see this truck."""

_SCHEMA_TEMPLATE = """\
Return ONE JSON object and nothing else. No markdown fence, no commentary.

{{
  "vehicle": {{
    "make": string|null,                 // read from badges/grille/styling, not from the seller
    "model": string|null,
    "body_type": "tractor_unit"|"rigid"|"other"|null,
    "cab_type": string|null,             // e.g. "high sleeper", "day cab"
    "axle_config": string|null,          // e.g. "4x2", "6x4"
    "approx_year_range": string|null,    // e.g. "2018-2022", from the generation you can see
    "odometer_km": integer|null,         // ONLY if you can actually read the display
    "odometer_photo_id": integer|null,
    "badges_seen": [string],             // literal text visible on the truck
    "confidence": 0.0-1.0
  }},
  "per_photo": [
    {{"photo_id": integer, "view": string, "legible": boolean, "notes": string}}
  ],
  "issues": [
    {{
      "photo_id": integer,              // REQUIRED. The photo this was seen in.
      "component": one of {components},
      "observation": string,            // what is visible, specific, truck vocabulary
      "severity": one of {severities},
      "confidence": 0.0-1.0,
      "price_impact": one of {impacts}
    }}
  ],
  "condition_summary": {{ {summary_keys} }},   // one sentence each; "not visible in these photos" if so
  "condition_grade": one of {grades},
  "coverage_gaps": [string],            // what you would need photographed to be sure
  "confidence": 0.0-1.0
}}

Rules, in order of importance:
1. Every entry in "issues" MUST carry a photo_id of a photo you were actually \
shown. If you cannot point at a photo, it is not an issue - put it in \
"coverage_gaps" instead.
2. Do not report a component you cannot see. An unphotographed fifth wheel is a \
coverage gap, not a clean fifth wheel.
3. "observation" must describe what is visible ("outer shoulder of the \
near-side steer tire is worn noticeably below the centre ribs"), not a verdict \
("tires bad").
4. Grade on what the photos support. A truck with no visible defects and thin \
coverage is not "excellent" - it is ungraded coverage, so say so in \
coverage_gaps and keep confidence low.
5. Cosmetic findings are worth reporting but must be marked severity \
"cosmetic" and price_impact "none" or "low".
6. Do NOT write photo numbers into prose. "photo_id" fields are the only place \
a photo is referenced; the renderer resolves them to filenames, and a number \
written into a sentence will not match what the buyer sees.
7. Keep it tight: at most 12 issues (the ones that move the price), one or two \
sentences per condition_summary value, and a "per_photo" entry ONLY for photos \
that are illegible or that carry a finding - not for every photo."""


def json_schema() -> dict:
    """Strict JSON Schema for backends that support constrained decoding.

    Mirrors `_SCHEMA_TEMPLATE` exactly. Every property is listed in `required`
    and nullable fields are typed `["x", "null"]` rather than omitted, because
    strict mode has no notion of an optional key - "not visible" has to be
    expressible as an explicit null.
    """
    nullable_str = {"type": ["string", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["vehicle", "per_photo", "issues", "condition_summary",
                     "condition_grade", "coverage_gaps", "confidence"],
        "properties": {
            "vehicle": {
                "type": "object",
                "additionalProperties": False,
                "required": ["make", "model", "body_type", "cab_type", "axle_config",
                             "approx_year_range", "odometer_km", "odometer_photo_id",
                             "badges_seen", "confidence"],
                "properties": {
                    "make": nullable_str,
                    "model": nullable_str,
                    "body_type": nullable_str,
                    "cab_type": nullable_str,
                    "axle_config": nullable_str,
                    "approx_year_range": nullable_str,
                    "odometer_km": {"type": ["integer", "null"]},
                    "odometer_photo_id": {"type": ["integer", "null"]},
                    "badges_seen": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
            },
            "per_photo": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["photo_id", "view", "legible", "notes"],
                    "properties": {
                        "photo_id": {"type": "integer"},
                        "view": {"type": "string"},
                        "legible": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                },
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["photo_id", "component", "observation", "severity",
                                 "confidence", "price_impact"],
                    "properties": {
                        "photo_id": {"type": "integer"},
                        "component": {"type": "string", "enum": COMPONENTS},
                        "observation": {"type": "string"},
                        "severity": {"type": "string", "enum": SEVERITIES},
                        "confidence": {"type": "number"},
                        "price_impact": {"type": "string", "enum": IMPACTS},
                    },
                },
            },
            "condition_summary": {
                "type": "object",
                "additionalProperties": False,
                "required": list(SUMMARY_KEYS),
                "properties": {k: {"type": "string"} for k in SUMMARY_KEYS},
            },
            "condition_grade": {"type": "string", "enum": GRADES + ["unknown"]},
            "coverage_gaps": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
    }


def schema_block() -> str:
    return _SCHEMA_TEMPLATE.format(
        components=json.dumps(COMPONENTS),
        severities=json.dumps(SEVERITIES),
        impacts=json.dumps(IMPACTS),
        grades=json.dumps(GRADES),
        summary_keys=", ".join(f'"{k}": string' for k in SUMMARY_KEYS),
    )


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
    picked, seen = [], set()
    while len(picked) < limit:
        progressed = False
        for view in order:
            bucket = by_view[view]
            if not bucket:
                continue
            check = bucket.pop(0)
            picked.append(check)
            seen.add(check.photo_id)
            progressed = True
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked


def build_prompt(declared: dict | None, selected: list, gate: GateReport) -> str:
    lines = [schema_block(), "", "---", ""]
    lines.append(f"You are looking at {len(selected)} photos of one vehicle, "
                 f"photo_id 0 to {len(selected) - 1}.")
    hints = ", ".join(f"photo_id {i}: {c.view.replace('_', ' ')}"
                      for i, c in enumerate(selected))
    lines.append(f"A zero-shot classifier tagged them as - {hints}. "
                 "Those tags are a hint and are sometimes wrong; trust the pixels.")

    soft = [f"photo_id {i}" for i, c in enumerate(selected)
            if any("soft focus" in r for r in c.reasons)]
    if soft:
        lines.append(f"{', '.join(soft)} are soft-focus - do not claim fine detail "
                     "such as tread depth from them; mark them not legible instead.")

    if declared:
        stated = ", ".join(f"{k}={v}" for k, v in declared.items() if v not in (None, ""))
        if stated:
            lines.append("")
            lines.append(f"The seller states: {stated}. Treat this as a claim, not a fact. "
                         "If the photos contradict it - a different generation of cab, an "
                         "odometer that does not match the stated kilometres - say so in "
                         "condition_summary and raise it as an issue with the photo_id that "
                         "shows the contradiction.")
    lines.append("")
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a model response. Tolerant, then strict."""
    cleaned = _FENCE.sub("", text or "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, depth, in_str, esc = cleaned.find("{"), 0, False, False
    if start < 0:
        raise ValueError("no JSON object in model response")
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start:i + 1])
    raise ValueError("unterminated JSON object in model response")


def _clamp(value, options, default):
    return value if value in options else default


def parse(text: str, selected: list) -> EvidenceReport:
    """Validate and normalise the model's JSON into an EvidenceReport.

    `selected[i]` is the gate PhotoCheck the model saw as photo_id i, so this
    is also where VLM-facing indices are translated back to the gate's ids.
    Anything that cannot be anchored to a real photo is dropped and recorded
    as a parse warning rather than silently kept.
    """
    report = EvidenceReport(raw_text=text)
    data = extract_json(text)
    n = len(selected)

    def to_gate_id(value) -> int | None:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            return None
        return selected[idx].photo_id if 0 <= idx < n else None

    v = data.get("vehicle") or {}
    odo_photo = to_gate_id(v.get("odometer_photo_id"))
    odo_km = v.get("odometer_km")
    try:
        odo_km = int(odo_km) if odo_km is not None else None
    except (TypeError, ValueError):
        odo_km = None
    report.vehicle = VehicleRead(
        make=v.get("make") or None,
        model=v.get("model") or None,
        body_type=v.get("body_type") or None,
        cab_type=v.get("cab_type") or None,
        axle_config=v.get("axle_config") or None,
        approx_year_range=v.get("approx_year_range") or None,
        odometer_km=odo_km,
        odometer_photo_id=odo_photo,
        badges_seen=[str(b) for b in (v.get("badges_seen") or [])][:8],
        confidence=float(v.get("confidence") or 0.0),
    )

    for entry in data.get("per_photo") or []:
        gid = to_gate_id(entry.get("photo_id"))
        if gid is None:
            report.parse_warnings.append(
                f"per_photo entry cites photo_id {entry.get('photo_id')!r}, which was not sent")
            continue
        report.per_photo.append(PhotoEvidence(
            photo_id=gid, view=str(entry.get("view") or "unknown"),
            legible=bool(entry.get("legible", True)), notes=str(entry.get("notes") or "")))

    for entry in data.get("issues") or []:
        gid = to_gate_id(entry.get("photo_id"))
        if gid is None:
            report.parse_warnings.append(
                f"dropped an uncited issue: {str(entry.get('observation'))[:80]!r}")
            continue
        component = str(entry.get("component") or "").strip()
        if component not in COMPONENTS:
            report.parse_warnings.append(f"unknown component {component!r} normalised to 'other'")
            component = component or "other"
        try:
            confidence = float(entry.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        report.issues.append(Issue(
            photo_id=gid, component=component,
            observation=str(entry.get("observation") or "").strip(),
            severity=_clamp(entry.get("severity"), SEVERITIES, "minor"),
            confidence=max(0.0, min(1.0, confidence)),
            price_impact=_clamp(entry.get("price_impact"), IMPACTS, "low"),
        ))

    summary = data.get("condition_summary") or {}
    report.condition_summary = {k: str(summary.get(k) or "not visible in these photos")
                                for k in SUMMARY_KEYS}
    report.condition_grade = _clamp(data.get("condition_grade"), GRADES, "unknown")
    report.coverage_gaps = [str(g) for g in (data.get("coverage_gaps") or [])][:10]
    try:
        report.confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        report.confidence = 0.0
    return report


def run(gate: GateReport, declared: dict | None = None, *,
        backend: str | None = None, limit: int = MAX_EVIDENCE_PHOTOS) -> EvidenceReport:
    t0 = time.time()
    selected = select_photos(gate, limit)
    if not selected:
        report = EvidenceReport(parse_warnings=["no usable photos to send"])
        report.elapsed_s = round(time.time() - t0, 2)
        return report

    client = vlm.resolve(backend)
    prompt = build_prompt(declared, selected, gate)
    paths = [Path(c.path) for c in selected]

    response = client.complete(prompt, paths, system=SYSTEM,
                               max_tokens=EVIDENCE_MAX_TOKENS,
                               json_schema=json_schema() if client.supports_structured_output else None)
    try:
        report = parse(response.text, selected)
    except (ValueError, json.JSONDecodeError) as exc:
        # One repair attempt, text-only: cheaper than re-uploading the photos
        # and it fixes the common failure, which is a truncated or fenced
        # object rather than a misunderstood task.
        repair = client.complete(
            "That was not parseable as a single JSON object "
            f"({type(exc).__name__}: {exc}). Return the same content as ONE valid "
            "JSON object, no fence, no commentary:\n\n" + response.text[:6000],
            [], system=SYSTEM, max_tokens=EVIDENCE_MAX_TOKENS,
            json_schema=json_schema() if client.supports_structured_output else None)
        report = parse(repair.text, selected)
        report.parse_warnings.insert(0, f"first response was unparseable ({exc}); repaired")
        response = repair

    report.backend = response.backend
    report.model = response.model
    report.elapsed_s = round(time.time() - t0, 2)
    report.raw_text = response.text
    return report
