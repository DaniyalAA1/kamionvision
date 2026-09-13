"""The three vision passes, and the parsers that keep each one honest.

  identity()   every selected photo at once, short output. The only pass that
               sees the whole set, which is what `same_vehicle` needs: a
               text-only synthesis cannot notice that photo 9 is a different
               truck, and `pipeline.pricing_blocker` rests on that answer.
  closeup()    one photo, one call, a view-specific checklist. Where the depth
               comes from, and where `photo_id` stops being something the model
               has to remember to cite and becomes a structural fact.
  synthesize() text only, no images. Rolls the per-photo notes into the system
               summary, the grade and the coverage gaps, and says which
               findings are the same defect seen twice.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageOps

from .. import gate as gate_stage
from ..condition import (CONFIDENCE_UNPARSEABLE, IMPACT_RANK, SEVERITY_RANK,
                         family_of)
from ..schema import Issue, PhotoCheck, PhotoFinding, VehicleRead
from . import prompts

# Padding added around the subject box before cropping, as a fraction of the
# box. A box hugging the bodywork loses the ground line and the vehicle beside
# it, and both are context a buyer reads.
CROP_PAD = 0.08
# Above this the subject already fills the frame and cropping buys nothing.
CROP_MAX_SUBJECT_FRAC = 0.70
# A crop this small is not a photograph of a truck any more. 40x40 px used to
# be accepted, written as a JPEG and handed a full tread-depth checklist.
CROP_MIN_SIDE_PX = 160
CROP_MIN_AREA_FRAC = 0.08


# --- shared parsing --------------------------------------------------------

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


def _odometer(value) -> int | None:
    """A reading, or nothing. Zero and negatives are a misread, not a new truck."""
    try:
        km = int(value)
    except (TypeError, ValueError):
        return None
    return km if 0 < km < 5_000_000 else None


def _confidence(value, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


# --- the subject crop ------------------------------------------------------

def wants_crop(check: PhotoCheck) -> bool:
    """Should this photo be sent as the subject crop rather than whole?

    Only when there is something else in the frame to be confused by. A tire
    close-up has no truck box at all and a lone truck filling the frame is
    already the crop, so both are sent untouched - cropping either would throw
    away context for no gain.
    """
    box = check.subject_box
    if not box or not check.width or not check.height:
        return False
    if not gate_stage.crop_is_safe(check):
        return False
    rect = subject_crop_rect(box, check.width, check.height)
    if rect is None:
        return False
    # The PADDED rect, which is the one write_subject_crop actually cuts. The
    # unpadded box was measured before: CROP_PAD is 1.35x on area, so a subject
    # at 0.60 of the frame passed the test and then yielded a crop covering
    # 0.81 of it - while the prompt told the model other vehicles had been
    # deliberately excluded.
    frac = (rect[2] - rect[0]) * (rect[3] - rect[1]) / float(check.width * check.height)
    if frac >= CROP_MAX_SUBJECT_FRAC:
        return False
    return gate_stage.competing_vehicles(check) >= 1


def subject_crop_rect(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    """The padded subject crop, in source pixels. Same rectangle
    `write_subject_crop` actually cuts, so a box the model returns on a crop
    can be mapped back."""
    if not box or not width or not height:
        return None
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    crop = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
            min(int(width), int(x2 + pad_x)), min(int(height), int(y2 + pad_y)))
    if min(crop[2] - crop[0], crop[3] - crop[1]) < CROP_MIN_SIDE_PX:
        return None
    if (crop[2] - crop[0]) * (crop[3] - crop[1]) < CROP_MIN_AREA_FRAC * width * height:
        return None
    return crop


def write_subject_crop(check: PhotoCheck, into: Path) -> Path | None:
    """Crop to the subject box, padded, into `into`. None if it cannot."""
    if not check.subject_box:
        return None
    try:
        pil = ImageOps.exif_transpose(Image.open(check.path)).convert("RGB")
    except Exception:
        return None
    crop = subject_crop_rect(check.subject_box, pil.width, pil.height)
    if not crop:
        return None
    out = into / f"crop_{check.photo_id}.jpg"
    pil.crop(crop).save(out, format="JPEG", quality=90)
    return out


def parse_box(value) -> list[float] | None:
    """Normalised [x, y, w, h] in 0-1, or None if missing or junk.

    A guessed region is worse than none: the screen would draw a yellow box
    on the wrong part of the truck and call it evidence.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None
    w, h = min(w, 1.0 - x), min(h, 1.0 - y)
    return [x, y, w, h] if w > 0.0 and h > 0.0 else None


def map_box_to_original(box: list[float], crop: tuple[int, int, int, int],
                        width: int, height: int) -> list[float] | None:
    """Move a box from crop-normalised space onto the original photograph."""
    cx1, cy1, cx2, cy2 = crop
    cw, ch = cx2 - cx1, cy2 - cy1
    if cw <= 0 or ch <= 0 or width <= 0 or height <= 0:
        return None
    x, y, w, h = box
    return parse_box([(cx1 + x * cw) / float(width),
                      (cy1 + y * ch) / float(height),
                      (w * cw) / float(width),
                      (h * ch) / float(height)])


def _observation_box(raw, check: PhotoCheck, *, cropped: bool) -> list[float] | None:
    box = parse_box(raw)
    if not box or not cropped:
        return box
    crop = subject_crop_rect(check.subject_box, check.width, check.height)
    return map_box_to_original(box, crop, check.width, check.height) if crop else None


# --- pass A: identity ------------------------------------------------------

def parse_identity(text: str) -> tuple[VehicleRead, bool, str]:
    data = extract_json(text)
    vehicle = VehicleRead(
        make=data.get("make") or None,
        model=data.get("model") or None,
        body_type=data.get("body_type") or None,
        cab_type=data.get("cab_type") or None,
        axle_config=data.get("axle_config") or None,
        approx_year_range=data.get("approx_year_range") or None,
        badges_seen=[str(b) for b in (data.get("badges_seen") or [])][:8],
        confidence=_confidence(data.get("confidence"), 0.0),
    )
    return (vehicle, bool(data.get("same_vehicle", True)),
            str(data.get("vehicle_mismatch") or "").strip())


def identity_context(declared: dict | None, selected: list) -> str:
    lines = [f"You are looking at {len(selected)} photos of what should be one vehicle, "
             f"photo_id 0 to {len(selected) - 1}."]
    hints = ", ".join(f"photo_id {i}: {c.view.replace('_', ' ')}"
                      for i, c in enumerate(selected))
    lines.append(f"A zero-shot classifier tagged them as - {hints}. "
                 "Those tags are a hint and are sometimes wrong; trust the pixels.")
    if declared:
        stated = ", ".join(f"{k}={v}" for k, v in declared.items() if v not in (None, ""))
        if stated:
            lines.append(f"The seller states: {stated}. Treat this as a claim, not a "
                         "fact - if the cab generation you can see contradicts it, "
                         "report what you see.")
    return "\n".join(lines)


def identity(client, selected: list, declared: dict | None, *, max_tokens: int):
    prompt = prompts.IDENTITY_PROMPT.format(
        context=identity_context(declared, selected))
    return client.complete(
        prompt, [Path(c.path) for c in selected], system=prompts.SYSTEM,
        max_tokens=max_tokens,
        json_schema=prompts.IDENTITY_SCHEMA if client.supports_structured_output else None)


# --- pass B: one photo, in depth -------------------------------------------

def parse_closeup(text: str, check: PhotoCheck, *, cropped: bool) -> PhotoFinding:
    """Model JSON for one photo into a PhotoFinding.

    `photo_id` comes from the caller, never from the response. The call was
    given exactly one photograph, so the binding between a claim and the frame
    it was seen in is structural here - there is no index for the model to get
    wrong and no uncited issue to detect and drop.
    """
    data = extract_json(text)
    finding = PhotoFinding(
        photo_id=check.photo_id, view=check.view,
        shows=str(data.get("shows") or "").strip(),
        legible=bool(data.get("legible", True)),
        cropped=cropped,
        # Uncapped, like `observations`. Truncating only the good news was
        # itself a bias: a report that can list thirty faults and six virtues
        # is not describing the same truck twice.
        strengths=[str(s).strip() for s in (data.get("strengths") or []) if str(s).strip()],
        cannot_tell=[str(s).strip() for s in (data.get("cannot_tell") or []) if str(s).strip()][:6],
        odometer_km=_odometer(data.get("odometer_km")),
        confidence=_confidence(data.get("confidence"), 0.0),
    )
    for entry in data.get("observations") or []:
        component = str(entry.get("component") or "").strip()
        if component not in prompts.COMPONENTS:
            component = component or "other"
        observation = str(entry.get("observation") or "").strip()
        if not observation:
            continue
        # Every default here rounds DOWN. An unreadable severity used to become
        # "minor" and an unreadable impact "low", so a field nobody could parse
        # was worth real money; an unreadable confidence became 0.5, which is
        # where the weight table says the model half believes it. The honest
        # state is "the model did not grade this finding" - kept, shown with
        # its photograph, and weighted at zero.
        severity = _clamp(entry.get("severity"), prompts.SEVERITIES, None)
        impact = _clamp(entry.get("price_impact"), prompts.IMPACTS, None)
        finding.issues.append(Issue(
            photo_id=check.photo_id, component=component, observation=observation,
            severity=severity or "cosmetic",
            confidence=_confidence(entry.get("confidence"), CONFIDENCE_UNPARSEABLE),
            price_impact=impact or "none",
            ungraded=severity is None or impact is None,
            box=_observation_box(entry.get("box"), check, cropped=cropped)))
    return finding


def closeup(client, check: PhotoCheck, vehicle_line: str, *,
            tmpdir: Path, max_tokens: int) -> PhotoFinding:
    """One photo, one call. Raises VLMError; the caller decides what that costs."""
    from ..vision import VIEW_LABELS

    t0 = time.time()
    cropped_path = write_subject_crop(check, tmpdir) if wants_crop(check) else None
    image = cropped_path or Path(check.path)
    pretty = {k: k.replace("_", " ") for k in VIEW_LABELS}.get(
        check.view, check.view.replace("_", " "))
    prompt = prompts.closeup_prompt(
        view=check.view, view_pretty=pretty, vehicle=vehicle_line,
        cropped=cropped_path is not None,
        soft=any("soft focus" in r for r in check.reasons))

    response = client.complete(
        prompt, [image], system=prompts.SYSTEM, max_tokens=max_tokens,
        json_schema=prompts.closeup_schema() if client.supports_structured_output else None)
    finding = parse_closeup(response.text, check, cropped=cropped_path is not None)
    finding.elapsed_s = round(time.time() - t0, 2)
    return finding


# --- pass C: synthesis -----------------------------------------------------

def notes_block(findings: list[PhotoFinding]) -> tuple[str, list[Issue]]:
    """The per-photo notes as text, plus the flat issue list they are numbered by."""
    lines, flat = [], []
    for finding in findings:
        if finding.error:
            continue
        head = f"Photo {finding.photo_id} ({finding.view.replace('_', ' ')})"
        if not finding.legible:
            head += " - marked not legible"
        lines.append(f"{head}: {finding.shows or 'no description returned'}")
        for issue in finding.issues:
            flat.append(issue)
            lines.append(f"  [{len(flat) - 1}] {issue.component}: {issue.observation} "
                         f"({issue.severity}, {issue.price_impact} price impact)")
        for good in finding.strengths:
            lines.append(f"  ok: {good}")
        for gap in finding.cannot_tell:
            lines.append(f"  cannot tell: {gap}")
        lines.append("")
    return "\n".join(lines), flat


def parse_synthesis(text: str) -> dict:
    data = extract_json(text)
    summary = data.get("condition_summary") or {}
    return {
        "condition_summary": {k: str(summary.get(k) or "not visible in these photos")
                              for k in prompts.SUMMARY_KEYS},
        "condition_grade": _clamp(data.get("condition_grade"), prompts.GRADES, "unknown"),
        "coverage_gaps": [str(g) for g in (data.get("coverage_gaps") or [])][:10],
        "headline": str(data.get("headline") or "").strip(),
        "confidence": _confidence(data.get("confidence"), 0.0),
        "duplicates": data.get("duplicates") or [],
    }


# Words that carry no information about which defect is being described, so
# they must not contribute to the overlap score.
_NOISE = {
    "the", "a", "an", "and", "or", "but", "with", "without", "from", "that",
    "this", "there", "here", "which", "where", "when", "are", "is", "was",
    "were", "has", "have", "had", "its", "it", "of", "on", "in", "at", "to",
    "for", "by", "as", "no", "not", "any", "some", "all", "both", "one", "two",
    "visible", "appears", "appear", "shows", "showing", "seen", "photo",
    "image", "frame", "side", "across", "along", "around", "near", "over",
    "under", "than", "more", "most", "less", "very", "rather", "still",
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _NOISE}


# Below this many content words, containment is meaningless - every token of a
# three-word note lands inside almost any long one.
CONTAINMENT_MIN_TOKENS = 4
# ASSUMED. Full containment of a short observation inside a long one scores
# this, so it is strong evidence of one defect but not automatically a merge.
CONTAINMENT_FACTOR = 0.75


def _overlap(a: str, b: str) -> float:
    """How likely two observations are the same defect. 1.0 is certain.

    Jaccard alone could not merge a short description with a long one, and the
    fan-out produces wildly different lengths for one defect because each call
    sees a different frame of it. Measured on a real pair - "outer shoulder
    worn to the wear bars" against a twenty-word version of the same sentence -
    Jaccard scores 0.23 and does not merge, while containment scores 1.00:
    every content word of the short note is inside the long one. So the score
    is the better of the two, with containment discounted.
    """
    left, right = (a or "").strip().lower(), (b or "").strip().lower()
    if left and left == right:
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    if min(len(ta), len(tb)) < CONTAINMENT_MIN_TOKENS:
        return jaccard
    containment = len(ta & tb) / min(len(ta), len(tb))
    return max(jaccard, CONTAINMENT_FACTOR * containment)


# Above this, two observations on the same component id are the same defect
# seen twice. Set deliberately high: merging two genuinely different defects
# hides one, which is worse than listing one twice. It stays where it was -
# per-family saturation has already capped what an unmerged duplicate can cost,
# so the price of under-merging has collapsed while the price of over-merging
# has not moved.
SAME_DEFECT = 0.42
# ASSUMED, and higher, because across two component ids the id itself no longer
# corroborates: `corrosion` and `chassis_frame` are one physical defect often
# enough to be worth folding, and two different defects often enough that the
# wording has to carry more of the argument.
SAME_DEFECT_CROSS_ID = 0.50
# ASSUMED. Corroborating witnesses here are the same model looking at the same
# truck, so they are correlated and a plain noisy-OR would overstate them. Each
# extra frame contributes this share of its own confidence.
CORROBORATION_WEIGHT = 0.35
CONFIDENCE_CEILING = 0.95


def corroborated_rank(ranks: list[int]) -> int:
    """The highest level two frames independently support.

    Not the max, which is the exaggeration machine - one call in sixteen
    calling a scuff `major` should not grade the truck. Not the median either,
    which throws away a corroborated escalation. And not "whichever photo came
    first", which is what the old merge kept by accident of ordering.

      [minor, major]                  -> minor
      [moderate, major]               -> moderate
      [major, major]                  -> major
      [minor, major, major]           -> major
      [minor, minor, moderate, major] -> moderate
    """
    if len(ranks) == 1:
        return ranks[0]
    return max(r for r in sorted(set(ranks))
               if sum(1 for x in ranks if x >= r) >= 2)


def merged_confidence(confidences: list[float]) -> float:
    """Discounted noisy-OR: 0.6 -> 0.60, twice -> 0.684, three times -> 0.750.

    Monotone, bounded, and it never reaches 1.0. Merge is the only stage
    allowed to move confidence UP, and only on corroboration; `reconcile` is
    the only stage allowed to move it down, and it records a `Correction` each
    time. Merge runs first, so the degradation head still gets the last word.
    """
    ordered = sorted((c for c in confidences
                      if isinstance(c, (int, float)) and math.isfinite(c)), reverse=True)
    if not ordered:
        return 0.0
    miss = 1.0 - ordered[0]
    for extra in ordered[1:]:
        miss *= 1.0 - CORROBORATION_WEIGHT * extra
    return min(CONFIDENCE_CEILING, round(1.0 - miss, 3))


def _same_defect(a: Issue, b: Issue) -> bool:
    """One physical defect, described twice.

    Gated on the component FAMILY rather than on an exact id. The 33-id enum
    has real semantic overlap - `corrosion` / `chassis_frame` / `undercarriage`
    are one place on the truck, as are `fluid_leaks` / `engine_bay` and
    `paint_finish` / `cab_exterior_panels` - so requiring the ids to match
    meant one defect reported under two of them could never merge, however
    identically it was worded.
    """
    if family_of(a.component) != family_of(b.component):
        return False
    threshold = SAME_DEFECT if a.component == b.component else SAME_DEFECT_CROSS_ID
    return _overlap(a.observation, b.observation) >= threshold


def _resolve(survivor: Issue, group: list[Issue]) -> None:
    """Give the survivor the severity, impact and confidence the group supports."""
    graded = [i for i in group if not i.ungraded]
    if graded:
        survivor.ungraded = False
        ranks = [SEVERITY_RANK[i.severity] for i in graded if i.severity in SEVERITY_RANK]
        if ranks:
            by_rank = {SEVERITY_RANK[s]: s for s in SEVERITY_RANK}
            survivor.severity = by_rank[corroborated_rank(ranks)]
            claimed = [by_rank[r] for r in sorted(set(ranks), reverse=True)]
            survivor.severity_span = claimed if len(claimed) > 1 else []
        impacts = [IMPACT_RANK[i.price_impact] for i in graded
                   if i.price_impact in IMPACT_RANK]
        if impacts:
            by_impact = {IMPACT_RANK[k]: k for k in IMPACT_RANK}
            survivor.price_impact = by_impact[corroborated_rank(impacts)]
    survivor.confidence = merged_confidence([i.confidence for i in group])


def merge_duplicates(flat: list[Issue], duplicates: list) -> list[Issue]:
    """Fold repeated sightings of one defect into a single cited finding.

    Sixteen per-photo calls each report the same worn drive tire in their own
    words. The synthesis pass is asked to name those duplicates, and it catches
    the obvious ones, but it missed three paraphrases of one shoulder-worn
    drive tire on the first real run - which then read as three major findings
    and dragged the condition grade down with them. So the model's answer is
    taken first and a deterministic pass over content-word overlap runs behind
    it, within a component family.

    The survivor keeps the photo it cites, because the report points a reader
    at a frame and that must not move. It does NOT keep its own severity: it
    takes the highest level two frames independently support, and the levels
    that lost are printed on `severity_span` rather than discarded. The
    corroborating photos land on `also_seen_in` - "seen in three photos" is a
    trust signal a buyer can check, which is more than a confidence decimal
    ever was.
    """
    merged_away: set[int] = set()
    extra: dict[int, list[int]] = {}
    members: dict[int, list[int]] = {}
    for entry in duplicates or []:
        try:
            keep = int(entry.get("keep"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not 0 <= keep < len(flat) or keep in merged_away:
            continue
        for other in entry.get("merge") or []:
            try:
                idx = int(other)
            except (TypeError, ValueError):
                continue
            if idx == keep or not 0 <= idx < len(flat) or idx in merged_away:
                continue
            merged_away.add(idx)
            extra.setdefault(keep, []).append(flat[idx].photo_id)
            members.setdefault(keep, []).append(idx)

    # Second pass: paraphrases of one defect the model did not pair up. Only
    # within a family, and only above a high overlap, because collapsing two
    # genuinely different defects loses one.
    survivors = [i for i in range(len(flat)) if i not in merged_away]
    for pos, i in enumerate(survivors):
        if i in merged_away:
            continue
        for j in survivors[pos + 1:]:
            if j in merged_away or not _same_defect(flat[i], flat[j]):
                continue
            merged_away.add(j)
            extra.setdefault(i, []).append(flat[j].photo_id)
            extra[i].extend(flat[j].also_seen_in or [])
            members.setdefault(i, []).append(j)

    out = []
    for i, issue in enumerate(flat):
        if i in merged_away:
            continue
        seen = [p for p in dict.fromkeys(extra.get(i, [])) if p != issue.photo_id]
        issue.also_seen_in = seen
        if members.get(i):
            _resolve(issue, [issue] + [flat[j] for j in members[i]])
        out.append(issue)
    return out


def synthesize(client, findings: list[PhotoFinding], vehicle_line: str, *,
               max_tokens: int):
    notes, flat = notes_block(findings)
    prompt = prompts.synthesis_prompt(
        n=len([f for f in findings if not f.error]), vehicle=vehicle_line, notes=notes)
    response = client.complete(
        prompt, [], system=prompts.SYSTEM, max_tokens=max_tokens,
        json_schema=prompts.synthesis_schema() if client.supports_structured_output else None)
    return response, flat


def vehicle_line(vehicle: VehicleRead) -> str:
    """One line naming the truck, threaded into passes B and C for context."""
    named = " ".join(x for x in (vehicle.make, vehicle.model) if x).strip()
    if not named:
        return "The make and model could not be read from the photos."
    spec = ", ".join(x for x in (vehicle.cab_type, vehicle.axle_config,
                                 vehicle.approx_year_range) if x)
    return (f"The vehicle has been identified from the full set as a {named}"
            + (f" ({spec})" if spec else "") + ".")


def tempdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="kamion-crop-")
