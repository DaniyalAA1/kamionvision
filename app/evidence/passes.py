"""The four vision passes, and the parsers that keep each one honest.

  identity()   every selected identity photo at once, short output. The only
               pass that sees a whole set, which is what `same_vehicle` needs:
               a text-only synthesis cannot notice that photo 9 is a different
               truck, and `pipeline.pricing_blocker` rests on that answer. Read
               `IDENTITY_SAMPLES` times by `sampling.identity_consensus`.
  badge()      one call, one crop: the badge band of the best whole-vehicle
               frame, at the highest resolution the backend will take. A second
               and independent witness to make and model, recorded on
               `VehicleRead.badge_*` and never allowed to overwrite the first -
               `app/identity.py` is the only thing that adjudicates between
               witnesses.
  closeup()    one photo, one call, a view-specific checklist. Where the depth
               comes from, and where `photo_id` stops being something the model
               has to remember to cite and becomes a structural fact.
  synthesize() text only, no images. Rolls the per-photo notes into the system
               summary, the grade and the coverage gaps, and says which
               findings are the same defect seen twice.

The fifth, `calibration.calibrate()`, lives next door because it runs after
`merge_duplicates` rather than inside this file's sequence.

Pass A and pass B no longer read the same photographs, and that is deliberate.
`stage.select_photos` round-robins on a condition-first view priority that
leads with a tire close-up; a tire contributes nothing to make, model or axle
count and costs a slot, while a side profile is the only view that can honestly
settle 4x2 against 6x2. `stage.select_identity_photos` picks for pass A instead.
Because the backends label images by POSITION - "photo_id={i}", in the order
attached - the index pass A answers with is an index into the set it was sent,
so `parse_identity` takes the real photo ids from the caller and maps them back
before `vehicle_mismatch` is put in front of a seller.

What is new here is `expectation_line`. Every close-up call used to be
byte-identical for a 2021 truck at 80,000 km and the same truck at 400,000 km,
so the only baseline the model had for "worn" was a new truck, and everything
was worn against that. The seller's year and distance are in hand before pass A
runs, and the distance goes in as a BAND: telling a close-up call the exact
figure invites it to echo that back as `odometer_km`, which would quietly
destroy the one cross-check - `reconcile._check_odometer` - that can catch a
seller understating the distance.
"""
from __future__ import annotations

import inspect
import json
import math
import re
import tempfile
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageOps

from .. import gate as gate_stage
from .. import modelspec
from ..condition import (CONFIDENCE_UNPARSEABLE, IMPACT_RANK, SEVERITY_RANK,
                         family_of)
from ..config import (BADGE_CROP_BOTTOM, BADGE_CROP_TOP, BADGE_EFFORT,
                      CLOSEUP_EFFORT, IDENTITY_EFFORT,
                      IDENTITY_IMAGE_LONG_EDGE, KM_PER_YEAR_TR,
                      SYNTHESIS_EFFORT)
from ..schema import Issue, PhotoCheck, PhotoFinding, VehicleRead
from . import calibration, prompts

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


# --- calling into other people's files -------------------------------------

def supported(fn, wanted: dict) -> dict:
    """The subset of `wanted` that `fn` will actually accept.

    `prompts.py` and `vlm/base.py` grow keywords on their own schedule, and a
    call site that hard-codes one that has not landed yet takes the whole
    appraisal down over a prompt refinement or a backend that predates the
    parameter. So the call site says what it would like to pass and this drops
    whatever cannot be heard yet - the same posture as `fell_back_from`, one
    layer down: degrading is allowed, guessing is not.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(wanted)
    return {k: v for k, v in wanted.items() if k in params}


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


# --- what this truck's distance already predicts ---------------------------

def _int(value) -> int | None:
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def km_band(km: int) -> tuple[int, int]:
    """The band a stated distance falls in. Never the figure itself.

    100,000 km wide below half a million and 250,000 above, because the
    difference between 120,000 and 180,000 km matters to a tire and the
    difference between 900,000 and 960,000 does not. A band cannot be read back
    out as a six-digit odometer reading, which is the entire point: the
    close-up call gets a reference distribution and `reconcile._check_odometer`
    keeps an independent read to check the seller against.
    """
    width = 100_000 if km < 500_000 else 250_000
    lo = (km // width) * width
    return lo, lo + width


_NO_ECHO = ('Do NOT use this band to fill in "odometer_km" - that field comes only '
            "from digits you can read in the photograph in front of you.")


def expectation_line(declared: dict | None, *, today: int | None = None) -> str:
    """What a truck of this age and distance should already look like.

    The anti-exaggeration baseline, and the case that will actually run on
    judging day is the last one: neither figure stated. It says do not assume
    old, which is the null bias that stops an unknown truck being graded
    against a new one.
    """
    from datetime import date

    declared = declared or {}
    year = _int(declared.get("year"))
    km = _int(declared.get("km"))
    now = today or date.today().year
    age = now - year if year and 1980 < year <= now + 1 else None
    q25, _median, q75 = KM_PER_YEAR_TR

    if age is None and km is None:
        return ("Neither the age nor the distance of this truck is known. Do not "
                "assume it is either new or old. Where a finding depends on how "
                'much work the truck has done, say so in "cannot_tell" rather '
                "than assuming the worst.")

    lo, hi = km_band(km) if km else (0, 0)
    band = f"{lo:,}-{hi:,} km"

    if age is None:
        return (f"The age of this truck is not stated. The seller states a distance "
                f"in the {band} band. Judge every component against the work that "
                f"distance represents: a consumable that has done its job over that "
                f"distance is on schedule and is not a finding. " + _NO_ECHO)

    aged = ("less than a year old" if age <= 0 else
            "about a year old" if age == 1 else f"about {age} years old")
    typical_lo, typical_hi = (round(age * q25, -4), round(age * q75, -4))
    if age <= 0 or typical_hi <= 0:
        typical = ""
    else:
        typical = (f"A tractor unit of this age in this market has typically covered "
                   f"{typical_lo:,.0f}-{typical_hi:,.0f} km")

    if km is None:
        if not typical:
            return (f"This truck is {aged} and the seller did not state a distance. "
                    "Judge against the age alone.")
        return (f"This truck is {aged}. The seller did not state a distance. "
                f"{typical}, but that is the spread of stock offered for sale and "
                f"not this truck. Judge against the age alone, and where a "
                f"component's state depends on distance rather than years, say in "
                f'"cannot_tell" that you would need the odometer to call it.')

    worked_less = typical and km < typical_lo
    if not typical:
        verdict = ""
    elif km > typical_hi:
        verdict = ", so this one has worked harder than most"
    elif worked_less:
        verdict = ", so this one has worked far less than most"
    else:
        verdict = ", so this one has done about the work its age predicts"

    if worked_less:
        judging = ("At that distance the original tires and the original cab trim "
                   "should still be in place and only lightly worn. Wear that would "
                   "be ordinary on a truck of this age at the usual distance is a "
                   "genuine finding on this truck - say so, and note anything that "
                   "looks like more use than the stated distance.")
    else:
        winters = "a winter" if age == 1 else f"{age} winters"
        judging = (f"Judge every component against a truck that has done that work: "
                   f"consumables at or past one full replacement cycle, a chassis "
                   f"that has stood outdoors for {winters}, a cab that has been "
                   f"lived in. A component in that state is on schedule and is not "
                   f"a finding.")

    head = f"This truck is {aged}. The seller states a distance in the {band} band."
    middle = f"{typical}{verdict}." if typical else ""
    return " ".join(x for x in (head, middle, judging, _NO_ECHO) if x)


def component_frame_counts(findings: list[PhotoFinding]) -> dict[str, int]:
    """How many of the frames that were read should show each component.

    The denominator pass D needs. A finding on the fifth wheel seen once, in
    the only frame that shows the fifth wheel, is not the same claim as the
    same finding seen once in four frames that all show it - the second is
    evidence the defect is local, and until now nothing computed either number.
    """
    counts: dict[str, int] = {}
    for finding in findings:
        if finding.error:
            continue
        for family in prompts.VIEW_FAMILIES.get(finding.view, ()):
            for component in prompts.ANCHOR_COMPONENTS.get(family, ()):
                counts[component] = counts.get(component, 0) + 1
    return counts


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


def _write_crop(check: PhotoCheck, into: Path, name: str, rect_of) -> Path | None:
    """Open the photo once, cut the rectangle `rect_of` wants, write it.

    `rect_of` is handed the dimensions of the image as PIL sees it after EXIF
    transposition, which is not always what the gate measured: a phone photo
    tagged "rotate 90" reports its stored size, and cropping a rect computed
    against that would cut the wrong part of the truck.
    """
    try:
        pil = ImageOps.exif_transpose(Image.open(check.path)).convert("RGB")
    except Exception:
        return None
    crop = rect_of(pil.width, pil.height)
    if not crop:
        return None
    out = into / name
    pil.crop(crop).save(out, format="JPEG", quality=90)
    return out


def write_subject_crop(check: PhotoCheck, into: Path) -> Path | None:
    """Crop to the subject box, padded, into `into`. None if it cannot."""
    if not check.subject_box:
        return None
    return _write_crop(check, into, f"crop_{check.photo_id}.jpg",
                       lambda w, h: subject_crop_rect(check.subject_box, w, h))


def write_badge_crop(check: PhotoCheck, into: Path) -> Path | None:
    """Crop to the badge band of the subject box. None if it cannot."""
    if not check.subject_box:
        return None
    return _write_crop(check, into, f"badge_{check.photo_id}.jpg",
                       lambda w, h: badge_crop_rect(check, width=w, height=h))


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

# "photo 3", "photo_id 3", "frame 3" - an index into the photos this call was
# actually sent, which is how the backends label them.
_PHOTO_REF = re.compile(r"\b(photo_id|photo|frame|image)\s*#?\s*(\d{1,3})\b",
                        re.IGNORECASE)


def remap_photo_refs(text: str, photo_ids: list[int] | None) -> str:
    """Rewrite an index into the sent set as the photo id a reader can open.

    Pass A is handed a subset of the frames chosen for identity, and every
    backend labels images by position within that subset. So "photo 1" from the
    model means the second photograph it was given, not `photo_id 1` - and
    `pipeline.pricing_blocker` puts this string in front of a seller, who has
    only the ids in the report to look it up by. An index nobody sent is left
    exactly as it came back rather than being mapped to something plausible.
    """
    if not text or not photo_ids:
        return text

    def swap(match: re.Match) -> str:
        index = int(match.group(2))
        if not 0 <= index < len(photo_ids):
            return match.group(0)
        return f"{match.group(1)} {photo_ids[index]}"

    return _PHOTO_REF.sub(swap, text)


# Answers that name nothing. The identity schema offers closed lists that end
# in "other" on purpose - a judge's truck may be a model this repo has never
# heard of and an honest "other" beats a forced wrong pick - but "other" is not
# a make, and carrying it forward would put it in front of a reader and into
# `modelspec.normalise_model` as if it were one. `body_type` is deliberately
# NOT folded this way: there "other" is a real answer with a real consequence.
_UNNAMED = {"", "other", "unknown", "none", "n/a", "na", "null"}


def _named(value) -> str | None:
    text = str(value or "").strip()
    return None if text.lower() in _UNNAMED else text


def parse_identity(text: str, photo_ids: list[int] | None = None
                   ) -> tuple[VehicleRead, bool, str]:
    """One identity read. `photo_ids` are the frames this call was sent, in order."""
    data = extract_json(text)
    make = _named(data.get("make"))
    model = _named(data.get("model"))
    vehicle = VehicleRead(
        make=make,
        model=model,
        # The raw answer is evidence about the read; the folded one is the key
        # the anchor row, the spec card and the price model all join on.
        model_canonical=modelspec.normalise_model(make, model),
        body_type=data.get("body_type") or None,
        cab_type=data.get("cab_type") or None,
        axle_config=data.get("axle_config") or None,
        approx_year_range=data.get("approx_year_range") or None,
        generation=_named(data.get("generation")),
        generation_conf=_confidence(data.get("generation_conf"), 0.0),
        year_evidence=str(data.get("year_evidence") or "").strip(),
        badges_seen=[str(b) for b in (data.get("badges_seen") or [])][:8],
        confidence=_confidence(data.get("confidence"), 0.0),
    )
    return (vehicle, bool(data.get("same_vehicle", True)),
            remap_photo_refs(str(data.get("vehicle_mismatch") or "").strip(), photo_ids))


# The only keys from the seller's form that are allowed into a prompt. An
# asking price is not one of them.
DECLARED_IN_PROMPTS = ("year", "km", "make", "model")


def identity_context(declared: dict | None, selected: list) -> str:
    lines = [f"You are looking at {len(selected)} photos of what should be one vehicle, "
             f"photo_id 0 to {len(selected) - 1}."]
    hints = ", ".join(f"photo_id {i}: {c.view.replace('_', ' ')}"
                      for i, c in enumerate(selected))
    lines.append(f"A zero-shot classifier tagged them as - {hints}. "
                 "Those tags are a hint and are sometimes wrong; trust the pixels.")
    if declared:
        # Whitelisted, not filtered: `declared` carries the seller's asking price
        # on the CLI path, and "the VLM never sees or emits a price" has to
        # survive someone passing --asking.
        stated = ", ".join(f"{k}={v}" for k, v in declared.items()
                           if k in DECLARED_IN_PROMPTS and v not in (None, ""))
        if stated:
            lines.append(f"The seller states: {stated}. Treat this as a claim, not a "
                         "fact - if the cab generation you can see contradicts it, "
                         "report what you see.")
    return "\n".join(lines)


def identity_schema(make: str | None = None) -> dict:
    """The identity schema, narrowed to the models this make actually offers.

    Wrapped rather than called directly so that the schema pass A is ASKED with
    and the schema a repair is CHECKED against are one object. Repairing
    against a wider schema than the call used lets a model "fix" its answer
    into a model name the narrowed call had deliberately excluded, which is the
    one case the enum exists to prevent.
    """
    builder = getattr(prompts, "identity_schema", None)
    return builder(make) if callable(builder) else prompts.IDENTITY_SCHEMA


def identity_prompt(*, context: str, make: str | None = None,
                    model: str | None = None) -> str:
    builder = getattr(prompts, "identity_prompt", None)
    if callable(builder):
        return builder(context=context, make=make, model=model)
    return prompts.IDENTITY_PROMPT.format(context=context)


def declared_identity(declared: dict | None) -> tuple[str | None, str | None]:
    """The seller's stated make and model, folded the way the prompts take them.

    One reading of `declared`, so the prompt, the schema and any repair of the
    answer are all narrowed by the same claim.
    """
    stated = declared or {}
    return _named(stated.get("make")), _named(stated.get("model"))


def identity(client, selected: list, declared: dict | None, *, max_tokens: int,
             effort: str | None = IDENTITY_EFFORT):
    """Every identity photo at once. Raises VLMError; the caller decides the cost.

    The seller's stated make and model go in as a claim rather than an answer -
    `identity_context` has always said so in words, and they are handed to the
    schema builder as well so a stated Ford narrows `model` to the models Ford
    makes. They are whitelisted through `DECLARED_IN_PROMPTS`, so an asking
    price passed on the CLI still cannot reach a prompt.
    """
    make, model = declared_identity(declared)
    prompt = identity_prompt(context=identity_context(declared, selected),
                             make=make, model=model)
    # Higher than EVIDENCE_IMAGE_LONG_EDGE: a model badge is small in frame and
    # 1024 px across a whole tractor leaves "F-MAX 500" a few pixels tall. Sent
    # only to a backend whose `complete` takes it - see `supported`.
    extra = supported(client.complete, {"long_edge": IDENTITY_IMAGE_LONG_EDGE})
    return client.complete(
        prompt, [Path(c.path) for c in selected], system=prompts.SYSTEM,
        max_tokens=max_tokens, effort=effort,
        json_schema=identity_schema(make) if client.supports_structured_output else None,
        **extra)


# --- pass A2: the badge read -----------------------------------------------
# A second, independent witness to make and model. `badges_seen` has been
# collected since the first fan-out and was only ever displayed - nothing
# checked that it corroborated anything, and it was read off a downscaled
# montage of eight photographs. This is one call on one crop, and what it says
# lands on `VehicleRead.badge_*` where `app/identity.py` can weigh it against
# the identity pass, the chassis plate and the trained head.

# Where a badge lives, best first. The maker's badge is on the grille, square
# on in the front and front three-quarter views; the model script is on the
# door and the side fairing.
BADGE_VIEW_PRIORITY = ["exterior_front_34", "exterior_front", "exterior_side",
                       "exterior_rear"]


def badge_crop_rect(check: PhotoCheck, *, width: int | None = None,
                    height: int | None = None) -> tuple[int, int, int, int] | None:
    """The badge band of the subject box, in source pixels.

    `BADGE_CROP_TOP` and `BADGE_CROP_BOTTOM` are fractions of the SUBJECT BOX
    height measured down from the box top, not of the frame: a tractor parked
    at the left of a dealer-lot photograph has its grille in the upper part of
    its own box and nowhere in particular in the frame.

    Padded and floored by `subject_crop_rect` rather than by a second cropper,
    so one rule decides what is too small to be worth cutting.
    """
    box = check.subject_box
    width = width or check.width
    height = height or check.height
    if not box or not width or not height:
        return None
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    if box_height <= 0:
        return None
    band = (x1, y1 + BADGE_CROP_TOP * box_height,
            x2, y1 + BADGE_CROP_BOTTOM * box_height)
    return subject_crop_rect(band, width, height)


def best_badge_frame(gate) -> PhotoCheck | None:
    """The frame most likely to have a legible badge in it, or None.

    Whole-vehicle views only, and only where the gate picked a subject: a crop
    of "the badge band" of a frame whose subject is a neighbour's lorry is a
    confident reading of the wrong truck.
    """
    ranked = []
    for check in gate.photos:
        if not check.usable or check.view not in BADGE_VIEW_PRIORITY:
            continue
        if badge_crop_rect(check) is None:
            continue
        ranked.append((BADGE_VIEW_PRIORITY.index(check.view),
                       -check.capture_quality, check.photo_id, check))
    ranked.sort(key=lambda row: row[:3])
    return ranked[0][3] if ranked else None


def parse_badge(text: str) -> dict:
    """The transcription, and what the pass made of it. No adjudication here."""
    data = extract_json(text)
    return {
        "badge_text": [str(t).strip() for t in (data.get("badge_text") or [])
                       if str(t).strip()][:8],
        "make": _named(data.get("make")),
        "model": _named(data.get("model")),
        "trim_or_power": _named(data.get("trim_or_power")),
        "legible": bool(data.get("legible", True)),
        "confidence": _confidence(data.get("confidence"), 0.0),
    }


def badge_available() -> bool:
    """Whether the badge prompt exists in this build. It is another file's."""
    return callable(getattr(prompts, "badge_prompt", None))


def badge(client, check: PhotoCheck, *, image: Path, max_tokens: int,
          effort: str | None = BADGE_EFFORT):
    """One crop, one call. Raises VLMError; a failed badge costs the badge.

    Deliberately blind to what pass A concluded. `prompts.badge_prompt` accepts
    a make and a model and ignores both, and nothing is passed here either: a
    witness told the answer in advance is a paraphrase of the first witness,
    and `app/identity.py` would then be weighing one read twice.
    """
    prompt = prompts.badge_prompt()
    schema = getattr(prompts, "BADGE_SCHEMA", None)
    extra = supported(client.complete, {"long_edge": IDENTITY_IMAGE_LONG_EDGE})
    return client.complete(
        prompt, [image], system=prompts.SYSTEM, max_tokens=max_tokens, effort=effort,
        json_schema=schema if client.supports_structured_output else None, **extra)


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
            # Unreadable fields round DOWN to zero weight and set `ungraded`;
            # they used to round up to minor/low/0.5, which was silent
            # inflation on a finding nobody could read.
            severity=severity or "cosmetic",
            confidence=_confidence(entry.get("confidence"), CONFIDENCE_UNPARSEABLE),
            price_impact=impact or "none",
            ungraded=severity is None or impact is None,
            magnitude=calibration.parse_magnitude(entry.get("magnitude")),
            box=_observation_box(entry.get("box"), check, cropped=cropped)))
    return finding


def closeup_image(check: PhotoCheck, tmpdir: Path) -> tuple[Path, bool]:
    """The image this photo is sent as, cut once rather than once per sample."""
    cropped = write_subject_crop(check, tmpdir) if wants_crop(check) else None
    return (cropped or Path(check.path)), cropped is not None


def weak_points_for_view(vehicle: VehicleRead | None, view: str) -> list[tuple[str, str]]:
    """What this model is known to go wrong on, filtered to what is in frame.

    The filter is what makes the card affordable: a close-up of a tire has no
    business being told about this model's AdBlue tank, and each line it does
    get is paid for on every one of the ~16 photos x `CLOSEUP_SAMPLES` calls.
    The view-to-component mapping is `prompts.view_components` and not a second
    copy of it - `closeup_prompt` builds its anchors from the same one, and two
    mappings that must agree are one silent divergence waiting to happen.

    An unknown view asks for nothing rather than for everything, because
    `modelspec.weak_points` reads an empty filter as no filter at all.
    """
    components = list(prompts.view_components(view))
    if vehicle is None or not components:
        return []
    return modelspec.weak_points(vehicle.make, vehicle.model, components=components)


def closeup(client, check: PhotoCheck, vehicle_line: str, *,
            tmpdir: Path | None = None, max_tokens: int,
            expectation: str = "", band: str | None = None,
            image: Path | None = None, cropped: bool | None = None,
            weak_points: list | tuple = (),
            effort: str | None = CLOSEUP_EFFORT, repair=None) -> PhotoFinding:
    """One photo, one call. Raises VLMError; the caller decides what that costs.

    `expectation` and `band` are the truck's own baseline, threaded in beside
    the vehicle line: without them this call grades every consumable against a
    new one. `weak_points` is the same argument at model level - what a truck
    of this exact model is known to go wrong on, already filtered to the
    components in this frame. `repair` is the text-only retry - safe here
    because `parse_closeup` takes `photo_id` from the caller and never from the
    response, so a repair cannot re-bind a claim to a photo that was not sent.
    """
    from ..vision import VIEW_LABELS

    t0 = time.time()
    if image is None:
        image, cropped = closeup_image(check, tmpdir)
    cropped = bool(cropped)
    pretty = {k: k.replace("_", " ") for k in VIEW_LABELS}.get(
        check.view, check.view.replace("_", " "))
    # The spec card's own lines travel inside `vehicle_line`, so they are
    # composed once per run rather than once per call; only the per-view weak
    # points are passed separately.
    prompt = prompts.closeup_prompt(
        view=check.view, view_pretty=pretty, vehicle=vehicle_line,
        cropped=cropped, expectation=expectation, band=band,
        soft=any("soft focus" in r for r in check.reasons),
        **supported(prompts.closeup_prompt, {"weak_points": list(weak_points)}))

    schema = prompts.closeup_schema() if client.supports_structured_output else None
    response = client.complete(prompt, [image], system=prompts.SYSTEM,
                               max_tokens=max_tokens, effort=effort, json_schema=schema)
    note = ""
    try:
        finding = parse_closeup(response.text, check, cropped=cropped)
    except (ValueError, json.JSONDecodeError) as exc:
        if repair is None:
            raise
        response = repair(client, response, exc, prompts.closeup_schema(), max_tokens)
        finding = parse_closeup(response.text, check, cropped=cropped)
        note = f"unparseable ({exc}); repaired"
    finding.backend = response.backend
    if note:
        finding.sample_errors.append(note)
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
                         f"({calibration.issue_suffix(issue, samples=finding.samples)})")
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


# Characters of one spec line this consumer takes. The budget belongs here
# rather than in `modelspec`: a card row is hand-written prose and the F-MAX
# row's `cab` and `driveline` fields are each a paragraph, which is right for
# `app.cli doctor` and for the identity prompt, where it is read once. This
# line is read on every close-up call and again by the synthesis and the
# calibration - a 1.3 KB spec block is ~330 tokens paid fifty times over - and
# what a close-up needs from it is the head of each fact, not the sales
# variants at the end of it. `textwrap.shorten` cuts on a word boundary.
SPEC_LINE_CHARS = 200


def synthesize(client, findings: list[PhotoFinding], vehicle_line: str, *,
               max_tokens: int, effort: str | None = SYNTHESIS_EFFORT):
    notes, flat = notes_block(findings)
    prompt = prompts.synthesis_prompt(
        n=len([f for f in findings if not f.error]), vehicle=vehicle_line, notes=notes)
    response = client.complete(
        prompt, [], system=prompts.SYSTEM, max_tokens=max_tokens, effort=effort,
        json_schema=prompts.synthesis_schema() if client.supports_structured_output else None)
    return response, flat


def vehicle_line(vehicle: VehicleRead) -> str:
    """What passes B, C and D know about this truck, in as few lines as possible.

    It was one sentence, and that sentence was the entire extent of it: a 2015
    Cargo tractor and a 2023 F-MAX at the same odometer got the same reference
    for what "on schedule" looks like. The model card adds the few facts that
    change how a photograph reads - the segment, the cab, the driveline, the
    axle configurations the model was actually sold in - and nothing else,
    because every line here is paid for on every close-up call.

    When the card has nothing to say this returns the byte-identical sentence
    it always did, which is the case for any brand the reference has not been
    curated for. A test pins that.
    """
    named = " ".join(x for x in (vehicle.make, vehicle.model) if x).strip()
    if not named:
        head = "The make and model could not be read from the photos."
    else:
        spec = ", ".join(x for x in (vehicle.cab_type, vehicle.axle_config,
                                     vehicle.approx_year_range) if x)
        head = (f"The vehicle has been identified from the full set as a {named}"
                + (f" ({spec})" if spec else "") + ".")
    # Never a substitute for the photograph: `spec_lines` says what the model
    # generally is, and the close-up prompt asks for what is actually visible.
    spec_lines = [textwrap.shorten(line, SPEC_LINE_CHARS, placeholder=" ...")
                  for line in modelspec.spec_lines(vehicle.make, vehicle.model,
                                                   generation=vehicle.generation)]
    return "\n".join([head] + spec_lines)


def tempdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="kamion-crop-")
