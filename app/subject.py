"""Which vehicle in the photographs is the one being sold.

Split out of `app/gate.py` because the answer stopped being a per-frame fact.
A frame can only ever say "there is a truck-shaped object here, this big, in
this position"; the thing that says *which* truck is being sold is the set -
the subject appears in fifteen frames and the neighbour's lorry appears in
one. So this module is two phases:

  phase 1  per frame, score every vehicle box on label prior x confidence x
           sqrt(area) x centrality x edge relief. Pure, pixel-free, and the
           whole of what a single photograph can know.
  phase 2  across the set, embed the candidate crops with the CLIP model the
           gate already runs, build an appearance prototype from the frames
           where the answer is unambiguous, and let that prototype resolve the
           frames where it is not.

The rule this replaces was "a box holding the centre of the frame wins
outright, with area breaking ties among those". It was written for a real
corpus failure - on `demo/tr_clean/000.jpg` the subject ran off the top of the
frame and measured 15% of the area against 23% for a whole tractor parked to
the left - and it solved that case by making centrality an override. An
override has no crossover point: any 0.26-confidence box straddling the centre
pixel eliminated a 0.95-confidence box filling a third of the frame, which is
how an engine-bay close-up came to be cropped to a lorry forty metres away and
handed to the close-up call under the line "this image has been cropped to the
one vehicle being sold".

Nothing here moves the gate verdict. `truck_dominant`, the refusal ladder and
`blocks_pricing` are computed in `gate.inspect` from the raw detector output
before any of this runs, deliberately: the measured 1-in-200 false refusal is
not something a rendering and cropping decision gets to change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import GATE_THRESHOLDS
from .schema import Detection, PhotoCheck

# --- the view vocabularies -------------------------------------------------
# These live here rather than in gate.py because every consumer of them is
# now a subject-selection question. `gate.py` re-exports them, so
# `gate.TRUCK_PART_VIEWS` still resolves.

# Views that establish "this is the whole vehicle" rather than a detail of it.
WHOLE_VEHICLE_VIEWS = {"exterior_front", "exterior_front_34", "exterior_side", "exterior_rear"}

# Close-up views that are still unmistakably part of a heavy vehicle. Used to
# tell "a real truck, photographed badly" apart from "not a truck at all" when
# no whole-vehicle frame exists, and to tell a component close-up apart from a
# photograph of a scene that happens to contain a vehicle.
TRUCK_PART_VIEWS = {"tire_wheel", "interior_cab", "dashboard_odometer", "engine_bay",
                    "chassis_undercarriage", "fifth_wheel"}
# `damage_detail` is deliberately excluded above: it is the taxonomy's catch-all
# and it matched a photo of a parked motorcycle at 0.42.
MIN_PART_VIEW_CONF = 0.50


# --- thresholds ------------------------------------------------------------

_THRESHOLDS: dict | None = None


def thresholds() -> dict:
    """Load the calibrated thresholds, falling back to the shipped defaults."""
    global _THRESHOLDS
    if _THRESHOLDS is None:
        if GATE_THRESHOLDS.exists():
            _THRESHOLDS = json.loads(GATE_THRESHOLDS.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(
                f"{GATE_THRESHOLDS} missing - run: .venv/bin/python -m app.calibrate_gate")
    return _THRESHOLDS


# --- the candidate pool ----------------------------------------------------

# COCO calls a cab-forward tractor a bus routinely and a tight cab shot a car,
# so all three are in the pool. `train` is not: it is the label YOLO puts on a
# dashboard filling the frame, and it is the input to the disqualifier path.
SUBJECT_LABELS = ("truck", "bus", "car")

# What reaches `check.detections` and the screen. Unchanged from the shipped
# gate; the candidate floor below is deliberately lower.
DETECTION_CONF = 0.25
# A clipped or half-occluded subject is a low-confidence box - the shipped
# calibration measured `vehicle_conf_min` at 0.193 across 200 real vehicles -
# so the candidate pool has to reach below what the screen draws.
CANDIDATE_CONF = 0.10

# ASSUMPTION. One physical vehicle, one box: COCO's per-class NMS leaves a
# tractor carrying truck 0.71 / bus 0.44 / car 0.31 at the same pixels, and the
# second and third then count as competitors and can be picked as the subject.
# Restricted to SUBJECT_LABELS on purpose - a truck box must never suppress an
# overlapping `motorcycle` or `train` box, because those are the entire input
# to COCO_DISQUALIFYING and to the "YOLO calls a dashboard a train" invariant.
DEDUP_IOU = 0.70


# --- the frame score -------------------------------------------------------

# ASSUMPTIONS, all of them, seeded to clear the two known corpus cases and
# swept against same-listing retrieval P@1 in scripts/eval_subject.py.

# A cab-forward tractor is a `bus` often enough that the prior has to admit it;
# a `car` is admitted to the pool but cannot win on its own (see car_eligible).
LABEL_PRIOR = {"truck": 1.00, "bus": 0.92, "car": 0.55}
# Confidence counts, but floored rather than linear: a 0.20 box on a subject
# clipped by the frame edge is still the subject.
CONF_FLOOR_W = 0.40
# Apparent size scales with linear extent, not area. Raw area is what let a
# whole background tractor beat a clipped foreground one.
AREA_EXPONENT = 0.5
# A box holding the centre pixel gets a bonus, NOT an override. The crossover
# is now statable: a centred box loses to a rival roughly 2.5-4x its area.
CENTRE_HIT_BONUS = 1.35
# How far off-centre a box can sit before it stops looking like the subject.
# A corner box keeps 45% of its score.
CENTRE_BIAS = 0.55
# A box that holds the centre of the frame AND runs off its edge is a subject
# the photographer could not fit in, and it is always under-measured. One
# clipped side is worth +18%, about treating the box as 1.39x its area.
#
# Only for a box that holds the centre. Relief for any box touching the edge
# is what got `demo/tr_clean/000.jpg` wrong on the real frame rather than the
# stylised one: there the BACKGROUND tractor is the box flush against x=0
# (truck 0.805 over 23% of the frame) and the subject the photographer framed
# sits clear of every edge (truck 0.454 over 15%). Unconditional relief handed
# +18% to the wrong truck and flipped a case the old rule got right. Touching
# the edge without holding the centre is not evidence of clipping; it is
# evidence of being at the side of the shot.
EDGE_RELIEF = 0.18
EDGE_EPS_PX = 2.0
# A candidate that looks nothing like the set's truck keeps 30% of its score.
# Never zero: the prototype can be wrong, and a hard veto on a 12-photo set
# with one odd frame is how you end up drawing no box at all.
PROTO_FLOOR = 0.30

# MEASURED. q25 of `view_conf` over the 1,374 whole-vehicle originals in
# data/metadata/images.csv. A correct `exterior_front` sits at 0.357 in that
# corpus, so 0.50 would refuse a quarter of the honest whole-vehicle frames a
# vote on who the subject is.
SEED_VIEW_CONF = 0.398
# ASSUMPTIONS. A frame only seeds the set's identity when its own answer is
# unambiguous; SEED_MARGIN is the point of the whole architecture - a
# dealer-lot frame with two comparable trucks does not get a vote, and is
# instead resolved by the frames that do.
SEED_CONF = 0.50
SEED_MARGIN = 0.85
# Seeds that disagree with the provisional mean by more than this many
# standard deviations are dropped and the mean is retaken.
SEED_TRIM_SIGMA = 2.0
# Cosine at which two winning crops stop being the same vehicle, for the
# reported cluster count. Reported, never enforced: `same_vehicle` is the
# VLM's call and `pipeline.pricing_blocker` rests on it.
CLUSTER_COSINE = 0.75

# ASSUMPTION. A competitor has to be worth cropping away. The old absolute 2%
# floor meant a parked hatchback 80 m behind a lone truck triggered a crop and
# threw away the ground line for nothing. An eighth of the subject's area is
# about a third of its linear extent.
COMPETITOR_REL_AREA = 0.08
COMPETITOR_ABS_AREA = 0.02
COMPETITOR_CONF = 0.40

# MEASURED, and carried in models/gate_thresholds.json as
# `detector.part_view_subject_area_frac`: q90 of the best vehicle box's
# area_frac over the 1,955 confident part-view originals in the corpus. Above
# it, the close-up frame itself is what YOLO drew a box around; below it, the
# box is the yard behind the component. The distribution is strongly bimodal -
# of the 773 part-view frames that carry a box at all, 140 sit under 20% of the
# frame and 455 sit above 80%.
PART_VIEW_SUBJECT_AREA_FALLBACK = 0.94
# ASSUMPTION. A truck forty metres away is small in pixels whatever the frame
# size; this rounds out the relative test on very large phone frames. The
# measured q05 of the short side of a boxed part-view candidate is 100 px.
PART_VIEW_MIN_SIDE_PX = 96.0

# Absolute CLIP cosines. Set by scripts/eval_subject.py --calibrate-rescue at
# the point giving <= 5% false rescue against other listings' prototypes, or
# left at 1.01 - unreachable, the clause disabled - when the two distributions
# do not separate. An unmeasured absolute CLIP cosine is the single most likely
# way for this to misbehave quietly.
CAR_RESCUE_SIM = 1.01
PART_VIEW_RESCUE_SIM = 1.01


@dataclass
class Candidate:
    """One vehicle box that could be the thing being sold."""
    photo_id: int
    det: Detection
    label: str
    confidence: float
    box: list[float]
    area_frac: float
    centre_hit: bool
    far: float                     # 0 at the centre, 1 at a corner
    clipped_sides: int
    min_side_px: float
    # Sharpness inside the box over sharpness outside it. The subject of a
    # photograph is what the photographer focused on; the yard behind it is
    # not. 1.0 means "no information" and is the safe default, so a frame we
    # could not measure is never suppressed on this signal.
    focus_ratio: float = 1.0
    emb: Any = None                # CLIP of the padded crop, L2-normalised
    sim_abs: float = 0.0
    sim_rel: float = 1.0
    score: float = 0.0
    attached: bool = True          # already present in check.detections


@dataclass
class SubjectIdentity:
    """What the SET concluded about which vehicle it is about."""
    prototype: Any = None          # 512-d, L2-normalised; never serialised
    method: str = "none"           # recurring_vehicle | single_frame | none
    seeded_from: list[int] = field(default_factory=list)
    frames_agreeing: int = 0
    mean_sim: float = 0.0
    clusters: int = 0


NO_IDENTITY = SubjectIdentity()


# --- phase 1 ---------------------------------------------------------------

def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = abs((ax2 - ax1) * (ay2 - ay1)) + abs((bx2 - bx1) * (by2 - by1)) - inter
    return inter / union if union > 0 else 0.0


def dedupe_vehicles(dets: list[Detection]) -> list[Detection]:
    """One physical vehicle, one box.

    Greedy, highest confidence first, over SUBJECT_LABELS only. Everything
    else - including `motorcycle` and `train` - passes through untouched, so
    the disqualifier path keeps its input.
    """
    keep: list[Detection] = []
    vehicles = sorted([d for d in dets if d.label in SUBJECT_LABELS],
                      key=lambda d: -d.confidence)
    survivors: list[Detection] = []
    for d in vehicles:
        if any(_iou(d.box, s.box) >= DEDUP_IOU for s in survivors):
            continue
        survivors.append(d)
    surviving = {id(d) for d in survivors}
    for d in dets:
        if d.label in SUBJECT_LABELS and id(d) not in surviving:
            continue
        keep.append(d)
    return keep


def _candidate(check: PhotoCheck, det: Detection, *, attached: bool = True) -> Candidate:
    w, h = float(check.width or 1), float(check.height or 1)
    x1, y1, x2, y2 = det.box
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    far = min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5 / (0.5 * 2 ** 0.5))
    clipped = sum((x1 <= EDGE_EPS_PX, y1 <= EDGE_EPS_PX,
                   x2 >= w - EDGE_EPS_PX, y2 >= h - EDGE_EPS_PX))
    return Candidate(
        photo_id=check.photo_id, det=det, label=det.label,
        confidence=float(det.confidence), box=list(det.box),
        area_frac=float(det.area_frac),
        centre_hit=bool(x1 <= w / 2 <= x2 and y1 <= h / 2 <= y2),
        far=far, clipped_sides=int(clipped),
        min_side_px=float(min(abs(x2 - x1), abs(y2 - y1))),
        attached=attached)


def build_candidates(check: PhotoCheck, extra: list[Detection] = (),
                     gray=None) -> list[Candidate]:
    """The frame's candidate pool: the vehicle boxes the screen draws plus the
    sub-threshold ones it does not, deduped as one set.

    `gray` is the frame in greyscale, passed in because `gate.inspect` already
    has it and decoding a second time per candidate would cost a pass over the
    pixels for nothing. Without it every candidate keeps the neutral focus
    ratio of 1.0 and the focus rule simply does not fire - which is what a unit
    test with synthetic boxes and no image wants.
    """
    dets = [d for d in check.detections if d.label in SUBJECT_LABELS] + list(extra)
    attached = {id(d) for d in check.detections}
    out = [_candidate(check, d, attached=id(d) in attached)
           for d in dedupe_vehicles(dets)]
    if gray is not None:
        for c in out:
            c.focus_ratio = round(focus_ratio(gray, c.box), 3)
    return out


def candidates(check: PhotoCheck) -> list[Candidate]:
    """Every box that could be the vehicle being sold, before the set has an
    opinion. Prefers the wider pool `gate.inspect` stashed on `_candidates`
    and falls back to `check.detections`, which is what a unit test with
    synthetic boxes and no detector run has."""
    stashed = getattr(check, "_candidates", None)
    if stashed:
        return list(stashed)
    return [_candidate(check, d) for d in check.detections if d.label in SUBJECT_LABELS]


def frame_score(c: Candidate, *, use_prototype: bool = False) -> float:
    """What one photograph, on its own, thinks of one box."""
    prior = LABEL_PRIOR.get(c.label, 0.0)
    conf = CONF_FLOOR_W + (1.0 - CONF_FLOOR_W) * c.confidence
    area = max(0.0, c.area_frac) ** AREA_EXPONENT
    centre = CENTRE_HIT_BONUS if c.centre_hit else (1.0 - CENTRE_BIAS * c.far)
    edge = 1.0 + EDGE_RELIEF * c.clipped_sides if c.centre_hit else 1.0
    proto = (PROTO_FLOOR + (1.0 - PROTO_FLOOR) * c.sim_rel) if use_prototype else 1.0
    return prior * conf * area * centre * edge * proto


def car_eligible(c: Candidate, pool: list[Candidate], identity: SubjectIdentity) -> bool:
    """A `car` box is the subject only when nothing better is on offer.

    COCO labels a tight cab shot `car`, so excluding the class outright loses
    real subjects; admitting it outright loses `test_a_car_is_never_the_subject`,
    where a car filling 87% of the frame sits behind a truck at 0.6. So: only
    when no truck or bus is in this frame at all, or when the set has an
    appearance prototype and this crop matches it.
    """
    if c.label != "car":
        return True
    if not any(o.label in ("truck", "bus") for o in pool if o is not c):
        return True
    if identity.prototype is None:
        return False
    return c.sim_abs >= CAR_RESCUE_SIM


def part_view_subject_area() -> float:
    """MEASURED: q90 of the best vehicle box's `area_frac` over the confident
    part-view originals. Written by app.calibrate_gate."""
    try:
        value = thresholds()["detector"].get("part_view_subject_area_frac")
    except Exception:
        value = None
    return float(value) if value else PART_VIEW_SUBJECT_AREA_FALLBACK


# Below this, a box is sharply less in focus than the rest of its frame and is
# the background rather than the subject. MEASURED: see
# `scripts/calibrate_focus.py`, which builds the two distributions - boxes on
# confident whole-vehicle frames, which should be in focus, against boxes under
# the part-view area floor - and places the cut where the false-suppression rate
# on the first is under 5%.
#
# This is the one rule in this module that does NOT consult the view tag, and
# that is the point of it. 36.8% of the corpus is tagged into an exterior view
# and the four exterior classes have median confidences of 0.34 to 0.55 - they
# are where a zero-shot classifier puts frames it cannot place. A Ford dashboard
# in `tr_truckmarket/14299/011.jpg` is tagged `exterior_front` at 0.38 and a
# stripped engine bay in `mascus/83AEA622/008.jpg` is tagged `exterior_rear` at
# 0.39. Every view-gated rule below misses both. Focus does not.
# MEASURED at 0.55 by scripts/calibrate_focus.py over 300 corpus frames: the
# largest cut whose false-suppression rate on confident whole-vehicle subject
# boxes stays under 5%. 0.62, which was the first guess, costs 7% there.
#
# Be honest about what this buys. It is PRECISE and it is NOT high-recall: at
# 0.55 it catches 11.5% of small boxes on confident part views and only 2.0% of
# the small boxes that carry no part-view tag at all. What it does catch is the
# worst of them - `us_selectrucks/232615/010.jpg`, an engine bay whose box was a
# lorry behind it at ratio 0.52, and `.../256409/014.jpg`, a dashboard whose box
# was a truck through the windscreen at 0.50. Most off-tag small boxes are
# genuinely in focus (median ratio 1.98) and are other trucks on a dealer lot,
# which the area floor already handles.
#
# So this rule is not the answer to the view tag being unreliable. It is a cheap
# second opinion that survives the tag being wrong, and the tag still needs
# fixing on its own account.
FOCUS_SCENERY_RATIO = 0.55
# Regions smaller than this cannot carry a meaningful Laplacian variance.
FOCUS_MIN_PIXELS = 400


def focus_ratio(gray, box: list[float]) -> float:
    """Sharpness inside `box` over the mean sharpness of the frame around it.

    Laplacian variance, the same statistic `gate.capture_metrics` already scores
    a whole frame with - reused rather than reinvented so a region score and a
    frame score mean the same thing. Returns 1.0, which is neutral, whenever
    there is not enough of either region to measure.
    """
    if gray is None:
        return 1.0
    import cv2
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if (x2 - x1) * (y2 - y1) < FOCUS_MIN_PIXELS:
        return 1.0
    inside = gray[y1:y2, x1:x2]
    strips = [gray[:y1, :], gray[y2:, :], gray[y1:y2, :x1], gray[y1:y2, x2:]]
    strips = [t for t in strips if t.size >= FOCUS_MIN_PIXELS]
    if not strips:
        return 1.0          # the box IS the frame; nothing to compare against
    outside = float(np.mean([cv2.Laplacian(t, cv2.CV_64F).var() for t in strips]))
    if outside <= 1e-9:
        return 1.0
    return float(cv2.Laplacian(inside, cv2.CV_64F).var() / outside)


def is_scenery(check: PhotoCheck, c: Candidate, identity: SubjectIdentity) -> bool:
    """On a close-up of one component, a small vehicle box is the yard behind it.

    The close-up pass is told the image it is holding IS the vehicle being sold
    - "other vehicles in the original frame were deliberately excluded, so
    describe only what is in front of you" - and is then asked the engine-bay
    checklist about it. Handing it a crop of a lorry forty metres away is the
    worst per-frame error this pipeline can make: every answer comes back
    confident, cited to a real `photo_id`, and about the wrong truck.

    Measured on the corpus: `us_selectrucks/256409/007.jpg` is a tire close-up
    whose subject box was a strip of eight lorries across the top 15% of the
    frame, and `.../014.jpg` a dashboard whose subject box was a truck seen
    through the windscreen at 3% of the frame.
    """
    # Focus first, and deliberately before the view tag is consulted. A box
    # markedly less sharp than the frame around it is the yard behind the
    # subject, and that is true whether or not the classifier managed to call
    # this frame a close-up. Measured on the two frames named above: the engine
    # bay's background box scores 0.51 and the dashboard's 0.56, against 2.33
    # and 7.07 for small boxes that really are in focus - so area and focus
    # catch different failures and both are needed.
    if c.focus_ratio < FOCUS_SCENERY_RATIO and c.area_frac < part_view_subject_area():
        return True
    if check.view not in TRUCK_PART_VIEWS or check.view_conf < MIN_PART_VIEW_CONF:
        return False                      # not a close-up; not this rule's business
    if c.area_frac >= part_view_subject_area():
        return False                      # the close-up frame itself is what got boxed
    if c.min_side_px < PART_VIEW_MIN_SIDE_PX:
        return True
    if identity.prototype is not None and c.sim_abs >= PART_VIEW_RESCUE_SIM:
        return False                      # genuinely our truck, seen whole, mistagged
    return True


def crop_is_safe(check: PhotoCheck) -> bool:
    """Never crop a confident close-up of a component.

    The crop can only remove the thing the per-view checklist is about - a
    fifth-wheel plate, a tread block, the DOT date on a sidewall - and
    CROP_PAD is not enough padding to protect it. A truck box on an engine-bay
    frame is the yard behind the engine, not the engine.
    """
    return not (check.view in TRUCK_PART_VIEWS and check.view_conf >= MIN_PART_VIEW_CONF)


def eligible(check: PhotoCheck, pool: list[Candidate],
             identity: SubjectIdentity = NO_IDENTITY) -> list[Candidate]:
    return [c for c in pool
            if LABEL_PRIOR.get(c.label, 0.0) > 0
            and car_eligible(c, pool, identity)
            and not is_scenery(check, c, identity)]


def _basis(check: PhotoCheck, best: Candidate | None, pool: list[Candidate],
           identity: SubjectIdentity) -> str:
    """Why that box, or why none. Same posture as `EvidenceReport.fell_back_from`:
    the gate may change its mind, it may not do it where nobody can see."""
    if best is None:
        if not pool:
            return ""
        biggest = max(pool, key=lambda c: c.area_frac)
        if check.view in TRUCK_PART_VIEWS and check.view_conf >= MIN_PART_VIEW_CONF:
            return (f"none: close-up of the {check.view.replace('_', ' ')}; the largest "
                    f"vehicle box is {biggest.area_frac * 100:.0f}% of the frame, which "
                    f"is the yard behind it")
        return "none: no vehicle box in this frame is the one being sold"
    if identity.prototype is not None and best.sim_abs:
        # Frame-local, deliberately: `frames_agreeing` is a set-level count and
        # is not known until every frame has been assigned.
        return (f"matches the vehicle the rest of the set is about "
                f"(cosine {best.sim_abs:.2f})")
    if len(pool) == 1:
        return "the only vehicle in frame"
    return (f"the largest of {len(pool)} vehicle boxes once position and "
            f"confidence are counted")


def assign(check: PhotoCheck, identity: SubjectIdentity = NO_IDENTITY) -> Candidate | None:
    """Choose this frame's subject and record it on the check."""
    pool = candidates(check)
    for d in check.detections:
        d.is_subject = False
    for c in pool:
        c.det.is_subject = False
    use_proto = identity.prototype is not None
    keep = eligible(check, pool, identity)
    for c in keep:
        c.score = frame_score(c, use_prototype=use_proto)
    best = max(keep, key=lambda c: c.score, default=None)
    check.subject_box = list(best.box) if best else None
    check.subject_score = round(best.score, 4) if best else 0.0
    check.subject_sim = round(best.sim_abs, 3) if best else 0.0
    check.subject_basis = _basis(check, best, pool, identity)
    if best is not None:
        _ensure_in_detections(check, best)
        best.det.is_subject = True
    return best


def _ensure_in_detections(check: PhotoCheck, best: Candidate) -> None:
    """The subject is always present in `check.detections`.

    A winning candidate below DETECTION_CONF would otherwise be a box the
    screen is asked to draw and cannot find - which is exactly the silent
    failure `Detection.is_subject` exists to end.
    """
    if best.det in check.detections:
        return
    check.detections.append(best.det)
    best.attached = True


def pick_subject(check: PhotoCheck) -> list[float] | None:
    """The one box that is the vehicle being sold, in this frame alone, or None.

    Decided here rather than in the browser. The old screen picked the largest
    vehicle box client-side while the vision model was handed the whole frame,
    so the box a viewer was shown and the pixels the model actually read were
    only coincidentally the same truck. On a dealer-lot photo they were not.
    """
    best = assign(check)
    return list(best.box) if best else None


def competing_vehicles(check: PhotoCheck) -> int:
    """Vehicle boxes other than the subject, big enough to be worth cropping
    away. Above zero, the frame has something in it to be confused by."""
    subject_area = 0.0
    for d in check.detections:
        if _is_subject(d, check):
            subject_area = max(subject_area, d.area_frac)
    floor = max(COMPETITOR_ABS_AREA, COMPETITOR_REL_AREA * subject_area)
    from . import vision
    return sum(1 for d in check.detections
               if d.label in vision.COCO_VEHICLES and d.confidence >= COMPETITOR_CONF
               and d.area_frac >= floor and not _is_subject(d, check))


def _is_subject(d: Detection, check: PhotoCheck) -> bool:
    # `is_subject` first; the coordinate match survives only for a PhotoCheck
    # assembled by hand, where `subject_box` was set without a selection run.
    return bool(d.is_subject) or (check.subject_box is not None
                                  and list(d.box) == list(check.subject_box))


# --- phase 2 ---------------------------------------------------------------

def _is_seed(check: PhotoCheck, c: Candidate, pool: list[Candidate]) -> bool:
    """May this frame define what the set's vehicle looks like?

    Only when its own answer is obvious. A dealer-lot frame with two
    comparable trucks does not get a vote on who the subject is; it is
    resolved by the frames that do. That is the whole architecture in a line.
    """
    if not (check.usable and check.content == "keep"):
        return False
    if check.view not in WHOLE_VEHICLE_VIEWS or check.view_conf < SEED_VIEW_CONF:
        return False
    if c.label not in ("truck", "bus") or c.confidence < SEED_CONF:
        return False
    if c.area_frac < thresholds()["detector"]["min_truck_area_frac"]:
        return False
    rest = [o for o in pool if o is not c]
    runner_up = max((frame_score(o) for o in rest), default=0.0)
    return runner_up < SEED_MARGIN * frame_score(c)


def _normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _prototype(seeds: list[Candidate]) -> np.ndarray | None:
    """Weighted mean of the seed crops, trimmed at 2 sigma.

    One neighbour's lorry photographed dead-centre in 1 of 12 frames gets
    outvoted rather than averaged in.
    """
    seeds = [s for s in seeds if s.emb is not None]
    if not seeds:
        return None
    def mean(group: list[Candidate]) -> np.ndarray:
        w = np.array([s.confidence * (s.area_frac ** 0.5) for s in group], dtype=np.float32)
        m = (np.stack([s.emb for s in group]) * w[:, None]).sum(axis=0)
        return _normalise(m.astype(np.float32))
    proto = mean(seeds)
    if len(seeds) >= 3:
        sims = np.array([float(s.emb @ proto) for s in seeds])
        floor = float(sims.mean() - SEED_TRIM_SIGMA * sims.std())
        kept = [s for s, v in zip(seeds, sims) if v >= floor]
        if kept and len(kept) < len(seeds):
            proto = mean(kept)
    return proto


def _clusters(winners: list[Candidate]) -> int:
    """Single-link agglomerative at CLUSTER_COSINE. Reported, never enforced."""
    embs = [w.emb for w in winners if w.emb is not None]
    if not embs:
        return 0
    groups: list[list[np.ndarray]] = []
    for e in embs:
        hit = next((g for g in groups
                    if any(float(e @ m) >= CLUSTER_COSINE for m in g)), None)
        if hit is None:
            groups.append([e])
        else:
            hit.append(e)
    return len(groups)


def ground(checks: list[PhotoCheck], embed=None) -> SubjectIdentity:
    """Decide the set's subject, then assign a box to every frame.

    `embed(crops) -> (n, d) float32` is injected so this stays testable
    without CLIP; `gate.inspect` passes `vision.clip().embed`.
    """
    pools = {c.photo_id: candidates(c) for c in checks}
    identity = SubjectIdentity(method="none")

    if embed is not None:
        _embed_candidates(checks, pools, embed)
        seeds = [c for check in checks for c in pools[check.photo_id]
                 if _is_seed(check, c, pools[check.photo_id])]
        if not seeds:
            seeds = [c for check in checks for c in pools[check.photo_id]
                     if _relaxed_seed(check, c)]
        proto = _prototype(seeds)
        if proto is not None and len({s.photo_id for s in seeds}) >= 2:
            identity = SubjectIdentity(prototype=proto, method="recurring_vehicle",
                                       seeded_from=sorted({s.photo_id for s in seeds}))
        elif len(checks) <= 1:
            identity.method = "single_frame"
    elif len(checks) <= 1:
        identity.method = "single_frame"

    if identity.prototype is not None:
        _score_similarity(pools, identity.prototype)

    winners = []
    for check in checks:
        check._candidates = pools[check.photo_id]
        best = assign(check, identity)
        if best is not None:
            winners.append(best)
    if identity.prototype is not None and winners:
        sims = [w.sim_abs for w in winners if w.emb is not None]
        identity.mean_sim = round(float(np.mean(sims)), 3) if sims else 0.0
        identity.frames_agreeing = sum(1 for s in sims if s >= identity.mean_sim * 0.85)
        identity.clusters = _clusters(winners)
    return identity


def _relaxed_seed(check: PhotoCheck, c: Candidate) -> bool:
    """Fallback ladder: a set whose whole-vehicle frames CLIP mistagged."""
    return (check.usable and c.label in ("truck", "bus") and c.confidence >= 0.35
            and c.area_frac >= 0.20 and c.centre_hit)


def _embed_candidates(checks, pools, embed) -> None:
    """One CLIP batch over every candidate crop in the set."""
    from PIL import Image, ImageOps
    jobs: list[tuple[Candidate, Any]] = []
    for check in checks:
        pool = [c for c in pools[check.photo_id]
                if c.area_frac >= 0.01 and c.min_side_px >= 32]
        if not pool:
            continue
        try:
            pil = ImageOps.exif_transpose(Image.open(check.path)).convert("RGB")
        except Exception:
            continue
        for c in pool:
            x1, y1, x2, y2 = c.box
            px, py = (x2 - x1) * 0.08, (y2 - y1) * 0.08
            jobs.append((c, pil.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                                      min(pil.width, int(x2 + px)),
                                      min(pil.height, int(y2 + py))))))
    if not jobs:
        return
    embs = embed([im for _, im in jobs])
    for (c, _), e in zip(jobs, embs):
        c.emb = _normalise(np.asarray(e, dtype=np.float32))


def _score_similarity(pools, proto: np.ndarray) -> None:
    """Absolute cosine, and its within-frame contrast.

    `sim_rel` rather than the raw cosine is what the score uses. CLIP ViT-B/32
    cosines between crops of any two trucks sit in a compressed band and the
    same-truck and different-truck distributions overlap heavily in absolute
    terms - but the decision here is never cross-frame. It is always "which of
    the boxes IN THIS FRAME is our truck", and a within-frame contrast is
    scale-free.
    """
    for pool in pools.values():
        for c in pool:
            c.sim_abs = float(c.emb @ proto) if c.emb is not None else 0.0
        sims = [c.sim_abs for c in pool if c.emb is not None]
        lo, hi = (min(sims), max(sims)) if sims else (0.0, 0.0)
        for c in pool:
            c.sim_rel = 1.0 if hi - lo < 1e-6 else (c.sim_abs - lo) / (hi - lo)


def evidence_sentence(identity: SubjectIdentity, checks: list[PhotoCheck]) -> str:
    """What the set concluded, in words, for GateReport.subject_evidence."""
    boxed = sum(1 for c in checks if c.subject_box)
    if identity.prototype is None:
        return (f"{boxed} of {len(checks)} frame(s) carry a box on the vehicle being sold; "
                f"no frame was unambiguous enough to establish what it looks like across "
                f"the set")
    extra = (f"; the crops fall into {identity.clusters} visually distinct vehicle(s)"
             if identity.clusters > 1 else "")
    return (f"the vehicle in {identity.frames_agreeing} of {boxed} boxed frame(s) recurs, "
            f"seeded from {len(identity.seeded_from)} unambiguous whole-vehicle frame(s) "
            f"at mean cosine {identity.mean_sim:.2f}{extra}")
