"""The pipeline's data contract.

These dataclasses are the product. The brief warns against "a thin wrapper
that sends photos to a vision API and prints whatever number comes back";
what separates this from that is that the vision model is never allowed to
emit free prose - it fills a fixed structure whose every condition claim
carries a `photo_id`, and the number is produced downstream by a regression
the model never sees.

Everything is plain dataclasses with `to_dict`, so an `Appraisal` round-trips
to JSON whole and the CLI and the web app stay dumb renderers.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Underscore-prefixed fields are carried in memory but never serialised.
        # A CLIP embedding is 512 floats per photo; on a 20-photo set that is
        # 10k numbers that would otherwise land in the JSON payload and in
        # every SSE frame, for no reader's benefit.
        return {f.name: _asdict(getattr(obj, f.name))
                for f in dataclasses.fields(obj) if not f.name.startswith("_")}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    return obj


class _Dict:
    def to_dict(self) -> dict:
        return _asdict(self)


# --- stage 1: gate --------------------------------------------------------

class GateDecision(str, Enum):
    PASS = "pass"
    ASK_MORE = "ask_more"          # it is a truck, but the photo set is short
    REFUSE_QUALITY = "refuse_quality"      # nothing legible in any frame
    REFUSE_NOT_A_TRUCK = "refuse_not_a_truck"
    REFUSE_NO_PHOTOS = "refuse_no_photos"


@dataclass
class Detection(_Dict):
    label: str
    confidence: float
    box: list[float]                # xyxy in pixels
    area_frac: float                # share of the frame
    # Whether the gate chose this box as the vehicle being sold. The screen
    # used to find the subject by exact float equality on all four
    # coordinates, which held only because `subject_box` was literally an
    # element of this list; a subject computed from anywhere else would have
    # drawn no box at all and told nobody.
    is_subject: bool = False


@dataclass
class PhotoCheck(_Dict):
    """Everything the deterministic gate knows about one photo."""
    photo_id: int
    path: str
    filename: str
    width: int = 0
    height: int = 0
    # classical capture metrics, same formulas the corpus was scored with
    blur_laplacian_var: float = 0.0
    brightness: float = 0.0
    contrast_rms: float = 0.0
    dark_clipped_frac: float = 0.0
    bright_clipped_frac: float = 0.0
    colourfulness: float = 0.0
    capture_quality: float = 0.0
    quality_bucket: str = "unknown"
    # detector + zero-shot tags
    detections: list[Detection] = field(default_factory=list)
    truck_conf: float = 0.0
    truck_area_frac: float = 0.0
    # Largest competing vehicle box (car/motorcycle/bus...). A truck box that
    # is smaller than this is scenery, not the subject being sold.
    competing_area_frac: float = 0.0
    truck_dominant: bool = False
    # The one detection the gate treats as the vehicle being sold, xyxy in
    # source pixels. Decided here rather than in the browser so the box a
    # viewer sees and the pixels the vision model is given are the same truck:
    # a dealer-lot photo has five, and averaging them describes none of them.
    subject_box: list[float] | None = None
    # Why that box, in a sentence: "recurs in 9 of 12 frames", "the only
    # vehicle in frame", "none: engine bay close-up, the only truck box is 4%
    # of the frame". Same posture as `EvidenceReport.fell_back_from` - the
    # gate is allowed to change its mind, not to do it where nobody can see.
    subject_basis: str = ""
    subject_score: float = 0.0
    # Cosine between the chosen crop and the set's appearance prototype.
    subject_sim: float = 0.0
    non_truck_subject: str | None = None
    view: str = "unknown"
    view_conf: float = 0.0
    content: str = "keep"
    content_conf: float = 0.0
    # Keep-mass, not top-1: the three "keep" prompts split the probability of
    # a genuine vehicle photo between them, so a legitimate cab interior can
    # top out at 0.4 on any single one of them.
    keep_mass: float = 0.0
    body_tag: str = "unknown"
    body_tag_conf: float = 0.0
    # The L2-normalised CLIP image embedding, carried for app.perception and
    # dropped on serialisation. Computed by ClipTagger.tag either way.
    _embedding: Any = None
    # Every vehicle box that could be the subject, with its crop embedding.
    # Underscore-prefixed for the same reason as `_embedding`: numpy arrays
    # never reach JSON or an SSE frame.
    _candidates: Any = None
    # verdict
    usable: bool = True
    reasons: list[str] = field(default_factory=list)


@dataclass
class GateReport(_Dict):
    decision: GateDecision = GateDecision.PASS
    headline: str = ""
    photos: list[PhotoCheck] = field(default_factory=list)
    usable_photo_ids: list[int] = field(default_factory=list)
    views_present: list[str] = field(default_factory=list)
    missing_views: list[str] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)   # "send me a shot of X"
    truck_evidence: str = ""
    # What the set concluded about which vehicle is being sold. Set-level,
    # because a background lorry appears in one frame and the subject appears
    # in fifteen - and that is evidence no single frame can hold.
    subject_evidence: str = ""
    subject_method: str = ""        # recurring_vehicle | single_frame | none
    subject_consistency: float = 0.0    # mean cosine of the chosen crops
    subject_frames: int = 0             # frames that agreed with the prototype
    # CLIP clusters among whole-vehicle seed crops. Calibrated before it may
    # block pricing; 0 means "not computed or not enough seeds".
    subject_clusters: int = 0
    # Zero-shot tractor vs rigid, pooled over whole-vehicle frames only.
    body_tag: str = "unknown"
    body_tag_conf: float = 0.0
    # Coverage can be too thin to defend a number while still being rich
    # enough to describe condition. That case re-asks instead of refusing.
    blocks_pricing: bool = False
    elapsed_s: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.decision.value.startswith("refuse")


# --- stage 1b: trained perception heads -----------------------------------

@dataclass
class PhotoPerception(_Dict):
    """What the trained heads make of one photo.

    Distinct from PhotoCheck on purpose: PhotoCheck is what a formula measured,
    this is what a fitted model predicted, and the report keeps them apart so a
    reader can see which is which.
    """
    photo_id: int = 0
    degraded_prob: float = 0.0
    severity: float = 0.0
    degradations: list[str] = field(default_factory=list)
    view: str = "unknown"
    view_conf: float = 0.0
    # False when the photo is too corrupted to support a claim resting on fine
    # detail. Consumed by app.reconcile, never by the gate - a degraded photo
    # is still a photo of the truck.
    fine_detail_ok: bool = True


@dataclass
class PerceptionReport(_Dict):
    photos: list[PhotoPerception] = field(default_factory=list)
    brand: str | None = None
    brand_conf: float = 0.0
    model_card: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def by_id(self, photo_id: int) -> "PhotoPerception | None":
        return next((p for p in self.photos if p.photo_id == photo_id), None)


@dataclass
class Correction(_Dict):
    """One disagreement between the trained heads and the vision model.

    Recorded rather than applied silently. The precedent is
    `EvidenceReport.fell_back_from`: the system is allowed to change its mind,
    it is not allowed to do so where nobody can see it.
    """
    # reconcile: unsupported_detail | identity_conflict | coverage_restored
    #            | odometer_recovered | odometer_conflict
    # evidence:  severity_calibrated | severity_raise_clamped
    #            | uncorroborated_finding | sample_disagreement
    kind: str = ""
    photo_id: int | None = None
    detail: str = ""
    before: str = ""
    after: str = ""


@dataclass
class ReconcileReport(_Dict):
    corrections: list[Correction] = field(default_factory=list)
    identity_conflict: str = ""
    # Widening this stage asks pricing for, as (label, factor) pairs.
    widening: list[list] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def n(self) -> int:
        return len(self.corrections)

    def of_kind(self, kind: str) -> list["Correction"]:
        return [c for c in self.corrections if c.kind == kind]


# --- stage 2: evidence ----------------------------------------------------

@dataclass
class Magnitude(_Dict):
    """What one photograph can honestly say about how big a defect is.

    Every field is ABSOLUTE - answerable from a single frame with no knowledge
    of what else is on the truck. The severity ordinal is not: asking one
    close-up call whether a tire is "moderate" is asking for a relative
    judgement from an absolute-only observation, sixteen times, and then
    summing the answers into a price. So the close-up reports magnitude and a
    provisional ordinal, and the set-aware calibration pass decides severity.
    """
    extent: str = ""        # spot | local | widespread | whole_component
    state: str = ""         # as_new | worn_in_service | end_of_life | failed
    consumable: bool = False
    blocks_use: str = ""    # no | maybe | yes | cannot_tell


@dataclass
class Issue(_Dict):
    """One defect, bound to the photo it was seen in.

    `photo_id` is mandatory by construction: an issue that cannot name a
    photo is dropped in `evidence.parse`, because an uncheckable claim is
    exactly what the brief means by reasoning a buyer cannot trust.
    """
    photo_id: int
    component: str          # steer_tires | drive_tires | fifth_wheel | ...
    observation: str        # what is visible, in truck vocabulary
    severity: str           # cosmetic | minor | moderate | major
    confidence: float
    price_impact: str = "none"   # none | low | medium | high
    # Other photos the same defect was seen in. Fourteen per-photo calls report
    # one worn steer tire three times; the synthesis pass folds those into one
    # finding and records the corroboration here rather than discarding it.
    # It is also the honest consumer-facing confidence signal - "seen in 3
    # photos" is checkable in a way that "confidence 0.87" is not.
    also_seen_in: list[int] = field(default_factory=list)
    # Normalised [x, y, w, h] in 0-1 of the original photograph. The close-up
    # pass is asked to point at the pixels that show the defect; missing or
    # junk boxes stay None and the screen opens the photo unmarked.
    box: list[float] | None = None
    # What one frame could say on its own, before the set had an opinion.
    magnitude: Magnitude | None = None
    # The ordinal the single-photo call proposed, kept when the calibration
    # pass revised it. A recalibrated finding is shown as recalibrated.
    severity_provisional: str = ""
    severity_reason: str = ""
    # One entry per sample of this photo that reported this defect. The spread
    # is the honest uncertainty; `confidence` is the agreement rate over it.
    severity_votes: list[str] = field(default_factory=list)
    # Reported by fewer samples than the quorum. Kept and shown - never
    # deleted - but it cannot be promoted to `major` on one vote.
    corroborated: bool = True
    # What the model said about its own claim, kept for the disclosure now
    # that `confidence` means cross-sample agreement instead.
    self_confidence: float = 0.0
    # A field that could not be read as its enum. Shown with its photo and
    # weighted at zero, because rounding an unreadable value up to "minor" was
    # silent inflation.
    ungraded: bool = False
    # ["minor", "moderate"] when two frames disagreed and the merge had to
    # pick. Printed, so a reader sees the disagreement rather than its winner.
    severity_span: list[str] = field(default_factory=list)


@dataclass
class PhotoFinding(_Dict):
    """One photo, read on its own, by its own vision call.

    The unit the fan-out produces and the unit the screen streams. `photo_id`
    is not a claim the model made here - the call was given exactly one photo,
    so the binding is structural. That is a strictly stronger guarantee than
    the single-call design, where an issue citing a photo that was never sent
    had to be detected and dropped.
    """
    photo_id: int
    view: str
    shows: str = ""             # one plain line: what this frame is of
    legible: bool = True
    issues: list[Issue] = field(default_factory=list)
    # Things this frame positively shows to be in good order. A report that can
    # only name faults is not an appraisal, it is a complaint, and a buyer
    # deciding whether to drive six hours needs the other half.
    strengths: list[str] = field(default_factory=list)
    cannot_tell: list[str] = field(default_factory=list)
    # Set only by a frame that actually shows a legible odometer. The reading
    # moved here from the set-level pass when evidence became a fan-out: asking
    # one call to read six digits off one of sixteen downscaled photos was
    # always the weakest link, and the dashboard close-up is looking straight
    # at it.
    odometer_km: int | None = None
    confidence: float = 0.0
    # True when the subject crop was sent rather than the whole frame. Surfaced
    # because "I looked at this truck, not the four behind it" is part of the
    # answer, not an implementation detail.
    cropped: bool = False
    # Which backend answered, and how many samples of this photo were taken.
    # A fallback inside the fan-out names the photo it happened on.
    backend: str = ""
    samples: int = 1
    sample_errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    # This photo's call failed and the appraisal carried on without it. Recorded
    # rather than swallowed: thirteen frames read and one lost is a different
    # answer from fourteen read, and the reader is entitled to know which.
    error: str = ""


@dataclass
class VehicleRead(_Dict):
    """What the photos say the truck is, independent of what the seller typed."""
    make: str | None = None
    model: str | None = None
    body_type: str | None = None
    cab_type: str | None = None
    axle_config: str | None = None
    approx_year_range: str | None = None
    odometer_km: int | None = None
    odometer_photo_id: int | None = None
    badges_seen: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class FamilyRollup(_Dict):
    """One subsystem of the truck, rolled up across every finding on it."""
    family: str
    demerit: float = 0.0        # saturated: the worst counts in full, the rest decay
    n_findings: int = 0
    worst: str = ""             # severity id
    seen: float = 0.0           # 0.0 | 0.5 | 1.0 - how well it was photographed
    credit: float = 0.0         # 0.0 | 0.5 | 1.0 - clean, and affirmed clean
    note: str = ""


@dataclass
class ConditionRollup(_Dict):
    """The one object the grade and the price are both derived from.

    They used to be computed by unrelated code from the same list under
    opposite rules - the grade was worst-of and count-blind, the price was an
    uncapped sum where count drove money - and nothing checked them against
    each other.
    """
    families: list[FamilyRollup] = field(default_factory=list)
    demerit: float = 0.0            # S, summed across families
    worst_family_demerit: float = 0.0
    coverage: float = 0.0           # C, in 0..1, from legible photos only
    merit: float = 0.0              # M, in 0..C by construction
    grade: str = "unknown"
    grade_reason: str = ""          # the term that decided it, in a sentence
    ungraded_findings: int = 0


@dataclass
class EvidenceReport(_Dict):
    vehicle: VehicleRead = field(default_factory=VehicleRead)
    # Whether every photo is of the SAME vehicle. A seller padding a listing
    # with photos of a tidier truck is a real marketplace failure, and averaging
    # condition across two vehicles produces a confident number about neither.
    same_vehicle: bool = True
    vehicle_mismatch: str = ""
    photo_findings: list[PhotoFinding] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    condition_summary: dict[str, str] = field(default_factory=dict)
    condition_grade: str = "unknown"      # excellent | good | fair | poor
    # The deterministic rollup the grade and the price both come from.
    condition: ConditionRollup | None = None
    # What the synthesis pass graded it. A recorded second opinion, shown and
    # not obeyed, because the deterministic one is reproducible and testable.
    condition_grade_model: str = ""
    grade_disagreement: str = ""
    # Severity changes the calibration pass made, in `reconcile`'s format:
    # before, after, reason, never deleted.
    corrections: list[Correction] = field(default_factory=list)
    calibration_note: str = ""
    # Components the photos positively showed to be in order. Collected on
    # every close-up call and, until now, worth nothing downstream.
    confirmed_sound: list[str] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    backend: str = ""
    model: str = ""
    # Backends that were tried and failed before this one answered. Surfaced on
    # the report and on screen: falling back is allowed, doing it quietly is not.
    fell_back_from: list[str] = field(default_factory=list)
    # One entry per vision call the fan-out made, in the order they were
    # issued: ("identity", 11.8), ("photo 3", 7.2), ("synthesis", 4.1). The
    # trace line and the disclosure both read this.
    calls: list[list] = field(default_factory=list)
    photos_read: int = 0
    photos_failed: int = 0
    elapsed_s: float = 0.0
    raw_text: str = ""
    parse_warnings: list[str] = field(default_factory=list)


# --- stage 3: price -------------------------------------------------------

@dataclass
class Comparable(_Dict):
    listing_id: str
    source: str
    make: str
    model: str
    year: int | None
    km: float | None
    price: float | None
    currency: str
    price_try: float | None
    price_usd: float | None
    url: str = ""
    distance: float = 0.0       # similarity distance in feature space
    why: str = ""


@dataclass
class ConditionAdjustment(_Dict):
    multiplier: float = 1.0
    pct: float = 0.0
    cap_pct: float = 0.0
    drivers: list[str] = field(default_factory=list)
    basis: str = ""
    # The cap and the weights are not the same kind of number and must not be
    # labelled as though they were. The cap is one out-of-fold residual sigma,
    # measured on 84 listings over 958 evaluations. The weight tables that
    # decide where inside the cap a truck lands are hand-set assumptions -
    # this corpus has no condition ground truth to fit them against.
    cap_basis: str = ""         # measured
    weights_basis: str = ""     # assumed
    coverage_pct: float = 0.0   # how much of the truck was photographed well
    merit_pct: float = 0.0      # how much of it was affirmed sound
    direction: str = "none"     # discount | premium | none
    notes: list[str] = field(default_factory=list)


@dataclass
class AskingVerdict(_Dict):
    """How the seller's own number compares to the evidence.

    Judged against the comparable-asking band, not the condition-adjusted one:
    the question "is this priced like other trucks of its age and mileage" and
    the question "is it worth that given its condition" are different, and a
    buyer needs both answered separately.
    """
    asking: float = 0.0
    currency: str = "TRY"
    vs_comparables_pct: float = 0.0
    vs_estimate_pct: float = 0.0
    inside_comparable_band: bool = False
    label: str = ""            # priced with the market | above | below
    summary: str = ""


@dataclass
class AnchorEstimate(_Dict):
    """What the truck cost new, depreciated - an estimate that needs no comparable.

    The comparables route cannot price a brand it has never seen, and 78 of the
    84 Turkish listings are Ford. This route only needs a published new price
    and a retention curve, so it still has an opinion about a Mercedes.
    """
    ok: bool = False
    reason: str = ""
    new_price: float = 0.0
    currency: str = "TRY"
    retention: float = 0.0
    point: float = 0.0
    # Share of the blended log-price this route contributed, by inverse variance.
    weight: float = 0.0
    matched: str = ""
    source: str = ""
    source_type: str = ""
    source_url: str = ""
    as_of: str = ""
    basis: str = ""


@dataclass
class PriceEstimate(_Dict):
    ok: bool = True
    reason: str = ""
    currency: str = "TRY"
    point: float = 0.0
    low: float = 0.0
    high: float = 0.0
    interval_level: float = 0.8
    point_usd: float = 0.0
    low_usd: float = 0.0
    high_usd: float = 0.0
    # The comparables-only estimate, before anything the photos said. Kept
    # separate because the measured interval coverage belongs to THIS band:
    # the model is fit on asking prices, so it predicts what this truck would
    # be asked for. The condition adjustment then deliberately departs from
    # that, and conflating the two would launder a measured number onto an
    # unmeasured one.
    baseline_point: float = 0.0
    baseline_low: float = 0.0
    baseline_high: float = 0.0
    anchor: AnchorEstimate | None = None
    adjustment: ConditionAdjustment = field(default_factory=ConditionAdjustment)
    asking: AskingVerdict | None = None
    comparables: list[Comparable] = field(default_factory=list)
    drivers: list[dict] = field(default_factory=list)   # feature contributions
    inputs: dict = field(default_factory=dict)
    inputs_provenance: dict = field(default_factory=dict)
    model_card: dict = field(default_factory=dict)
    widened: list[str] = field(default_factory=list)    # why the band is wider
    caveats: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


# --- the whole thing ------------------------------------------------------

@dataclass
class TraceStep(_Dict):
    step: str
    detail: str
    elapsed_s: float


@dataclass
class Appraisal(_Dict):
    status: str = "ok"            # ok | refused | need_more_photos
    headline: str = ""
    gate: GateReport = field(default_factory=GateReport)
    perception: PerceptionReport | None = None
    evidence: EvidenceReport | None = None
    reconcile: ReconcileReport | None = None
    price: PriceEstimate | None = None
    requests: list[str] = field(default_factory=list)
    declared: dict = field(default_factory=dict)
    trace: list[TraceStep] = field(default_factory=list)
    elapsed_s: float = 0.0
    version: str = ""
