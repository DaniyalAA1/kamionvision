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
    non_truck_subject: str | None = None
    view: str = "unknown"
    view_conf: float = 0.0
    content: str = "keep"
    content_conf: float = 0.0
    # Keep-mass, not top-1: the three "keep" prompts split the probability of
    # a genuine vehicle photo between them, so a legitimate cab interior can
    # top out at 0.4 on any single one of them.
    keep_mass: float = 0.0
    # The L2-normalised CLIP image embedding, carried for app.perception and
    # dropped on serialisation. Computed by ClipTagger.tag either way.
    _embedding: Any = None
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
    kind: str = ""              # unsupported_detail | identity_conflict | view_conflict | coverage_restored
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


@dataclass
class PhotoEvidence(_Dict):
    photo_id: int
    view: str
    legible: bool = True
    notes: str = ""


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
class EvidenceReport(_Dict):
    vehicle: VehicleRead = field(default_factory=VehicleRead)
    # Whether every photo is of the SAME vehicle. A seller padding a listing
    # with photos of a tidier truck is a real marketplace failure, and averaging
    # condition across two vehicles produces a confident number about neither.
    same_vehicle: bool = True
    vehicle_mismatch: str = ""
    per_photo: list[PhotoEvidence] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    condition_summary: dict[str, str] = field(default_factory=dict)
    condition_grade: str = "unknown"      # excellent | good | fair | poor
    coverage_gaps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    backend: str = ""
    model: str = ""
    # Backends that were tried and failed before this one answered. Surfaced on
    # the report and on screen: falling back is allowed, doing it quietly is not.
    fell_back_from: list[str] = field(default_factory=list)
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
