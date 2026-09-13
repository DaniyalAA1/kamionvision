"""One rollup of the photo evidence, and both the grade and the price from it.

The grade and the price used to be computed by unrelated code from the same
list of findings, under opposite rules. The grade was pure worst-of - one
`moderate` anywhere across sixteen photos graded the whole truck `fair`, and
`SYNTHESIS_PROMPT` rule 3 explicitly forbids count from moving it. The price
was an uncapped sum over the same list, so count was the only thing that moved
money. Nothing checked one against the other, and the two disagreed about which
truck was worse: ten minor findings spread over ten subsystems priced at -1.8%
and graded `good`, while one moderate scuff on a fuel-tank strap priced at
-0.5% and graded `fair`.

So there is one intermediate object now. `rollup()` turns findings into a
`ConditionRollup`; `grade_of()` and `net_score()` are both pure functions of
it. They cannot disagree, the happy path and the synthesis-failure path run the
same code, and `tests/test_condition.py` asserts the agreement directly.

Three layers, one knob each:

  per finding   severity x stated price impact x confidence, with a floor.
                An unreadable enum weighs ZERO and is marked `ungraded` - it
                used to round up to minor/low, which was silent inflation.
  per family    the worst defect on a subsystem counts in full, the second
                counts 40%, the third 16%, and no subsystem can cost more than
                the single worst finding this vocabulary can express. That is
                how an appraiser quotes work: the second scuff on a panel is
                already inside the repaint.
  across        a plain sum. Tires and a cracked frame rail are two repairs
                and they should add.

The positive side is new and is deliberately hard to earn. Coverage is counted
only from photos that were read successfully AND marked legible, merit is
bounded by coverage (`M <= C`) in the arithmetic rather than by a rule, and one
finding the model itself calls real cancels the premium outright. Absence of
evidence therefore cannot become evidence of excellence, and a blurred phone
set loses its premium and widens its band instead of accumulating false
defects.

Where this file sits matters: `app/pricing/model.py` imports it, and importing
anything under `app.evidence` executes its `__init__`, which pulls in `app.vlm`
and PIL. Pricing is importable today without the vision stack and stays that
way. Precedent: `app/reconcile.py`. For the same reason the component and view
vocabularies are restated here as literals rather than imported from
`app.evidence.prompts` / `app.vision` (which imports torch); a test asserts
both restatements agree with the originals, exactly as `js/elevation.js` is
pinned against `vision.VIEW_LABELS`.

Everything here is pure: no I/O, no model, no network.
"""
from __future__ import annotations

import math

from .schema import ConditionRollup, FamilyRollup, Issue, PhotoFinding

# --- the vocabularies ------------------------------------------------------

SEVERITY_RANK = {"cosmetic": 0, "minor": 1, "moderate": 2, "major": 3}
IMPACT_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
GRADE_RANK = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}

# ASSUMED. Severity and stated price impact combine multiplicatively, then
# scale by confidence in the finding: a low-confidence major defect should not
# move the price like a certain one. These tables decide where inside the
# measured cap a truck lands and there is nothing on this corpus to fit them
# against - `data/DATASET_CARD.md` limitation 3, no condition ground truth of
# any kind. `scripts/probe_condition_residual.py` is the experiment that would
# turn them into measurements, and it is expected to fail its own null.
SEVERITY_WEIGHT = {"cosmetic": 0.15, "minor": 0.40, "moderate": 1.00, "major": 2.20}
IMPACT_WEIGHT = {"none": 0.0, "low": 0.40, "medium": 1.00, "high": 2.00}

WEIGHTS_BASIS = (
    "assumed, not fitted: the severity x impact weight table, the per-subsystem "
    "saturation decay, the family value weights and the coverage thresholds are "
    "all hand-set. This corpus carries no condition ground truth to fit them "
    "against, so they are stated rather than measured"
)

# ASSUMED, and the same floor `reconcile.CONFIDENCE_FLOOR` uses: a finding the
# model barely believes still costs something, because a downgrade must never
# become a silent deletion.
CONFIDENCE_FLOOR = 0.2
# ASSUMED. An unparseable confidence is weaker evidence than a stated 0.5 -
# 0.5 is where the weight table says "the model half believes it" - so the
# default has to sit below it. The exact value is a guess.
CONFIDENCE_UNPARSEABLE = 0.3

# ASSUMED. The second finding on one subsystem counts 40%, the third 16%.
DECAY = 0.40
# DERIVED, not guessed: the largest demerit a single finding can produce
# (major x high x confidence 1.0). So the family cap says "a subsystem is never
# worth more than its own worst possible single defect".
FAMILY_CAP = max(SEVERITY_WEIGHT.values()) * max(IMPACT_WEIGHT.values())   # 4.40

# ASSUMED. Demerit at which the discount reaches ~76% of its cap. Four
# confident moderate/medium findings on four different subsystems.
DEMERIT_SCALE = 4.0
# ASSUMED. Merit is already in 0..1, so this is a straight pass-through; it
# exists as a named constant so the eval harness can sweep the two sides
# independently.
MERIT_SCALE = 1.0
# ASSUMED. One confident moderate/medium finding is worth exactly this much
# demerit, and it is the point at which the premium is fully cancelled. You do
# not pay a premium for a truck that has one thing badly wrong, however clean
# the rest of it is.
MERIT_BLOCK = 1.0

# ASSUMED, calibrated to discriminate rather than to be correct: the coverage
# index over all 200 corpus vehicles is mean 0.81, median 0.85, p10 0.61. About
# 93% of the corpus clears C_MIN and about half clears C_GOOD, so the gate
# separates within the observed distribution instead of saturating at one end.
C_MIN, C_GOOD = 0.60, 0.85
# ASSUMED. Below this the set is not describable as a condition at all.
C_UNGRADED = 0.35
C_EXCELLENT = 0.75
M_EXCELLENT = 0.55
S_EXCELLENT = 0.25
S_FAIR = 1.8
S_POOR = 6.0
# ASSUMED. A finding real enough to set the grade on its own must be asserted
# confidently or corroborated by a second frame.
QUALIFIED_CONFIDENCE = 0.75

# ASSUMED. Above this many findings on one photograph the model has produced a
# list rather than an inspection. Nothing is truncated - capping deletes
# evidence - it is recorded as a parse warning instead.
FINDINGS_PER_PHOTO_WARN = 12


# --- the 14 component families --------------------------------------------
# Money, not prose. `prompts.COMPONENT_SUMMARY` groups findings for a reader
# and puts `cab_steps` with `paint_finish`; this groups them for a price and
# does not, because a bent step is a bolt-on and a scuffed panel is a bodyshop
# job. Both mappings stay.
#
# The three floating ids - `corrosion`, `paint_finish`, `fluid_leaks` - have no
# geometry of their own and land in the family they are physically observed on.
# That cross-id folding is the merge the old exact-id gate could never do.
# `warning_lights` is deliberately a family of one: a lit fault lamp must not
# be diluted into "cab interior".

FAMILIES: dict[str, tuple[float, tuple[str, ...]]] = {
    "steer_tires":     (0.07, ("steer_tires",)),
    "drive_tires":     (0.10, ("drive_tires",)),
    "wheels_hubs":     (0.05, ("wheels_rims", "brakes_hubs")),
    "fifth_wheel":     (0.08, ("fifth_wheel", "coupling_airlines")),
    "frame_corrosion": (0.14, ("chassis_frame", "undercarriage", "corrosion")),
    "suspension_air":  (0.07, ("air_suspension", "air_tanks_lines")),
    "body_panels":     (0.09, ("cab_exterior_panels", "front_bumper_valance",
                               "fairings_skirts", "doors_handles", "roof_deflector",
                               "paint_finish")),
    "steps_guards":    (0.02, ("cab_steps", "mudflaps_guards")),
    "glass_lights":    (0.05, ("windscreen_glass", "grille_headlights", "mirrors_visor")),
    "tanks":           (0.04, ("fuel_tank", "adblue_tank")),
    "engine_leaks":    (0.15, ("engine_bay", "fluid_leaks", "exhaust_dpf")),
    "cab_seats_trim":  (0.05, ("cab_interior_seats", "bunk_sleeper", "cab_floor_trim")),
    "cab_controls":    (0.04, ("steering_wheel_controls", "dashboard_instruments")),
    "warning_lights":  (0.05, ("warning_lights",)),
}

# Anything the model invented rather than chose from the enum. It still costs
# money - a defect is a defect - but it earns no coverage and no merit, because
# nobody can say which part of the truck was affirmed sound.
OTHER = "other"

COMPONENT_FAMILY = {c: f for f, (_, members) in FAMILIES.items() for c in members}

FAMILY_LABEL = {
    "steer_tires": "steer tires", "drive_tires": "drive tires",
    "wheels_hubs": "wheels and hubs", "fifth_wheel": "fifth wheel and coupling",
    "frame_corrosion": "frame and corrosion", "suspension_air": "suspension and air system",
    "body_panels": "body panels and paint", "steps_guards": "steps and guards",
    "glass_lights": "glass and lights", "tanks": "fuel and AdBlue tanks",
    "engine_leaks": "engine bay and leaks", "cab_seats_trim": "cab seats and trim",
    "cab_controls": "dash and controls", "warning_lights": "warning lights",
    OTHER: "other",
}

# Which views show a family well enough to say it is sound. Primary earns full
# coverage, secondary earns half - a tire close-up shows the tire, a 3/4 front
# shot shows it but not the way a close-up does. `damage_detail` grants nothing
# to anything: it can show any part, so crediting it would credit everything.
#
# View ids are `app.vision.VIEW_LABELS`, restated here so this file stays free
# of torch. `tests/test_condition.py` pins both directions.
VIEW_COVERAGE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "steer_tires":     (("tire_wheel",), ("exterior_side", "exterior_front_34",
                                          "exterior_front", "exterior_rear")),
    "drive_tires":     (("tire_wheel",), ("exterior_side", "exterior_front_34",
                                          "exterior_front", "exterior_rear")),
    "wheels_hubs":     (("tire_wheel",), ("exterior_side", "exterior_front_34",
                                          "exterior_front", "exterior_rear")),
    "fifth_wheel":     (("fifth_wheel",), ("exterior_rear", "chassis_undercarriage")),
    "frame_corrosion": (("chassis_undercarriage",), ("exterior_side", "exterior_rear")),
    "suspension_air":  (("chassis_undercarriage",), ("tire_wheel",)),
    "body_panels":     (("exterior_side", "exterior_front_34"),
                        ("exterior_front", "exterior_rear")),
    "steps_guards":    (("exterior_side",), ("exterior_front_34", "exterior_rear")),
    "glass_lights":    (("exterior_front", "exterior_front_34"), ("interior_cab",)),
    "tanks":           (("exterior_side",), ("chassis_undercarriage", "exterior_front_34")),
    "engine_leaks":    (("engine_bay",), ("chassis_undercarriage",)),
    "cab_seats_trim":  (("interior_cab",), ()),
    "cab_controls":    (("dashboard_odometer",), ("interior_cab",)),
    "warning_lights":  (("dashboard_odometer",), ("interior_cab",)),
}


def family_of(component: str) -> str:
    return COMPONENT_FAMILY.get(component, OTHER)


# --- layer 1: one finding --------------------------------------------------

def demerit(issue: Issue) -> float:
    """What one finding is worth, before any saturation.

    Every default here rounds DOWN. An unreadable severity is not a minor
    defect - it is a finding nobody graded - so it weighs nothing and is still
    printed with its photograph. The old `.get(..., 0.4)` fallbacks turned a
    garbage observation into real money.
    """
    if getattr(issue, "ungraded", False):
        return 0.0
    sw = SEVERITY_WEIGHT.get(issue.severity)
    iw = IMPACT_WEIGHT.get(issue.price_impact)
    if sw is None or iw is None:
        return 0.0
    try:
        conf = float(issue.confidence)
    except (TypeError, ValueError):
        conf = CONFIDENCE_UNPARSEABLE
    if not math.isfinite(conf):
        conf = CONFIDENCE_UNPARSEABLE
    return sw * iw * max(CONFIDENCE_FLOOR, min(1.0, conf))


def family_demerit(values: list[float]) -> float:
    """The worst defect on a subsystem counts in full, the rest decay.

    Sixteen per-photo calls describing one worn drive tire sixteen ways used to
    cost sixteen times one worn drive tire. Now the second statement is worth
    40% of the first, the third 16%, and the subsystem is capped at the worst
    single finding the vocabulary can express.
    """
    ordered = sorted((v for v in values if v > 0.0), reverse=True)
    total = sum(v * DECAY ** i for i, v in enumerate(ordered))
    return min(total, FAMILY_CAP)


def qualified(issue: Issue, rank: int) -> bool:
    """A finding real enough to set the grade on its own.

    Severity alone used to do it, which is how a scuffed fuel-tank strap graded
    a whole truck `fair`. The model's own stated price impact has to be medium
    or better, and the claim has to be either confidently asserted or seen in a
    second frame. Count still cannot escalate a grade - and now neither can one
    throwaway line.
    """
    if getattr(issue, "ungraded", False):
        return False
    if SEVERITY_RANK.get(issue.severity, -1) < rank:
        return False
    if IMPACT_RANK.get(issue.price_impact, -1) < IMPACT_RANK["medium"]:
        return False
    try:
        conf = float(issue.confidence)
    except (TypeError, ValueError):
        conf = 0.0
    if not math.isfinite(conf):
        conf = 0.0
    return conf >= QUALIFIED_CONFIDENCE or bool(issue.also_seen_in)


# --- coverage and merit ----------------------------------------------------

def coverage_gate(coverage: float) -> float:
    """How much of an earned premium thin coverage is allowed to keep."""
    if C_GOOD <= C_MIN:
        return 1.0
    return max(0.0, min(1.0, (coverage - C_MIN) / (C_GOOD - C_MIN)))


def _views_read(findings: list[PhotoFinding], gate=None) -> tuple[set[str], dict[str, set[str]]]:
    """Views a photo was actually read from, and the views that affirmed something.

    A view counts only from a frame that returned an answer AND was marked
    legible. That one line routes the whole degraded-photo story through
    coverage rather than through findings: a blurred tire shot earns no tire
    coverage, so the set loses its premium and widens its band instead of
    accumulating defects nobody can see.
    """
    by_id = {}
    if gate is not None:
        by_id = {c.photo_id: c.view for c in getattr(gate, "photos", []) or []}
    seen: set[str] = set()
    affirmed: dict[str, set[str]] = {}
    for finding in findings or []:
        if getattr(finding, "error", "") or not getattr(finding, "legible", True):
            continue
        view = finding.view or by_id.get(finding.photo_id) or "unknown"
        seen.add(view)
        if finding.strengths:
            affirmed.setdefault(view, set()).update(finding.strengths)
    return seen, affirmed


def _seen_level(family: str, views: set[str]) -> float:
    primary, secondary = VIEW_COVERAGE.get(family, ((), ()))
    if any(v in views for v in primary):
        return 1.0
    if any(v in views for v in secondary):
        return 0.5
    return 0.0


# --- the rollup ------------------------------------------------------------

def rollup(issues: list[Issue], findings: list[PhotoFinding] | None = None,
           *, gate=None) -> ConditionRollup:
    """Findings in, one object out. The grade and the price both read this."""
    findings = list(findings or [])
    issues = list(issues or [])
    photos_read = sum(1 for f in findings if not getattr(f, "error", ""))
    views, affirmed_views = _views_read(findings, gate)

    per_family: dict[str, list[Issue]] = {}
    ungraded = 0
    for issue in issues:
        if getattr(issue, "ungraded", False):
            ungraded += 1
        per_family.setdefault(family_of(issue.component), []).append(issue)

    rows: list[FamilyRollup] = []
    for family in list(FAMILIES) + [OTHER]:
        members = per_family.get(family, [])
        weight, _ = FAMILIES.get(family, (0.0, ()))
        values = [demerit(i) for i in members]
        d = family_demerit(values)
        seen = _seen_level(family, views) if family != OTHER else 0.0
        worst = ""
        graded = [i for i in members if not getattr(i, "ungraded", False)]
        if graded:
            worst = max(graded, key=lambda i: SEVERITY_RANK.get(i.severity, -1)).severity
        # Merit is for a subsystem nothing at all was flagged on. A cosmetic
        # scuff the model itself rates `none` for price impact costs nothing -
        # but it is still something somebody saw, so the panel it is on stops
        # being one of the panels this truck is being paid for.
        if not members and seen > 0.0:
            primary, secondary = VIEW_COVERAGE.get(family, ((), ()))
            was_affirmed = any(v in affirmed_views for v in (*primary, *secondary))
            credit = min(1.0 if was_affirmed else 0.5, seen)
        else:
            credit = 0.0
        if not members and seen == 0.0:
            continue
        rows.append(FamilyRollup(
            family=family, demerit=round(d, 4), n_findings=len(members),
            worst=worst, seen=seen, credit=credit,
            note=_family_note(family, members, values, d, seen, credit)))

    total = sum(r.demerit for r in rows)
    coverage = sum(FAMILIES[r.family][0] * r.seen for r in rows if r.family in FAMILIES)
    merit = sum(FAMILIES[r.family][0] * r.credit for r in rows if r.family in FAMILIES)

    out = ConditionRollup(
        families=rows,
        demerit=round(total, 4),
        worst_family_demerit=round(max((r.demerit for r in rows), default=0.0), 4),
        coverage=round(coverage, 4),
        merit=round(merit, 4),
        ungraded_findings=ungraded,
    )
    out.grade, out.grade_reason = grade_of(out, issues, photos_read=photos_read)
    return out


def _family_note(family: str, members: list[Issue], values: list[float],
                 d: float, seen: float, credit: float) -> str:
    label = FAMILY_LABEL.get(family, family.replace("_", " "))
    if not members:
        if credit >= 1.0:
            return f"{label}: photographed and positively called sound"
        if seen > 0.0:
            return (f"{label}: {'photographed' if seen >= 1.0 else 'only glimpsed'}, "
                    f"nothing flagged")
        return f"{label}: not photographed"
    graded = sum(1 for v in values if v > 0.0)
    if len(members) == 1:
        body = "one finding"
    elif d >= FAMILY_CAP - 1e-9:
        body = (f"{len(members)} findings, capped at the worst single defect this "
                f"vocabulary can express")
    else:
        body = f"{len(members)} findings, the worst counts in full and the rest decay"
    if graded < len(members):
        body += f" ({len(members) - graded} reported but not graded)"
    return f"{label}: {body}"


# --- the grade -------------------------------------------------------------

def grade_of(r: ConditionRollup, issues: list[Issue] | None = None, *,
             photos_read: int = 0) -> tuple[str, str]:
    """The ladder, in order. Deterministic, and authoritative on both paths.

    The old rule was `max(severity)` with `default=-1` and no `-1` key, so
    sixteen spotless photos graded `unknown` and finding ONE cosmetic scuff
    upgraded them to `good`. The exaggeration bug and the no-premium bug were
    the same bug from two sides.
    """
    issues = list(issues or [])
    if photos_read <= 0:
        return "unknown", ("unknown: no photograph was read, so there is nothing "
                           "to grade")
    if r.coverage < C_UNGRADED:
        return "unknown", (f"unknown: only {r.coverage:.0%} of the truck by value was "
                           f"photographed legibly, which is too little to describe a "
                           f"condition at all")

    worst_major = next((i for i in issues if qualified(i, SEVERITY_RANK["major"])), None)
    if worst_major is not None:
        return "poor", _qualified_reason("poor", worst_major)
    if r.demerit >= S_POOR:
        return "poor", (f"poor: no single defect is severe enough on its own, but the "
                        f"findings across {sum(1 for f in r.families if f.n_findings)} "
                        f"subsystems total {r.demerit:.1f} against a {S_POOR:.0f} "
                        f"threshold")

    worst_moderate = next((i for i in issues if qualified(i, SEVERITY_RANK["moderate"])), None)
    if worst_moderate is not None:
        return "fair", _qualified_reason("fair", worst_moderate)
    if r.demerit >= S_FAIR:
        return "fair", (f"fair: nothing individually serious, but the wear adds up to "
                        f"{r.demerit:.1f} across "
                        f"{sum(1 for f in r.families if f.n_findings)} subsystems")

    above_cosmetic = [i for i in issues
                      if not getattr(i, "ungraded", False)
                      and SEVERITY_RANK.get(i.severity, 0) > SEVERITY_RANK["cosmetic"]]
    gated_merit = r.merit * coverage_gate(r.coverage)
    if (r.coverage >= C_EXCELLENT and gated_merit >= M_EXCELLENT
            and r.demerit <= S_EXCELLENT and not above_cosmetic):
        found = (f"the only {len(issues)} finding(s) are cosmetic" if issues
                 else "nothing was found")
        return "excellent", (f"excellent: {r.coverage:.0%} of the truck by value was "
                             f"photographed legibly, {r.merit:.0%} of it was positively "
                             f"called sound, and {found}")

    if not issues:
        return "good", (f"good: nothing was flagged, but only {r.merit:.0%} of the truck "
                        f"by value was positively affirmed sound "
                        f"({r.coverage:.0%} photographed), which is short of what an "
                        f"excellent grade needs")
    return "good", (f"good: {len(issues)} finding(s), none of them severe enough or "
                    f"confident enough to set a grade on its own; the wear totals "
                    f"{r.demerit:.1f} against a {S_FAIR} threshold for fair")


def _qualified_reason(grade: str, issue: Issue) -> str:
    where = FAMILY_LABEL.get(family_of(issue.component),
                             issue.component.replace("_", " "))
    seen = (f" seen in {1 + len(issue.also_seen_in)} frames"
            if issue.also_seen_in else "")
    return (f"{grade}: the {where} carry a {issue.severity} defect the model rates "
            f"{issue.price_impact} price impact at {issue.confidence:.2f} "
            f"confidence{seen}")


# --- what pricing reads ----------------------------------------------------

def premium_block(r: ConditionRollup, issues: list[Issue] | None = None) -> float:
    """How much of an earned premium survives the defects that were found.

    Without this, a cracked frame rail on an otherwise immaculate, fully
    photographed truck nets out to about -1.3%: the clean bodywork pays for the
    frame. With it the premium is zero and the truck prices on the frame alone.
    In words - you do not pay a premium for a truck that has one thing badly
    wrong, however clean the rest of it is.

    The grade ladder makes the hard case unreachable on its own (a qualified
    moderate cannot coexist with an `excellent` grade), so this is belt and
    braces. It still earns its keep on the soft case: a subsystem carrying
    several cosmetic findings shaves the premium in proportion.
    """
    for issue in issues or []:
        if qualified(issue, SEVERITY_RANK["moderate"]):
            return 0.0
    return max(0.0, 1.0 - r.worst_family_demerit / MERIT_BLOCK)


def net_score(r: ConditionRollup, issues: list[Issue] | None = None) -> float:
    """What `tanh` is applied to. Positive is a premium, negative a discount.

    A premium is paid only to a truck this system is willing to call
    `excellent`, and that is the whole justification for paying one at all: the
    comparable baseline is AVERAGE-condition asking prices, so `good` means
    average and belongs at the baseline. A truck with ten minor findings on it
    is a normal truck, not a bargain and not a premium one.

    Above that line the premium ramps continuously from zero at the excellent
    threshold to its cap at a fully photographed, fully affirmed truck, so
    there is no cliff at the grade boundary. Below it the discount is the
    saturated demerit and nothing else. Which makes the two halves agree by
    construction: `excellent => multiplier >= 1`, everything else
    `=> multiplier <= 1`, and strictly below whenever anything was found.

    The demerit is not subtracted again on the premium side. It is already in
    there: a subsystem with any finding on it loses its merit credit outright,
    so the defects an excellent truck does have have already cost it merit.
    """
    if r.grade != "excellent":
        return -r.demerit / DEMERIT_SCALE
    span = max(1e-9, 1.0 - M_EXCELLENT)
    ramp = (r.merit * coverage_gate(r.coverage) - M_EXCELLENT) / span
    return max(0.0, ramp) * premium_block(r, issues) / MERIT_SCALE
