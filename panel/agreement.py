"""How much three independent readers actually agreed.

An α you cannot verify is worse than no α, so this is a from-scratch
Krippendorff implementation with the published worked example pinned in
`tests/test_panel.py` - nominal, ordinal and interval on the same data, plus a
second, differently-organised computation of the same quantity.

Two things to keep straight when reading the output:

  * `not_visible` is MISSING, not a fifth low severity. A component nobody
    could see is an absent reading; folding it into "none" would manufacture
    agreement out of two readers both failing to look. Krippendorff's α takes
    missing data natively - a unit with fewer than two remaining readings drops
    out - so the ordinal α is computed over the components at least two
    panelists could see. `visibility_agreement` reports the other half.

  * α corrects for chance, and when almost every answer is the same value the
    expected disagreement D_e collapses toward zero and α goes unstable or
    undefined even at 95% raw agreement. That is the α paradox and it is the
    LIKELY regime here: a well-calibrated panel writes "none" most of the time.
    So every α is returned next to its raw percent agreement and its value
    counts, and D_e == 0 returns α = None rather than a fabricated 1.0.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Sequence

from app.evidence import prompts as P
from panel.protocol import NOT_VISIBLE, SEVERITY_SCALE


@dataclass
class Alpha:
    alpha: float | None
    d_observed: float
    d_expected: float
    n_units: int          # units with >= 2 readings
    n_observations: int
    metric: str
    counts: dict          # value -> marginal count
    note: str = ""

    def __str__(self) -> str:
        a = "undefined" if self.alpha is None else f"{self.alpha:.3f}"
        return (f"alpha({self.metric})={a} over {self.n_units} units / "
                f"{self.n_observations} readings")


def _delta(metric: str, values: Sequence, counts: dict) -> Callable[[int, int], float]:
    """Squared difference between two value INDICES, per Krippendorff."""
    if metric == "nominal":
        return lambda i, j: 0.0 if i == j else 1.0
    if metric == "interval":
        return lambda i, j: float((i - j) ** 2)
    if metric == "ordinal":
        n = [counts.get(v, 0.0) for v in values]

        def ordinal(i: int, j: int) -> float:
            lo, hi = (i, j) if i <= j else (j, i)
            run = sum(n[lo:hi + 1]) - (n[lo] + n[hi]) / 2.0
            return float(run ** 2)
        return ordinal
    raise ValueError(f"unknown metric {metric!r}")


def krippendorff_alpha(units: Sequence[Sequence], *, metric: str = "ordinal",
                       values: Sequence | None = None) -> Alpha:
    """α over `units`, each a list of that unit's readings. Missing = omitted.

    `values` fixes the value order (required for ordinal); it defaults to the
    sorted set observed.
    """
    usable = [[v for v in u if v is not None] for u in units]
    usable = [u for u in usable if len(u) >= 2]
    if values is None:
        values = sorted({v for u in usable for v in u})
    index = {v: i for i, v in enumerate(values)}
    k = len(values)

    # coincidence matrix: each unit contributes its ordered pairs, weighted 1/(m-1)
    o = [[0.0] * k for _ in range(k)]
    for unit in usable:
        m = len(unit)
        for a, b in combinations(unit, 2):
            i, j = index[a], index[b]
            o[i][j] += 1.0 / (m - 1)
            o[j][i] += 1.0 / (m - 1)

    marginal = [sum(row) for row in o]
    n = sum(marginal)
    counts = {v: marginal[i] for i, v in enumerate(values)}
    n_obs = sum(len(u) for u in usable)
    if n < 2:
        return Alpha(None, 0.0, 0.0, len(usable), n_obs, metric, counts,
                     "fewer than two coincident readings")

    delta = _delta(metric, values, counts)
    d_o = sum(o[i][j] * delta(i, j) for i in range(k) for j in range(k)) / n
    d_e = sum(marginal[i] * marginal[j] * delta(i, j)
              for i in range(k) for j in range(k)) / (n * (n - 1))
    if d_e == 0:
        return Alpha(None, d_o, d_e, len(usable), n_obs, metric, counts,
                     "expected disagreement is zero - every reading is the same "
                     "value, so alpha is undefined rather than 1.0")
    return Alpha(1.0 - d_o / d_e, d_o, d_e, len(usable), n_obs, metric, counts)


def percent_agreement(units: Sequence[Sequence]) -> tuple[float | None, int]:
    """Mean over units of the share of rater PAIRS that gave the same value."""
    pairs = hits = 0
    for unit in units:
        seen = [v for v in unit if v is not None]
        for a, b in combinations(seen, 2):
            pairs += 1
            hits += (a == b)
    return (hits / pairs if pairs else None), pairs


# --- the two panel-specific views -----------------------------------------

def component_units(vectors_by_vehicle: dict[str, list[dict[str, str]]],
                    *, components: Sequence[str] = tuple(P.COMPONENTS)
                    ) -> dict[tuple[str, str], list[str]]:
    """(vehicle, component) -> one state per panelist."""
    return {(vid, c): [vec.get(c, NOT_VISIBLE) for vec in vectors]
            for vid, vectors in vectors_by_vehicle.items()
            for c in components}


def severity_alpha(vectors_by_vehicle: dict[str, list[dict[str, str]]]) -> Alpha:
    """Ordinal α over severity, with `not_visible` treated as missing."""
    units = [[None if s == NOT_VISIBLE else s for s in states]
             for states in component_units(vectors_by_vehicle).values()]
    return krippendorff_alpha(units, metric="ordinal", values=SEVERITY_SCALE)


def state_alpha(vectors_by_vehicle: dict[str, list[dict[str, str]]]) -> Alpha:
    """Nominal α over the full six-word vocabulary, `not_visible` included.

    A companion, not a substitute: it answers "did they pick the same word",
    which is a different and easier question than "did they read the same
    severity".
    """
    units = list(component_units(vectors_by_vehicle).values())
    return krippendorff_alpha(units, metric="nominal",
                              values=SEVERITY_SCALE + [NOT_VISIBLE])


def visibility_alpha(vectors_by_vehicle: dict[str, list[dict[str, str]]]) -> Alpha:
    """Nominal α on the prior question: could this component be seen at all."""
    units = [[s == NOT_VISIBLE for s in states]
             for states in component_units(vectors_by_vehicle).values()]
    return krippendorff_alpha(units, metric="nominal", values=[False, True])


@dataclass
class GradeAgreement:
    unanimous: float | None            # share of vehicles where all raters matched
    pairwise: float | None             # share of rater pairs that matched
    n_vehicles: int
    confusion: dict = field(default_factory=dict)   # (a, b) -> count, a <= b
    needs_adjudication: list = field(default_factory=list)
    alpha: Alpha | None = None
    distribution: dict = field(default_factory=dict)


def adjudication_required(grades: Sequence[str]) -> bool:
    """All three panelists differing on grade. The trigger, alone and testable."""
    seen = [g for g in grades if g]
    return len(seen) >= 3 and len(set(seen)) == len(seen)


def grade_agreement(grades_by_vehicle: dict[str, list[str]]) -> GradeAgreement:
    units = list(grades_by_vehicle.values())
    pairwise, _ = percent_agreement(units)
    unan = [all(g == u[0] for g in u) for u in units if len(u) >= 2]
    confusion: Counter = Counter()
    for unit in units:
        for a, b in combinations(unit, 2):
            confusion[tuple(sorted((a, b), key=P.GRADES.index))] += 1
    return GradeAgreement(
        unanimous=(sum(unan) / len(unan) if unan else None),
        pairwise=pairwise,
        n_vehicles=len(units),
        confusion=dict(confusion),
        needs_adjudication=[vid for vid, g in grades_by_vehicle.items()
                            if adjudication_required(g)],
        alpha=krippendorff_alpha(units, metric="ordinal", values=list(P.GRADES)),
        distribution=dict(Counter(g for unit in units for g in unit)),
    )


def consensus(vectors: Sequence[dict[str, str]]) -> dict[str, str]:
    """Per component, the median visible severity; not_visible only if the
    majority could not see it. Median, not mean: the scale is ordinal, and a
    median never invents a level nobody wrote."""
    out = {}
    for c in P.COMPONENTS:
        states = [v.get(c, NOT_VISIBLE) for v in vectors]
        visible = sorted((s for s in states if s != NOT_VISIBLE),
                         key=SEVERITY_SCALE.index)
        # a tie goes to the reading, not to the blank: two of three seeing a
        # component is enough to call it seen.
        out[c] = visible[(len(visible) - 1) // 2] \
            if len(visible) * 2 >= len(states) and visible else NOT_VISIBLE
    return out


def consensus_grade(grades: Sequence[str]) -> str:
    """Majority grade; with no majority, the ordinal median - never the worst."""
    counts = Counter(grades)
    top, n = counts.most_common(1)[0]
    if n * 2 > len(grades):
        return top
    ordered = sorted(grades, key=P.GRADES.index)
    return ordered[(len(ordered) - 1) // 2]
