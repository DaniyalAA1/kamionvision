"""Suite 3 - monotonicity of condition against kilometres and age.

Shares its corpus sweep with `distribution`: one pass over N vehicles produces
the cached findings and BOTH suites score off them. That halves the headline
cost of the pair, and the budget below reports zero incremental calls for
exactly that reason rather than hiding the sharing in a footnote.

    rho_km        Spearman(S, km) over vehicles, S = the rollup demerit
    rho_age       Spearman(S, age)
    rho_partial   Spearman(S, km | age, capture_quality) via rank residuals
    null          permute km across vehicles WITHIN (market, brand) strata,
                  2000 draws

`capture_quality` is not an optional covariate. Older trucks are photographed
worse, and without partialling it out this test measures photography rather
than wear. The column is already in `images.csv`.

The pre-registered interpretation, which MUST be printed in the scorecard so
nobody over-reads the number:

    The corpus is reconditioned OEM stock (DATASET_CARD limitation 2), so the
    true km -> visible-wear relation is attenuated by construction. This is a
    direction check and a regression guard, not a validation. A significantly
    NEGATIVE rho is a bug. A near-zero rho is expected and uninformative. A
    strongly positive rho is suspicious and should be checked for the
    photography confound first.

The stronger monotonicity evidence in this harness is the dose-response curve
inside `twin_fp.by_severity_band`, where the ordering of the stimulus is known
by construction rather than inferred from an odometer. Rank it above this.
"""
from __future__ import annotations

import math
import random

import numpy as np

from ..scorecard import DIAGNOSTIC, MEASURED, Metric, SuiteResult
from .base import Budget

NAME = "monotonic"


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True) -> Budget:
    from . import distribution
    shared = distribution.plan(tier, seed=seed, model_id=model_id, effort=effort,
                               structured=structured, samples=samples,
                               require_files=require_files)
    return Budget(name=NAME, tier=tier, seed=seed, n_calls=0, units=shared.units,
                  params={"shares_corpus_with": distribution.NAME,
                          "vehicles": shared.units},
                  note="zero incremental calls: scores off the cached findings the "
                       "distribution sweep already paid for")


def run(plan_obj, client, **kwargs) -> list[dict]:
    """Reuse distribution's records; this suite never initiates vision calls."""
    shared = kwargs.get("distribution_records")
    return list(shared) if shared is not None else []


def score(records, plan_obj, cache, model_id):
    """Compute rank correlations and a stratified permutation null offline."""
    rows = [_row(record) for record in records]
    rows = [r for r in rows if r is not None]
    result = SuiteResult(name=NAME, caveats=[INTERPRETATION])
    if len(rows) < 3:
        result.status = "skipped"
        result.detail = {"reason": "fewer than three usable distribution records",
                         "records": len(records), "vehicles": len(rows)}
        return result

    s = np.asarray([r["demerit"] for r in rows], dtype=float)
    km = np.asarray([r["km"] for r in rows], dtype=float)
    age = np.asarray([r["age"] for r in rows], dtype=float)
    quality = np.asarray([r["capture_quality"] for r in rows], dtype=float)
    rho_km = _spearman(s, km)
    rho_age = _spearman(s, age)
    rho_partial = _partial_spearman(s, km, age, quality)

    draws = int(plan_obj.params.get("null_draws", 2000))
    null = _permutation_null(rows, s, age, quality, draws=draws, seed=plan_obj.seed)
    p_negative = ((1 + sum(v <= rho_partial for v in null)) / (len(null) + 1)
                  if null and rho_partial is not None else None)
    null_ci = (_percentiles(null, 0.025, 0.975) if null else None)
    result.metrics = [
        Metric(name="rho_km", value=_round(rho_km), n=len(rows),
               headline=True, note="Spearman(condition demerit, kilometres)"),
        Metric(name="rho_age", value=_round(rho_age), n=len(rows),
               note="Spearman(condition demerit, age)"),
        Metric(name="rho_partial", value=_round(rho_partial), n=len(rows),
               headline=True,
               note="Spearman rank residual correlation controlling for age and "
                    "capture_quality"),
        Metric(name="null_mean", value=_round(_mean(null)), ci=null_ci,
               n=len(null), basis=DIAGNOSTIC,
               note="km permuted within (market, make) strata"),
        Metric(name="negative_tail_p", value=_round(p_negative), n=len(null),
               basis=MEASURED,
               note="one-sided stratified permutation probability of a correlation "
                    "at least this negative"),
    ]
    result.strata = {"market": _by_market(rows)}
    result.detail = {
        "vehicles": len(rows), "null_draws": len(null),
        "null_strata": "(market, make)",
        "shared_with": "distribution",
    }
    return result


INTERPRETATION = (
    "The corpus is reconditioned OEM stock (DATASET_CARD limitation 2), so the "
    "true km -> visible-wear relation is attenuated by construction. This is a "
    "direction check and a regression guard, not a validation. A significantly "
    "NEGATIVE rho is a bug. A near-zero rho is expected and uninformative. A "
    "strongly positive rho is suspicious and should be checked for the photography "
    "confound first. Rank twin_fp.by_severity_band above this result.")


def _row(record: dict) -> dict | None:
    appraisal = record.get("appraisal") or {}
    evidence = appraisal.get("evidence") or {}
    condition = evidence.get("condition") or {}
    values = {
        "demerit": _finite(condition.get("demerit")),
        "km": _finite(record.get("km")),
        "year": _finite(record.get("year")),
        "capture_quality": _finite(record.get("capture_quality")),
    }
    if any(value is None for value in values.values()):
        return None
    return {
        **values,
        "age": -values["year"],  # ranks identically to age without a clock dependency
        "market": str(record.get("market") or "unknown"),
        "make": str(record.get("make") or "unknown"),
    }


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    ac, bc = a - a.mean(), b - b.mean()
    denom = math.sqrt(float(ac @ ac) * float(bc @ bc))
    return float(ac @ bc / denom) if denom > 0 else None


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    return _pearson(_rank(a), _rank(b))


def _residual(target: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    design = np.column_stack([np.ones(len(target)), *[_rank(c) for c in controls]])
    fitted = design @ np.linalg.lstsq(design, _rank(target), rcond=None)[0]
    return _rank(target) - fitted


def _partial_spearman(s, km, age, quality) -> float | None:
    return _pearson(_residual(s, [age, quality]),
                    _residual(km, [age, quality]))


def _permutation_null(rows, s, age, quality, *, draws: int, seed: int) -> list[float]:
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["market"], row["make"]), []).append(i)
    original = [row["km"] for row in rows]
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        permuted = list(original)
        for indices in groups.values():
            values = [permuted[i] for i in indices]
            rng.shuffle(values)
            for i, value in zip(indices, values):
                permuted[i] = value
        value = _partial_spearman(
            s, np.asarray(permuted, dtype=float), age, quality)
        if value is not None and math.isfinite(value):
            out.append(value)
    return out


def _by_market(rows):
    out = {}
    for market in sorted({r["market"] for r in rows}):
        bucket = [r for r in rows if r["market"] == market]
        if len(bucket) < 2:
            continue
        out[market] = {
            "n": len(bucket),
            "rho_km": _round(_spearman(
                np.asarray([r["demerit"] for r in bucket]),
                np.asarray([r["km"] for r in bucket]))),
        }
    return out


def _percentiles(values, low, high):
    ordered = sorted(values)
    return [_round(ordered[int((len(ordered) - 1) * low)]),
            _round(ordered[int((len(ordered) - 1) * high)])]


def _mean(values):
    return sum(values) / len(values) if values else None


def _round(value):
    return None if value is None else round(float(value), 4)
