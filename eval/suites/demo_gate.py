"""Suite 6 - the nine rehearsed demo cases as a regression gate.

`app/demo.py:run_demo` prints and returns 0/1 rather than returning results, so
the first move here is twenty additive lines in that file: extract
`run_case(case, backend) -> dict`, and have `run_demo` and this suite both call
it. No behaviour change, and it is what stops the demo script and the regression
gate from drifting apart.

Existing status expectations stay. What this adds is condition-side assertions
that would have caught the whole bug class:

    tr_clean        grade in {good, excellent} AND multiplier >= 0.97
                    - a clean dealer set must not be punished
    tr_phone        |grade_rank - grade_rank(tr_clean)| <= 1 AND
                    multiplier >= multiplier(tr_clean) - 0.02
    closeups_only   C < 0.60, no premium, pricing still declined
    odometer_lie    unchanged: conflict surfaced, band widened
    unseen_brand    band widened, still prices

`tr_phone` is a DIFFERENT vehicle's twins, so it is unpaired and therefore a
weak check. Add a ninth case, `tr_clean_degraded`: the degraded twins of the
same listing that backs `tr_clean`. One `_copy` in `build_fixtures`, and it puts
the headline defence - same truck, worse photos, same grade - on stage in front
of judges instead of only in a scorecard.
"""
from __future__ import annotations

import math

from eval.scorecard import Gate, Metric, SuiteResult

from .base import Budget, samples_for

NAME = "demo_gate"

N_CASES = 9


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True) -> Budget:
    from .base import calls_per_vehicle
    # Two cases refuse at the gate and never reach a vision call; the rest are
    # small fixture sets rather than full 16-photo appraisals.
    priced, photos_each = N_CASES - 2, 8
    per = calls_per_vehicle(photos_each, samples=samples_for(tier))
    return Budget(name=NAME, tier=tier, seed=seed, n_calls=priced * per,
                  units=N_CASES, unit="case",
                  params={"cases": N_CASES, "reaching_vision": priced,
                          "photos_each": photos_each, "calls_per_case": per},
                  note="cheapest suite - run it FIRST so a broken build fails "
                       "before the expensive suites spend anything")


def run(plan_obj, client, **kwargs) -> list[dict]:
    from app.demo import resolved_cases, run_case

    records = []
    for case in resolved_cases():
        try:
            record = dict(run_case(case, backend=client))
            # Useful at the terminal, but redundant with the structured appraisal
            # and needlessly large in the replay index.
            record.pop("report_text", None)
            records.append(record)
        except Exception as exc:
            records.append({
                "case_id": case["id"],
                "title": case["title"],
                "folder": case["folder"],
                "expected": case["expect"],
                "missing": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return records


def score(records, plan_obj, cache, model_id):
    rows = {r.get("case_id"): r for r in records}
    missing = [r for r in records if r.get("missing")]
    errors = [r for r in records if r.get("error")]
    usable = [r for r in records if not r.get("missing") and not r.get("error")
              and r.get("appraisal")]
    result = SuiteResult(name=NAME)
    result.detail = {
        "cases": {r.get("case_id", "?"): {
            "status": r.get("status"),
            "expected": r.get("expected"),
            "missing": bool(r.get("missing")),
            "error": r.get("error", ""),
        } for r in records}
    }
    if missing:
        names = ", ".join(r.get("case_id", "?") for r in missing)
        result.caveats.append(
            f"Skipped missing demo fixture folder(s): {names}. "
            "Run `python -m app.demo --build` to create them.")
    if errors:
        result.caveats.append(
            "Case errors: " + "; ".join(
                f"{r.get('case_id', '?')}: {r['error']}" for r in errors))
    if not usable:
        result.status = "skipped"
        return result

    checks: dict[str, bool | None] = {}
    expected = []
    for row in usable:
        passed = row.get("status") in set(row["expected"].split("|"))
        expected.append(passed)
        result.detail["cases"][row["case_id"]]["status_expected"] = passed
    checks["status_expectations"] = all(expected)

    clean = _appraisal(rows.get("tr_clean"))
    clean_grade = _grade(clean)
    clean_multiplier = _multiplier(clean)
    checks["tr_clean"] = (
        clean_grade in {"good", "excellent"}
        and clean_multiplier is not None and clean_multiplier >= 0.97
    ) if clean else None

    phone = _appraisal(rows.get("tr_phone"))
    phone_grade = _grade(phone)
    phone_multiplier = _multiplier(phone)
    checks["tr_phone"] = (
        clean_grade in GRADE_RANK and phone_grade in GRADE_RANK
        and abs(GRADE_RANK[phone_grade] - GRADE_RANK[clean_grade]) <= 1
        and clean_multiplier is not None and phone_multiplier is not None
        and phone_multiplier >= clean_multiplier - 0.02
    ) if phone and clean else None

    closeups = _appraisal(rows.get("closeups_only"))
    closeup_coverage = _coverage(closeups)
    closeup_multiplier = _multiplier(closeups, default=1.0)
    checks["closeups_only"] = (
        closeup_coverage is not None and closeup_coverage < 0.60
        and closeup_multiplier <= 1.0 and _pricing_declined(closeups)
    ) if closeups else None

    odometer = _appraisal(rows.get("odometer_lie"))
    checks["odometer_lie"] = (
        _has_correction(odometer, "odometer_conflict")
        and _band_widened(odometer)
    ) if odometer else None

    unseen = _appraisal(rows.get("unseen_brand"))
    checks["unseen_brand"] = (
        _prices(unseen) and _band_widened(unseen)
    ) if unseen else None

    for name, passed in checks.items():
        result.metrics.append(Metric(
            name=name, value=None if passed is None else int(passed),
            n=0 if passed is None else 1, unit="pass",
            headline=name == "status_expectations"))
        result.gates.append(Gate(
            name=name, value=None if passed is None else float(passed),
            threshold=1.0, direction="above"))

    failed = [name for name, passed in checks.items() if passed is False]
    if failed or errors:
        result.status = "failed"
    result.detail["checks"] = {
        **checks,
        "tr_clean_grade": clean_grade,
        "tr_clean_multiplier": clean_multiplier,
        "tr_phone_grade": phone_grade,
        "tr_phone_multiplier": phone_multiplier,
        "closeups_coverage": closeup_coverage,
        "closeups_multiplier": closeup_multiplier,
    }
    return result


GRADE_RANK = {"unknown": 0, "poor": 1, "fair": 2, "good": 3, "excellent": 4}


def _appraisal(row: dict | None) -> dict | None:
    if not row or row.get("missing") or row.get("error"):
        return None
    return row.get("appraisal")


def _grade(appraisal: dict | None) -> str:
    return ((appraisal or {}).get("evidence") or {}).get("condition_grade", "unknown")


def _multiplier(appraisal: dict | None, default=None):
    price = (appraisal or {}).get("price") or {}
    adjustment = price.get("adjustment") or {}
    return adjustment.get("multiplier", default)


def _coverage(appraisal: dict | None):
    evidence = (appraisal or {}).get("evidence") or {}
    condition = evidence.get("condition") or {}
    return condition.get("coverage")


def _pricing_declined(appraisal: dict | None) -> bool:
    price = (appraisal or {}).get("price")
    return not price or not price.get("ok", False)


def _prices(appraisal: dict | None) -> bool:
    price = (appraisal or {}).get("price") or {}
    return bool(price.get("ok") and price.get("point", 0) > 0)


def _has_correction(appraisal: dict | None, kind: str) -> bool:
    reconcile = (appraisal or {}).get("reconcile") or {}
    return any(c.get("kind") == kind for c in reconcile.get("corrections") or [])


def _band_widened(appraisal: dict | None) -> bool:
    price = (appraisal or {}).get("price") or {}
    if any("widen" in reason.lower() for reason in price.get("widened") or []):
        return True
    low, high = price.get("low", 0), price.get("high", 0)
    base_low, base_high = price.get("baseline_low", 0), price.get("baseline_high", 0)
    if min(low, high, base_low, base_high) <= 0:
        return False
    return math.log(high / low) > math.log(base_high / base_low) + 1e-9
