"""Suite 4 - the grade histogram over the corpus.

Runs the full pipeline over a stratified vehicle sample and reports the grade
distribution with Wilson intervals, split by market, by `quality_bucket`
(dealer / mixed / phone, using `gallery.py`'s own definition so the buckets mean
on the scorecard what they mean on screen) and by variant.

The prior, written as a falsifiable acceptance band:

    on the TR OEM subset:   P(good or excellent) >= 0.70   and   P(poor) <= 0.10

That is an ASSUMPTION derived from DATASET_CARD limitation 2 - the Turkish
corpus is reconditioned dealer stock offered for sale - and the scorecard must
label it that way. It is the strongest cheap prior available, and the panel
reference set is what converts it into a measurement. The expectation is that
the current system fails it badly; that failing number, before and after, is
the single most persuasive line the harness produces.

Also free and high value:
    grade_agreement_model_vs_rollup   how often the synthesis's own grade equals
                                      the deterministic one, and the SIGNED bias
                                      between them

This suite's cached findings are also what `monotonic` and `panel` score off,
so it is the one place where the budget is genuinely shared three ways.
"""
from __future__ import annotations

from app import pipeline

from .. import cases
from ..scorecard import ASSUMED, MEASURED, Gate, Metric, SuiteResult, wilson_ci
from .base import Budget, samples_for

NAME = "distribution"

TIER_VEHICLES = {"smoke": 8, "standard": 40, "full": 200}
STRATA = ("market", "quality_bucket", "km_tercile")


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True) -> Budget:
    from .base import calls_per_vehicle
    n = TIER_VEHICLES.get(tier, TIER_VEHICLES["standard"])
    per = calls_per_vehicle(samples=samples_for(tier))
    return Budget(name=NAME, tier=tier, seed=seed, n_calls=n * per, units=n,
                  params={"vehicles": n, "calls_per_vehicle": per, "strata": list(STRATA),
                          "prior_tr_good_or_excellent": 0.70, "prior_tr_poor": 0.10},
                  note="the acceptance band is an ASSUMPTION from DATASET_CARD "
                       "limitation 2, not a measurement; the panel set is what "
                       "would convert it")


def run(plan_obj, client, **kwargs) -> list[dict]:
    """Run one appraisal per sampled vehicle and retain everything scoring needs."""
    table = cases.vehicles().set_index("listing_id", drop=False)
    ids = cases.sample_vehicles(
        int(plan_obj.params.get("vehicles", plan_obj.units)),
        seed=plan_obj.seed, strata=tuple(plan_obj.params.get("strata", STRATA)))
    images = cases.load_images()
    images = images[images["variant"] == "original"]
    records = []
    for listing_id in ids:
        row = table.loc[str(listing_id)]
        group = images[images["listing_id"] == str(listing_id)].sort_values("image_index")
        photos = [cases.full_path(str(path)) for path in group["path"]]
        meta = {
            "listing_id": str(listing_id),
            "source_key": str(row.get("source_key", "")),
            "market": str(row.get("market", "")),
            "make": str(row.get("make", "")),
            "year": _number(row.get("year")),
            "km": _number(row.get("km")),
            "quality_bucket": str(row.get("quality_bucket", "unknown")),
            "capture_quality": _number(row.get("mean_capture_quality")),
        }
        if not photos or not all(path.exists() for path in photos):
            records.append({**meta, "missing": True,
                            "error": "one or more corpus photos are missing"})
            continue
        declared = {k: meta[k] for k in ("make", "year", "km") if meta[k] is not None}
        try:
            appraisal = pipeline.appraise(
                photos, declared, market="TR", backend=client).to_dict()
            records.append({**meta, "missing": False, "appraisal": appraisal})
        except Exception as exc:
            records.append({**meta, "missing": False,
                            "error": f"{type(exc).__name__}: {exc}"})
    return records


def score(records, plan_obj, cache, model_id):
    """Build the grade histogram and model-vs-rollup comparison offline."""
    result = SuiteResult(name=NAME)
    usable = [_normalise(r) for r in records if _normalise(r) is not None]
    errors = [r for r in records if r.get("error")]
    missing = [r for r in records if r.get("missing")]
    result.detail = {
        "vehicles_planned": int(plan_obj.params.get("vehicles", plan_obj.units)),
        "vehicles_scored": len(usable),
        "missing": len(missing),
        "errors": len(errors),
    }
    if errors:
        result.caveats.append(f"{len(errors)} appraisal(s) failed and were excluded.")
    if missing:
        result.caveats.append(f"{len(missing)} vehicle photo set(s) were missing on disk.")
    if not usable:
        result.status = "skipped"
        result.detail["reason"] = "no usable corpus appraisals"
        return result

    grades = ("excellent", "good", "fair", "poor", "unknown")
    for grade in grades:
        k = sum(r["grade"] == grade for r in usable)
        result.metrics.append(Metric(
            name=f"grade_{grade}", value=round(k / len(usable), 4),
            ci=wilson_ci(k, len(usable)), n=len(usable), unit="share",
            headline=grade in ("good", "poor")))

    paired = [r for r in usable if r["grade"] in GRADE_RANK
              and r["model_grade"] in GRADE_RANK]
    exact = sum(r["grade"] == r["model_grade"] for r in paired)
    bias = (_mean([GRADE_RANK[r["model_grade"]] - GRADE_RANK[r["grade"]]
                   for r in paired]) if paired else None)
    result.metrics.extend([
        Metric(name="grade_agreement_model_vs_rollup",
               value=round(exact / len(paired), 4) if paired else None,
               ci=wilson_ci(exact, len(paired)), n=len(paired), unit="share"),
        Metric(name="model_grade_bias", value=_round(bias), n=len(paired),
               unit="grade ranks",
               note="signed mean rank(model synthesis - deterministic rollup); "
                    "positive means the model is more optimistic"),
    ])

    tr_oem = [r for r in usable if r["market"] == "TR"
              and r["source_key"].startswith("tr_")]
    good_n = sum(r["grade"] in ("good", "excellent") for r in tr_oem)
    poor_n = sum(r["grade"] == "poor" for r in tr_oem)
    good_rate = good_n / len(tr_oem) if tr_oem else None
    poor_rate = poor_n / len(tr_oem) if tr_oem else None
    result.metrics.extend([
        Metric(name="tr_oem_good_or_excellent", value=_round(good_rate),
               ci=wilson_ci(good_n, len(tr_oem)), n=len(tr_oem), unit="share",
               basis=ASSUMED, headline=True,
               note="acceptance band is an assumption from DATASET_CARD limitation 2"),
        Metric(name="tr_oem_poor", value=_round(poor_rate),
               ci=wilson_ci(poor_n, len(tr_oem)), n=len(tr_oem), unit="share",
               basis=ASSUMED, headline=True,
               note="acceptance band is an assumption from DATASET_CARD limitation 2"),
    ])
    result.gates = [
        Gate(name="tr_oem_good_or_excellent", value=_round(good_rate),
             threshold=float(plan_obj.params.get("prior_tr_good_or_excellent", 0.70)),
             direction="above", basis=ASSUMED),
        Gate(name="tr_oem_poor", value=_round(poor_rate),
             threshold=float(plan_obj.params.get("prior_tr_poor", 0.10)),
             direction="below", basis=ASSUMED),
    ]
    result.strata = {
        field: _histogram(usable, field, grades)
        for field in ("market", "quality_bucket", "variant")
    }
    result.caveats.append(
        "The TR OEM acceptance band is an ASSUMPTION derived from the corpus being "
        "reconditioned dealer stock, not condition ground truth.")
    result.detail["records"] = usable
    return result


GRADE_RANK = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}


def _number(value):
    try:
        value = float(value)
        return value if value == value else None
    except (TypeError, ValueError):
        return None


def _normalise(record: dict) -> dict | None:
    appraisal = record.get("appraisal") or {}
    evidence = appraisal.get("evidence") or {}
    condition = evidence.get("condition") or {}
    if not evidence or not condition:
        return None
    return {
        **{k: record.get(k) for k in (
            "listing_id", "source_key", "market", "make", "year", "km",
            "quality_bucket", "capture_quality")},
        "variant": record.get("variant", "original"),
        "grade": evidence.get("condition_grade") or condition.get("grade", "unknown"),
        "model_grade": evidence.get("condition_grade_model", ""),
        "demerit": _number(condition.get("demerit")),
    }


def _histogram(rows: list[dict], field: str, grades) -> dict:
    out = {}
    for label in sorted({str(r.get(field) or "unknown") for r in rows}):
        bucket = [r for r in rows if str(r.get(field) or "unknown") == label]
        out[label] = {"n": len(bucket)}
        for grade in grades:
            k = sum(r["grade"] == grade for r in bucket)
            out[label][grade] = round(k / len(bucket), 4)
            out[label][f"{grade}_ci"] = wilson_ci(k, len(bucket))
    return out


def _mean(values):
    return sum(values) / len(values) if values else None


def _round(value):
    return None if value is None else round(float(value), 4)
