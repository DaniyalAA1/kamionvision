"""Suite 2 - test-retest variance.

No backend in `app/vlm/` exposes a temperature or a seed - verified in
`openai_backend.py` and `anthropic_backend.py`, both of which send neither - so
repeated calls genuinely vary at provider defaults. That variance is not a
nuisance to be averaged away here: it is the measurement.

The payoff is concrete and is why this suite pays for itself. `multiplier_sd_log`
is the spread of log(condition multiplier) across repeats of the SAME vehicle,
and it is written straight into `models/price_model.json` as the MEASURED
condition widening. Today that widening is a stated assumption.

Photo tier (24 photos x N=5 repeats):
    findings_count_cv        sd/mean of finding count per photo
    severity_disagreement    P(two repeats assign different severity to a
                             matched finding), over all C(5,2)=10 repeat pairs
    presence_instability     mean over findings of (1 - times_seen/N) - how
                             often does a finding simply vanish
    krippendorff_alpha       ordinal alpha over severity on matched findings

Vehicle tier (4 vehicles x N=3 full appraisals):
    grade_instability        P(two repeats give a different DETERMINISTIC grade)
    model_grade_instab       the same for the synthesis's own grade. The gap
                             between these two is the evidence for preferring
                             the deterministic ladder.
    multiplier_sd_log        sd of log(condition multiplier)  <- the shipped number

Implementation notes for whoever fills this in:
  * Repeats are `repeat_index` in the cache key. They are ALREADY separate
    entries, so a retest run over photos the twin suite has touched is partly
    free - but only for repeat indices below `config.CLOSEUP_SAMPLES`.
  * Match findings across repeats with `twin_fp.cluster_observations`; do not
    write a third matcher.
  * The vehicle tier must drive the real `pipeline.appraise` through a
    `CachedBackend`, because the whole point is the variance of the shipped
    path, not of pass B alone.
"""
from __future__ import annotations

import itertools
import math
import statistics

from app import config, pipeline
from app.evidence import passes, prompts

from .. import cases
from ..cache import CachedBackend, ResponseCache
from ..scorecard import MEASURED, Metric, SuiteResult, bootstrap_ci, wilson_ci
from . import twin_fp
from .base import Budget, samples_for

NAME = "retest"

# (photos, photo repeats, vehicles, vehicle repeats)
TIER_PLAN = {"smoke": (8, 3, 1, 2), "standard": (24, 5, 4, 3), "full": (48, 5, 8, 3)}


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True) -> Budget:
    from .base import calls_per_vehicle
    photos, photo_n, vehicles, vehicle_n = TIER_PLAN.get(tier, TIER_PLAN["standard"])
    photo_calls = photos * photo_n
    vehicle_calls = vehicles * vehicle_n * calls_per_vehicle(samples=samples_for(tier))
    return Budget(name=NAME, tier=tier, seed=seed,
                  n_calls=photo_calls + vehicle_calls,
                  units=photos + vehicles, unit="photo+vehicle",
                  params={"photos": photos, "photo_repeats": photo_n,
                          "vehicles": vehicles, "vehicle_repeats": vehicle_n,
                          "photo_calls": photo_calls, "vehicle_calls": vehicle_calls,
                          "calls_per_vehicle": calls_per_vehicle(samples=samples_for(tier)),
                          "effort": effort or config.CLOSEUP_EFFORT,
                          "model_id": model_id},
                  note="repeats are distinct repeat_index keys, so indices below "
                       f"CLOSEUP_SAMPLES={config.CLOSEUP_SAMPLES} may already be cached")


def run(plan_obj, client, **kwargs) -> list[dict]:
    """Collect pass-B photo repeats and repeated full shipped appraisals."""
    pairs = cases.sample_twin_pairs(
        max(int(plan_obj.params["vehicles"]), int(plan_obj.params["photos"])),
        1, seed=plan_obj.seed,
        require_files=kwargs.get("require_files", True))[:int(plan_obj.params["photos"])]
    records = _run_photos(
        pairs, int(plan_obj.params["photo_repeats"]), client, plan_obj)
    records.extend(_run_vehicles(
        int(plan_obj.params["vehicles"]),
        int(plan_obj.params["vehicle_repeats"]), client, plan_obj))
    return records


def score(records, plan_obj, cache, model_id):
    """Score cached photo responses and serialized appraisals without a client."""
    photo_units = _photo_units(
        [r for r in records if r.get("tier") == "photo"], cache, model_id,
        int(plan_obj.params["photo_repeats"]))
    vehicle_units = _vehicle_units(
        [r for r in records if r.get("tier") == "vehicle"])
    result = SuiteResult(name=NAME)
    if not photo_units and not vehicle_units:
        result.status = "skipped"
        result.detail = {"reason": "no usable photo or vehicle repeats",
                         "records": len(records)}
        return result

    count_cvs, presence, alpha_units = [], [], []
    severity_disagree = severity_pairs = 0
    for unit in photo_units:
        counts = [len(unit["by_repeat"].get(i, []))
                  for i in range(unit["planned_repeats"])]
        mean = _mean(counts)
        if mean:
            count_cvs.append(statistics.stdev(counts) / mean
                             if len(counts) > 1 else 0.0)
        observations = [obs for values in unit["by_repeat"].values() for obs in values]
        for cluster in twin_fp.cluster_observations(observations):
            by_repeat = _cluster_ranks(cluster)
            presence.append(1.0 - len(by_repeat) / unit["planned_repeats"])
            alpha_units.append(by_repeat)
            for a, b in itertools.combinations(sorted(by_repeat), 2):
                severity_pairs += 1
                severity_disagree += by_repeat[a] != by_repeat[b]

    grade_diff = grade_pairs = model_diff = model_pairs = 0
    multiplier_sds = []
    for unit in vehicle_units:
        for a, b in itertools.combinations(unit, 2):
            if a["grade"] and b["grade"]:
                grade_pairs += 1
                grade_diff += a["grade"] != b["grade"]
            if a["model_grade"] and b["model_grade"]:
                model_pairs += 1
                model_diff += a["model_grade"] != b["model_grade"]
        logs = [math.log(r["multiplier"]) for r in unit if r["multiplier"] > 0]
        if len(logs) > 1:
            multiplier_sds.append(statistics.stdev(logs))

    def add(name, value, *, n=0, ci=None, unit="", headline=False, note=""):
        result.metrics.append(Metric(
            name=name, value=_round(value), n=n, ci=ci, unit=unit,
            headline=headline, note=note, basis=MEASURED))

    add("findings_count_cv", _mean(count_cvs), n=len(count_cvs), headline=True,
        note="mean per-photo sample SD / mean finding count")
    severity_rate = severity_disagree / severity_pairs if severity_pairs else None
    add("severity_disagreement", severity_rate, n=severity_pairs,
        ci=wilson_ci(severity_disagree, severity_pairs), headline=True)
    add("presence_instability", _mean(presence), n=len(presence), headline=True,
        note="mean over matched finding clusters of 1 - repeats_seen/N")
    add("krippendorff_alpha", _ordinal_alpha(alpha_units), n=len(alpha_units),
        headline=True, note="ordinal alpha over severity; absent ratings omitted")
    grade_rate = grade_diff / grade_pairs if grade_pairs else None
    model_rate = model_diff / model_pairs if model_pairs else None
    add("grade_instability", grade_rate, n=grade_pairs,
        ci=wilson_ci(grade_diff, grade_pairs), headline=True)
    add("model_grade_instab", model_rate, n=model_pairs,
        ci=wilson_ci(model_diff, model_pairs))
    add("multiplier_sd_log", _mean(multiplier_sds), n=len(multiplier_sds),
        ci=bootstrap_ci(multiplier_sds, _mean, seed=plan_obj.seed),
        unit="log multiplier", headline=True,
        note="mean within-vehicle sample SD of log(condition multiplier)")
    result.detail = {
        "photos_scored": len(photo_units), "vehicles_scored": len(vehicle_units),
        "matching": "twin_fp.cluster_observations",
    }
    return result


def _run_photos(pairs, repeats, client, plan_obj):
    rows = []
    schema = prompts.closeup_schema() if client.supports_structured_output else None
    effort = plan_obj.params.get("effort", config.CLOSEUP_EFFORT)
    for pair in pairs:
        prompt, image = twin_fp.build_prompt(pair), pair.path("original")
        for repeat in range(repeats):
            row = {"tier": "photo", "unit": pair.unit,
                   "listing_id": pair.listing_id, "image_index": pair.image_index,
                   "view": pair.view, "repeat": repeat, "ok": False, "error": ""}
            args = dict(system=prompts.SYSTEM, json_schema=schema, effort=effort,
                        repeat_index=repeat)
            try:
                row["key"] = client.key_for(prompt, [image], **args)
                client.complete(prompt, [image], max_tokens=config.CLOSEUP_MAX_TOKENS,
                                **args)
                row["ok"] = True
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


class _AppraisalRepeatBackend(CachedBackend):
    """Put each full appraisal in a distinct cache repeat namespace."""

    def __init__(self, source: CachedBackend, offset: int):
        super().__init__(source.inner, cache=source.cache, model_id=source.model_id,
                         read_only=source.read_only, record=source.record)
        self.offset = offset

    def key_for(self, prompt, images, *, system="", json_schema=None, effort=None,
                repeat_index=0):
        return super().key_for(
            prompt, images, system=system, json_schema=json_schema, effort=effort,
            repeat_index=self.offset + int(repeat_index))


def _as_cached(client) -> CachedBackend:
    if isinstance(client, CachedBackend):
        return client
    return CachedBackend(client, cache=ResponseCache(),
                         model_id=getattr(client, "model", "unknown"))


def _run_vehicles(count, repeats, client, plan_obj):
    cached = _as_cached(client)
    table = cases.vehicles().set_index("listing_id", drop=False)
    images = cases.load_images()
    images = images[images["variant"] == "original"]
    rows = []
    for listing_id in cases.sample_vehicles(count, seed=plan_obj.seed):
        meta = table.loc[str(listing_id)]
        group = images[images["listing_id"] == str(listing_id)].sort_values("image_index")
        photos = [cases.full_path(str(path)) for path in group["path"]]
        declared = {"make": str(meta.get("make", "")),
                    "year": _number(meta.get("year")), "km": _number(meta.get("km"))}
        for repeat in range(repeats):
            row = {"tier": "vehicle", "unit": str(listing_id),
                   "listing_id": str(listing_id), "repeat": repeat,
                   "ok": False, "error": ""}
            if not photos or not all(path.exists() for path in photos):
                row["error"] = "one or more corpus photos are missing"
                rows.append(row)
                continue
            backend = _AppraisalRepeatBackend(cached, (repeat + 1) * 10_000)
            try:
                appraisal = pipeline.appraise(
                    photos, declared, market="TR", backend=backend).to_dict()
                row.update(ok=True, appraisal=appraisal)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def _photo_units(rows, cache, model_id, planned_repeats):
    grouped = {}
    for row in rows:
        entry = cache.get(model_id, row["key"]) \
            if cache and row.get("ok") and row.get("key") else None
        if entry is None:
            continue
        pair = cases.TwinPair(
            listing_id=str(row["listing_id"]), image_index=int(row["image_index"]),
            view=str(row.get("view") or "unknown"),
            market=str(row.get("market") or "unknown"), original_rel="unused")
        try:
            finding = passes.parse_closeup(
                entry.get("text", ""), twin_fp.photo_check(pair, "original"),
                cropped=False)
        except Exception:
            continue
        sample = int(row["repeat"])
        observations = [
            twin_fp.Obs(sample=sample, component=i.component,
                        family=twin_fp.family_of(i.component),
                        observation=i.observation, severity=i.severity,
                        impact=i.price_impact, confidence=float(i.confidence or 0.0))
            for i in finding.issues]
        unit = grouped.setdefault(
            row["unit"], {"by_repeat": {}, "planned_repeats": planned_repeats})
        unit["by_repeat"][sample] = observations
    return list(grouped.values())


def _cluster_ranks(cluster):
    by_repeat = {}
    for member in cluster.members:
        by_repeat.setdefault(member.sample, []).append(
            twin_fp.SEVERITY_RANK.get(member.severity, 0))
    return {sample: sorted(values)[(len(values) - 1) // 2]
            for sample, values in by_repeat.items()}


def _ordinal_alpha(units):
    usable = [list(unit.values()) for unit in units if len(unit) >= 2]
    if not usable:
        return None
    values = [value for ratings in usable for value in ratings]
    frequencies = {rank: values.count(rank) for rank in set(values)}

    def distance(a, b):
        lo, hi = sorted((a, b))
        if lo == hi:
            return 0.0
        between = sum(frequencies.get(rank, 0) for rank in range(lo, hi + 1))
        return (between - (frequencies[lo] + frequencies[hi]) / 2.0) ** 2

    observed_num = 0.0
    observed_den = 0
    for ratings in usable:
        observed_den += len(ratings)
        observed_num += sum(
            distance(a, b) / (len(ratings) - 1)
            for i, a in enumerate(ratings) for j, b in enumerate(ratings) if i != j)
    observed = observed_num / observed_den

    n = len(values)
    if n < 2:
        return None
    expected = sum(
        count_a * count_b * distance(a, b)
        for a, count_a in frequencies.items()
        for b, count_b in frequencies.items() if a != b) / (n * (n - 1))
    return 1.0 - observed / expected if expected else 1.0


def _vehicle_units(rows):
    grouped = {}
    for row in rows:
        appraisal = row.get("appraisal") or {}
        evidence = appraisal.get("evidence") or {}
        adjustment = ((appraisal.get("price") or {}).get("adjustment") or {})
        multiplier = _number(adjustment.get("multiplier"))
        if not row.get("ok") or not evidence or multiplier is None:
            continue
        grouped.setdefault(row["unit"], []).append({
            "grade": evidence.get("condition_grade", ""),
            "model_grade": evidence.get("condition_grade_model", ""),
            "multiplier": multiplier,
        })
    return list(grouped.values())


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _mean(values):
    return sum(values) / len(values) if values else None


def _round(value):
    return None if value is None else round(float(value), 4)
