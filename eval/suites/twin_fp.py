"""Suite 1 - the degraded-twin false-positive rate. The primary metric.

A degraded twin is the SAME TRUCK. `degrade_images.py` applied motion blur,
mud, glare and JPEG crush to a copy; it did not wear the tires down. So a
finding that appears on the twin and has no counterpart on the original is a
false positive **by construction**, and no label was needed to say so. That is
the whole idea: this corpus has no condition ground truth, and the pairing is
the closest thing to one that exists in it.

Everything here is built to kill a confound, because in a paired design a
confound is the entire result:

  pairing        `(listing_id, image_index)` and nothing else.
  the view tag   the ORIGINAL's, on both halves. The zero-shot view classifier
                 moves under degradation - a mud-covered rear three-quarter
                 reads as `trailer_only` - and a different view tag is a
                 different question bank, which would make this a comparison of
                 prompts rather than of photographs.
  the soft line  `soft=False` on both. The as-shipped prompt adds a soft-focus
                 caution when the gate saw blur, which fires on twins far more
                 often than on originals. That line is a real and probably
                 useful defence, so measuring it is worth doing - but as its
                 own paid diagnostic, not silently inside the headline.
  the crop       `cropped=False` on both. `wants_crop` depends on gate
                 detections, and YOLO does not find the same boxes in a blurred
                 frame, so the shipped rule would send a crop of one member and
                 the whole frame of the other.
  the vehicle    the same one-line vehicle context on both, built from the
                 corpus record, so pass A is not spent and cannot differ.

The match threshold is `TWIN_MATCH`, set deliberately BELOW
`passes.SAME_DEFECT`: a more generous matcher pairs more twin findings with
originals and therefore reports FEWER false positives. The instrument is biased
against its own hypothesis, and the number it prints is a lower bound. Say
which way an instrument points; do not just say it is calibrated.

The honest counter-argument, which is in the artifact and not only here: a
degraded twin carries strictly less information than its original, so a
twin-only finding COULD be a genuine defect the clean call happened to miss
rather than a fabrication. Three things answer that. The strict definition
requires the finding in a quorum of twin samples and in ZERO original samples.
`original_only_rate` - the mirror statistic, which is expected to be positive -
is reported beside it. And `by_transform` separates the hardest case of all: mud
and rain put real pixels on the truck, so "mud on the lower panel" is correct
perception and a false CONDITION finding at the same time.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from app import config
from app.evidence import passes, prompts
from app.condition import SEVERITY_RANK
from app.schema import PhotoCheck
from app.vlm.base import VLMError

from .. import cases
from ..cache import PlannedCall
from ..scorecard import (ASSUMED, DIAGNOSTIC, MEASURED, Gate, Metric, SuiteResult,
                         bootstrap_ci, wilson_ci)
from .base import samples_for

NAME = "twin_fp"

# (vehicles, photos per vehicle). Pairs = vehicles x photos; calls = pairs x 2
# halves x CLOSEUP_SAMPLES.
TIER_PLAN = {"smoke": (15, 4), "standard": (40, 6), "full": (60, 10)}

# Below passes.SAME_DEFECT on purpose - see the module docstring. Module-level
# so `eval/replay.py --sweep` can move it for free against the cache.
TWIN_MATCH = 0.30

IMPACT_RANK = {name: i for i, name in enumerate(prompts.IMPACTS)}

# The design's pre-registered gate. An assumption, labelled as one: nobody has
# measured what this number is yet, which is exactly what the baseline run is
# for.
SEVERITY_INFLATION_GATE = 0.15

# Components only a truck has. Lifted from app/vlm/bench.py so its one
# defensible signal survives its deletion - as a diagnostic, never a selection
# criterion. See scorecard.BENCH_NOTE.
TRUCK_SPECIFIC = {"steer_tires", "drive_tires", "fifth_wheel", "coupling_airlines",
                  "chassis_frame", "air_suspension", "air_tanks_lines", "adblue_tank",
                  "exhaust_dpf", "bunk_sleeper", "mudflaps_guards", "fairings_skirts",
                  "roof_deflector", "cab_steps", "undercarriage", "brakes_hubs"}

VARIANTS = ("original", "degraded")


# --- the family partition, which may not exist yet -------------------------

def family_of(component: str) -> str:
    """`app.condition.family` when it lands; the component id until then.

    The design matches findings by family rather than by component because
    inter-annotator agreement on `corrosion` versus `chassis_frame` is exactly
    the ambiguity the condition workstream removes. That module does not exist
    yet, so this degrades to identity - which is STRICTER (fewer matches, more
    twin-only findings, a higher reported FP rate). The artifact records which
    partition was in force, because the two are not comparable.
    """
    try:
        from app import condition                      # noqa: PLC0415
    except ImportError:
        return component
    fn = getattr(condition, "family", None) or getattr(condition, "family_of", None)
    if fn is None:
        return component
    try:
        return str(fn(component) or component)
    except Exception:
        return component


def family_partition_name() -> str:
    return "app.condition" if family_of("drive_tires") != "drive_tires" else "component-identity"


# --- the prompt, forced identical across the pair --------------------------

def vehicle_line(pair: cases.TwinPair) -> str:
    """The pass-A line, taken from the corpus record instead of a vision call.

    Identical on both halves by construction, which is the point; it also
    removes pass A from the budget entirely, so the suite's unit is two calls
    per sample rather than two calls plus a shared identity call whose cost
    would have to be amortised over a set this suite never assembles.
    """
    named = " ".join(x for x in (pair.make, pair.model_name) if x).strip()
    if not named:
        return "The make and model could not be read from the photos."
    year = f" (registered {pair.year})" if pair.year else ""
    return f"The vehicle has been identified from the full set as a {named}{year}."


def build_prompt(pair: cases.TwinPair) -> str:
    from app.vision import VIEW_LABELS
    pretty = {k: k.replace("_", " ") for k in VIEW_LABELS}.get(
        pair.view, pair.view.replace("_", " "))
    return prompts.closeup_prompt(view=pair.view, view_pretty=pretty,
                                  vehicle=vehicle_line(pair),
                                  cropped=False, soft=False)


def photo_check(pair: cases.TwinPair, variant: str) -> PhotoCheck:
    """A PhotoCheck good enough for `parse_closeup`, built without the gate.

    `parse_closeup` reads only `photo_id`, `view` and - for a cropped call, which
    this suite never makes - the subject box and frame size. Constructing it here
    rather than running YOLO keeps scoring pure and offline, which is what makes
    `--replay` free.
    """
    path = pair.path(variant)
    return PhotoCheck(photo_id=pair.image_index, path=str(path), filename=path.name,
                      view=pair.view)


# --- planning --------------------------------------------------------------

@dataclass
class Plan:
    name: str
    tier: str
    seed: int
    samples: int
    pairs: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def to_dict(self) -> dict:
        return {"name": self.name, "tier": self.tier, "seed": self.seed,
                "samples": self.samples, "pairs": len(self.pairs),
                "calls": len(self.calls), "params": self.params,
                "sampling": cases.describe(self.pairs)}


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True,
         n_vehicles: int | None = None, per_vehicle: int | None = None,
         resolve_keys: bool = True) -> Plan:
    """Resolve the sample and every cache key it implies, spending nothing.

    Keys are computable offline here precisely BECAUSE the prompt is forced:
    no gate, no identity pass and no crop decision stand between the file on
    disk and the bytes the provider would see. That is what makes `--estimate`
    an exact count rather than a guess.
    """
    from ..cache import cache_key, image_digest

    tier_vehicles, tier_photos = TIER_PLAN.get(tier, TIER_PLAN["standard"])
    n_vehicles = tier_vehicles if n_vehicles is None else int(n_vehicles)
    per_vehicle = tier_photos if per_vehicle is None else int(per_vehicle)
    samples = int(samples or samples_for(tier))
    effort = effort if effort is not None else config.CLOSEUP_EFFORT
    pairs = cases.sample_twin_pairs(n_vehicles, per_vehicle, seed=seed,
                                    require_files=require_files)
    # `data/images/` is gitignored, so a fresh clone samples zero pairs and the
    # budget silently reads 0. Record the collapse rather than let the estimate
    # look like a free run.
    available = len(pairs)
    if require_files:
        wanted = len(cases.sample_twin_pairs(n_vehicles, per_vehicle, seed=seed,
                                             require_files=False))
    else:
        wanted = available
    schema = prompts.closeup_schema() if structured else None

    calls: list[PlannedCall] = []
    for pair in pairs:
        prompt = build_prompt(pair)
        for variant in VARIANTS:
            path = pair.path(variant)
            try:
                # `resolve_keys=False` skips the JPEG re-encode. A caller who only
                # wants the count does not need the keys, and the encode is ~50 ms
                # a frame - 3,600 of them is half a minute of pixels for a number
                # that was already known.
                digest = image_digest(path) if resolve_keys else f"unresolved:{path}"
            except VLMError:
                # Missing pixels: plan the call so the budget is honest, and let
                # run() record the failure rather than silently shrinking n.
                digest = f"missing:{path}"
            for sample in range(samples):
                calls.append(PlannedCall(
                    key=cache_key(image_digests=[digest], prompt=prompt,
                                  system=prompts.SYSTEM, model_id=model_id,
                                  effort=effort, schema=schema, repeat_index=sample),
                    unit=pair.unit, label=f"{pair.unit}:{variant}:{sample}",
                    meta={"listing_id": pair.listing_id, "variant": variant,
                          "sample": sample, "view": pair.view,
                          "path": str(path), "missing": digest.startswith("missing:")}))

    return Plan(name=NAME, tier=tier, seed=seed, samples=samples, pairs=pairs,
                calls=calls,
                params={"n_vehicles": n_vehicles, "per_vehicle": per_vehicle,
                        "keys_resolved": resolve_keys,
                        "pairs_available": available, "pairs_wanted": wanted,
                        "pairs_missing_on_disk": wanted - available,
                        "samples": samples, "effort": effort, "structured": structured,
                        "model_id": model_id, "twin_match": TWIN_MATCH,
                        "same_defect": passes.SAME_DEFECT,
                        "family_partition": family_partition_name(),
                        "prompt_held": ["view=original", "soft=False", "cropped=False",
                                        "vehicle_line=corpus"]})


# --- running ---------------------------------------------------------------

def run(plan_obj: Plan, client, *, concurrency: int | None = None,
        gate_checks=None, on_call=None) -> list[dict]:
    """Make (or serve from cache) every planned call. Returns the unit index.

    The index is metadata only: the response TEXT lives in the cache, and
    `score()` reads it back from there. That separation is what lets a parser
    change be re-scored for nothing.
    """
    workers = max(1, min(concurrency or config.EVIDENCE_CONCURRENCY, len(plan_obj.calls) or 1))
    by_unit = {p.unit: p for p in plan_obj.pairs}
    schema = prompts.closeup_schema() if client.supports_structured_output else None

    dropped: dict[str, str] = {}
    if gate_checks is not None:
        dropped = _gate_drop(plan_obj.pairs, gate_checks)

    records: list[dict] = []

    def one(call: PlannedCall) -> dict:
        pair = by_unit[call.unit]
        row = {"unit": call.unit, "key": call.key, **call.meta,
               "image_index": pair.image_index, "ok": False, "error": ""}
        drop = dropped.get(f"{call.unit}:{call.meta['variant']}")
        if drop:
            row["error"] = f"gate_dropped: {drop}"
            row["gate_dropped"] = True
            return row
        if call.meta.get("missing"):
            row["error"] = "image missing on disk"
            return row
        prompt, image = build_prompt(pair), pair.path(call.meta["variant"])
        args = dict(system=prompts.SYSTEM, json_schema=schema,
                    effort=plan_obj.params.get("effort"),
                    repeat_index=call.meta["sample"])
        # Take the key from the CLIENT rather than the plan. They agree whenever
        # the plan was built for this model at this effort, and when they do not,
        # the record must name the entry that was actually written or `score()`
        # reads back nothing and reports a silent zero.
        row["key"] = client.key_for(prompt, [image], **args)
        row["planned_key_matched"] = row["key"] == call.key
        try:
            response = client.complete(prompt, [image],
                                       max_tokens=config.CLOSEUP_MAX_TOKENS, **args)
            row["ok"] = True
            row["elapsed_s"] = response.elapsed_s
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, c): c for c in plan_obj.calls}
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            if on_call:
                on_call(row)
    records.sort(key=lambda r: (r["unit"], r["variant"], r["sample"]))
    return records


def _gate_drop(pairs: list, gate_checks) -> dict[str, str]:
    """Which halves the real gate refuses. Reported, never counted as an FP.

    A twin so degraded the gate will not send it is the product working: the
    refusal path is a feature of this system, and charging it as a false
    positive would measure the opposite of what is intended.
    """
    paths, index = [], []
    for pair in pairs:
        for variant in VARIANTS:
            path = pair.path(variant)
            if path.exists():
                paths.append(path)
                index.append(f"{pair.unit}:{variant}")
    if not paths:
        return {}
    out: dict[str, str] = {}
    for label, check in zip(index, gate_checks(paths)):
        if not check.usable:
            out[label] = "; ".join(check.reasons) or "unusable"
    return out


# --- observations, clusters, matching --------------------------------------

@dataclass
class Obs:
    sample: int
    component: str
    family: str
    observation: str
    severity: str
    impact: str
    confidence: float


@dataclass
class Cluster:
    family: str
    members: list = field(default_factory=list)

    @property
    def support(self) -> int:
        """Distinct SAMPLES this defect was reported in. Not member count."""
        return len({m.sample for m in self.members})

    @property
    def rep(self) -> Obs:
        return self.members[0]

    def _median_rank(self, table: dict, key) -> float:
        ranks = sorted(table.get(key(m), 0) for m in self.members)
        return float(ranks[(len(ranks) - 1) // 2])   # lower middle: symmetric, deterministic

    @property
    def severity_rank(self) -> float:
        return self._median_rank(SEVERITY_RANK, lambda m: m.severity)

    @property
    def impact_rank(self) -> float:
        return self._median_rank(IMPACT_RANK, lambda m: m.impact)

    @property
    def confidence(self) -> float:
        return sum(m.confidence for m in self.members) / len(self.members)


def quorum_for(samples: int) -> int:
    """>= 2/3 of the samples, and at least one. With CLOSEUP_SAMPLES=3 this is 2."""
    return max(1, math.ceil(2 * samples / 3))


def cluster_observations(observations: list[Obs], threshold: float = None) -> list[Cluster]:
    """Fold repeated sightings of one defect across samples into one cluster.

    Same primitive the product merges with - `passes._overlap`, Jaccard over
    content words - because a second, differently-tuned matcher living in the
    eval harness would measure the harness.
    """
    threshold = TWIN_MATCH if threshold is None else threshold
    clusters: list[Cluster] = []
    for obs in observations:
        best, best_score = None, threshold
        for cluster in clusters:
            if cluster.family != obs.family:
                continue
            score = passes._overlap(cluster.rep.observation, obs.observation)
            if score >= best_score:
                best, best_score = cluster, score
        if best is None:
            clusters.append(Cluster(family=obs.family, members=[obs]))
        else:
            best.members.append(obs)
    return clusters


def greedy_match(left: list[Cluster], right: list[Cluster],
                 threshold: float = None) -> list[tuple]:
    """Maximum-weight bipartite matching, greedily, by descending overlap.

    Greedy rather than Hungarian because the sets are tiny and because the
    property that matters is one-to-one: without it a single original finding
    absorbs three twin paraphrases and the false-positive count silently drops
    by two. Box IoU is not consulted - the degradations include perspective
    skew, so boxes legitimately move, and requiring them to agree would
    manufacture false positives out of geometry.
    """
    threshold = TWIN_MATCH if threshold is None else threshold
    scored = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if a.family != b.family:
                continue
            score = passes._overlap(a.rep.observation, b.rep.observation)
            if score >= threshold:
                scored.append((score, i, j))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_l: set[int] = set()
    used_r: set[int] = set()
    out = []
    for score, i, j in scored:
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        out.append((left[i], right[j], score))
    return out


def matches_any(cluster: Cluster, observations: list[Obs],
                threshold: float = None) -> int:
    """How many distinct samples of the other half support this cluster."""
    threshold = TWIN_MATCH if threshold is None else threshold
    hits = set()
    for obs in observations:
        if obs.family != cluster.family:
            continue
        if passes._overlap(cluster.rep.observation, obs.observation) >= threshold:
            hits.add(obs.sample)
    return len(hits)


# --- scoring, which is pure and replayable ---------------------------------

def _read(records: list[dict], cache, model_id: str, pair, variant: str,
          ) -> tuple[list[Obs], dict]:
    """Parse one half of a pair back out of the cache. No network, ever."""
    observations: list[Obs] = []
    meta = {"samples": 0, "illegible": 0, "failed": 0, "gate_dropped": 0}
    check = photo_check(pair, variant)
    for row in records:
        if row["unit"] != pair.unit or row["variant"] != variant:
            continue
        if row.get("gate_dropped"):
            meta["gate_dropped"] += 1
            continue
        entry = cache.get(model_id, row["key"])
        if entry is None or not row.get("ok", False):
            meta["failed"] += 1
            continue
        try:
            finding = passes.parse_closeup(entry.get("text", ""), check, cropped=False)
        except Exception:
            meta["failed"] += 1
            continue
        meta["samples"] += 1
        meta["illegible"] += (not finding.legible)
        for issue in finding.issues:
            observations.append(Obs(
                sample=int(row["sample"]), component=issue.component,
                family=family_of(issue.component), observation=issue.observation,
                severity=issue.severity, impact=issue.price_impact,
                confidence=float(issue.confidence or 0.0)))
    return observations, meta


@dataclass
class PairResult:
    pair: object
    twin_clusters: int = 0
    twin_only: int = 0
    orig_clusters: int = 0
    orig_only: int = 0
    twin_findings: int = 0
    orig_findings: int = 0
    twin_samples: int = 0
    orig_samples: int = 0
    twin_illegible: int = 0
    orig_illegible: int = 0
    failed: int = 0
    gate_dropped: int = 0
    truck_specific: int = 0
    total_components: int = 0
    severity_deltas: list = field(default_factory=list)
    impact_deltas: list = field(default_factory=list)
    confidence_deltas: list = field(default_factory=list)
    twin_only_examples: list = field(default_factory=list)

    @property
    def scored(self) -> bool:
        """Both halves produced at least one usable sample."""
        return self.twin_samples > 0 and self.orig_samples > 0


def score_pairs(records: list[dict], pairs: list, cache, model_id: str,
                samples: int) -> list[PairResult]:
    quorum = quorum_for(samples)
    results = []
    for pair in pairs:
        twin_obs, twin_meta = _read(records, cache, model_id, pair, "degraded")
        orig_obs, orig_meta = _read(records, cache, model_id, pair, "original")
        res = PairResult(pair=pair,
                         twin_findings=len(twin_obs), orig_findings=len(orig_obs),
                         twin_samples=twin_meta["samples"], orig_samples=orig_meta["samples"],
                         twin_illegible=twin_meta["illegible"],
                         orig_illegible=orig_meta["illegible"],
                         failed=twin_meta["failed"] + orig_meta["failed"],
                         gate_dropped=twin_meta["gate_dropped"] + orig_meta["gate_dropped"])
        for obs in twin_obs + orig_obs:
            res.total_components += 1
            res.truck_specific += obs.component in TRUCK_SPECIFIC
        if not res.scored:
            results.append(res)
            continue

        twin_clusters = [c for c in cluster_observations(twin_obs) if c.support >= quorum]
        orig_clusters = [c for c in cluster_observations(orig_obs) if c.support >= quorum]
        res.twin_clusters = len(twin_clusters)
        res.orig_clusters = len(orig_clusters)

        # The strict rule: a quorum on the twin and ZERO supporting samples on
        # the original. Counted against every original sample, not against the
        # original's surviving clusters, so a finding the original mentioned
        # once is not a false positive.
        for cluster in twin_clusters:
            if matches_any(cluster, orig_obs) == 0:
                res.twin_only += 1
                if len(res.twin_only_examples) < 3:
                    res.twin_only_examples.append(
                        {"unit": pair.unit, "component": cluster.rep.component,
                         "severity": cluster.rep.severity,
                         "observation": cluster.rep.observation[:180],
                         "degradations": pair.degradations})
        for cluster in orig_clusters:
            if matches_any(cluster, twin_obs) == 0:
                res.orig_only += 1

        # One-to-one, so one original finding cannot absorb two twin paraphrases.
        for twin_c, orig_c, _ in greedy_match(twin_clusters, orig_clusters):
            res.severity_deltas.append(twin_c.severity_rank - orig_c.severity_rank)
            res.impact_deltas.append(twin_c.impact_rank - orig_c.impact_rank)
            res.confidence_deltas.append(twin_c.confidence - orig_c.confidence)
        results.append(res)
    return results


# --- aggregation -----------------------------------------------------------

def _by_vehicle(results: list[PairResult]) -> dict[str, list[PairResult]]:
    out: dict[str, list[PairResult]] = {}
    for r in results:
        out.setdefault(r.pair.listing_id, []).append(r)
    return out


def _macro(units: list[list[PairResult]], numer, denom) -> float | None:
    """Average a rate over VEHICLES, not over findings.

    A vehicle whose twin produced eleven findings would otherwise outvote three
    vehicles that produced one each, and eleven findings off one truck are not
    eleven independent observations.
    """
    values = []
    for unit in units:
        d = sum(denom(r) for r in unit)
        if d:
            values.append(sum(numer(r) for r in unit) / d)
    return sum(values) / len(values) if values else None


def _pooled_mean(units: list[list[PairResult]], pull) -> float | None:
    """Mean of a paired delta, weighting each VEHICLE equally."""
    per_vehicle = []
    for unit in units:
        values = [v for r in unit for v in pull(r)]
        if values:
            per_vehicle.append(sum(values) / len(values))
    return sum(per_vehicle) / len(per_vehicle) if per_vehicle else None


def _strata(results: list[PairResult], key) -> dict:
    buckets: dict[str, list[PairResult]] = {}
    for r in results:
        for label in key(r.pair):
            buckets.setdefault(str(label), []).append(r)
    out = {}
    for label, rows in sorted(buckets.items()):
        units = list(_by_vehicle(rows).values())
        deltas = [v for r in rows for v in r.severity_deltas]
        out[label] = {
            "pairs": len(rows),
            "twin_fp_rate": _round(_macro(units, lambda r: r.twin_only,
                                          lambda r: r.twin_clusters)),
            "twin_only_per_photo": _round(_macro(units, lambda r: r.twin_only,
                                                 lambda r: 1 if r.scored else 0)),
            "orig_only_rate": _round(_macro(units, lambda r: r.orig_only,
                                            lambda r: r.orig_clusters)),
            "severity_inflation": _round(sum(deltas) / len(deltas) if deltas else None),
            "matched": len(deltas),
        }
    return out


def _round(value, places: int = 4):
    return None if value is None else round(float(value), places)


CAVEATS = [
    "A degraded twin carries strictly LESS information than its original, so a "
    "twin-only finding could in principle be a genuine defect the clean call "
    "missed rather than a fabrication. That is why the definition is strict "
    "(a quorum of twin samples AND zero supporting original samples) and why "
    "`orig_only_rate` - which is expected to be positive - is reported beside "
    "it rather than buried.",
    "Mud, rain and glare put REAL pixels on the truck. A model reporting 'mud "
    "on the lower panel' is perceiving correctly and producing a false "
    "CONDITION finding at the same time. `by_transform` is what separates "
    "those two readings, and it is free because the transform list is already "
    "in the manifest.",
    f"TWIN_MATCH={TWIN_MATCH} is deliberately below passes.SAME_DEFECT="
    f"{passes.SAME_DEFECT}. A more generous matcher pairs more twin findings "
    f"with originals and therefore reports FEWER false positives, so every "
    f"rate here is a LOWER BOUND.",
    "Prompts are forced identical across the pair: the original's view tag on "
    "both halves, soft=False on both, cropped=False on both, and one vehicle "
    "line from the corpus record. The shipped prompt differs between the "
    "halves on all three, so this is the confound-free measurement rather than "
    "the as-shipped one; the difference between them is itself worth a paid "
    "diagnostic run.",
    "The dose-response curve in `by_severity_band` is the strongest "
    "monotonicity evidence this harness produces, because the ORDERING OF THE "
    "STIMULUS is known by construction. Rank it above the km correlation in "
    "the monotonic suite, which is attenuated by reconditioned OEM stock.",
]


def score(records: list[dict], plan_obj: Plan, cache, model_id: str) -> SuiteResult:
    results = score_pairs(records, plan_obj.pairs, cache, model_id, plan_obj.samples)
    scored = [r for r in results if r.scored]
    units = list(_by_vehicle(scored).values())
    n_vehicles = len(units)

    result = SuiteResult(name=NAME, caveats=list(CAVEATS))
    if plan_obj.samples < 2:
        result.caveats.insert(0, (
            f"This run read each half ONCE (samples={plan_obj.samples}), so the "
            f">=2/3 quorum degenerates to >=1 and a single-sample fluke counts "
            f"as a finding. Smoke-tier numbers are a machinery check, not a "
            f"measurement, and must not be diffed against a standard-tier run."))
    if not scored:
        result.status = "skipped"
        result.detail = {"reason": "no pair produced usable samples on both halves",
                         "pairs_planned": len(plan_obj.pairs),
                         "records": len(records)}
        return result

    def add(name, value, *, ci=None, n=0, unit="", basis=MEASURED, note="",
            headline=False):
        result.metrics.append(Metric(name=name, value=_round(value), ci=ci, n=n,
                                     unit=unit, basis=basis, note=note,
                                     headline=headline))

    fp_rate = _macro(units, lambda r: r.twin_only, lambda r: r.twin_clusters)
    per_photo = _macro(units, lambda r: r.twin_only, lambda r: 1)
    orig_rate = _macro(units, lambda r: r.orig_only, lambda r: r.orig_clusters)
    sev = _pooled_mean(units, lambda r: r.severity_deltas)
    imp = _pooled_mean(units, lambda r: r.impact_deltas)
    conf = _pooled_mean(units, lambda r: r.confidence_deltas)
    n_matched = sum(len(r.severity_deltas) for r in scored)

    seed = plan_obj.seed
    sev_ci = bootstrap_ci(units, lambda u: _pooled_mean(u, lambda r: r.severity_deltas),
                          seed=seed)

    add("twin_fp_rate", fp_rate, n=n_vehicles, headline=True,
        ci=bootstrap_ci(units, lambda u: _macro(u, lambda r: r.twin_only,
                                                lambda r: r.twin_clusters), seed=seed),
        note="quorum-supported twin findings with zero supporting original samples, "
             "over all quorum-supported twin findings, averaged over vehicles. "
             "A lower bound - see the TWIN_MATCH caveat.")
    add("severity_inflation", sev, ci=sev_ci, n=n_matched, unit="severity ranks",
        headline=True,
        note="mean(rank_twin - rank_orig) over MATCHED findings. Paired, so it "
             "controls for the truck, the view and the defect. This is literally "
             "the exaggeration being chased.")
    add("twin_only_per_photo", per_photo, n=n_vehicles, unit="findings/photo",
        ci=bootstrap_ci(units, lambda u: _macro(u, lambda r: r.twin_only, lambda r: 1),
                        seed=seed),
        note="false positives per twin photograph. Falls against the before-run "
             "is half the pre-registered success condition.")
    add("original_only_rate", orig_rate, n=n_vehicles,
        ci=bootstrap_ci(units, lambda u: _macro(u, lambda r: r.orig_only,
                                                lambda r: r.orig_clusters), seed=seed),
        note="the mirror. EXPECTED to be positive: a blurred photo legitimately "
             "supports fewer claims. Context, not a defect.")
    add("impact_inflation", imp, n=n_matched, unit="impact ranks",
        ci=bootstrap_ci(units, lambda u: _pooled_mean(u, lambda r: r.impact_deltas),
                        seed=seed))
    add("confidence_delta", conf, n=n_matched,
        ci=bootstrap_ci(units, lambda u: _pooled_mean(u, lambda r: r.confidence_deltas),
                        seed=seed))

    twin_photos = len(scored)
    add("findings_per_photo_twin",
        sum(r.twin_findings for r in scored) / max(1, sum(r.twin_samples for r in scored)),
        n=twin_photos, unit="findings/call")
    add("findings_per_photo_original",
        sum(r.orig_findings for r in scored) / max(1, sum(r.orig_samples for r in scored)),
        n=twin_photos, unit="findings/call")

    twin_ill = sum(r.twin_illegible for r in scored)
    twin_calls = sum(r.twin_samples for r in scored)
    orig_ill = sum(r.orig_illegible for r in scored)
    orig_calls = sum(r.orig_samples for r in scored)
    add("twin_illegible_rate", twin_ill / twin_calls if twin_calls else None,
        ci=wilson_ci(twin_ill, twin_calls), n=twin_calls,
        note="share of twin calls the model marked not legible. Under the intended "
             "mechanism a twin should LOSE COVERAGE rather than gain findings, so "
             "this rising while twin_only_per_photo falls is the shape of a fix.")
    add("original_illegible_rate", orig_ill / orig_calls if orig_calls else None,
        ci=wilson_ci(orig_ill, orig_calls), n=orig_calls)

    dropped = sum(r.gate_dropped for r in results)
    planned = len(plan_obj.calls)
    add("gate_dropped_rate", dropped / planned if planned else 0.0,
        ci=wilson_ci(dropped, planned), n=planned,
        note="halves the real gate refused. Correct behaviour, not a false positive.")
    failed = sum(r.failed for r in results)
    add("call_failure_rate", failed / planned if planned else 0.0,
        ci=wilson_ci(failed, planned), n=planned)

    total_components = sum(r.total_components for r in scored)
    add("truck_vocab_share",
        sum(r.truck_specific for r in scored) / total_components if total_components else None,
        n=total_components, basis=DIAGNOSTIC,
        note="carried over from app/vlm/bench.py, where it was a SELECTION "
             "criterion. Here it is a diagnostic and nothing else: a model that "
             "fabricates truck-flavoured findings scores well on it.")

    result.strata = {
        "transform": _strata(scored, lambda p: p.degradations or ["none"]),
        "severity_band": _strata(scored, lambda p: [p.severity_band]),
        "view": _strata(scored, lambda p: [p.view]),
        "market": _strata(scored, lambda p: [p.market]),
    }
    result.detail = {
        "pairs_scored": len(scored), "pairs_planned": len(plan_obj.pairs),
        "vehicles": n_vehicles, "samples_per_half": plan_obj.samples,
        "quorum": quorum_for(plan_obj.samples),
        "family_partition": family_partition_name(),
        "twin_match": TWIN_MATCH,
        "twin_only_examples": [e for r in scored for e in r.twin_only_examples][:12],
    }
    result.gates = [
        Gate(name="twin_severity_inflation", value=_round(sev),
             threshold=SEVERITY_INFLATION_GATE, direction="below", ci=sev_ci,
             basis=ASSUMED,
             note="pre-registered in the harness design before any baseline "
                  "existed. The stronger reading of the same number is whether "
                  "its CI contains 0 or lies below it."),
        Gate(name="severity_inflation_ci_excludes_positive",
             value=0.0 if (sev_ci and sev_ci[1] <= 0) else 1.0,
             threshold=0.0, direction="below", ci=sev_ci, basis=MEASURED,
             note="0 when the 95% interval lies at or below 0 - the "
                  "pre-registered success condition - and 1 when it does not, "
                  "i.e. the twin is graded harsher than the same truck "
                  "photographed cleanly."),
    ]
    return result
