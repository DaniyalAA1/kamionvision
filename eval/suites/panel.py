"""Suite 5 - scoring against the Opus panel reference set.

The only external reference in the harness, and the only one that can turn the
distribution suite's prior into a measurement. Consumed from
`eval/reference/panel_v1.json`, produced by the panel workstream, contract
frozen here so it can be built against:

    {"version": "panel_v1", "rubric_version": "...", "panelists": 3,
     "built_at": "...",
     "split": {"dev": ["<listing_id>", ...], "test": [...]},
     "vehicles": [
       {"listing_id": "...", "source_key": "...",
        "grade": "good", "grade_votes": ["good", "good", "fair"],
        "grade_agreement": 0.67,
        "findings": [{"family": "drive_tires", "severity": "moderate",
                      "impact": "medium", "photo_ids": [3, 7],
                      "observation": "...", "votes": 2, "n_panelists": 3}],
        "clean_families": [...], "coverage_families": [...]}]}

Two requirements on the panel workstream, both load-bearing:

  1. Panelists label FAMILIES, not the 33 component ids. Inter-annotator
     agreement on `corrosion` versus `chassis_frame` is precisely the ambiguity
     the condition workstream removes; asking humans to resolve it manufactures
     disagreement and then scores the pipeline against it.
  2. Votes are recorded PER FINDING, so this suite can weight by agreement and
     can compute its own ceiling.

Every metric is reported against the leave-one-panelist-out ceiling, never
against 1.0:

    grade_exact / grade_within_one
    grade_bias        mean(rank_pipeline - rank_panel)   <- THE signed
                      exaggeration number, and the one to quote
    finding_recall    panel findings (votes >= 2) matched by the pipeline
    finding_precision pipeline findings matched by any panel finding (votes >= 1)
    severity_mae / severity_bias on matched findings

CLAUDE.md already carries this exact lesson for the view head - "never report
the 0.772 teacher-agreement as accuracy". A pipeline at 0.65 exact-grade against
panelists who agree with each other 0.70 is near-ceiling, not failing.

Contamination control, which matters more here than any single metric because
this workstream introduces a dozen tunable constants: the panel set splits 60/40
by VEHICLE into `panel_dev` - where SAME_DEFECT, TWIN_MATCH, DECAY, the ladder
thresholds and the isotonic severity map may be tuned - and `panel_test`,
touched only to report. The split is seeded, disclosed in the scorecard, and the
panel must never see pipeline output. Panel vehicles are drawn FROM the
standard-tier sample so panel scoring reuses findings the distribution sweep
already cached - which is why this suite budgets zero incremental calls.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.condition import GRADE_RANK, SEVERITY_RANK
from app.config import REPO

from ..scorecard import MEASURED, Metric, SuiteResult, wilson_ci
from . import twin_fp
from .base import Budget

NAME = "panel"

REFERENCE = REPO / "eval" / "reference" / "panel_v1.json"
DEV_FRACTION = 0.60


def available() -> bool:
    return REFERENCE.exists()


def plan(tier: str = "standard", *, seed: int = 7, model_id: str = "unknown",
         effort: str | None = None, structured: bool = True,
         samples: int | None = None, require_files: bool = True) -> Budget:
    return Budget(name=NAME, tier=tier, seed=seed, n_calls=0, units=0,
                  params={"reference": str(Path(REFERENCE).relative_to(REPO)),
                          "available": available(), "dev_fraction": DEV_FRACTION},
                  note="zero incremental calls: draws its vehicles from the "
                       "distribution sweep and scores the findings that sweep "
                       "already cached"
                       + ("" if available() else "; reference set not built yet"))


def run(plan_obj, client, **kwargs) -> list[dict]:
    """Pair panel vehicles with appraisals the distribution sweep already ran."""
    ref = _load_reference()
    if ref is None:
        return []
    by_id = {}
    for row in kwargs.get("distribution_records") or []:
        lid = str(row.get("listing_id") or "")
        if lid:
            by_id[lid] = row
    split = ref.get("split") or {}
    dev = {str(x) for x in split.get("dev") or []}
    test = {str(x) for x in split.get("test") or []}
    records = []
    for vehicle in ref.get("vehicles") or []:
        lid = str(vehicle.get("listing_id") or "")
        if lid in test:
            part = "test"
        elif lid in dev:
            part = "dev"
        else:
            part = "unspecified"
        src = by_id.get(lid) or {}
        records.append({
            "listing_id": lid,
            "source_key": vehicle.get("source_key") or src.get("source_key"),
            "split": part,
            "panel": vehicle,
            "appraisal": src.get("appraisal"),
            "missing": src.get("appraisal") is None,
        })
    return records


def score(records, plan_obj, cache, model_id):
    """Score pipeline appraisals against panel consensus. Pure; no client."""
    result = SuiteResult(name=NAME)
    units = [_unit(r) for r in records]
    units = [u for u in units if u is not None]
    result.detail = {
        "reference": str(Path(REFERENCE).relative_to(REPO)),
        "available": available(),
        "records": len(records),
        "vehicles_scored": len(units),
        "dev_fraction": DEV_FRACTION,
    }
    if not units:
        result.status = "skipped"
        result.detail["reason"] = ("no panel records; reference set not built yet"
                                   if not available() else
                                   "no panel vehicles had a pipeline appraisal")
        return result

    by_split: dict[str, list] = {}
    for unit in units:
        by_split.setdefault(unit["split"], []).append(unit)
    result.strata = {name: _split_block(rows) for name, rows in by_split.items()}
    # Headline numbers are the held-out slice when it exists; otherwise all.
    report = by_split.get("test") or units
    block = _split_block(report)
    result.detail["ceiling"] = {
        "grade_exact": block["ceiling_grade_exact"],
        "grade_within_one": block["ceiling_grade_within_one"],
        "n_loo": block["n_loo"],
    }
    result.detail["split_used"] = "test" if "test" in by_split else "all"
    n = block["n"]
    for name, value, headline, note in (
        ("grade_exact", block["grade_exact"], True,
         f"leave-one-panelist-out ceiling {block['ceiling_grade_exact']}"),
        ("grade_within_one", block["grade_within_one"], True,
         f"leave-one-panelist-out ceiling {block['ceiling_grade_within_one']}"),
        ("grade_bias", block["grade_bias"], True,
         "mean(rank_pipeline - rank_panel); positive is pipeline optimism"),
        ("finding_recall", block["finding_recall"], True,
         "panel findings with votes >= 2 matched by the pipeline"),
        ("finding_precision", block["finding_precision"], False,
         "pipeline findings matched by any panel finding with votes >= 1"),
        ("severity_mae", block["severity_mae"], False,
         "ordinal |pipeline - panel| on matched findings"),
        ("severity_bias", block["severity_bias"], False,
         "mean(rank_pipeline - rank_panel) on matched findings"),
    ):
        k = None
        ci = None
        if name in ("grade_exact", "grade_within_one") and value is not None:
            k = int(round(value * n))
            ci = wilson_ci(k, n)
        elif name == "finding_recall" and block["panel_quorum"] and value is not None:
            k = int(round(value * block["panel_quorum"]))
            ci = wilson_ci(k, block["panel_quorum"])
        elif name == "finding_precision" and block["pipeline_findings"] and value is not None:
            k = int(round(value * block["pipeline_findings"]))
            ci = wilson_ci(k, block["pipeline_findings"])
        result.metrics.append(Metric(
            name=name, value=value, ci=ci,
            n=n if name.startswith("grade") else (block["matched"]
                                                  if name.startswith("severity")
                                                  else (block["panel_quorum"]
                                                        if name == "finding_recall"
                                                        else block["pipeline_findings"])),
            unit="share" if name.startswith("grade") or name.startswith("finding")
            else "severity-rank",
            basis=MEASURED, note=note, headline=headline))
    return result


def _load_reference() -> dict | None:
    if not available():
        return None
    try:
        return json.loads(REFERENCE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _unit(row: dict) -> dict | None:
    panel = row.get("panel")
    appraisal = row.get("appraisal")
    if not panel or not appraisal:
        return None
    evidence = appraisal.get("evidence") or {}
    pipeline_grade = evidence.get("condition_grade")
    if pipeline_grade not in GRADE_RANK:
        return None
    votes = [v for v in (panel.get("grade_votes") or []) if v in GRADE_RANK]
    consensus = panel.get("grade")
    if consensus not in GRADE_RANK:
        if not votes:
            return None
        consensus = Counter(votes).most_common(1)[0][0]
    return {
        "listing_id": str(row.get("listing_id") or panel.get("listing_id") or ""),
        "split": str(row.get("split") or "unspecified"),
        "pipeline_grade": pipeline_grade,
        "panel_grade": consensus,
        "votes": votes,
        "panel_findings": list(panel.get("findings") or []),
        "pipeline_findings": _pipeline_findings(evidence),
    }


def _pipeline_findings(evidence: dict) -> list[dict]:
    issues = evidence.get("issues") or evidence.get("findings") or []
    out = []
    for issue in issues:
        family = issue.get("family") or twin_fp.family_of(issue.get("component") or "")
        out.append({
            "family": family,
            "severity": issue.get("severity") or "minor",
            "impact": issue.get("impact") or issue.get("price_impact") or "medium",
            "observation": issue.get("observation") or "",
        })
    return out


def _split_block(units: list[dict]) -> dict:
    n = len(units)
    exact = sum(u["pipeline_grade"] == u["panel_grade"] for u in units)
    within = sum(_within_one(u["pipeline_grade"], u["panel_grade"]) for u in units)
    bias = _mean([GRADE_RANK[u["pipeline_grade"]] - GRADE_RANK[u["panel_grade"]]
                  for u in units])
    loo_exact, loo_within, n_loo = _loo_ceiling([u["votes"] for u in units])
    recall_hits = recall_n = prec_hits = prec_n = 0
    sev_abs = []
    sev_signed = []
    for unit in units:
        quorum = [f for f in unit["panel_findings"] if int(f.get("votes") or 0) >= 2]
        any_vote = [f for f in unit["panel_findings"] if int(f.get("votes") or 0) >= 1]
        quorum_clusters = _clusters(quorum)
        any_clusters = _clusters(any_vote)
        pipeline_clusters = _clusters(unit["pipeline_findings"])
        recall_pairs = twin_fp.greedy_match(quorum_clusters, pipeline_clusters)
        prec_pairs = twin_fp.greedy_match(pipeline_clusters, any_clusters)
        recall_n += len(quorum_clusters)
        recall_hits += len(recall_pairs)
        prec_n += len(pipeline_clusters)
        prec_hits += len(prec_pairs)
        for panel_c, pipe_c, _ in recall_pairs:
            if (panel_c.rep.severity in SEVERITY_RANK
                    and pipe_c.rep.severity in SEVERITY_RANK):
                delta = (SEVERITY_RANK[pipe_c.rep.severity]
                         - SEVERITY_RANK[panel_c.rep.severity])
                sev_abs.append(abs(delta))
                sev_signed.append(delta)
    return {
        "n": n,
        "grade_exact": _ratio(exact, n),
        "grade_within_one": _ratio(within, n),
        "grade_bias": bias,
        "ceiling_grade_exact": loo_exact,
        "ceiling_grade_within_one": loo_within,
        "n_loo": n_loo,
        "finding_recall": _ratio(recall_hits, recall_n),
        "finding_precision": _ratio(prec_hits, prec_n),
        "panel_quorum": recall_n,
        "pipeline_findings": prec_n,
        "matched": len(sev_abs),
        "severity_mae": _mean(sev_abs),
        "severity_bias": _mean(sev_signed),
    }


def _clusters(findings: list[dict]) -> list:
    observations = []
    for i, finding in enumerate(findings):
        observations.append(twin_fp.Obs(
            sample=i, component=str(finding.get("family") or "unknown"),
            family=str(finding.get("family") or "unknown"),
            observation=str(finding.get("observation") or ""),
            severity=str(finding.get("severity") or "minor"),
            impact=str(finding.get("impact") or "medium"),
            confidence=1.0))
    return twin_fp.cluster_observations(observations)


def _loo_ceiling(vote_lists: list[list[str]]) -> tuple[float | None, float | None, int]:
    exact = within = n = 0
    for votes in vote_lists:
        usable = [v for v in votes if v in GRADE_RANK]
        if len(usable) < 2:
            continue
        for i, held in enumerate(usable):
            rest = usable[:i] + usable[i + 1:]
            maj = Counter(rest).most_common(1)[0][0]
            exact += held == maj
            within += _within_one(held, maj)
            n += 1
    return _ratio(exact, n), _ratio(within, n), n


def _within_one(a: str, b: str) -> bool:
    return abs(GRADE_RANK[a] - GRADE_RANK[b]) <= 1


def _ratio(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
