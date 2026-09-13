"""Suite 5 - scoring against the Opus panel reference set. STUB.

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

from pathlib import Path

from app.config import REPO

from .base import Budget, NotImplementedSuite

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
    raise NotImplementedSuite("panel.run is not implemented yet")


def score(records, plan_obj, cache, model_id):
    raise NotImplementedSuite("panel.score is not implemented yet")
