"""The artifact: one row per vehicle, every panelist's answer kept.

Consensus is stored, but it is stored NEXT TO the three answers it was derived
from, never instead of them. The rubric will be revised - that is the point of
running a pilot - and a revision has to be re-scorable from what the panel
actually said. A file that kept only the consensus would make every rubric
change a re-run of the panel.

Stamped like `data/reference/new_prices_tr.json`: what it is, why it exists,
how it was made, when, and under which rubric hash. A row whose `rubric_sha`
does not match the current one is not wrong - it is graded under a different
rubric, and `check_rubric` says so rather than letting it pass silently.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from panel import agreement as ag
from panel.protocol import RUBRIC_SHA, SCHEMA_VERSION, severity_vector
from panel.select import DATA, Vehicle

ARTIFACT = DATA / "reference" / "condition_panel_v1.jsonl"
CARD = DATA / "reference" / "condition_panel_card.md"
PANEL_MODEL = "claude-opus-5"


def _relative(path: str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(DATA))
    except ValueError:
        return str(p)


def build_row(vehicle: Vehicle, panelists: list[dict], *,
              model_id: str = PANEL_MODEL, run_id: str,
              graded_at: str | None = None) -> dict:
    """One vehicle's row. `panelists` is [{panelist, response, errors, ...}]."""
    responses = [p["response"] for p in panelists if p.get("response")]
    vectors = [severity_vector(r) for r in responses]
    grades = [r.get("grade") for r in responses if r.get("grade")]
    confidences = [r.get("confidence") for r in responses
                   if isinstance(r.get("confidence"), (int, float))]

    per_component = {(vehicle.listing_id, c): [v.get(c) for v in vectors]
                     for c in (vectors[0] if vectors else {})}
    units = [[None if s == "not_visible" else s for s in states]
             for states in per_component.values()]
    within = ag.krippendorff_alpha(units, metric="ordinal",
                                   values=ag.SEVERITY_SCALE) if vectors else None
    raw, pairs = ag.percent_agreement(list(per_component.values()))

    return {
        "schema": SCHEMA_VERSION,
        "listing_id": vehicle.listing_id,
        "source_key": vehicle.source_key,
        "market": vehicle.market,
        "make": vehicle.make,
        "model": vehicle.model,
        "year": vehicle.year,
        "km": vehicle.km,
        "wear_band": vehicle.wear_band,
        "stratum": vehicle.stratum,
        "n_photos": vehicle.n_photos,
        "photos": [_relative(p) for p in vehicle.photos],
        "rubric_sha": RUBRIC_SHA,
        "model_id": model_id,
        "run_id": run_id,
        "graded_at": graded_at or date.today().isoformat(),
        "panelists": panelists,
        "consensus": {
            "components": ag.consensus(vectors) if vectors else {},
            "grade": ag.consensus_grade(grades) if grades else None,
            "confidence": round(sum(confidences) / len(confidences), 3)
            if confidences else None,
        },
        "agreement": {
            "grades": grades,
            "unanimous_grade": bool(grades) and len(set(grades)) == 1,
            "adjudication_required": ag.adjudication_required(grades),
            "within_vehicle_severity_alpha": None if within is None or within.alpha is None
            else round(within.alpha, 4),
            "component_state_agreement": None if raw is None else round(raw, 4),
            "component_pairs": pairs,
        },
    }


def write_rows(rows: list[dict], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_rows(path: Path = ARTIFACT) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def check_rubric(rows: list[dict]) -> list[str]:
    """Rows graded under a rubric that is no longer the current one."""
    return sorted({row["rubric_sha"] for row in rows} - {RUBRIC_SHA})


def panel_view(rows: list[dict]) -> tuple[dict, dict]:
    """(vectors by vehicle, grades by vehicle) - the input to `agreement`."""
    vectors, grades = {}, {}
    for row in rows:
        responses = [p["response"] for p in row["panelists"] if p.get("response")]
        if len(responses) < 2:
            continue
        vectors[row["listing_id"]] = [severity_vector(r) for r in responses]
        grades[row["listing_id"]] = [r.get("grade") for r in responses]
    return vectors, grades
