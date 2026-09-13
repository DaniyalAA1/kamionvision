"""Re-score a finished run against the current code, at zero vision calls.

This is what the cache is FOR. A run writes two things: the response texts, in
`eval/cache/`, and a unit index naming the keys, in `eval/runs/<id>/raw/`.
Neither holds a parsed object, so every change downstream of pass B -
`parse_closeup`, `merge_duplicates`, the rollup, the grade ladder, the price
multiplier, the matcher in this package, the scorecard's own arithmetic - can
be re-measured on exactly the same responses for nothing.

That partitions the workstreams cleanly. A prompt or model change busts the keys
and its own workstream pays for re-collection, visibly, as a 0% hit rate on the
next scorecard. A change to anything after the call costs a second of CPU.

`--sweep` follows from the same property: if re-scoring is free, then so is
re-scoring at fifty values of a constant. The guessed constants in this system -
`SAME_DEFECT`, `TWIN_MATCH`, the decay, the corroboration weight, the coverage
floor - become fitted ones, on `panel_dev` once that exists, and against the
twin suite's own false-positive rate before then.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path

from .cache import ResponseCache
from .cases import TwinPair
from .scorecard import Scorecard, load
from .suites import SUITES
from .suites.twin_fp import Plan as TwinPlan


def raw_path(run_dir: Path | str, suite: str) -> Path:
    return Path(run_dir) / "raw" / f"{suite}.json"


def write_raw(run_dir: Path, suite: str, plan_obj, records: list[dict]) -> Path:
    """The unit index. Metadata and keys only - the text stays in the cache."""
    path = raw_path(run_dir, suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = [p.to_dict() for p in getattr(plan_obj, "pairs", []) or []]
    path.write_text(json.dumps(
        {"suite": suite, "plan": plan_obj.to_dict(), "pairs": pairs,
         "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


_PAIR_FIELDS = {f.name for f in dataclasses.fields(TwinPair)}


def _rebuild_plan(suite: str, blob: dict):
    """Rebuild the plan object from the raw index, not from the corpus.

    Deliberately not re-derived from (tier, seed): the sample IS deterministic,
    but a replay that silently re-sampled because `images.csv` moved underneath
    it would report a different set of trucks under the same run id, and nothing
    would say so.
    """
    plan_blob = blob.get("plan") or {}
    pairs = [TwinPair(**{k: v for k, v in d.items() if k in _PAIR_FIELDS})
             for d in blob.get("pairs") or []]
    if suite == "twin_fp":
        return TwinPlan(name=suite, tier=plan_blob.get("tier", "unknown"),
                        seed=int(plan_blob.get("seed", 0)),
                        samples=int(plan_blob.get("samples", 1)),
                        pairs=pairs, calls=[], params=plan_blob.get("params") or {})
    from .suites.base import Budget
    budget = Budget(name=suite, tier=plan_blob.get("tier", "unknown"),
                    seed=int(plan_blob.get("seed", 0)),
                    n_calls=int(plan_blob.get("calls", 0)),
                    units=int(plan_blob.get("units", 0)),
                    params=plan_blob.get("params") or {})
    budget.pairs = pairs
    budget.samples = int(plan_blob.get("samples", 1))
    return budget


def replay(run_dir: Path | str, *, cache: ResponseCache | None = None,
           suites: list[str] | None = None) -> Scorecard:
    run_dir = Path(run_dir)
    before = load(run_dir)
    store = cache or ResponseCache()
    model_id = (before.get("model") or {}).get("id", "unknown")

    card = Scorecard.start(tier=before.get("tier", "unknown"),
                           seed=int(before.get("seed", 0)),
                           model=before.get("model") or {})
    card.run_id = f"{card.run_id}-replay"
    card.sampling = before.get("sampling") or {}
    card.notes.append(f"replay of {before.get('run_id')} - zero vision calls by "
                      f"construction; the client is offline and a miss raises.")

    served = missing = 0
    for suite in (suites or list(before.get("suites") or {})):
        path = raw_path(run_dir, suite)
        if not path.exists() or suite not in SUITES:
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        records = blob.get("records") or []
        plan_obj = _rebuild_plan(suite, blob)
        # Counted here rather than inside ResponseCache.get, because `score()`
        # reads a key more than once and the interesting number is how much of
        # the run survived, not how many dictionary lookups happened.
        for row in records:
            if row.get("key") and store.has(model_id, row["key"]):
                served += 1
            elif not row.get("gate_dropped"):
                missing += 1
        result = SUITES[suite].score(records, plan_obj, store, model_id)
        card.add(result)

    card.calls = {"new": 0, "cached": served, "failed": missing,
                  "hit_rate": round(served / (served + missing), 4) if served + missing else 0.0,
                  "replay_of": before.get("run_id")}
    if missing:
        card.notes.append(
            f"{missing} of {served + missing} recorded responses are no longer in "
            f"the cache. The metrics below are scored on what survived, which is "
            f"a different sample from the one the original run reported.")
    return card


# --- sweeps ----------------------------------------------------------------

def _resolve(target: str):
    """`app.evidence.passes.SAME_DEFECT` -> (module, attribute)."""
    module_path, _, attr = target.rpartition(".")
    if not module_path:
        raise ValueError(f"{target!r} is not a dotted path to a module attribute")
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        raise ValueError(f"{module_path} has no attribute {attr!r}")
    return module, attr


def parse_values(spec: str) -> list[float]:
    """`0.20:0.60:0.05` (inclusive range) or `0.2,0.3,0.42`."""
    if ":" in spec:
        start, stop, step = (float(x) for x in spec.split(":"))
        out, value = [], start
        while value <= stop + 1e-9:
            out.append(round(value, 6))
            value += step
        return out
    return [float(x) for x in spec.split(",") if x.strip()]


def sweep(run_dir: Path | str, target: str, values: list[float], *,
          cache: ResponseCache | None = None,
          suites: list[str] | None = None) -> list[dict]:
    """Re-score the same cached responses at each value of one constant.

    Restores the original in a `finally`: a sweep that leaves `SAME_DEFECT` at
    0.17 because it raised halfway through would silently change the product.
    """
    module, attr = _resolve(target)
    original = getattr(module, attr)
    rows = []
    try:
        for value in values:
            setattr(module, attr, value)
            card = replay(run_dir, cache=cache, suites=suites).to_dict()
            row = {"target": target, "value": value}
            for suite, blob in card.get("suites", {}).items():
                for metric in blob.get("metrics", []):
                    if metric.get("headline"):
                        row[f"{suite}.{metric['name']}"] = metric.get("value")
            rows.append(row)
    finally:
        setattr(module, attr, original)
    return rows


def render_sweep(rows: list[dict]) -> str:
    if not rows:
        return "no sweep rows"
    cols = [c for c in rows[0] if c not in ("target", "value")]
    L = [f"# sweep over `{rows[0]['target']}`", "",
         "| value | " + " | ".join(f"`{c}`" for c in cols) + " |",
         "|" + "---|" * (len(cols) + 1)]
    for row in rows:
        L.append(f"| {row['value']} | " + " | ".join(
            "-" if row.get(c) is None else f"{row[c]:.4g}" for c in cols) + " |")
    flat = [c for c in cols
            if len({None if r.get(c) is None else round(float(r[c]), 6)
                    for r in rows}) <= 1]
    if flat and len(rows) > 1:
        L += ["", f"**`{'`, `'.join(flat)}` did not move at any value.** Check that "
                  f"`{rows[0]['target']}` is actually read by the suites being "
                  f"scored before reading this as insensitivity - a sweep over a "
                  f"constant nobody consumes produces exactly this table."]
    L += ["", "Zero vision calls: every row is the same cached responses, "
              "re-scored. Tune on `panel_dev` only - see eval/suites/panel.py "
              "on contamination control."]
    return "\n".join(L)
