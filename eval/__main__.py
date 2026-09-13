"""The harness CLI.

    python -m eval --estimate --tier full        print the bill, spend nothing
    python -m eval --tier smoke                  run every suite at that tier
    python -m eval --suite twin_fp --tier standard
    python -m eval --replay eval/runs/<id>       re-score cached text, 0 calls
    python -m eval --replay <id> --sweep eval.suites.twin_fp.TWIN_MATCH=0.1:0.6:0.05
    python -m eval --diff eval/runs/A eval/runs/B

`--estimate` is a feature, not a nicety. A run prints "3,612 vision calls (2,140
cached, 1,472 new)" and REFUSES to proceed above `eval.CONFIRM_ABOVE_CALLS`
without `--yes`. Cost control belongs in the tool that spends the money, not in
the operator's memory of what a tier costs - and the numbers moved once already,
when `config.CLOSEUP_SAMPLES` went to 3 and tripled every per-photo suite
against the figures written into the design.

Everything except an actual run works with no credentials at all, which is the
point of `--estimate` and of `--replay`: the expensive half is opt-in.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app import config
from app.config import REPO

from . import CONFIRM_ABOVE_CALLS, TIERS
from .cache import CACHE_ROOT, CachedBackend, ResponseCache, resolve_cached
from .replay import parse_values, render_sweep, replay, sweep, write_raw
from .scorecard import Scorecard, diff, load, render_diff
from .suites import DEFAULT_ORDER, SUITES
from .suites.base import NotImplementedSuite

RUNS = REPO / "eval" / "runs"

MODEL_BY_BACKEND = {"cursor": config.CURSOR_MODEL, "openai": config.OPENAI_MODEL,
                    "anthropic": config.ANTHROPIC_MODEL}


def default_model_id(backend: str | None = None) -> str:
    """The model a run WOULD use, read from config without probing anything.

    `--estimate` must work with zero API keys, and probing a backend to learn
    its model id would make the cheapest command in the harness the one that
    needs credentials.
    """
    name = (backend or config.BACKEND_OVERRIDE or config.BACKEND_CHAIN[0]).lower()
    return MODEL_BY_BACKEND.get(name, name)


def chosen_suites(names: list[str] | None) -> list[str]:
    if not names:
        return list(DEFAULT_ORDER)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; have {sorted(SUITES)}")
    return [n for n in DEFAULT_ORDER if n in names]


# --- estimate --------------------------------------------------------------

def estimate(tier: str, names: list[str], seed: int, model_id: str,
             effort: str | None, cache: ResponseCache) -> tuple[list[dict], dict]:
    rows, totals = [], {"planned": 0, "cached": 0, "new": 0, "unknown": 0}
    for name in names:
        plan_obj = SUITES[name].plan(tier, seed=seed, model_id=model_id, effort=effort,
                                     require_files=False)
        calls = list(getattr(plan_obj, "calls", []) or [])
        if calls and getattr(plan_obj, "params", {}).get("keys_resolved", True):
            cached, new = resolve_cached(calls, model_id, cache)
            unknown = 0
        else:
            # Keys unresolvable ahead of the run (the suite needs the gate, pass A
            # or a crop decision first). Counted as NEW, which over-states the
            # bill - the safe direction for a number whose job is to stop a run.
            cached, new, unknown = 0, plan_obj.n_calls, plan_obj.n_calls
        params = getattr(plan_obj, "params", {}) or {}
        rows.append({"suite": name, "planned": plan_obj.n_calls, "cached": cached,
                     "new": new, "keys_known": not unknown,
                     "missing": params.get("pairs_missing_on_disk", 0),
                     "wanted": params.get("pairs_wanted", 0),
                     "note": getattr(plan_obj, "params", {}).get("note", "")
                             or getattr(plan_obj, "note", "")})
        totals["planned"] += plan_obj.n_calls
        totals["cached"] += cached
        totals["new"] += new
        totals["unknown"] += unknown
    return rows, totals


def print_estimate(rows: list[dict], totals: dict, tier: str, model_id: str) -> None:
    print(f"\ntier {tier}  model {model_id}  cache {CACHE_ROOT}")
    print(f"{'suite':<14}{'planned':>9}{'cached':>9}{'new':>9}  keys")
    print("-" * 50)
    for row in rows:
        keys = ("shared - no calls of its own" if not row["planned"]
                else "resolved" if row["keys_known"]
                else "unknown -> counted as new")
        print(f"{row['suite']:<14}{row['planned']:>9}{row['cached']:>9}{row['new']:>9}"
              f"  {keys}")
    print("-" * 50)
    print(f"{'TOTAL':<14}{totals['planned']:>9}{totals['cached']:>9}{totals['new']:>9}")
    print(f"\n{totals['planned']:,} vision calls ({totals['cached']:,} cached, "
          f"{totals['new']:,} new)")
    for row in [r for r in rows if r.get("missing")]:
        print(f"\n{row['suite']}: {row['missing']} of {row['wanted']} sampled photo "
              f"pairs are not on this disk (data/images/ is gitignored - rebuild "
              f"with scripts/download_images.py). The budget above covers only "
              f"what IS here, so it under-states a full run.")
    if totals["unknown"]:
        print(f"{totals['unknown']:,} of those are suites whose keys cannot be "
              f"resolved before the run; they are counted as new, so this is an "
              f"upper bound.")


# --- run -------------------------------------------------------------------

def do_run(args) -> int:
    cache = ResponseCache(args.cache)
    model_id = args.model or default_model_id(args.backend)
    names = chosen_suites(args.suite)
    rows, totals = estimate(args.tier, names, args.seed, model_id, args.effort, cache)
    print_estimate(rows, totals, args.tier, model_id)

    if args.estimate:
        return 0
    if totals["new"] > CONFIRM_ABOVE_CALLS and not args.yes:
        print(f"\nrefusing to start: {totals['new']:,} new vision calls is above the "
              f"{CONFIRM_ABOVE_CALLS:,}-call confirmation threshold. Re-run with "
              f"--yes if that is what you meant.")
        return 2

    from app import vlm
    inner = vlm.resolve_chain(args.backend)[0]
    client = CachedBackend(inner, cache=cache, model_id=args.model or inner.model)
    print(f"\nbackend {inner.name}/{client.model_id}, effort "
          f"{args.effort or config.CLOSEUP_EFFORT}")

    gate_checks = None
    if args.gate:
        from app import gate as gate_stage
        gate_checks = gate_stage.inspect

    t0 = time.time()
    card = Scorecard.start(tier=args.tier, seed=args.seed,
                           model={"backend": inner.name, "id": client.model_id,
                                  "effort": args.effort or config.CLOSEUP_EFFORT})
    out_dir = Path(args.out or RUNS) / card.run_id
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    for name in names:
        suite = SUITES[name]
        plan_obj = suite.plan(args.tier, seed=args.seed, model_id=client.model_id,
                              effort=args.effort)
        try:
            records = suite.run(plan_obj, client, gate_checks=gate_checks)
            write_raw(out_dir, name, plan_obj, records)
            result = suite.score(records, plan_obj, cache, client.model_id)
        except NotImplementedSuite as exc:
            from .scorecard import SuiteResult
            result = SuiteResult(name=name, status="not_implemented",
                                 detail={"reason": str(exc),
                                         "planned_calls": plan_obj.n_calls})
        card.add(result)
        if name == "twin_fp":
            card.sampling = plan_obj.to_dict().get("sampling", {})
        print(f"  {name:<14} {result.status}  "
              f"({cache.stats.new} new / {cache.stats.cached} cached so far)")

    card.calls = cache.stats.to_dict()
    card.elapsed_s = time.time() - t0
    written = card.write(Path(args.out or RUNS))
    print(f"\n{written}/scorecard.md")
    print(f"calls: {card.calls['new']} new, {card.calls['cached']} cached, "
          f"{card.calls['failed']} failed - hit rate {100*card.calls['hit_rate']:.0f}%")
    return 0


def do_replay(args) -> int:
    run_dir = Path(args.replay)
    if not run_dir.exists():
        run_dir = RUNS / args.replay
    cache = ResponseCache(args.cache)
    if args.sweep:
        target, _, spec = args.sweep.partition("=")
        rows = sweep(run_dir, target.strip(), parse_values(spec), cache=cache,
                     suites=args.suite)
        print(render_sweep(rows))
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return 0
    card = replay(run_dir, cache=cache, suites=args.suite)
    written = card.write(Path(args.out or RUNS))
    print(card.markdown())
    print(f"\n{written}/scorecard.md  (0 vision calls)")
    return 0


def do_diff(args) -> int:
    a, b = (Path(p) if Path(p).exists() else RUNS / p for p in args.diff)
    before, after = load(a), load(b)
    rows = diff(before, after)
    text = render_diff(before, after, rows)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval",
                                 description="the appraisal evaluation harness")
    ap.add_argument("--tier", default="standard", choices=TIERS)
    ap.add_argument("--suite", action="append",
                    help=f"one of {sorted(SUITES)}; repeatable. Default: all.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", default=None, help="pin a vision backend")
    ap.add_argument("--model", default=None, help="override the model id used for cache keys")
    ap.add_argument("--effort", default=None, help="reasoning effort for the close-up pass")
    ap.add_argument("--cache", default=CACHE_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--estimate", action="store_true",
                    help="print the call budget and exit, spending nothing")
    ap.add_argument("--yes", action="store_true",
                    help=f"proceed past {CONFIRM_ABOVE_CALLS} new calls")
    ap.add_argument("--gate", dest="gate", action="store_true", default=True,
                    help="run the real gate over each photo (default)")
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="skip the gate; loads no torch and makes every planned call")
    ap.add_argument("--replay", default=None, metavar="RUN",
                    help="re-score a finished run from the cache. Zero calls.")
    ap.add_argument("--sweep", default=None, metavar="DOTTED.PATH=SPEC",
                    help="with --replay: re-score at each value, e.g. "
                         "app.evidence.passes.SAME_DEFECT=0.2:0.6:0.05")
    ap.add_argument("--diff", nargs=2, default=None, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args(argv)

    if args.diff:
        return do_diff(args)
    if args.replay:
        return do_replay(args)
    return do_run(args)


if __name__ == "__main__":
    sys.exit(main())
