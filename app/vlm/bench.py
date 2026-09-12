"""Compare vision backends/models on the real evidence task.

    .venv/bin/python -m app.vlm.bench
    .venv/bin/python -m app.vlm.bench --models gpt-5.6-sol gpt-5.6-terra --backend openai

Picks the shipped default by measurement rather than by which id sounds
newest. Scores what the report actually depends on: did it come back
parseable, did it cite photos that exist, did it read the odometer, and did it
use heavy-vehicle components rather than a generic dent/scratch taxonomy.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

from .. import evidence as ev_stage
from .. import gate as gate_stage
from ..config import REPO

# Components that only a truck has. Their share of the findings is the cheap
# proxy for "sensible to someone who knows trucks" - a model describing a
# tractor unit in car vocabulary scores near zero here.
TRUCK_SPECIFIC = {"steer_tires", "drive_tires", "fifth_wheel", "coupling_airlines",
                  "chassis_frame", "air_suspension", "air_tanks_lines", "adblue_tank",
                  "exhaust_dpf", "bunk_sleeper", "mudflaps_guards", "fairings_skirts",
                  "roof_deflector", "cab_steps", "undercarriage", "brakes_hubs"}

DEFAULT_FIXTURE = "demo/tr_clean"


def score(report) -> dict:
    issues = report.issues
    truck_specific = sum(1 for i in issues if i.component in TRUCK_SPECIFIC)
    return {
        "seconds": report.elapsed_s,
        "issues": len(issues),
        "truck_specific": truck_specific,
        "truck_vocab_share": round(truck_specific / len(issues), 2) if issues else 0.0,
        "gaps": len(report.coverage_gaps),
        "grade": report.condition_grade,
        "confidence": report.confidence,
        "odometer_km": report.vehicle.odometer_km,
        "make": report.vehicle.make,
        "warnings": len(report.parse_warnings),
        "mean_obs_chars": round(statistics.mean(
            [len(i.observation) for i in issues]), 0) if issues else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--models", nargs="*",
                    default=["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"])
    ap.add_argument("--effort", default="low")
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    photos = sorted(p for p in (REPO / args.fixture).glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not photos:
        print(f"no photos in {args.fixture} - run: python -m app.demo --build")
        return 2
    print(f"fixture: {args.fixture} ({len(photos)} photos)\n")
    gate = gate_stage.run(photos)

    env_key = {"openai": "KAMION_OPENAI_MODEL",
               "anthropic": "KAMION_ANTHROPIC_MODEL",
               "cursor": "KAMION_CURSOR_MODEL"}[args.backend]
    effort_key = {"openai": "KAMION_OPENAI_EFFORT",
                  "anthropic": "KAMION_ANTHROPIC_EFFORT"}.get(args.backend)

    header = f"{'model':18s} {'sec':>6s} {'iss':>4s} {'truck%':>7s} {'gaps':>5s} " \
             f"{'grade':>10s} {'conf':>5s} {'odo':>9s} {'warn':>5s}"
    print(header)
    print("-" * len(header))
    rows = []
    for model in args.models:
        os.environ[env_key] = model
        if effort_key:
            os.environ[effort_key] = args.effort
        for _ in range(args.repeats):
            # config is read at import time, so re-read it for each model.
            import importlib
            from .. import config as cfg
            importlib.reload(cfg)
            import app.vlm.openai_backend as ob
            import app.vlm.anthropic_backend as ab
            importlib.reload(ob)
            importlib.reload(ab)

            t0 = time.time()
            try:
                report = ev_stage.run(gate, {"year": 2020, "make": "Ford Trucks"},
                                      backend=args.backend)
                s = score(report)
                rows.append((model, s))
                print(f"{model:18s} {s['seconds']:6.1f} {s['issues']:4d} "
                      f"{s['truck_vocab_share'] * 100:6.0f}% {s['gaps']:5d} "
                      f"{s['grade']:>10s} {s['confidence']:5.2f} "
                      f"{str(s['odometer_km'] or '-'):>9s} {s['warnings']:5d}")
            except Exception as exc:
                print(f"{model:18s} FAILED {type(exc).__name__}: {str(exc)[:90]}")
                print(f"  (elapsed {time.time() - t0:.1f}s)")
    print("\nissues = findings returned, truck% = share using heavy-vehicle components,")
    print("odo = odometer km read off the dash, warn = schema violations repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
