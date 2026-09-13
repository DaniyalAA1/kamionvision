"""How many CLIP clusters do known-single listings produce?

A threshold that flags mixed vehicles must almost never fire on the 200
corpus vehicles, each of which is one tractor. Front vs rear of one F-MAX
can look like two clusters if every winner is counted; this script measures
`cluster_seed_winners` (whole-vehicle seeds only).

    .venv/bin/python scripts/calibrate_mixed_vehicle.py
    .venv/bin/python scripts/calibrate_mixed_vehicle.py --limit 10

Writes models/mixed_vehicle_thresholds.json. pricing_blocker only reads a
row measured on all 200 originals with false-positive rate <= 0.5%.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import gate as gate_stage
from app.config import DATA, IMAGES_CSV, MODELS, REPO
from app.subject import WHOLE_VEHICLE_VIEWS

OUT = MODELS / "mixed_vehicle_thresholds.json"
# At most 1 of 200 known-single vehicles may sit at or above the threshold.
FP_BUDGET = 0.005


def vehicles(limit: int | None) -> list[tuple[str, str, list[Path]]]:
    df = pd.read_csv(IMAGES_CSV, low_memory=False)
    df = df[df.variant.astype(str) == "original"]
    out = []
    for (source_key, listing_id), group in df.groupby(["source_key", "listing_id"], sort=True):
        paths = [DATA / p for p in group.sort_values("image_index").path]
        paths = [p for p in paths if p.is_file()]
        if paths:
            out.append((str(source_key), str(listing_id), paths))
    if limit:
        out = out[:limit]
    return out


def count_one(paths: list[Path]) -> dict:
    checks, identity = gate_stage.inspect(paths)
    whole = sum(1 for c in checks if c.usable and c.view in WHOLE_VEHICLE_VIEWS)
    return {
        "clusters": int(identity.clusters),
        "seeds": len(identity.seeded_from),
        "method": identity.method,
        "whole_vehicle_frames": whole,
        "body_tag": "",
        "body_tag_conf": 0.0,
    }


def choose_threshold(counts: list[int]) -> tuple[int | None, float]:
    """Smallest k such that P(count >= k) <= FP_BUDGET. None if no such k."""
    n = len(counts)
    if not n:
        return None, 1.0
    for k in range(2, max(counts) + 2):
        fp = sum(1 for c in counts if c >= k) / n
        if fp <= FP_BUDGET:
            return k, fp
    return None, sum(1 for c in counts if c >= 2) / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="first N vehicles only (smoke). Artifact will not be enforceable.")
    args = p.parse_args()

    rows = vehicles(args.limit)
    print(f"calibrating mixed-vehicle clusters on {len(rows)} vehicles")
    results = []
    hist: Counter[int] = Counter()
    for i, (src, lid, paths) in enumerate(rows, 1):
        rec = count_one(paths)
        rec.update(source_key=src, listing_id=lid, n_photos=len(paths))
        results.append(rec)
        hist[rec["clusters"]] += 1
        print(f"  {i:3d}/{len(rows)} {src}/{lid}  clusters={rec['clusters']} "
              f"seeds={rec['seeds']} whole={rec['whole_vehicle_frames']}")

    counts = [r["clusters"] for r in results]
    flag, fp = choose_threshold(counts)
    demo_clusters = None
    mixed = REPO / "demo" / "rigid_truck"
    if mixed.is_dir():
        demo_paths = sorted(p for p in mixed.iterdir() if p.suffix.lower() in
                            {".jpg", ".jpeg", ".png", ".webp"})
        if demo_paths:
            demo_clusters = count_one(demo_paths)["clusters"]
            print(f"demo/rigid_truck (true-positive check, not training): "
                  f"clusters={demo_clusters}")

    artifact = {
        "min_clusters_to_flag": flag,
        "seed_frames_min": 2,
        "false_positive_rate": round(fp, 4),
        "n_vehicles": len(results),
        "histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "demo_rigid_truck_clusters": demo_clusters,
        "basis": ("measured on known-single corpus vehicles, whole-vehicle "
                  "seeds only" + ("" if not args.limit else
                                  f"; --limit {args.limit}, not enforceable")),
        "fitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    OUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps(artifact, indent=2))
    if flag is None:
        print("no threshold meets the 0.5% false-positive budget; "
              "pricing_blocker will not use this artifact")


if __name__ == "__main__":
    main()
