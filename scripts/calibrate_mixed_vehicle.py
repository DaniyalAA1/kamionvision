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
from app.schema import GateDecision
from app.subject import WHOLE_VEHICLE_VIEWS

OUT = MODELS / "mixed_vehicle_thresholds.json"
GATE_THRESHOLDS = MODELS / "gate_thresholds.json"
# At most 1 of 200 known-single vehicles may sit at or above the threshold.
FP_BUDGET = 0.005
DEMO_CASES = ("rigid_truck", "mixed_vehicles")


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
    report = gate_stage.run(paths)
    whole = sum(1 for c in report.photos
                if c.usable and c.view in WHOLE_VEHICLE_VIEWS)
    return {
        "clusters": int(report.subject_clusters),
        "seeds": int(report.subject_frames),
        "method": report.subject_method,
        "whole_vehicle_frames": whole,
        "body_tag": report.body_tag,
        "body_tag_conf": float(report.body_tag_conf),
        "decision": report.decision.value if isinstance(
            report.decision, GateDecision) else str(report.decision),
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


def summarise_body_tags(results: list[dict], demos: dict) -> dict:
    """False-rigid budget: 0 of 200 known tractors at the blocking confidence."""
    tags = Counter(r["body_tag"] or "unknown" for r in results)
    rigids = [r for r in results if r["body_tag"] == "rigid"]
    max_fp_conf = max((r["body_tag_conf"] for r in rigids), default=0.0)
    demo = demos.get("rigid_truck") or {}
    demo_hit = demo.get("body_tag") == "rigid"
    demo_conf = float(demo.get("body_tag_conf") or 0.0)
    block_conf = None
    if demo_hit:
        # Must sit strictly above every corpus rigid, and at or below the demo.
        floor = round(max_fp_conf + 0.01, 2) if rigids else 0.55
        if demo_conf >= floor:
            block_conf = min(demo_conf, max(floor, 0.55))
            block_conf = round(block_conf, 2)
    blocks = bool(
        block_conf is not None
        and demo_hit
        and demo_conf >= block_conf
        and all(r["body_tag_conf"] < block_conf for r in rigids)
    )
    return {
        "histogram": {k: int(v) for k, v in sorted(tags.items())},
        "corpus_rigid_n": len(rigids),
        "corpus_max_rigid_conf": round(max_fp_conf, 3),
        "demo_rigid_truck_tag": demo.get("body_tag"),
        "demo_rigid_truck_conf": round(demo_conf, 3) if demo else None,
        "block_conf": block_conf if blocks else None,
        "blocks": blocks,
        "basis": ("zero-shot CLIP on whole-vehicle frames; block only if 0 of "
                  "200 known tractors are tagged rigid at block_conf and "
                  "demo/rigid_truck is a hit"),
    }


def write_body_tag_thresholds(body: dict) -> None:
    if not GATE_THRESHOLDS.exists():
        return
    data = json.loads(GATE_THRESHOLDS.read_text(encoding="utf-8"))
    data["body_tag"] = body
    GATE_THRESHOLDS.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"updated {GATE_THRESHOLDS} body_tag.blocks={body['blocks']} "
          f"block_conf={body['block_conf']}")


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
              f"seeds={rec['seeds']} whole={rec['whole_vehicle_frames']} "
              f"body={rec['body_tag']}@{rec['body_tag_conf']:.2f}",
              flush=True)

    counts = [r["clusters"] for r in results]
    flag, fp = choose_threshold(counts)
    demos = {}
    for name in DEMO_CASES:
        folder = REPO / "demo" / name
        if not folder.is_dir():
            continue
        demo_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in
                            {".jpg", ".jpeg", ".png", ".webp"})
        if not demo_paths:
            continue
        rec = count_one(demo_paths)
        demos[name] = rec
        print(f"demo/{name} (true-positive check, not training): "
              f"clusters={rec['clusters']} body={rec['body_tag']}"
              f"@{rec['body_tag_conf']:.3f}")

    body = summarise_body_tags(results, demos)
    artifact = {
        "min_clusters_to_flag": flag,
        "seed_frames_min": 2,
        "false_positive_rate": round(fp, 4),
        "n_vehicles": len(results),
        "histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "demo_rigid_truck_clusters": (demos.get("rigid_truck") or {}).get("clusters"),
        "demo_mixed_vehicles_clusters": (demos.get("mixed_vehicles") or {}).get("clusters"),
        "body_tag": body,
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
    if len(results) == 200 and not args.limit:
        write_body_tag_thresholds(body)


if __name__ == "__main__":
    main()
