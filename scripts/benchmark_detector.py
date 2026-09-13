"""Which COCO weight should the gate run, and at what input size?

Set expectations before reading the numbers. Every candidate is COCO-80, so a
bigger model will NOT stop calling a cab-forward tractor a `bus` or a tight cab
shot a `car`. That confusion is fixed by app/subject.py, not by the weights.
What a bigger weight buys is better boxes, fewer cross-class duplicates and
better recall on clipped and partly-occluded subjects - the failure that
produced `vehicle_conf_min = 0.193` and the one vehicle in 200 with no truck
box anywhere.

Per-photo truck-box rate is deliberately NOT the objective. It rewards a model
that finds MORE trucks, which is the direction of the reported bug. Stated
before running, in the style of scripts/probe_residual_signal.py:

    Ship the weight x imgsz that maximises same-listing retrieval precision@1
    on the selected subject crops, subject to (a) part_view_subject_rate no
    higher than the incumbent's, (b) vehicles_with_no_truck_box no higher than
    the incumbent's, and (c) duplicate_vehicle_pairs_per_frame no higher than
    the incumbent's.

Two stages, because the full grid over 7,458 images is hours of MPS:

    stage 1   every arm over a vehicle-grouped subsample of both populations,
              on the cheap constraint metrics from app.calibrate_gate.sweep -
              the same function the shipped calibration runs, so a weight
              comparison is never a comparison of two implementations.
    stage 2   the shortlist plus the incumbent over all 7,458, and retrieval
              P@1 via scripts/eval_subject.py.

    .venv/bin/python scripts/benchmark_detector.py --weights-dir /path/to/pts
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import vision  # noqa: E402
from app.calibrate_gate import sweep  # noqa: E402
from app.config import DATA, IMAGES_CSV, META  # noqa: E402

OUT = META / "detector_benchmark.json"

# Settings held identical across every arm and matching the shipped call:
# conf at the candidate floor, no `classes=` filter (the disqualifier
# vocabulary needs motorcycle/train/boat), agnostic_nms off (a truck box must
# never suppress an overlapping motorcycle or train), max_det bounded.
PREDICT = {"iou": 0.7, "agnostic_nms": False, "max_det": 50, "half": False}

GRID = [
    ("yolov8n.pt", 640),        # incumbent, at its shipped input size
    ("yolov8n.pt", 960),
    ("yolov8n.pt", 1280),
    ("yolov8m.pt", 640),
    ("yolov8m.pt", 960),
    ("yolo11m.pt", 640),
    ("yolo11m.pt", 960),
    ("yolo11m.pt", 1280),
    ("yolo11l.pt", 640),
    ("yolo11l.pt", 960),
]


def rows_for(im: pd.DataFrame, frac: float, seed: int) -> list[tuple]:
    """Vehicle-grouped subsample: every one of the 200 vehicles keeps a share
    of its frames, so `vehicles_with_no_truck_box` stays interpretable."""
    rng = random.Random(seed)
    out = []
    for _, group in im.groupby(["listing_id", "variant"]):
        rows = list(group.itertuples())
        if frac < 1.0:
            k = max(1, int(round(len(rows) * frac)))
            rows = rng.sample(rows, k)
        out.extend((str(r.listing_id), str(DATA / str(r.path)),
                    str(r.view), float(r.view_conf)) for r in rows)
    out.sort(key=lambda t: t[1])
    return out


def load(weight: str, weights_dir: Path | None):
    from ultralytics import YOLO
    for candidate in ((weights_dir / weight) if weights_dir else None,
                      Path("models") / weight):
        if candidate and candidate.exists():
            return YOLO(str(candidate))
    return YOLO(weight)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-dir", type=Path, default=None)
    ap.add_argument("--stage1-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    print(__doc__.strip().splitlines()[0])
    print("objective: same-listing retrieval P@1, subject to part_view_subject_rate,")
    print("           vehicles_with_no_truck_box and duplicate pairs not rising.\n")

    im = pd.read_csv(IMAGES_CSV)
    rows = rows_for(im, args.stage1_frac, args.seed)
    print(f"stage 1: {len(rows)} frames "
          f"({args.stage1_frac:.0%} of every vehicle, both variants), "
          f"{len(GRID)} arms\n")

    results = []
    for weight, imgsz in GRID:
        yolo = load(weight, args.weights_dir)
        t0 = time.time()
        stats = sweep(yolo, rows, conf=vision_candidate_conf(), imgsz=imgsz, **PREDICT)
        stats.update({"weights": weight, "imgsz": imgsz,
                      "wall_s": round(time.time() - t0, 1)})
        results.append(stats)
        print(f"{weight:12s} @{imgsz:5d}  "
              f"truck_box@.25 {stats['photo_frac_truck_box_at_0.25']:.3f}  "
              f"no_box_vehicles {stats['vehicles_with_no_truck_box_at_0.25']:3d}  "
              f"dup_pairs {stats['duplicate_vehicle_pairs_per_frame']:.3f}  "
              f"part_q90 {stats['part_view_area_q90']}  "
              f"{stats['wall_s']:.0f}s", flush=True)

    base = next(r for r in results if r["weights"] == "yolov8n.pt" and r["imgsz"] == 640)
    for r in results:
        r["passes_constraints"] = bool(
            r["vehicles_with_no_truck_box_at_0.25"] <= base["vehicles_with_no_truck_box_at_0.25"]
            and r["duplicate_vehicle_pairs_per_frame"] <= base["duplicate_vehicle_pairs_per_frame"])

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "objective": ("same-listing retrieval P@1 on selected subject crops, subject to "
                      "part_view_subject_rate, vehicles_with_no_truck_box and "
                      "duplicate_vehicle_pairs_per_frame not rising above the incumbent"),
        "note": ("Stage 1 only decides the shortlist. Per-photo truck-box rate is "
                 "reported because it is the number in gate_thresholds.json and a "
                 "reader will look for it, but it is not the objective: it rewards a "
                 "model that finds more trucks, which is the direction of the bug."),
        "predict_kwargs": PREDICT,
        "stage1_frac": args.stage1_frac,
        "stage1_frames": len(rows),
        "incumbent": {"weights": "yolov8n.pt", "imgsz": 640},
        "stage1": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    short = sorted([r for r in results if r["passes_constraints"]],
                   key=lambda r: -r["photo_frac_truck_box_at_0.25"])[:3]
    print("shortlist for stage 2: "
          + ", ".join(f"{r['weights']}@{r['imgsz']}" for r in short))


def vision_candidate_conf() -> float:
    from app import subject
    return subject.CANDIDATE_CONF


if __name__ == "__main__":
    main()
