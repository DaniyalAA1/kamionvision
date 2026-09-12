"""Derive the gate's reject thresholds from the corpus instead of guessing them.

The brief's input is "a seller with a phone and no photography skills", and
`scripts/degrade_images.py` already built a 1:1 degraded twin of every clean
photo to stand in for exactly that. That gives a labelled pair of populations:

    originals  photos a dealer took on a prepped lot  -> should pass
    degraded   the same trucks, phone-quality          -> should mostly pass,
               with only the genuinely illegible tail rejected

So each threshold is set at a stated quantile of the *degraded* population and
then reported against both, which turns "photo too blurry" from a magic number
into a measured false-reject rate. Written to models/gate_thresholds.json.

    .venv/bin/python -m app.calibrate_gate
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GATE_THRESHOLDS, IMAGES, IMAGES_CSV, MODELS

# Quantile of the degraded population each floor is placed at. 0.04 says: we
# accept losing the worst 4% of phone-quality photos to buy a hard refusal
# path for genuinely unreadable input.
REJECT_QUANTILE = 0.04
WARN_QUANTILE = 0.15

# YOLO sample size per population. The detector is the expensive part of
# calibration; 500 photos is enough to pin the operating point to ~2%.
YOLO_SAMPLE = 500

# A truck box has to fill at least this much of the frame to count as the
# subject of the photo rather than scenery behind it.
MIN_TRUCK_AREA_FRAC = 0.12


def _pct(series: pd.Series, predicate) -> float:
    return round(100.0 * float(predicate(series).mean()), 2)


def main() -> None:
    im = pd.read_csv(IMAGES_CSV)
    orig = im[im.variant == "original"]
    degr = im[im.variant == "degraded"]

    blur_floor = float(np.round(degr.blur_laplacian_var.quantile(REJECT_QUANTILE), 1))
    blur_warn = float(np.round(degr.blur_laplacian_var.quantile(WARN_QUANTILE), 1))
    dark_floor = float(np.round(degr.brightness.quantile(REJECT_QUANTILE), 4))
    bright_ceil = float(np.round(degr.brightness.quantile(1 - REJECT_QUANTILE), 4))
    contrast_floor = float(np.round(degr.contrast_rms.quantile(REJECT_QUANTILE), 4))
    dark_clip_ceil = float(np.round(degr.dark_clipped_frac.quantile(1 - REJECT_QUANTILE), 4))
    bright_clip_ceil = float(np.round(degr.bright_clipped_frac.quantile(1 - REJECT_QUANTILE), 4))

    thresholds = {
        "blur_laplacian_var_floor": blur_floor,
        "blur_laplacian_var_warn": blur_warn,
        "brightness_floor": dark_floor,
        "brightness_ceiling": bright_ceil,
        "contrast_rms_floor": contrast_floor,
        "dark_clipped_frac_ceiling": dark_clip_ceil,
        "bright_clipped_frac_ceiling": bright_clip_ceil,
    }

    def unusable(df: pd.DataFrame) -> pd.Series:
        return (
            (df.blur_laplacian_var < blur_floor)
            | (df.brightness < dark_floor)
            | (df.brightness > bright_ceil)
            | (df.contrast_rms < contrast_floor)
            | (df.dark_clipped_frac > dark_clip_ceil)
            | (df.bright_clipped_frac > bright_clip_ceil)
        )

    measured = {
        "originals_n": int(len(orig)),
        "degraded_n": int(len(degr)),
        "originals_rejected_pct": _pct(orig, unusable),
        "degraded_rejected_pct": _pct(degr, unusable),
    }
    # Per-vehicle is the number that actually matters: a whole photo set is
    # only refused when every frame in it fails, so report that rate too.
    for name, df in (("originals", orig), ("degraded", degr)):
        flag = unusable(df)
        per_vehicle = df.assign(_bad=flag).groupby("listing_id")._bad.mean()
        measured[f"{name}_vehicles_fully_rejected"] = int((per_vehicle == 1.0).sum())
        measured[f"{name}_vehicles_n"] = int(per_vehicle.size)

    # --- detector operating point ----------------------------------------
    # Run over EVERY image, grouped by vehicle. The sampled per-photo rate is
    # not the quantity of interest: the gate refuses a photo SET, so the
    # number to defend on stage is "how many real vehicles did it refuse".
    from . import vision
    from .vision import COCO_VEHICLES
    yolo = vision.yolo()
    det_stats: dict = {}
    for name, df in (("originals", orig), ("degraded", degr)):
        t0 = time.time()
        rows = [(str(r.listing_id), str(IMAGES.parent / str(r.path))) for r in df.itertuples()]
        rows = [(lid, p) for lid, p in rows if Path(p).exists()]
        best_per_photo: list[float] = []
        best_per_vehicle: dict[str, float] = {}
        for i in range(0, len(rows), 64):
            chunk = rows[i:i + 64]
            preds = yolo.predict([p for _, p in chunk], verbose=False, conf=0.05,
                                 device=vision.device())
            for (lid, _), r in zip(chunk, preds):
                names = r.names
                h, w = r.orig_shape
                frame_area = float(h * w) or 1.0
                # Same dominance rule the gate ships: a truck box only counts
                # when it is big enough to be the subject and is not smaller
                # than a competing vehicle in the same frame.
                best, best_area, competing = 0.0, 0.0, 0.0
                for c, f, box in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
                    label, conf = names[int(c)], float(f)
                    x1, y1, x2, y2 = (float(v) for v in box)
                    area = abs((x2 - x1) * (y2 - y1)) / frame_area
                    if label in ("truck", "bus"):
                        if conf > best:
                            best, best_area = conf, area
                    elif label in COCO_VEHICLES and conf >= 0.5:
                        competing = max(competing, area)
                if best_area < MIN_TRUCK_AREA_FRAC or best_area < competing:
                    best = 0.0
                best_per_photo.append(best)
                best_per_vehicle[lid] = max(best_per_vehicle.get(lid, 0.0), best)
        arr = np.array(best_per_photo) if best_per_photo else np.zeros(1)
        veh = np.array(list(best_per_vehicle.values())) if best_per_vehicle else np.zeros(1)
        det_stats[name] = {
            "photos": len(arr),
            "vehicles": len(veh),
            "seconds": round(time.time() - t0, 1),
            "photo_frac_truck_box_at_0.25": round(float((arr >= 0.25).mean()), 3),
            "photo_frac_truck_box_at_0.50": round(float((arr >= 0.50).mean()), 3),
            # The gate's actual false-refusal rate on known-real trucks.
            "vehicles_with_no_truck_box_at_0.25": int((veh < 0.25).sum()),
            "vehicles_with_no_truck_box_at_0.50": int((veh < 0.50).sum()),
            "vehicle_conf_min": round(float(veh.min()), 3),
            "vehicle_conf_q05": round(float(np.quantile(veh, 0.05)), 3),
        }

    # Per-photo truck detection is NOT a rejection rule: a tire close-up has no
    # truck-shaped object in it and is still a photo of the truck. The rule is
    # set-level - somewhere in the upload there must be a frame with a truck
    # box. These are the operating points for that decision.
    detector = {
        "truck_conf_confirm": 0.50,
        "truck_conf_weak": 0.25,
        "disqualify_conf": 0.55,
        "min_truck_area_frac": MIN_TRUCK_AREA_FRAC,
        "disqualify_area_frac": 0.20,
        "note": ("Set-level, not per-photo. A close-up of a tire or a dashboard "
                 "legitimately contains no truck box and is never rejected on "
                 "that basis; the photo SET is refused only when no frame "
                 "anywhere carries a truck/bus box above truck_conf_weak."),
        "measured": det_stats,
    }

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": __doc__.strip().splitlines()[0],
        "reject_quantile_of_degraded": REJECT_QUANTILE,
        "warn_quantile_of_degraded": WARN_QUANTILE,
        "capture": thresholds,
        "capture_measured": measured,
        "detector": detector,
        "coverage": {
            # Which canonical views a buyer actually needs before a number is
            # defensible. Ranked by what a Kamion buyer would drive six hours
            # to check; the first three are hard requirements for a full price.
            "required": ["exterior_front_34", "tire_wheel", "dashboard_odometer"],
            "wanted": ["interior_cab", "exterior_side", "chassis_undercarriage", "fifth_wheel"],
            "min_photos": 3,
            "min_usable_photos": 2,
        },
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    GATE_THRESHOLDS.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {GATE_THRESHOLDS}")
    print(json.dumps({k: payload[k] for k in ("capture", "capture_measured")}, indent=2))
    print(json.dumps(det_stats, indent=2))


if __name__ == "__main__":
    main()
