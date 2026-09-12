"""Generate paired "seller with a phone" variants of the clean listing photos.

The OEM-dealer sources (TruckMarket, SelecTrucks) shoot in reasonable light on a
prepped lot, so their photos under-represent the brief: a seller holding a phone,
bad light, awkward angle, mud on the panel. Rather than throw those photos away,
each one gets a degraded twin carrying the same listing metadata and the same
view tag, plus a record of exactly which corruptions were applied and how hard.

That pairing is the useful part - it is supervision for the quality gate (this
image and that image show the same truck; one is usable, one is not) that no
amount of scraping produces on its own.

Recipes are sampled per image from a seeded RNG, so the output is reproducible.
"""

import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P
import argparse
import collections
import importlib.util
import json
import os
import pathlib
import random
from concurrent.futures import ProcessPoolExecutor

import albumentations as A
import cv2
import numpy as np

# Reuse the exact metric definitions the cleaning pass uses, so a degraded twin's
# score is directly comparable to its clean original rather than a parallel
# implementation that might drift.
_spec = importlib.util.spec_from_file_location(
    "clean_dataset", os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_dataset.py"))
_cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cd)
capture_metrics, quality_score = _cd.capture_metrics, _cd.quality_score

SEED = 20260912

# Each entry: (name, weight, builder). Weights reflect how often the failure
# mode actually shows up in phone photos of vehicles.
def build_ops():
    return [
        ("motion_blur",     1.0, lambda s: A.MotionBlur(blur_limit=(5, 9 + 2 * int(7 * s)), p=1)),
        ("defocus",         0.7, lambda s: A.Defocus(radius=(2, 3 + int(6 * s)), p=1)),
        ("underexposed",    1.0, lambda s: A.RandomBrightnessContrast(
            brightness_limit=(-0.20 - 0.35 * s, -0.10), contrast_limit=(-0.3, -0.05), p=1)),
        ("overexposed",     0.6, lambda s: A.RandomBrightnessContrast(
            brightness_limit=(0.12, 0.20 + 0.35 * s), contrast_limit=(-0.2, 0.1), p=1)),
        ("low_light_noise", 0.9, lambda s: A.ISONoise(
            color_shift=(0.01, 0.02 + 0.06 * s), intensity=(0.2, 0.3 + 0.7 * s), p=1)),
        ("sensor_noise",    0.6, lambda s: A.GaussNoise(std_range=(0.05, 0.08 + 0.20 * s), p=1)),
        ("jpeg_crush",      1.0, lambda s: A.ImageCompression(
            quality_range=(max(8, 40 - int(32 * s)), max(12, 55 - int(32 * s))), p=1)),
        ("downscaled",      0.8, lambda s: A.Downscale(
            scale_range=(max(0.15, 0.55 - 0.4 * s), max(0.25, 0.75 - 0.4 * s)), p=1)),
        ("awkward_angle",   1.0, lambda s: A.Perspective(
            scale=(0.04, 0.06 + 0.14 * s), keep_size=True, p=1)),
        ("sun_glare",       0.5, lambda s: A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5), src_radius=int(120 + 260 * s), p=1)),
        ("harsh_shadow",    0.5, lambda s: A.RandomShadow(p=1)),
        ("mud_spatter",     0.8, lambda s: A.Spatter(mode="mud", p=1)),
        ("rain",            0.3, lambda s: A.RandomRain(p=1)),
        ("colour_cast",     0.7, lambda s: A.ColorJitter(
            brightness=(0.85, 1.05), contrast=(0.8, 1.1),
            saturation=(0.5, 1.0), hue=(-0.06 - 0.06 * s, 0.06 + 0.06 * s), p=1)),
    ]



def _apply(job):
    """Apply one precomputed recipe. Runs in a worker process."""
    row, chosen, severity, jpeg_q, args = job
    img = cv2.imread(str(P.resolve(row["path"])), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > args.max_side:
        sc = args.max_side / max(h, w)
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)

    builders = {o[0]: o[2] for o in build_ops()}
    pipeline = A.Compose([builders[n](severity) for n in chosen])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_bgr = cv2.cvtColor(pipeline(image=rgb)["image"], cv2.COLOR_RGB2BGR)

    out_dir = pathlib.Path(args.out_dir) / row["source_key"] / str(row["listing_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{row['image_index']:03d}_deg.jpg"
    # Re-encode at low quality: the final insult of a messaging-app upload.
    cv2.imwrite(str(dst), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])

    metrics = capture_metrics(out_bgr)
    return {
        "path": P.to_bundle(dst), "variant": "degraded",
        **metrics,
        "capture_quality": quality_score(metrics, out_bgr.shape[1], out_bgr.shape[0]),
        "clean_capture_quality": row.get("capture_quality"),
        "clean_path": row["path"], "listing_id": row["listing_id"],
        "image_index": row["image_index"], "source_key": row["source_key"],
        "view": row.get("view"), "is_whole_vehicle": row.get("is_whole_vehicle"),
        "degradations": chosen, "severity": round(severity, 3),
        "width": out_bgr.shape[1], "height": out_bgr.shape[0],
        "bytes": dst.stat().st_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(P.CLEAN_INDEX))
    ap.add_argument("--out-dir", default=str(P.DEGRADED_DIR))
    ap.add_argument("--out", default=str(P.DEGRADED_INDEX))
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of clean images to generate a twin for")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    rng = random.Random(SEED)
    ops = build_ops()
    names = [o[0] for o in ops]
    weights = [o[1] for o in ops]
    builders = {o[0]: o[2] for o in ops}

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    rng.shuffle(rows)
    rows = rows[:int(len(rows) * args.fraction)]
    print(f"generating degraded twins for {len(rows)} images")

    # Recipes are drawn here, single-threaded from the seeded RNG, so the output
    # is reproducible no matter how many workers apply them.
    jobs = []
    for row in rows:
        severity = rng.triangular(0.15, 1.0, 0.55)
        k = rng.choices([2, 3, 4], weights=[0.4, 0.4, 0.2])[0]
        chosen, pool, pool_w = [], list(names), list(weights)
        for _ in range(k):
            pick = rng.choices(pool, weights=pool_w)[0]
            idx = pool.index(pick)
            pool.pop(idx)
            pool_w.pop(idx)
            chosen.append(pick)
        if "underexposed" in chosen and "overexposed" in chosen:
            chosen.remove("overexposed")
        jobs.append((row, sorted(chosen), severity, rng.randint(45, 80), args))

    counts = collections.Counter()
    written = 0
    with open(args.out, "w", encoding="utf-8") as fh, \
            ProcessPoolExecutor(max_workers=args.workers) as pool_x:
        for i, rec in enumerate(pool_x.map(_apply, jobs, chunksize=16), 1):
            if rec:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts.update(rec["degradations"])
                written += 1
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    print(f"wrote {written} degraded images -> {args.out}")
    print("degradation frequency:", dict(counts.most_common()))


if __name__ == "__main__":
    main()
