"""Did the new subject-grounding actually pick better boxes?

A diagnostic in the mould of scripts/probe_residual_signal.py: it decides
nothing, it measures. Every metric runs twice, `--algorithm old` and
`--algorithm new`, over the same detector output on the same rows, so the
algorithm delta and any future weight delta are never confounded.

    .venv/bin/python scripts/eval_subject.py --algorithm old
    .venv/bin/python scripts/eval_subject.py --algorithm new
    .venv/bin/python scripts/eval_subject.py --calibrate-rescue

The corpus is 100% trucks and carries no negatives, but the per-listing
supervision lets two labelled families be constructed for free:

  distractor injection    two listings' whole-vehicle frames pasted side by
                          side; the correct box is on the half belonging to
                          the set the other frames come from.
  background-truck        a small whole-truck crop pasted into the top fifth
  injection               of a close-up; the correct answer is NO box at all.
                          This is the reported bug, reproduced on demand.

Primary objective, stated before running, in the style of the residual probe:

    Ship the algorithm that maximises same-listing retrieval precision@1 on
    the selected subject crops, subject to `whole_vehicle_miss_rate` no
    higher than the incumbent's.

Retrieval P@1 is label-free and uses the per-listing same-truck supervision
exactly: if the algorithm crops the neighbour's lorry on photo 7, that crop's
nearest neighbour is some *other* listing's truck.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import subject as subject_stage  # noqa: E402
from app import vision  # noqa: E402
from app.config import DATA, IMAGES_CSV, META  # noqa: E402
from app.evidence import passes  # noqa: E402
from app.schema import Detection, PhotoCheck  # noqa: E402

OUT = META / "subject_eval.json"
CANDIDATE_CONF = subject_stage.CANDIDATE_CONF
DETECTION_CONF = subject_stage.DETECTION_CONF


# --- the shipped algorithm, kept verbatim so the delta is honest -----------

OLD_CENTRE_BIAS = 0.55


def old_pick_subject(check: PhotoCheck) -> list[float] | None:
    """app/gate.py before this workstream: a box holding the centre of the
    frame wins outright, area x centrality only breaks ties among those."""
    boxes = [d for d in check.detections if d.label in ("truck", "bus")]
    if not boxes:
        return None
    cx, cy = check.width / 2, check.height / 2
    centred = [d for d in boxes
               if d.box[0] <= cx <= d.box[2] and d.box[1] <= cy <= d.box[3]]
    pool = centred or boxes

    def score(box):
        x1, y1, x2, y2 = box
        frame = float(check.width * check.height) or 1.0
        area = abs((x2 - x1) * (y2 - y1)) / frame
        bx, by = (x1 + x2) / 2 / (check.width or 1), (y1 + y2) / 2 / (check.height or 1)
        far = ((bx - 0.5) ** 2 + (by - 0.5) ** 2) ** 0.5 / (0.5 * 2 ** 0.5)
        return area * (1.0 - OLD_CENTRE_BIAS * min(1.0, far))

    return list(max(pool, key=lambda d: score(d.box)).box)


def old_competing_vehicles(check: PhotoCheck) -> int:
    subject = check.subject_box
    return sum(1 for d in check.detections
               if d.label in vision.COCO_VEHICLES and d.confidence >= 0.4
               and d.area_frac >= 0.02 and list(d.box) != subject)


def old_wants_crop(check: PhotoCheck) -> bool:
    box = check.subject_box
    if not box or not check.width or not check.height:
        return False
    x1, y1, x2, y2 = box
    frac = abs((x2 - x1) * (y2 - y1)) / float(check.width * check.height)
    if frac >= passes.CROP_MAX_SUBJECT_FRAC:
        return False
    return old_competing_vehicles(check) >= 1


# --- the detector + CLIP pass, shared by both algorithms -------------------

def analyse(paths: list[Path], batch: int = 32) -> list[PhotoCheck]:
    """Detections and view tags for every path. No selection, no verdict."""
    yolo, clip = vision.yolo(), vision.clip()
    checks: list[PhotoCheck] = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        pils, ok = [], []
        for p in chunk:
            try:
                pils.append(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
                ok.append(p)
            except Exception:
                continue
        if not pils:
            continue
        preds = yolo.predict(pils, verbose=False, conf=0.05, device=vision.device())
        tags = clip.tag(pils)
        for p, pil, r, tag in zip(ok, pils, preds, tags):
            c = PhotoCheck(photo_id=len(checks), path=str(p), filename=p.name)
            c.width, c.height = pil.size
            frame = float(pil.width * pil.height) or 1.0
            for cls, f, box in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
                label, conf = r.names[int(cls)], float(f)
                x1, y1, x2, y2 = (float(v) for v in box)
                if conf < CANDIDATE_CONF:
                    continue
                c.detections.append(Detection(
                    label=label, confidence=round(conf, 3),
                    box=[round(v, 1) for v in (x1, y1, x2, y2)],
                    area_frac=round(abs((x2 - x1) * (y2 - y1)) / frame, 4)))
            c.view, c.view_conf = tag["view"], round(tag["view_conf"], 3)
            c.content, c.content_conf = tag["content"], round(tag["content_conf"], 3)
            c.keep_mass = round(tag["keep_mass"], 3)
            checks.append(c)
    return checks


def _split(checks: list[PhotoCheck]) -> None:
    """Move the sub-threshold vehicle boxes off `detections` onto the candidate
    pool, which is what gate.inspect does. The old algorithm never sees them."""
    for c in checks:
        faint = [d for d in c.detections if d.confidence < DETECTION_CONF]
        c.detections = [d for d in c.detections if d.confidence >= DETECTION_CONF]
        c._candidates = subject_stage.build_candidates(
            c, [d for d in faint if d.label in subject_stage.SUBJECT_LABELS])


def select(checks: list[PhotoCheck], algorithm: str, embed=None) -> None:
    """Run one algorithm over one listing's frames, in place."""
    if algorithm == "old":
        for c in checks:
            for d in c.detections:
                d.is_subject = False
            c.subject_box = old_pick_subject(c)
            c.subject_basis = ""
            for d in c.detections:
                if c.subject_box is not None and list(d.box) == c.subject_box:
                    d.is_subject = True
                    break
    else:
        subject_stage.ground(checks, embed=embed)


def cropped(check: PhotoCheck, algorithm: str) -> bool:
    return old_wants_crop(check) if algorithm == "old" else passes.wants_crop(check)


# --- metrics ---------------------------------------------------------------

PART, WHOLE = subject_stage.TRUCK_PART_VIEWS, subject_stage.WHOLE_VEHICLE_VIEWS
MIN_VIEW_CONF = subject_stage.MIN_PART_VIEW_CONF


def crop_pil(pil: Image.Image, box, pad: float = 0.08) -> Image.Image:
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return pil.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                     min(pil.width, int(x2 + px)), min(pil.height, int(y2 + py))))


def rates(rows: list[dict], scenery_floor: float) -> dict:
    part = [r for r in rows if r["view"] in PART and r["view_conf"] >= MIN_VIEW_CONF]
    whole = [r for r in rows if r["view"] in WHOLE and r["view_conf"] >= MIN_VIEW_CONF]
    by_vehicle: dict[str, int] = {}
    for r in rows:
        by_vehicle[r["listing_id"]] = by_vehicle.get(r["listing_id"], 0) + bool(r["box"])
    return {
        "frames": len(rows),
        "vehicles": len(by_vehicle),
        # The reported bug, as a number, and the sharper version of it: a box
        # smaller than 95% of genuine whole-vehicle subject boxes on a frame
        # that is a close-up of one component is the yard behind it.
        "part_view_subject_rate": round(sum(1 for r in part if r["box"]) / max(1, len(part)), 4),
        "part_view_scenery_rate": round(
            sum(1 for r in part if r["box"] and r["area"] < scenery_floor) / max(1, len(part)), 4),
        "part_view_crop_rate": round(sum(1 for r in part if r["crop"]) / max(1, len(part)), 4),
        "part_view_frames": len(part),
        # The opposite failure, and the one with no headroom: the control is
        # already at 97.5% boxed, so nothing may be bought by trading it away.
        "whole_vehicle_miss_rate": round(
            sum(1 for r in whole if not r["box"]) / max(1, len(whole)), 4),
        "whole_vehicle_crop_rate": round(
            sum(1 for r in whole if r["crop"]) / max(1, len(whole)), 4),
        "whole_vehicle_frames": len(whole),
        "view_agreement": round(
            sum(1 for r in part + whole
                if bool(r["box"]) == (r["view"] in WHOLE)) / max(1, len(part) + len(whole)), 4),
        "vehicles_with_no_subject_anywhere": int(sum(1 for v in by_vehicle.values() if not v)),
        "crop_rate": round(sum(1 for r in rows if r["crop"]) / max(1, len(rows)), 4),
        "subject_rate": round(sum(1 for r in rows if r["box"]) / max(1, len(rows)), 4),
    }


def retrieval(embs: np.ndarray, listing: list[str], phash: list[str] | None,
              hamming_floor: int = 8) -> dict:
    """Same-listing precision@1 over the selected subject crops."""
    if len(embs) < 2:
        return {"p_at_1": 0.0, "n": len(embs)}
    sim = embs @ embs.T
    np.fill_diagonal(sim, -2.0)
    hit = sim.argmax(axis=1)
    p1 = float(np.mean([listing[i] == listing[j] for i, j in enumerate(hit)]))
    out = {"p_at_1": round(p1, 4), "n": int(len(embs))}
    if phash is None:
        return out
    # Hardened: a same-listing neighbour taken one second later is a freebie.
    same = np.array([[listing[i] == listing[j] for j in range(len(listing))]
                     for i in range(len(listing))])
    near = np.array([[_hamming(phash[i], phash[j]) < hamming_floor for j in range(len(phash))]
                     for i in range(len(phash))])
    masked = sim.copy()
    masked[same & near] = -2.0
    hit2 = masked.argmax(axis=1)
    out["p_at_1_phash_hardened"] = round(
        float(np.mean([listing[i] == listing[j] for i, j in enumerate(hit2)])), 4)
    return out


def _hamming(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except (TypeError, ValueError):
        return 64


def consistency(embs: np.ndarray, listing: list[str], seed: int = 7,
                foils: int = 20) -> dict:
    """Within-listing agreement, and its control against other listings."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for i, lid in enumerate(listing):
        groups.setdefault(lid, []).append(i)
    keys = [k for k, v in groups.items() if len(v) >= 2]
    cons, sep = [], []
    for k in keys:
        idx = groups[k]
        block = embs[idx] @ embs[idx].T
        iu = np.triu_indices(len(idx), k=1)
        c = float(block[iu].mean())
        cons.append(c)
        others = [o for o in groups if o != k]
        if not others:
            continue
        foil = np.mean([float((embs[idx] @ embs[groups[rng.choice(others)]].T).mean())
                        for _ in range(min(foils, len(others)))])
        sep.append(c - float(foil))
    return {"consistency": round(float(np.mean(cons)), 4) if cons else 0.0,
            "separation": round(float(np.mean(sep)), 4) if sep else 0.0,
            "listings": len(keys)}


# --- the two synthetic families -------------------------------------------

def build_composites(im: pd.DataFrame, out_dir: Path, n: int, seed: int) -> dict:
    """Ground truth the corpus does not carry, constructed from the per-listing
    same-truck supervision with a few lines of PIL."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    whole = im[im.view.isin(WHOLE) & (im.view_conf >= MIN_VIEW_CONF)]
    part = im[im.view.isin(PART) & (im.view_conf >= MIN_VIEW_CONF)]
    by_whole = {k: list(v.path) for k, v in whole.groupby("listing_id")}
    by_part = {k: list(v.path) for k, v in part.groupby("listing_id")}
    usable = [k for k in by_whole if len(by_whole[k]) >= 3]
    rng.shuffle(usable)
    distractor, background = [], []
    for lid in usable[:n]:
        foil = rng.choice([k for k in usable if k != lid])
        a = DATA / rng.choice(by_whole[lid])
        b = DATA / rng.choice(by_whole[foil])
        if not (a.exists() and b.exists()):
            continue
        pa = ImageOps.exif_transpose(Image.open(a)).convert("RGB").resize((800, 600))
        pb = ImageOps.exif_transpose(Image.open(b)).convert("RGB").resize((800, 600))
        canvas = Image.new("RGB", (1600, 600))
        left = rng.random() < 0.5
        canvas.paste(pa if left else pb, (0, 0))
        canvas.paste(pb if left else pa, (800, 0))
        p = out_dir / f"distract_{lid}.jpg"
        canvas.save(p, quality=88)
        distractor.append({"listing_id": str(lid), "path": str(p),
                           "context": [str(DATA / q)
                                       for q in by_whole[lid] if str(DATA / q) != str(a)][:8],
                           "correct_half": "left" if left else "right"})

        if lid not in by_part:
            continue
        base = DATA / rng.choice(by_part[lid])
        if not base.exists():
            continue
        pc = ImageOps.exif_transpose(Image.open(base)).convert("RGB").resize((1200, 900))
        tw = pb.resize((int(1200 * 0.16), int(1200 * 0.16 * 0.75)))
        pc.paste(tw, (rng.randint(0, 1200 - tw.width), rng.randint(0, int(900 * 0.2))))
        p = out_dir / f"background_{lid}.jpg"
        pc.save(p, quality=88)
        background.append({"listing_id": str(lid), "path": str(p),
                           "context": [str(DATA / q) for q in by_whole[lid]][:8]})
    return {"distractor": distractor, "background": background}


def run_composites(cases: dict, algorithm: str, embed) -> dict:
    """distractor_resisted_rate and background_truck_rejected_rate."""
    resisted = total_d = 0
    for case in cases["distractor"]:
        checks = analyse([Path(p) for p in case["context"]] + [Path(case["path"])])
        if not checks or checks[-1].width != 1600:
            continue
        _split(checks)
        select(checks, algorithm, embed=embed)
        box = checks[-1].subject_box
        total_d += 1
        if box is None:
            continue
        mid = (box[0] + box[2]) / 2
        on_left = mid < 800
        resisted += int(on_left == (case["correct_half"] == "left"))
    rejected = total_b = 0
    for case in cases["background"]:
        checks = analyse([Path(p) for p in case["context"]] + [Path(case["path"])])
        if not checks:
            continue
        _split(checks)
        select(checks, algorithm, embed=embed)
        total_b += 1
        rejected += int(checks[-1].subject_box is None)
    return {
        "distractor_resisted_rate": round(resisted / max(1, total_d), 4),
        "distractor_n": total_d,
        "background_truck_rejected_rate": round(rejected / max(1, total_b), 4),
        "background_n": total_b,
    }


# --- rescue-cosine calibration --------------------------------------------

def calibrate_rescue(bundles: list[tuple[str, list[PhotoCheck]]], embed,
                     false_rescue: float = 0.05) -> dict:
    """The two distributions the absolute-cosine rescue clauses rest on.

    A candidate crop against its OWN set's prototype, and against another
    set's. The threshold goes where the second distribution gives at most 5%
    false rescue. If they do not separate, the clauses are disabled outright:
    an unmeasured absolute CLIP cosine is the single most likely way for this
    to misbehave quietly.
    """
    protos: dict[str, np.ndarray] = {}
    crops: dict[str, list[np.ndarray]] = {}
    for lid, checks in bundles:
        identity = subject_stage.ground(checks, embed=embed)
        if identity.prototype is None:
            continue
        protos[lid] = identity.prototype
        crops[lid] = [c.emb for ch in checks for c in (ch._candidates or [])
                      if c.emb is not None]
    keys = [k for k in protos if crops.get(k)]
    if len(keys) < 5:
        return {"separable": False, "reason": f"only {len(keys)} listings produced a prototype"}
    rng = random.Random(11)
    same = np.concatenate([np.stack(crops[k]) @ protos[k] for k in keys])
    other = np.concatenate([np.stack(crops[k]) @ protos[rng.choice([o for o in keys if o != k])]
                            for k in keys])
    cut = float(np.quantile(other, 1 - false_rescue))
    recall = float((same >= cut).mean())
    return {
        "separable": bool(recall >= 0.25 and cut < 0.999),
        "threshold_at_5pct_false_rescue": round(cut, 4),
        "true_rescue_rate_at_that_threshold": round(recall, 4),
        "same_listing": {"n": int(same.size), "mean": round(float(same.mean()), 4),
                         "q05": round(float(np.quantile(same, 0.05)), 4),
                         "q50": round(float(np.quantile(same, 0.50)), 4),
                         "q95": round(float(np.quantile(same, 0.95)), 4)},
        "other_listing": {"n": int(other.size), "mean": round(float(other.mean()), 4),
                          "q50": round(float(np.quantile(other, 0.50)), 4),
                          "q95": round(float(np.quantile(other, 0.95)), 4),
                          "q99": round(float(np.quantile(other, 0.99)), 4)},
    }


# --- main ------------------------------------------------------------------

def load(im: pd.DataFrame, limit: int | None, cache: Path | None,
         seed: int) -> list[tuple[str, list[PhotoCheck]]]:
    if cache and cache.exists():
        with cache.open("rb") as fh:
            bundles = pickle.load(fh)
        print(f"cache hit: {len(bundles)} listings from {cache}", flush=True)
        return bundles[:limit] if limit else bundles
    listings = sorted(im.listing_id.astype(str).unique())
    if limit:
        random.Random(seed).shuffle(listings)
        listings = listings[:limit]
    bundles = []
    t0 = time.time()
    for i, lid in enumerate(listings, 1):
        rows = im[im.listing_id.astype(str) == lid]
        paths = [DATA / str(p) for p in rows.path]
        checks = analyse([p for p in paths if p.exists()])
        if checks:
            bundles.append((lid, checks))
        if i % 20 == 0:
            print(f"  {i}/{len(listings)} listings, {time.time() - t0:.0f}s", flush=True)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as fh:
            pickle.dump(bundles, fh)
    return bundles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", choices=("old", "new"), default="new")
    ap.add_argument("--limit", type=int, default=None, help="listings, for a quick pass")
    ap.add_argument("--cache", type=Path, default=None,
                    help="pickle of the detector+CLIP pass, shared across arms")
    ap.add_argument("--composites", type=Path, default=None)
    ap.add_argument("--composite-n", type=int, default=200)
    ap.add_argument("--skip-composites", action="store_true")
    ap.add_argument("--calibrate-rescue", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    print(__doc__.strip().splitlines()[0])
    im = pd.read_csv(IMAGES_CSV)
    im = im[im.variant == "original"]
    bundles = load(im, args.limit, args.cache, args.seed)
    embed = vision.clip().embed if args.algorithm == "new" else None

    if args.calibrate_rescue:
        for _, checks in bundles:
            _split(checks)
        result = calibrate_rescue(bundles, embed=vision.clip().embed)
        print(json.dumps(result, indent=2))
        return

    phash_by_path = {str(DATA / str(r.path)): str(r.phash) for r in im.itertuples()}

    rows, crop_jobs, crop_meta = [], [], []
    # MEASURED: q05 of the box area over genuine whole-vehicle frames. A box
    # smaller than 95% of real subject boxes, on a frame that is a close-up of
    # one component, is the yard behind it.
    scenery_floor = (subject_stage.thresholds()["detector"]["measured"]["originals"]
                     .get("whole_vehicle_area_q05_of_boxed") or 0.31)
    t0 = time.time()
    for lid, checks in bundles:
        _split(checks)
        select(checks, args.algorithm, embed=embed)
        for c in checks:
            area = 0.0
            if c.subject_box:
                x1, y1, x2, y2 = c.subject_box
                area = abs((x2 - x1) * (y2 - y1)) / float(c.width * c.height or 1)
            rows.append({"listing_id": lid, "path": c.path, "view": c.view,
                         "view_conf": c.view_conf, "box": c.subject_box,
                         "area": area, "crop": bool(cropped(c, args.algorithm))})
            if c.subject_box:
                crop_meta.append((lid, c.path))
                crop_jobs.append((c.path, list(c.subject_box)))
    print(f"selection over {len(rows)} frames in {time.time() - t0:.0f}s", flush=True)

    embs = []
    clip = vision.clip()
    for i in range(0, len(crop_jobs), 64):
        pils = []
        for path, box in crop_jobs[i:i + 64]:
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            pils.append(crop_pil(pil, box))
        embs.append(clip.embed(pils))
    matrix = np.concatenate(embs) if embs else np.zeros((0, 512), dtype=np.float32)
    listing = [lid for lid, _ in crop_meta]
    phash = [phash_by_path.get(p, "") for _, p in crop_meta]

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "algorithm": args.algorithm,
        "objective": ("same-listing retrieval precision@1 on selected subject crops, "
                      "subject to whole_vehicle_miss_rate no higher than the incumbent"),
        "detector": vision.yolo_name() if hasattr(vision, "yolo_name") else "yolov8n.pt",
        "retrieval": retrieval(matrix, listing, phash),
        "within_listing": consistency(matrix, listing),
        "rates": rates(rows, scenery_floor),
    }

    if not args.skip_composites:
        comp_dir = args.composites or (Path(args.cache).parent / "composites"
                                       if args.cache else Path("/tmp/kv_composites"))
        cases_file = Path(comp_dir) / "cases.json"
        if cases_file.exists():
            cases = json.loads(cases_file.read_text(encoding="utf-8"))
        else:
            cases = build_composites(im, Path(comp_dir), args.composite_n, args.seed)
            cases_file.write_text(json.dumps(cases, indent=1))
        payload["composites"] = run_composites(cases, args.algorithm, embed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    existing[args.algorithm] = payload
    args.out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
