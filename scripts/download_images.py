"""Download listing images to data/images/<source>/<listing_id>/.

Resumable: skips files already on disk with a plausible size. Verifies each
download decodes as an image and records real dimensions, so the cleaning pass
starts from known-good files.

Usage: download_images.py <meta.jsonl> <out_subdir> [max_listings] [--tractor-only]
"""

import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
# Body types that are semi-tractors. Everything else (tipper, mixer, road/
# construction bodies, rigid trucks) is out of scope for this dataset.
TRACTOR_TYPES = {"çekici", "cekici", "tractor", "truck-tractor", "truck tractor"}

lock = threading.Lock()
stats = {"ok": 0, "skip": 0, "fail": 0}


def grab(job):
    url, rel = job
    path = str(P.resolve(rel))
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        with lock:
            stats["skip"] += 1
        return None
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            if r.status_code != 200 or len(r.content) < 2000:
                continue
            tmp = path + ".part"
            with open(tmp, "wb") as fh:
                fh.write(r.content)
            with Image.open(tmp) as im:
                im.verify()
            with Image.open(tmp) as im:
                w, h = im.size
            os.replace(tmp, path)
            with lock:
                stats["ok"] += 1
                done = stats["ok"] + stats["skip"] + stats["fail"]
                if done % 250 == 0:
                    print(f"  {done} ... ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}", flush=True)
            return {"path": rel, "url": url, "width": w, "height": h,
                    "bytes": os.path.getsize(path)}
        except Exception:
            continue
    with lock:
        stats["fail"] += 1
    return None


def main():
    meta, subdir = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
    tractor_only = "--tractor-only" in sys.argv

    records = [json.loads(l) for l in open(meta, encoding="utf-8")]
    if tractor_only:
        before = len(records)
        records = [r for r in records
                   if str(r.get("body_type", "")).strip().lower() in TRACTOR_TYPES]
        print(f"tractor filter: {before} -> {len(records)} listings")
    records = [r for r in records if r.get("image_urls")]
    records.sort(key=lambda r: -r.get("n_images", 0))
    if limit:
        records = records[:limit]

    jobs, index = [], []
    for rec in records:
        d = P.IMAGES / subdir / str(rec["listing_id"])
        d.mkdir(parents=True, exist_ok=True)
        for i, url in enumerate(rec["image_urls"]):
            ext = ".jpg" if ".png" not in url.lower() else ".png"
            rel = P.to_bundle(d / f"{i:03d}{ext}")
            jobs.append((url, rel))
            index.append({"listing_id": rec["listing_id"], "image_index": i,
                          "path": rel, "url": url})

    print(f"{len(records)} listings, {len(jobs)} images -> data/images/{subdir}/")
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(grab, jobs))

    out = P.image_index(subdir)
    kept = 0
    with open(out, "w", encoding="utf-8") as fh:
        for row in index:
            full = str(P.resolve(row["path"]))
            if os.path.exists(full) and os.path.getsize(full) > 5000:
                try:
                    with Image.open(full) as im:
                        row["width"], row["height"] = im.size
                except Exception:
                    continue
                row["bytes"] = os.path.getsize(full)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    print(f"done: ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}")
    print(f"index: {kept} images -> {out}")


if __name__ == "__main__":
    main()
