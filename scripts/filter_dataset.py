"""Narrow the corpus to photos that are unambiguously tied to one truck, then
keep the better-photographed half of the fleet.

Two filters, in order.

1. **Attribution.** 128 distinct photographs are published on more than one
   listing - a dealer shooting one unit and reusing the frames on other adverts
   that carry their own mileage and asking price. Deduplication already collapsed
   the copies, but it kept each photo under whichever listing happened to be
   processed first, and that choice is arbitrary. Those photos are removed
   outright, because nothing in the metadata ties them to a specific vehicle.

   Only the shared photos go. The affected truck keeps the rest of its gallery,
   which *is* unambiguously its own. Dropping whole trucks instead was measured
   and rejected: it left 5 of 173 Turkish vehicles standing and turned a
   two-market dataset into a US-only one.

2. **Coverage.** Of the vehicles that survive, keep the half with the most
   photos - ranked *within each source*, not across the pool. Ranking is by
   photo count descending, then listing id, so ties on the boundary split
   deterministically and re-running picks the same set.

   Per-source matters because the sources are not photographed alike, and
   because harvest settings differ: Mascus was captured with a 15-photo cap
   while TruckMarket galleries run to 22. A single global cut-off landed at 18
   and silently deleted all 91 Mascus vehicles - the entire independent-dealer
   layer, and the only non-OEM photography in the corpus - as an artefact of
   that cap rather than anything about the trucks. Cutting within each source
   makes every source lose its own worst-photographed half.

Excluded image files are deleted. Every image URL is retained in
data/metadata/sources/, so `download_images.py` restores them.
"""
import collections
import json
import pathlib as _pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P

import imagehash
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 200_000_000
SOURCE_KEYS = ("tr_truckmarket", "us_selectrucks", "mascus")


def _hash(rec):
    try:
        im = ImageOps.exif_transpose(Image.open(P.resolve(rec["path"]))).convert("RGB")
        return rec["source_key"], str(rec["listing_id"]), str(imagehash.phash(im, hash_size=8))
    except Exception:
        return None


def shared_photo_hashes():
    """pHashes that appear under more than one listing, across every download."""
    downloaded = []
    for src in SOURCE_KEYS:
        idx = P.image_index(src)
        if not idx.exists():
            continue
        for line in open(idx, encoding="utf-8"):
            r = json.loads(line)
            r["source_key"] = src
            downloaded.append(r)
    print(f"hashing {len(downloaded)} downloaded originals ...")
    with ProcessPoolExecutor() as pool:
        results = [r for r in pool.map(_hash, downloaded, chunksize=64) if r]
    by_hash = collections.defaultdict(set)
    for src, lid, h in results:
        by_hash[h].add((src, lid))
    return {h for h, listings in by_hash.items() if len(listings) > 1}


def main():
    force = "--force" in sys.argv
    report_now = json.load(open(P.CLEANING_REPORT, encoding="utf-8"))
    if "corpus_filter" in report_now and not force:
        print("This corpus has already been filtered - cleaning_report.json records\n"
              "a corpus_filter block. Re-running would apply the 50% coverage cut a\n"
              "second time and halve the dataset again. Re-run clean_dataset.py first,\n"
              "or pass --force if that is genuinely what you want.")
        return

    shared = shared_photo_hashes()
    print(f"photos published on more than one listing: {len(shared)}")

    clean = [json.loads(l) for l in open(P.CLEAN_INDEX, encoding="utf-8")]
    before_imgs = len(clean)
    before_veh = len({(r["source_key"], str(r["listing_id"])) for r in clean})

    # Filter 1 - attribution.
    unambiguous = [r for r in clean if r.get("phash") not in shared]
    dropped_ambiguous = before_imgs - len(unambiguous)

    # Filter 2 - coverage, applied within each source. Deterministic:
    # count desc, then listing id.
    per_vehicle = collections.Counter(
        (r["source_key"], str(r["listing_id"])) for r in unambiguous)
    by_source = collections.defaultdict(list)
    for key, count in per_vehicle.items():
        by_source[key[0]].append((key, count))

    keep_vehicles, cutoff = set(), {}
    for src, entries in sorted(by_source.items()):
        ranked = sorted(entries, key=lambda kv: (-kv[1], kv[0]))
        n = (len(ranked) + 1) // 2
        keep_vehicles |= {k for k, _ in ranked[:n]}
        cutoff[src] = ranked[n - 1][1] if ranked else 0
        print(f"  {src:16} {len(ranked):>3} vehicles -> keep {n:>3} "
              f"(cut-off {cutoff[src]} photos)")

    kept = [r for r in unambiguous
            if (r["source_key"], str(r["listing_id"])) in keep_vehicles]
    dropped_coverage = len(unambiguous) - len(kept)

    keep_paths = {r["path"] for r in kept}
    removed = []

    # Rewrite the clean index.
    with open(P.CLEAN_INDEX, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Rewrite the degraded index; a twin follows its original.
    kept_deg, survivors = 0, []
    if P.DEGRADED_INDEX.exists():
        rows = [json.loads(l) for l in open(P.DEGRADED_INDEX, encoding="utf-8")]
        survivors = [r for r in rows if r.get("clean_path") in keep_paths]
        kept_deg = len(survivors)
        for r in rows:
            if r.get("clean_path") not in keep_paths:
                removed.append(r["path"])
        with open(P.DEGRADED_INDEX, "w", encoding="utf-8") as fh:
            for r in survivors:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    removed += [r["path"] for r in clean if r["path"] not in keep_paths]

    # Delete the excluded files.
    freed = 0
    for rel in removed:
        f = P.resolve(rel)
        if f.exists():
            freed += f.stat().st_size
            f.unlink()
    # Prune directories the deletions emptied.
    for d in sorted(P.IMAGES.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    print(f"\nvehicles {before_veh} -> {len(keep_vehicles)}")
    print(f"originals {before_imgs} -> {len(kept)}")
    print(f"  dropped, photo shared across listings : {dropped_ambiguous}")
    print(f"  dropped, bottom 50% by photo count    : {dropped_coverage}")
    print(f"degraded twins kept: {kept_deg}")
    print(f"deleted {len(removed)} files, freed {freed / 1e9:.2f} GB")

    report = json.load(open(P.CLEANING_REPORT, encoding="utf-8"))
    report["corpus_filter"] = {
        "attribution": {
            "rule": "drop any photo whose perceptual hash appears under more than one listing",
            "shared_photos": len(shared),
            "images_dropped": dropped_ambiguous,
            "note": ("the affected vehicle keeps its remaining photos; dropping whole "
                     "vehicles was measured and rejected because it left 5 of 173 "
                     "Turkish vehicles and made the corpus US-only"),
        },
        "coverage": {
            "rule": ("keep the top 50% of vehicles by photo count WITHIN EACH SOURCE, "
                     "ties broken by listing id"),
            "vehicles_before": len(per_vehicle),
            "vehicles_kept": len(keep_vehicles),
            "photo_count_cutoff_per_source": cutoff,
            "images_dropped": dropped_coverage,
            "note": ("cutting across the pooled corpus instead deleted all 91 Mascus "
                     "vehicles, because that source was harvested with a 15-photo cap "
                     "and the global cut-off landed at 18"),
        },
        "files_deleted": len(removed),
        "bytes_freed": freed,
        "recovery": "image URLs retained in data/metadata/sources/; re-run download_images.py",
    }
    json.dump(report, open(P.CLEANING_REPORT, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    import prune_orphans
    prune_orphans.prune()
    print("cleaning_report.json updated")


if __name__ == "__main__":
    main()
