"""Delete image files that no manifest row references.

The cleaning pass drops images from the index - corpus duplicates, documents,
photographs of diagnostic screens - but never deletes the files. They then sit
inside `data/images/` belonging to no dataset row, which matters for two
reasons: sharing the bundle ships images that are not part of the dataset, and
the on-disk count of originals no longer matches the count of degraded twins
even though the manifest itself is strictly 1:1.

Idempotent and safe to re-run: it only ever removes files absent from the clean
and degraded indexes, and every image URL stays in data/metadata/sources/ so
download_images.py can restore them.
"""
import json
import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P


def referenced_paths():
    refs = set()
    for index in (P.CLEAN_INDEX, P.DEGRADED_INDEX):
        if index.exists():
            for line in open(index, encoding="utf-8"):
                refs.add(json.loads(line)["path"])
    return refs


def prune(dry_run=False):
    refs = referenced_paths()
    if not refs:
        print("no index found; refusing to prune")
        return 0, 0
    orphans = [f for f in P.IMAGES.rglob("*.jpg") if P.to_bundle(f) not in refs]
    freed = sum(f.stat().st_size for f in orphans)
    if not dry_run:
        for f in orphans:
            f.unlink()
        for d in sorted(P.IMAGES.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    verb = "would prune" if dry_run else "pruned"
    print(f"{verb} {len(orphans)} orphaned files ({freed / 1e9:.2f} GB)")
    return len(orphans), freed


if __name__ == "__main__":
    prune(dry_run="--dry-run" in sys.argv)
