"""Apply the human-in-the-loop visual review to the cleaned manifests.

Every original image in the corpus was tiled into numbered contact sheets and
inspected; each cell a reviewer flagged was then re-checked individually at full
resolution before anything was deleted. That second check mattered: half the
flags were wrong. Real Freightliner Cascadias carrying a dealer banner or corner
watermark had been called "placeholder graphics", and a dealer decal
photographed on the truck's own bodywork had been called a "logo". Those are
kept, and annotated instead.

Nothing is ever excluded for photographic quality. Blur, darkness, blown
highlights, mud, glare and bad framing are the point of this dataset.

Removes both the offending original and its degraded twin, from the manifests
and from disk.
"""

import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P
import json
import os

REVIEW = P.REVIEW
CLEAN = P.CLEAN_INDEX
DEGRADED = P.DEGRADED_INDEX
REPORT = P.CLEANING_REPORT


def main():
    review = json.load(open(REVIEW, encoding="utf-8"))
    drop = {e["path"] for e in review["exclude"]}
    overlay = set(review["annotate_marketing_overlay"])
    watermark = set(review["annotate_dealer_watermark"])

    removed_files = []

    def rewrite(path, path_key):
        if not os.path.exists(path):
            return 0, 0
        kept, dropped = [], 0
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            # A degraded twin is dropped when its clean original is dropped.
            if row.get(path_key) in drop:
                dropped += 1
                full = P.resolve(row["path"])
                if full.exists():
                    full.unlink()
                    removed_files.append(row["path"])
                continue
            src = row.get("clean_path") or row.get("path")
            row["marketing_overlay"] = src in overlay
            row["dealer_watermark"] = src in watermark
            kept.append(row)
        with open(path, "w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(kept), dropped

    n_clean, d_clean = rewrite(CLEAN, "path")
    n_deg, d_deg = rewrite(DEGRADED, "clean_path")
    print(f"originals: kept {n_clean}, removed {d_clean}")
    print(f"degraded : kept {n_deg}, removed {d_deg}")
    print(f"files deleted from disk: {len(removed_files)}")

    rep = json.load(open(REPORT, encoding="utf-8"))
    rep["kept"] = n_clean
    rep["dropped"] = rep["candidates"] - n_clean
    rep.setdefault("drop_reasons", {})["drop_manual_review_not_a_truck"] = d_clean
    rep["manual_visual_review"] = {
        "method": ("all originals tiled into 264 numbered 5x5 contact sheets, "
                   "reviewed by 24 parallel reviewers, every flagged cell then "
                   "re-verified individually at full resolution"),
        "images_reviewed": review["images_reviewed"],
        "flagged_by_reviewers": review["cells_flagged_by_reviewers"],
        "confirmed_and_removed": d_clean,
        "reviewer_false_positives_kept": len(review["reviewer_false_positives_kept"]),
        "excluded_for_image_quality": 0,
        "annotations_added": {"marketing_overlay": len(overlay),
                              "dealer_watermark": len(watermark)},
    }
    json.dump(rep, open(REPORT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("cleaning_report.json updated")


if __name__ == "__main__":
    main()
