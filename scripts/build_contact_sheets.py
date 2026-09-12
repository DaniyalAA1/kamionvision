"""Tile every original image into numbered contact sheets for visual review.

Each sheet is a 5x5 grid of 25 images. Every cell is stamped with its index in a
high-contrast corner badge so a reviewer can name a specific cell unambiguously.

Cells are letterboxed onto a neutral grey canvas rather than cropped: cropping
would hide exactly the framing problems (truck half out of frame, subject at the
edge) that matter here. Aspect ratio is preserved for the same reason.
"""

import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P
import json
import os
import sys

from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = 200_000_000

CELL = 360
GRID = 5
PER_SHEET = GRID * GRID
PAD = 4
BG = (48, 48, 48)
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "sheets"
MANIFEST = P.CLEAN_INDEX


def main():
    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    # Group by source then listing so a reviewer sees coherent context per sheet.
    rows.sort(key=lambda r: (r["source_key"], str(r["listing_id"]), r["image_index"]))
    os.makedirs(OUT_DIR, exist_ok=True)

    index = {}
    n_sheets = (len(rows) + PER_SHEET - 1) // PER_SHEET
    for s in range(n_sheets):
        chunk = rows[s * PER_SHEET:(s + 1) * PER_SHEET]
        size = GRID * CELL + (GRID + 1) * PAD
        sheet = Image.new("RGB", (size, size), BG)
        draw = ImageDraw.Draw(sheet)
        cells = {}
        for i, row in enumerate(chunk):
            cx, cy = i % GRID, i // GRID
            x0 = PAD + cx * (CELL + PAD)
            y0 = PAD + cy * (CELL + PAD)
            try:
                im = ImageOps.exif_transpose(
                    Image.open(P.resolve(row["path"]))).convert("RGB")
                im.thumbnail((CELL, CELL), Image.LANCZOS)
                sheet.paste(im, (x0 + (CELL - im.width) // 2, y0 + (CELL - im.height) // 2))
            except Exception:
                draw.rectangle([x0, y0, x0 + CELL, y0 + CELL], fill=(90, 0, 0))
            # Index badge, drawn last so it is never covered by the image.
            draw.rectangle([x0, y0, x0 + 38, y0 + 22], fill=(255, 235, 0))
            draw.text((x0 + 6, y0 + 6), f"{i:02d}", fill=(0, 0, 0))
            cells[str(i)] = {
                "path": row["path"], "listing_id": row["listing_id"],
                "source_key": row["source_key"], "view": row.get("view"),
            }
        name = f"sheet_{s:04d}.jpg"
        sheet.save(os.path.join(OUT_DIR, name), quality=88)
        index[name] = cells
        if (s + 1) % 25 == 0:
            print(f"  {s + 1}/{n_sheets} sheets", flush=True)

    with open(os.path.join(OUT_DIR, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    print(f"wrote {n_sheets} sheets covering {len(rows)} images -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
