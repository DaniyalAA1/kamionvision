"""The unified truck gallery behind the intake screen.

One grid, every vehicle in the corpus, no market split. The eight rehearsed
demo cases pin to the front so a refusal is always one click away on stage, and
the 200 real listings follow, filterable by make, year, kilometres and how the
photos were taken.

The market selector this replaces was a trap: the price model is `tr_only`, so
choosing "United States" asked a Turkish fit to price in dollars. Everything
here prices in lira, and an American truck takes the unseen-brand widening and
says why - which is a rehearsed, measured path rather than a hidden one.

Reads the packaged corpus only. No network, no image decoding at import: the
covers are resolved lazily and cached, because a grid of 200 dealer originals
is 100 MB of JPEG nobody asked for.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from .config import DATA, IMAGES_CSV, LISTINGS_CSV

# Long edge for a gallery cover. Big enough to read a cab shape and a colour on
# a retina grid, small enough that 200 of them are a few megabytes in total.
COVER_LONG_EDGE = 480

# The corpus spells one make several ways - FORD and Ford, FREIGHTLINER and
# Freightliner - because three harvesters wrote it. Filter chips have to agree
# or the same manufacturer appears twice in the row.
_MAKE_FIXUPS = {"WESTERN STAR": "Western Star", "MAN": "MAN", "DAF": "DAF"}

_LOCK = threading.Lock()
_CARDS: list[dict] | None = None
_PHOTOS: dict[tuple[str, str], list[str]] | None = None
_COVER_PATHS: dict[str, str] | None = None
_THUMBS: Path | None = None


def pretty_make(raw: object) -> str:
    """One spelling per manufacturer, or "Unknown" when the harvest had none.

    `str(nan)` is the string "nan", which is truthy and title-cases to "Nan" -
    one Mascus row with no make produced a filter chip spelled that way.
    """
    name = str(raw or "").strip()
    if not name or name.lower() in ("nan", "none", "null"):
        return "Unknown"
    upper = name.upper()
    return _MAKE_FIXUPS.get(upper, name.title())


def _quality_label(buckets: list[str]) -> str:
    """How this set was shot, in the words the filter row uses."""
    good = sum(1 for b in buckets if b == "good")
    poor = sum(1 for b in buckets if b == "poor")
    total = max(1, len(buckets))
    if good / total >= 0.6:
        return "dealer"
    if poor / total >= 0.5:
        return "phone"
    return "mixed"


def _load() -> tuple[list[dict], dict]:
    listings = pd.read_csv(LISTINGS_CSV, low_memory=False)
    images = pd.read_csv(IMAGES_CSV, low_memory=False)
    images = images[images.variant == "original"]

    photos: dict[tuple[str, str], list[str]] = {}
    covers: dict[tuple[str, str], int] = {}
    cover_paths: dict[str, str] = {}
    quality: dict[tuple[str, str], list[str]] = {}
    for key, group in images.groupby(["source_key", "listing_id"]):
        group = group.sort_values("image_index")
        paths = [str(p) for p in group.path]
        photos[(str(key[0]), str(key[1]))] = paths
        quality[(str(key[0]), str(key[1]))] = [str(b) for b in group.quality_bucket]
        # Prefer a whole-vehicle frame for the cover: a grid of tire close-ups
        # is unbrowsable. Fall back to the sharpest frame of any view.
        whole = group[group.is_whole_vehicle == True]  # noqa: E712 - pandas mask
        pick = (whole if len(whole) else group).sort_values(
            "capture_quality", ascending=False).iloc[0]
        covers[(str(key[0]), str(key[1]))] = paths.index(str(pick.path))
        cover_paths[str(key[1])] = str(pick.path)

    cards = []
    for row in listings.itertuples():
        key = (str(row.source_key), str(row.listing_id))
        if key not in photos:
            continue
        km = None if pd.isna(row.km) else float(row.km)
        year = None if pd.isna(row.year) else int(row.year)
        cards.append({
            "id": f"{key[0]}:{key[1]}",
            "source_key": key[0],
            "listing_id": key[1],
            "make": pretty_make(row.make),
            "model": "" if pd.isna(row.model) else str(row.model),
            "year": year,
            "km": km,
            "n_photos": len(photos[key]),
            "capture": _quality_label(quality[key]),
            "cover": f"/api/corpus-photo/{key[0]}/{key[1]}/{covers[key]}",
            "demo": False,
        })
    cards.sort(key=lambda c: (c["make"], -(c["year"] or 0)))
    return cards, photos, cover_paths


def _case_cover(case: dict, photos: list) -> int:
    """Which frame of a rehearsed case to show on its card.

    Matched by size rather than by filename. `app.demo._copy` renumbers the
    frames it copies while the corpus folder has gaps where cleaning dropped a
    photo, so demo/tr_clean/013.jpg and images/.../013.jpg are different
    pictures - matching on the name put a dashboard on the card. The bytes are
    a straight `shutil.copyfile`, so the size is exact.

    Four of the cases are backed by a real listing and get that listing's
    whole-vehicle frame. The other four are the point of their own case - a
    motorcycle, a night shot of nothing, three different rigids, a set with no
    whole-vehicle frame in it at all - and their first frame is exactly what
    belongs on the card.
    """
    listing_id = str(case.get("listing_id") or "")
    wanted = (_COVER_PATHS or {}).get(listing_id)
    if not wanted:
        return 0
    source = DATA / wanted
    if source.exists():
        size = source.stat().st_size
        for i, path in enumerate(photos):
            if path.stat().st_size == size:
                return i
    # The phone-quality case is built from the degraded twins, so no size will
    # ever match. The twins are 1:1 with the originals and `_copy` preserves
    # order, so the cover's position in the corpus folder is its position here.
    for (_, lid), paths in (_PHOTOS or {}).items():
        if lid == listing_id and wanted in paths:
            index = paths.index(wanted)
            if index < len(photos):
                return index
    return 0


def _demo_cards() -> list[dict]:
    """The rehearsed cases, pinned to the front of the same grid.

    They keep their own ids and their own upload route because three of them -
    the motorcycle, the night shot, the mixed rigids - are not corpus vehicles
    at all. A gallery that could only offer real tractors could not demonstrate
    a refusal, and refusing well is a line in the brief's rubric.
    """
    from .demo import resolved_cases
    from .config import REPO
    from . import pipeline

    out = []
    for case in resolved_cases():
        folder = REPO / case["folder"]
        photos = pipeline.collect_photos(folder) if folder.exists() else []
        declared = case.get("declared") or {}
        out.append({
            "id": f"case:{case['id']}",
            "case_id": case["id"],
            "demo": True,
            "title": case["title"],
            "blurb": case["blurb"],
            "expect": case["expect"],
            "make": pretty_make(declared.get("make")) if declared.get("make") else "—",
            "model": "",
            "year": declared.get("year"),
            "km": declared.get("km"),
            "n_photos": len(photos),
            "capture": "phone" if case["id"] == "tr_phone" else "mixed",
            "cover": (f"/api/case-photo/{case['id']}/{_case_cover(case, photos)}"
                      if photos else ""),
            "available": bool(photos),
        })
    return out


def cards() -> list[dict]:
    """Every card in the gallery: rehearsed cases first, then the corpus."""
    global _CARDS, _PHOTOS, _COVER_PATHS
    with _LOCK:
        if _CARDS is None:
            corpus, photos, cover_paths = _load()
            _PHOTOS, _COVER_PATHS = photos, cover_paths
            # A vehicle that backs a rehearsed case appears once, as the case.
            from .demo import resolved_cases
            backing = {str(c.get("listing_id")) for c in resolved_cases()
                       if c.get("listing_id")}
            corpus = [c for c in corpus if c["listing_id"] not in backing]
            _CARDS = _demo_cards() + corpus
        return _CARDS


def photo_paths(source_key: str, listing_id: str) -> list[Path]:
    """Absolute paths to one vehicle's original frames, in capture order."""
    cards()
    entries = (_PHOTOS or {}).get((source_key, listing_id), [])
    return [DATA / p for p in entries]


def facets() -> dict:
    """The filter vocabulary, derived rather than hardcoded.

    A re-harvest that adds a make must not need a matching edit in the browser,
    so the chips come from the same rows the cards do.
    """
    rows = [c for c in cards() if not c["demo"]]
    makes: dict[str, int] = {}
    for card in rows:
        if card["make"] == "Unknown":
            continue
        makes[card["make"]] = makes.get(card["make"], 0) + 1
    years = sorted({c["year"] for c in rows if c["year"]})
    return {
        "makes": [{"name": k, "n": v} for k, v in
                  sorted(makes.items(), key=lambda kv: (-kv[1], kv[0]))],
        "year_min": years[0] if years else None,
        "year_max": years[-1] if years else None,
        "total": len(rows),
    }


def thumb(path: Path) -> Path:
    """A cached, downscaled copy of one corpus frame.

    Generated on demand into the system temp directory. The corpus originals
    are 3-5 MP dealer photographs and the grid asks for 200 of them at once;
    serving those raw makes the first paint of the gallery take longer than an
    appraisal does.
    """
    global _THUMBS
    with _LOCK:
        if _THUMBS is None:
            import tempfile
            _THUMBS = Path(tempfile.gettempdir()) / "kamionvision-thumbs"
            _THUMBS.mkdir(parents=True, exist_ok=True)
    try:
        rel = path.resolve().relative_to(DATA.resolve())
        out = _THUMBS / str(rel).replace("/", "_")
    except ValueError:
        out = _THUMBS / path.name
    if out.exists():
        return out
    from PIL import Image, ImageOps
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(pil.size) > COVER_LONG_EDGE:
        scale = COVER_LONG_EDGE / max(pil.size)
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
    pil.save(out, format="JPEG", quality=82, optimize=True)
    return out
