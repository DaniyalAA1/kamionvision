"""The six rehearsed demo cases, and the fixtures that back them.

    .venv/bin/python -m app.demo --build     # create demo/ fixtures
    .venv/bin/python -m app.demo             # run every case
    .venv/bin/python -m app.demo --case not_a_truck

"Does it know its limits" is its own line in the brief's rubric, so the
refusal and re-ask paths are rehearsed cases with expected outcomes, not
something to improvise on stage. Each case declares what it should produce
and the runner checks it, so a regression shows up here rather than in front
of a judge.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import IMAGES, REPO

DEMO = REPO / "demo"

CASES = [
    {
        "id": "tr_clean",
        "title": "Turkish Ford F-MAX, full photo set",
        "folder": "demo/tr_clean",
        "blurb": "The ordinary case: a complete dealer photo set with the odometer visible.",
        "expect": "ok",
        "declared": {"year": 2020, "km": 420000, "make": "Ford Trucks"},
    },
    {
        "id": "tr_phone",
        "title": "Same class of truck, phone-quality photos",
        "folder": "demo/tr_phone",
        "blurb": "Degraded twins - motion blur, underexposure, mud, awkward angles. "
                 "Some frames get dropped; the appraisal still lands.",
        "expect": "ok|ok_with_requests",
        "declared": {"year": 2021, "make": "Ford Trucks"},
    },
    {
        "id": "closeups_only",
        "title": "Tire and cab close-ups, no whole-vehicle shot",
        "folder": "demo/closeups_only",
        "blurb": "Real truck, but never photographed whole. Describes condition, "
                 "refuses to price, asks for the shot it needs.",
        "expect": "need_more_photos",
        "declared": {},
    },
    {
        "id": "not_a_truck",
        "title": "Not a truck",
        "folder": "demo/not_a_truck",
        "blurb": "The failure the brief calls out by name - confidently pricing a "
                 "motorcycle. Refused before a token is spent.",
        "expect": "refused",
        "declared": {},
    },
    {
        "id": "dark_blur",
        "title": "Night shot of nothing",
        "folder": "demo/dark_blur",
        "blurb": "Unreadably dark and blurry. Refused on capture quality, with the "
                 "measured threshold behind it.",
        "expect": "refused",
        "declared": {},
    },
    {
        "id": "unseen_brand",
        "title": "A make the model has never priced",
        "folder": "demo/unseen_brand",
        "blurb": "An American conventional dropped into the Turkish market. Prices it, "
                 "but widens the band 1.65x and says why.",
        "expect": "ok|ok_with_requests",
        "declared": {},
    },
]


# --- fixture construction -------------------------------------------------

def _copy(paths: list[Path], dest: Path, limit: int | None = None) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*"):
        old.unlink()
    n = 0
    for src in paths[:limit] if limit else paths:
        shutil.copyfile(src, dest / f"{n:03d}{src.suffix.lower()}")
        n += 1
    return n


def _pick_listing(images_csv, market: str, variant: str, need_views: set[str] | None = None):
    import pandas as pd
    df = pd.read_csv(images_csv)
    df = df[(df.market == market) & (df.variant == variant)]
    best, best_score = None, -1
    for lid, group in df.groupby("listing_id"):
        views = set(group.view)
        score = len(views) * 10 + len(group)
        if need_views and not need_views.issubset(views):
            continue
        if score > best_score:
            best, best_score = (lid, group), score
    return best


def build_fixtures(force: bool = False) -> None:
    import cv2
    import numpy as np
    import pandas as pd

    from .config import IMAGES_CSV

    DEMO.mkdir(parents=True, exist_ok=True)
    need = {"exterior_front_34", "tire_wheel", "dashboard_odometer", "interior_cab"}

    # 1 + 3: a complete Turkish set, and the close-ups from it on their own.
    lid, group = _pick_listing(IMAGES_CSV, "TR", "original", need)
    paths = [IMAGES.parent / p for p in group.path]
    print(f"tr_clean       <- TR listing {lid}: {_copy(paths, DEMO / 'tr_clean')} photos")

    closeup_views = {"tire_wheel", "interior_cab", "dashboard_odometer", "engine_bay"}
    closeups = [IMAGES.parent / r.path for r in group.itertuples()
                if r.view in closeup_views]
    print(f"closeups_only  <- same listing, detail frames only: "
          f"{_copy(closeups, DEMO / 'closeups_only', limit=8)} photos")

    # 2: phone-quality twins of a different Turkish vehicle.
    tr_deg = pd.read_csv(IMAGES_CSV)
    tr_deg = tr_deg[(tr_deg.market == "TR") & (tr_deg.variant == "degraded")
                    & (tr_deg.listing_id != lid)]
    lid2, g2 = max(tr_deg.groupby("listing_id"), key=lambda kv: len(set(kv[1].view)))
    paths2 = [IMAGES.parent / p for p in g2.path]
    print(f"tr_phone       <- TR listing {lid2} degraded twins: "
          f"{_copy(paths2, DEMO / 'tr_phone')} photos")

    # 5: synthesised from a real frame, so the refusal is reproducible offline.
    dark = DEMO / "dark_blur"
    dark.mkdir(parents=True, exist_ok=True)
    for old in dark.glob("*"):
        old.unlink()
    for i, src in enumerate(paths[:4]):
        img = cv2.imread(str(src))
        img = cv2.GaussianBlur(img, (61, 61), 0)
        img = np.clip(img.astype(np.float32) * 0.06, 0, 255).astype(np.uint8)
        img = np.clip(img.astype(np.int16)
                      + np.random.default_rng(i).normal(0, 6, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        cv2.imwrite(str(dark / f"{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 30])
    print(f"dark_blur      <- 4 frames blurred to sigma 61 and dimmed to 6% exposure")

    # 6: a US conventional, priced as if it turned up in Türkiye.
    us = _pick_listing(IMAGES_CSV, "US", "original")
    if us:
        lid3, g3 = us
        import pandas as pd
        listings = pd.read_csv(REPO / "data/metadata/listings.csv")
        row = listings[listings.listing_id.astype(str) == str(lid3)]
        make = row.make.iloc[0] if len(row) else "?"
        paths3 = [IMAGES.parent / p for p in g3.path]
        print(f"unseen_brand   <- US listing {lid3} ({make}): "
              f"{_copy(paths3, DEMO / 'unseen_brand')} photos")

    # 4: not a truck. Kept out of git and fetched, because the corpus is 100%
    # trucks by construction - there is no negative in it to borrow.
    nt = DEMO / "not_a_truck"
    if force or not any(nt.glob("*")):
        fetch_non_truck(nt)
    else:
        print(f"not_a_truck    <- kept {len(list(nt.glob('*')))} existing photos")


# The corpus is 100% tractor units by construction, so there is no negative in
# it to borrow. These are resolved live from Wikimedia Commons rather than
# hardcoded as URLs, because Commons thumbnail paths are not stable - and they
# are the three inputs the brief warns about by name.
NON_TRUCK_QUERIES = [
    ("motorcycle", "motorcycle parked side view street"),
    ("passenger_car", "hatchback car parked front three quarter"),
    ("no_vehicle", "empty living room interior furniture"),
]

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "kamionvision-hackathon/1.0 (demo fixtures; contact via repo)"


def fetch_non_truck(dest: Path) -> None:
    """Pull three freely-licensed non-truck photos into the refusal fixture."""
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    got, attribution = 0, []
    for name, query in NON_TRUCK_QUERIES:
        try:
            r = session.get(COMMONS_API, timeout=30, params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
                "gsrlimit": 1, "prop": "imageinfo",
                "iiprop": "url|extmetadata", "iiurlwidth": 1280})
            r.raise_for_status()
            pages = (r.json().get("query") or {}).get("pages") or {}
            page = next(iter(pages.values()))
            info = page["imageinfo"][0]
            url = info.get("thumburl") or info["url"]
            blob = session.get(url, timeout=60)
            blob.raise_for_status()
            (dest / f"{name}.jpg").write_bytes(blob.content)
            licence = (info.get("extmetadata") or {}).get(
                "LicenseShortName", {}).get("value", "see Commons")
            attribution.append(f"{name}.jpg  {page['title']}  ({licence})")
            got += 1
        except Exception as exc:
            print(f"  could not fetch {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if attribution:
        (dest / "ATTRIBUTION.txt").write_text(
            "Non-truck fixtures, from Wikimedia Commons:\n\n"
            + "\n".join(attribution) + "\n", encoding="utf-8")
    print(f"not_a_truck    <- fetched {got}/{len(NON_TRUCK_QUERIES)} non-truck photos")
    if got == 0:
        print("  no network? drop any non-truck photo into demo/not_a_truck/ by hand",
              file=sys.stderr)


# --- runner ---------------------------------------------------------------

def run_demo(only: str | None = None, backend: str | None = None) -> int:
    from . import pipeline, report

    cases = [c for c in CASES if only is None or c["id"] == only]
    if not cases:
        print(f"no such case {only!r}; have {[c['id'] for c in CASES]}", file=sys.stderr)
        return 2

    failures = []
    for case in cases:
        folder = REPO / case["folder"]
        photos = pipeline.collect_photos(folder)
        print("\n" + "#" * 78)
        print(f"# {case['id']}  —  {case['title']}")
        print(f"# expect: {case['expect']}   photos: {len(photos)}")
        print("#" * 78)
        if not photos:
            print(f"  MISSING fixtures at {case['folder']} - run: python -m app.demo --build")
            failures.append((case["id"], "no fixtures"))
            continue

        result = pipeline.appraise(photos, case.get("declared") or {},
                                   backend=backend,
                                   on_step=lambda s, d: print(f"  ... {s}: {d}", flush=True))
        print(report.render_text(result))
        expected = set(case["expect"].split("|"))
        if result.status not in expected:
            failures.append((case["id"], f"got {result.status}, expected {case['expect']}"))
            print(f"\n  ** UNEXPECTED: got {result.status}, expected {case['expect']} **")

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} case(s) did not match expectations:")
        for cid, why in failures:
            print(f"  {cid}: {why}")
        return 1
    print(f"all {len(cases)} case(s) behaved as expected")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(re)create the fixtures")
    ap.add_argument("--force", action="store_true", help="re-fetch non-truck photos too")
    ap.add_argument("--case")
    ap.add_argument("--backend")
    args = ap.parse_args()
    if args.build:
        build_fixtures(force=args.force)
        return 0
    return run_demo(only=args.case, backend=args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
