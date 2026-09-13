"""The nine rehearsed demo cases, and the fixtures that back them.

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
        # Declared values are filled in from the real listing by --build, so
        # the demo never quietly invents a spec. An earlier hardcoded year of
        # 2021 on a 2018 truck was flagged as a major finding by the model,
        # which was correct of it and wrong of the fixture.
        "declared_from_listing": ["year", "km", "make", "asking_price"],
    },
    {
        "id": "tr_phone",
        "title": "Same class of truck, phone-quality photos",
        "folder": "demo/tr_phone",
        "blurb": "Degraded twins - motion blur, underexposure, mud, awkward angles. "
                 "Some frames get dropped; the appraisal still lands.",
        "expect": "ok|ok_with_requests",
        "declared_from_listing": ["year", "km", "make", "asking_price"],
    },
    {
        "id": "tr_clean_degraded",
        "title": "Same Turkish Ford F-MAX, degraded twins",
        "folder": "demo/tr_clean_degraded",
        "blurb": "The exact tr_clean truck through phone-quality degradation, so "
                 "condition stability is a paired comparison.",
        "expect": "ok|ok_with_requests",
        "declared_from_listing": ["year", "km", "make", "asking_price"],
    },
    {
        "id": "odometer_lie",
        "title": "Seller's kilometres contradict the odometer",
        "folder": "demo/tr_clean",
        "blurb": "Same photos, but the mileage typed into the form is deliberately "
                 "falsified. The dashboard is read, the conflict is surfaced, and the "
                 "band widens instead of the number quietly moving.",
        "expect": "ok|ok_with_requests",
        "declared_from_listing": ["year", "make", "asking_price"],
        # The one fabricated input in the whole demo set, and it is fabricated
        # on purpose: the brief says sellers "sometimes lie", so the odometer
        # cross-check needs a lie to catch.
        "declared_override": {"km": 420000},
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
        "id": "mixed_vehicles",
        "title": "Three different trucks in one listing",
        "folder": "demo/rigid_truck",
        "blurb": "Rigid box trucks, and not even the same one twice. COCO calls them "
                 "all trucks and the gate lets them through, so the vehicle itself is "
                 "read from the photos: different trucks, and the wrong body type.",
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
        "declared_from_listing": ["year", "km", "make", "asking_price"],
    },
]

FIXTURE_MANIFEST = DEMO / "fixtures.json"


def resolved_cases() -> list[dict]:
    """CASES with declared values filled in from the real listings.

    `--build` records which listing backed each fixture; this reads the true
    year/km/make back out of listings.csv so a rehearsed case is never priced
    against a spec that was made up.
    """
    import json as _json
    manifest = {}
    if FIXTURE_MANIFEST.exists():
        manifest = _json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    out = []
    for case in CASES:
        resolved = dict(case)
        declared = dict(manifest.get(case["id"], {}).get("declared", {}))
        wanted = case.get("declared_from_listing") or []
        declared = {k: v for k, v in declared.items() if k in wanted}
        declared.update(case.get("declared_override") or {})
        resolved["declared"] = declared
        resolved["listing_id"] = manifest.get(case["id"], {}).get("listing_id")
        out.append(resolved)
    return out


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
    import json as _json

    import cv2
    import numpy as np
    import pandas as pd

    from .config import IMAGES_CSV

    DEMO.mkdir(parents=True, exist_ok=True)
    listings = pd.read_csv(REPO / "data/metadata/listings.csv")
    manifest: dict = {}

    def record(case_id: str, listing_id) -> None:
        """Store the listing a fixture came from, and its true spec."""
        row = listings[listings.listing_id.astype(str) == str(listing_id)]
        declared: dict = {}
        if len(row):
            r = row.iloc[0]
            declared = {"year": int(r.year), "km": int(r.km), "make": str(r.make).title()}
            if not pd.isna(r.price):
                # The listing's own asking price, so the demo judges a real
                # number rather than an invented one.
                declared["asking_price"] = float(r.price)
        manifest[case_id] = {"listing_id": str(listing_id), "declared": declared}

    need = {"exterior_front_34", "tire_wheel", "dashboard_odometer", "interior_cab"}

    # A complete Turkish set, its degraded twins, and its close-ups on their own.
    lid, group = _pick_listing(IMAGES_CSV, "TR", "original", need)
    paths = [IMAGES.parent / p for p in group.path]
    record("tr_clean", lid)
    record("odometer_lie", lid)
    print(f"tr_clean       <- TR listing {lid}: {_copy(paths, DEMO / 'tr_clean')} photos "
          f"({manifest['tr_clean']['declared']})")

    degraded = pd.read_csv(IMAGES_CSV)
    degraded = degraded[(degraded.market == "TR")
                        & (degraded.variant == "degraded")
                        & (degraded.listing_id.astype(str) == str(lid))]
    degraded_paths = [IMAGES.parent / p for p in degraded.path]
    record("tr_clean_degraded", lid)
    print(f"tr_clean_degraded <- same listing, degraded twins: "
          f"{_copy(degraded_paths, DEMO / 'tr_clean_degraded')} photos")

    closeup_views = {"tire_wheel", "interior_cab", "dashboard_odometer", "engine_bay"}
    closeups = [IMAGES.parent / r.path for r in group.itertuples()
                if r.view in closeup_views]
    record("closeups_only", lid)
    print(f"closeups_only  <- same listing, detail frames only: "
          f"{_copy(closeups, DEMO / 'closeups_only', limit=8)} photos")

    # 2: phone-quality twins of a different Turkish vehicle.
    tr_deg = pd.read_csv(IMAGES_CSV)
    tr_deg = tr_deg[(tr_deg.market == "TR") & (tr_deg.variant == "degraded")
                    & (tr_deg.listing_id != lid)]
    lid2, g2 = max(tr_deg.groupby("listing_id"), key=lambda kv: len(set(kv[1].view)))
    paths2 = [IMAGES.parent / p for p in g2.path]
    record("tr_phone", lid2)
    print(f"tr_phone       <- TR listing {lid2} degraded twins: "
          f"{_copy(paths2, DEMO / 'tr_phone')} photos "
          f"({manifest['tr_phone']['declared']})")

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
        record("unseen_brand", lid3)
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

    rigid = DEMO / "rigid_truck"
    if force or not any(rigid.glob("*")):
        fetch_non_truck(rigid, RIGID_QUERIES, label="rigid_truck")
    else:
        print(f"rigid_truck    <- kept {len(list(rigid.glob('*')))} existing photos")

    FIXTURE_MANIFEST.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {FIXTURE_MANIFEST.relative_to(REPO)}")


# The corpus is 100% tractor units by construction, so there is no negative in
# it to borrow. These are resolved live from Wikimedia Commons rather than
# hardcoded as URLs, because Commons thumbnail paths are not stable - and they
# are the three inputs the brief warns about by name.
NON_TRUCK_QUERIES = [
    ("motorcycle", "motorcycle parked side view street"),
    ("passenger_car", "hatchback car parked front three quarter"),
    ("no_vehicle", "empty living room interior furniture"),
]

# A rigid is a truck - COCO says so, and so does the gate. It is the wrong
# *kind* of truck, which only the vision model can tell you, so this fixture
# exercises the body-type check rather than the gate.
RIGID_QUERIES = [
    ("rigid_atego", "Mercedes Atego box"),
    ("rigid_fridge", "rigid lorry"),
    ("rigid_box", "box truck"),
]

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "kamionvision-hackathon/1.0 (demo fixtures; contact via repo)"


def _commons_get(session, **params):
    """One Commons request, paced and retried. Commons returns 429 readily."""
    import time

    import requests

    for attempt in range(4):
        r = session.get(COMMONS_API, timeout=30, params=params)
        if r.status_code == 429:
            time.sleep(2 ** attempt * 1.5)
            continue
        r.raise_for_status()
        return r
    raise requests.HTTPError("Commons kept returning 429; try again in a minute")


def fetch_non_truck(dest: Path, queries=None, label: str = "not_a_truck") -> None:
    """Pull freely-licensed photos from Wikimedia Commons into a fixture."""
    import time

    import requests

    queries = queries or NON_TRUCK_QUERIES
    dest.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    got, attribution = 0, []
    for i, (name, query) in enumerate(queries):
        if i:
            time.sleep(1.5)          # be a good citizen; this is someone's free API
        try:
            r = _commons_get(session, **{
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
                "gsrlimit": 1, "prop": "imageinfo",
                "iiprop": "url|extmetadata", "iiurlwidth": 1280})
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
f"{label} fixtures, from Wikimedia Commons:\n\n"
            + "\n".join(attribution) + "\n", encoding="utf-8")
    print(f"{label:14s} <- fetched {got}/{len(queries)} photos from Commons")
    if got == 0:
        print(f"  no network? drop photos into demo/{label}/ by hand", file=sys.stderr)


# --- runner ---------------------------------------------------------------

class CaseResult(dict):
    """Serializable demo record with the live appraisal kept out of its keys."""

    appraisal_object = None


def run_case(case: dict, backend=None) -> dict:
    """Run one resolved demo case and return a JSON-safe evaluation record."""
    from . import pipeline, report

    folder = REPO / case["folder"]
    photos = pipeline.collect_photos(folder)
    record = CaseResult({
        "case_id": case["id"],
        "title": case["title"],
        "folder": case["folder"],
        "expected": case["expect"],
        "photos": len(photos),
        "declared": case.get("declared") or {},
    })
    if not photos:
        record.update({"missing": True, "note": (
            f"MISSING fixtures at {case['folder']} - run: python -m app.demo --build")})
        return record

    result = pipeline.appraise(
        photos, record["declared"], backend=backend,
        on_step=lambda s, d: print(f"  ... {s}: {d}", flush=True))
    record.update({
        "missing": False,
        "status": result.status,
        "appraisal": result.to_dict(),
        "report_text": report.render_text(result),
    })
    record.appraisal_object = result
    return record


def run_demo(only: str | None = None, backend: str | None = None,
             export: str | None = None) -> int:
    from . import pipeline

    cases = [c for c in resolved_cases() if only is None or c["id"] == only]
    if not cases:
        print(f"no such case {only!r}; have {[c['id'] for c in CASES]}", file=sys.stderr)
        return 2

    failures, exported = [], []
    for case in cases:
        photos = pipeline.collect_photos(REPO / case["folder"])
        print("\n" + "#" * 78)
        print(f"# {case['id']}  —  {case['title']}")
        print(f"# expect: {case['expect']}   photos: {len(photos)}"
              + (f"   declared: {case['declared']}" if case.get("declared") else ""))
        print("#" * 78)
        record = run_case(case, backend=backend)
        if record["missing"]:
            print(f"  {record['note']}")
            failures.append((case["id"], "no fixtures"))
            continue

        print(record["report_text"])

        if export:
            from .export import write_html
            result = record.appraisal_object
            out = write_html(result, Path(export) / f"{case['id']}.html",
                             title=f"KamionVision — {case['title']}")
            exported.append({"file": out.name, "title": case["title"],
                             "status": result.status,
                             "headline": result.headline[:160]})
            print(f"\n  exported {out} ({out.stat().st_size / 1024:.0f} KB)")

        expected = set(case["expect"].split("|"))
        if record["status"] not in expected:
            failures.append((case["id"], f"got {record['status']}, expected {case['expect']}"))
            print(f"\n  ** UNEXPECTED: got {record['status']}, expected {case['expect']} **")

    if exported:
        from .export import write_index
        index = write_index(exported, export)
        print(f"\nwrote {index} — {len(exported)} offline reports, no server needed")

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
    ap.add_argument("--export", metavar="DIR",
                    help="also freeze each case as a standalone offline HTML report")
    args = ap.parse_args()
    if args.build:
        build_fixtures(force=args.force)
        return 0
    return run_demo(only=args.case, backend=args.backend, export=args.export)


if __name__ == "__main__":
    raise SystemExit(main())
