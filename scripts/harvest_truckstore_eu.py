"""Harvest Mercedes-Benz TruckStore's European tractor-unit stock.

Why a European source in a Türkiye-only build: the Turkish corpus is 78/84
Ford, because Ford Trucks' own OEM channel is the only reachable Turkish
source (see README "Source vetting"). That leaves the price model unable to
estimate a brand effect for the makes a judge is most likely to arrive with -
Actros, TGX, Scania R/S, DAF XF, Volvo FH.

This source does NOT fix the Turkish sample. TruckStore's Turkish storefront
sells European stock: the `country` parameter sets currency, not vehicle
location, and the `center` facet offers exactly one option, "Europe". Every
row here is labelled with the country it is actually in, never with the
storefront's.

What it is for is the experiment in `app/pricing/train.py`: do age, mileage
and brand slopes fitted on European tractor units transfer to Turkish ones?
Pooling is not free - the TR+US pool already measured worse on Turkish trucks
than TR alone - so the pooled fit only ships if it beats TR-only on held-out
Turkish listings.

    .venv/bin/python scripts/harvest_truckstore_eu.py --pages 40

Access: the endpoint is the one the public search widget calls, and
proxyprod.tso-aws.com/robots.txt allows it (it disallows /api/account,
/api/users, /api/logs, /management and /v3/api-docs only).
"""

import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P
import argparse
import json
import re
import time

import requests

API = "https://proxyprod.tso-aws.com/tsoApp/widget/truck/search/en_TR"
SOURCE_KEY = "truckstore_eu"
USER_AGENT = "kamionvision-hackathon/1.0 (dataset collection; contact via repo)"
PAGE_PAUSE = 0.6

# Longest first: "Mercedes-Benz Trucks Actros" must not match bare "Mercedes".
MAKES = [
    "Mercedes-Benz Trucks", "Mercedes-Benz", "MAN", "Scania", "DAF", "Volvo",
    "Renault Trucks", "Renault", "Iveco", "Ford Trucks", "Ford", "Fuso",
    "Mitsubishi", "Isuzu", "Nissan", "Tatra", "Sisu", "Kamaz", "Hyundai",
]

# TruckStore lists tractors alongside semitrailers and bodies. Only tractor
# units are comparable to the Turkish corpus, which is Çekici throughout.
WANTED_VEHICLE_TYPE = "tractor units"


def parse_make_model(title: str) -> tuple[str, str]:
    clean = " ".join((title or "").split())
    for make in MAKES:
        if clean.lower().startswith(make.lower()):
            model = clean[len(make):].strip(" -")
            # "Mercedes-Benz Trucks" is the brand as TruckStore writes it; the
            # Turkish corpus and every buyer call it Mercedes-Benz.
            canonical = "Mercedes-Benz" if make.startswith("Mercedes") else make
            canonical = "Renault" if canonical == "Renault Trucks" else canonical
            canonical = "Ford" if canonical == "Ford Trucks" else canonical
            return canonical.upper(), model or clean
    first, _, rest = clean.partition(" ")
    return first.upper(), rest or clean


def parse_int(value) -> int | None:
    if value in (None, ""):
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    return int(digits) if digits else None


def parse_year(value) -> int | None:
    """`dateOfRegistration` is "M/YYYY"; the month is not used anywhere."""
    m = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(m.group(0)) if m else None


def fetch_page(session, page: int, currency: str) -> dict:
    for attempt in range(4):
        r = session.post(API, timeout=45, json={
            "currency": currency, "country": "TR", "pageNumber": page})
        if r.status_code in (429, 502, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"page {page}: upstream kept failing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=0, help="0 = all reported pages")
    ap.add_argument("--currency", default="EUR",
                    help="EUR is the native quote currency for these dealers; "
                         "asking for TRY makes TruckStore convert first")
    args = ap.parse_args()

    P.ensure_dirs()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT,
                            "Origin": "https://www.truckstore.com",
                            "Referer": "https://www.truckstore.com/",
                            "Content-Type": "application/json"})

    first = fetch_page(session, 0, args.currency)
    total_pages = int(first.get("totalPageNumber") or 1)
    pages = min(args.pages, total_pages) if args.pages else total_pages
    print(f"{total_pages} pages reported, fetching {pages} "
          f"({first.get('pageSize')} per page, currency {first.get('currency')})")

    rows, seen, skipped = [], set(), {"sold": 0, "wrong_type": 0, "no_price": 0,
                                      "no_year_or_km": 0, "duplicate": 0}
    for page in range(pages):
        payload = first if page == 0 else fetch_page(session, page, args.currency)
        for r in payload.get("results", []):
            uvid = str(r.get("uvid") or "")
            if not uvid or uvid in seen:
                skipped["duplicate"] += 1
                continue
            if str(r.get("vehicleType", "")).strip().lower() != WANTED_VEHICLE_TYPE:
                skipped["wrong_type"] += 1
                continue
            if r.get("sold"):
                skipped["sold"] += 1
                continue
            # priceNet, not priceTotal: priceTotal is VAT-inclusive and these
            # dealers sit in different VAT regimes (Romania, Italy, Austria).
            # A market dummy absorbs a constant offset between markets; it
            # cannot absorb variance *within* one.
            price = parse_int(r.get("priceNet") or r.get("priceTotal"))
            year, km = parse_year(r.get("dateOfRegistration")), parse_int(r.get("km"))
            if not price:
                skipped["no_price"] += 1
                continue
            if not year or not km:
                skipped["no_year_or_km"] += 1
                continue
            seen.add(uvid)
            centre = r.get("center") or {}
            make, model = parse_make_model(r.get("title"))
            rows.append({
                "listing_id": uvid,
                "source": "truckstore.com",
                "source_type": "oem_dealer",
                # Labelled by where the VEHICLE is, never by the storefront. The
                # Turkish storefront sells European stock and the `center` facet
                # has exactly one option, "Europe".
                "market": "EU",
                "country": centre.get("country") or "Europe",
                "city": centre.get("city") or None,
                "currency": args.currency,
                "price": price,
                "price_gross": parse_int(r.get("priceTotal")),
                "make": make,
                "model": model,
                "title": " ".join((r.get("title") or "").split()),
                "body_type": r.get("body") or None,
                "vehicle_type": r.get("vehicleType"),
                "year": year,
                "km": km,
                # Stated by the dealer, not inferred from year - the one column
                # the Turkish source never provides.
                "euro_norm": (r.get("emmissionsStandard") or "").strip() or None,
                "engine_type": r.get("engineType") or None,
                "url": f"https://www.truckstore.com/TR/en/detail.html#/uvid={uvid}",
                "image_urls": [r["image"]] if r.get("image") else [],
                "n_images": 1 if r.get("image") else 0,
            })
        if page and page % 10 == 0:
            print(f"  page {page}: {len(rows)} tractor units kept")
        time.sleep(PAGE_PAUSE)

    out = P.listing_meta(SOURCE_KEY)
    with open(out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} listings -> {out}")
    print("skipped:", skipped)
    if rows:
        from collections import Counter
        print("makes:", dict(Counter(r["make"] for r in rows).most_common(12)))
        print("countries:", dict(Counter(r["country"] for r in rows).most_common(8)))
        print("euro norms:", dict(Counter(r["euro_norm"] for r in rows).most_common()))
        years = [r["year"] for r in rows]
        print(f"years {min(years)}-{max(years)}; "
              f"median price {sorted(r['price'] for r in rows)[len(rows) // 2]:,} {args.currency}")


if __name__ == "__main__":
    main()
