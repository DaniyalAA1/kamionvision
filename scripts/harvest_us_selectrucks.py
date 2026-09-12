"""Harvest SelecTrucks (Daimler Truck North America's OEM used network).

Enumerates listings from sitemap.xml and parses each rendered detail page for
price, mileage, VIN, configuration specs and the full photo gallery. VINs are
then decoded through NHTSA vPIC, whose BodyClass field is the authoritative
"is this actually a truck-tractor" filter.

Access posture (verified 2026-09-12): robots.txt is `User-agent: * / Allow: /`.

Two gotchas, both cost real debugging time:
  * Detail URLs REQUIRE a trailing slash; without one the site 301s to an empty
    body.
  * The site's own /api/v1/inventory/search/ endpoint ignores BOTH skip/count
    and page/displayRows, returning an identical 28-item page every time while
    reporting total=961. It cannot be paginated, so the sitemap is the only
    complete enumeration.
"""

import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = "https://www.selectrucks.com"
SITEMAP = f"{BASE}/sitemap.xml"
OUT = P.listing_meta("us_selectrucks")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
VPIC = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"
URL_RE = re.compile(r"/truck/(\d{4})/([^/]+)/(\d+)/(\d+)/([^/<]+)")


def discover():
    r = requests.get(SITEMAP, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    out, seen = [], set()
    for loc in re.findall(r"<loc>([^<]*/truck/[^<]+)</loc>", r.text):
        m = URL_RE.search(loc)
        if m and m.group(3) not in seen:
            seen.add(m.group(3))
            year, make, truck_id, dealer_id, stock = m.groups()
            out.append({"url": loc.rstrip("/") + "/", "year": int(year),
                        "make": requests.utils.unquote(make), "truck_id": int(truck_id),
                        "dealer_id": int(dealer_id), "stock_number": stock})
    return out


def detail(item):
    rec = {"listing_id": item["truck_id"], "source": "selectrucks.com",
           "source_type": "oem_dealer", "market": "US", "country": "United States",
           "currency": "USD", "url": item["url"], "year": item["year"],
           "make": item["make"], "stock_number": item["stock_number"],
           "dealer_id": item["dealer_id"], "image_urls": [], "n_images": 0}
    for attempt in range(3):
        try:
            r = requests.get(item["url"], headers={"User-Agent": UA, "Referer": f"{BASE}/trucks/"},
                             timeout=60)
            if r.status_code == 200 and r.text:
                break
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    else:
        return rec
    h = r.text

    if (m := re.search(r"<h1[^>]*>\s*(.*?)\s*</h1>", h, re.S)):
        title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))).strip()
        rec["title"] = title
        parts = title.split()
        if len(parts) > 2 and parts[0].isdigit():
            rec["model"] = " ".join(parts[2:])
    if (m := re.search(r'<p class="[^"]*#EC1C24[^"]*">\s*([^<]+?)\s*</p>', h)):
        rec["dealer_name"] = html.unescape(m.group(1)).strip()

    # Price + mileage render as adjacent spans: <span>$55,900</span> ... <span>367,925 </span>
    if (blk := re.search(r'<!-- Price Display -->(.*?)</div>', h, re.S)):
        nums = re.findall(r"<span>\s*\$?([\d,]+)\s*(?:&nbsp;)?\s*</span>", blk.group(1))
        vals = [int(n.replace(",", "")) for n in nums if n.replace(",", "").isdigit()]
        if vals:
            rec["price"] = vals[0]
        if len(vals) > 1:
            rec["mileage_mi"] = vals[1]
            rec["km"] = round(vals[1] * 1.609344)

    for val in re.findall(r"writeText\('([^']+)'\)", h):
        if re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", val):
            rec["vin"] = val

    for label, value in re.findall(r'class="selt-p-lg">\s*([^:<]{3,40}):\s*([^<]*)</span>', h):
        key = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
        if (val := html.unescape(value).strip()):
            rec[f"spec_{key}"] = val

    gallery = re.findall(rf'https://selectrucks\.imgix\.net/{item["dealer_id"]}/[^"?\'\s\\]+', h)
    seen, urls = set(), []
    for u in gallery:
        if "not-available" not in u and u not in seen:
            seen.add(u)
            urls.append(u)
    rec["image_urls"], rec["n_images"] = urls, len(urls)
    return rec


def vpic_enrich(records):
    have = [r for r in records if r.get("vin")]
    print(f"vPIC: decoding {len(have)} VINs")
    for i in range(0, len(have), 50):
        chunk = have[i:i + 50]
        try:
            resp = requests.post(VPIC, data={"format": "json",
                                             "data": ";".join(r["vin"] for r in chunk)}, timeout=120)
            results = {x.get("VIN"): x for x in resp.json().get("Results", [])}
        except Exception as e:
            print(f"  vPIC chunk {i} failed: {e}", file=sys.stderr)
            continue
        for r in chunk:
            if not (d := results.get(r["vin"])):
                continue
            for src, dst in [("BodyClass", "vpic_body_class"), ("GVWR", "vpic_gvwr"),
                             ("Make", "vpic_make"), ("Model", "vpic_model"),
                             ("ModelYear", "vpic_year"), ("EngineModel", "vpic_engine"),
                             ("DriveType", "vpic_drive_type"), ("Manufacturer", "vpic_manufacturer"),
                             ("PlantCountry", "vpic_plant_country"), ("EngineHP", "vpic_engine_hp"),
                             ("FuelTypePrimary", "vpic_fuel")]:
                if d.get(src):
                    r[dst] = d[src]
        print(f"  vPIC {min(i + 50, len(have))}/{len(have)}", flush=True)


def main():
    items = discover()
    print(f"sitemap: {len(items)} unique truck listings")
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = []
        for i, rec in enumerate(pool.map(detail, items), 1):
            records.append(rec)
            if i % 100 == 0:
                print(f"  detail {i}/{len(items)}", flush=True)
    print(f"fetched {len(records)} detail pages")
    vpic_enrich(records)
    with open(OUT, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} -> {OUT}")
    print(f"total gallery images: {sum(r['n_images'] for r in records)}")


if __name__ == "__main__":
    main()
