"""Harvest Mascus tractor-unit listings for the real-amateur layer.

Mascus is the counterweight to the two OEM-certified channels: independent
dealers uploading their own phone photos, so the capture quality actually varies
the way the brief describes, and the inventory includes damaged and inoperable
units that a manufacturer-certified lot would never list.

Fetching goes through the Firecrawl CLI, not requests. Direct parallel fetching
earns a hard 403 from Mascus within a few hundred requests and the block
outlasts a back-off; Firecrawl's pool is the practical way in.

Scope note, measured 2026-09-12: /transportation/tractor-units/tr,country.html
renders but returns ZERO listings - Mascus currently has no Turkey-located
tractor units. So this source contributes US vehicles only, and the Turkish side
of the corpus stays OEM-sourced. Country is taken from the per-listing `country`
field, never inferred from page language or currency.

Seller PII in assetDetails (email, phone, mobile, WhatsApp, names) is never
copied into the output; the PII_PREFIXES assertion enforces that.
"""

import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

OUT = P.listing_meta("mascus_tractors")
INDEX = "https://www.mascus.com/transportation/tractor-units/{cc},country.html"
LISTING_RE = re.compile(r"https://www\.mascus\.com/transportation/tractor-units/[a-z0-9-]+/[a-z0-9]+\.html")
COUNTRY_NAME = {"US": "United States", "TR": "Turkey"}
PII_PREFIXES = ("seller", "otherInformationDetected", "userID")


def firecrawl(url, fmt="rawHtml", attempts=4):
    """Scrape one URL through the Firecrawl CLI, returning its text output.

    Retries with back-off. Firecrawl enforces a small parallel-scrape limit and
    rejects overflow immediately rather than queueing, so a failed call usually
    means "too many in flight", not "page is bad" - without the retry most of a
    batch silently returns nothing.
    """
    for attempt in range(attempts):
        with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as tf:
            tmp = tf.name
        try:
            r = subprocess.run(["firecrawl", "scrape", url, "--format", fmt, "-o", tmp],
                               capture_output=True, text=True, timeout=180)
            if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 500:
                with open(tmp, encoding="utf-8", errors="replace") as fh:
                    return fh.read()
        except subprocess.TimeoutExpired:
            pass
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        time.sleep(2 * (attempt + 1))
    return None


def discover(country, pages):
    urls, base = [], INDEX.format(cc=country.lower())
    for page in range(1, pages + 1):
        u = base if page == 1 else f"{base}?page={page}"
        body = firecrawl(u, fmt="links")
        if not body:
            print(f"  index page {page}: FAILED")
            continue
        try:
            found = [x for x in json.loads(body) if LISTING_RE.fullmatch(x)]
        except json.JSONDecodeError:
            found = LISTING_RE.findall(body)
        new = [u for u in dict.fromkeys(found) if u not in urls]
        urls.extend(new)
        print(f"  index page {page}: +{len(new)} (total {len(urls)})", flush=True)
        if not new:
            break
    return urls


def parse(url, html, max_images):
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None
    try:
        a = json.loads(m.group(1))["props"]["pageProps"]["assetDetails"]
    except (KeyError, json.JSONDecodeError):
        return None
    country = (a.get("country") or "").upper()
    if country not in COUNTRY_NAME:
        return None

    km = a.get("meterReadoutKilometers")
    price = a.get("priceOriginal")
    rec = {
        "listing_id": a.get("mascusID"), "source": "mascus.com",
        "source_type": "marketplace_dealer", "market": country,
        "country": COUNTRY_NAME[country], "url": url,
        "make": a.get("brand"), "model": a.get("model"),
        "model_group": a.get("modelGroup"), "year": a.get("year"),
        "km": round(km) if isinstance(km, (int, float)) and km else None,
        "hours": a.get("meterReadoutHours"),
        "price": price if price else None,           # 0 means "price on request"
        "currency": a.get("priceOriginalUnit"), "price_eur": a.get("priceEURO") or None,
        "city": a.get("customLocationCityName") or a.get("companyCity"),
        "region": a.get("customLocationRegionName") or a.get("companyRegion"),
        "location": re.sub(r"\s*<br\s*/?>\s*", ", ", a.get("location") or "").strip(", ") or None,
        "latitude": a.get("latitude"), "longitude": a.get("longitude"),
        "damaged": a.get("damaged"), "body_type": "tractor",
        "category": a.get("qCategoryName"),
        "dealer_company": a.get("companyName"),
        "create_date": a.get("createDate"), "modify_date": a.get("modifyDate"),
    }

    # imagesLargeUrls are the WATERMARKED /imagetilewm/ tiles and unusable.
    # imagesSmallUrls are /image/product/medium/ thumbnails; swapping the size
    # segment to /large/ yields the clean full-resolution original.
    srcs = a.get("imagesSmallUrls") or []
    if isinstance(srcs, str):
        srcs = [srcs]
    seen, imgs = set(), []
    for u in srcs:
        if not u or "imagetilewm" in u or "no_pic" in u:
            continue
        u = u if u.startswith("http") else "https:" + u
        u = u.replace("/image/product/medium/", "/image/product/large/")
        if u not in seen:
            seen.add(u)
            imgs.append(u)
    rec["image_urls"] = imgs[:max_images]
    rec["n_images"] = len(rec["image_urls"])
    rec["n_images_available"] = len(imgs)
    assert not any(k.startswith(PII_PREFIXES) for k in rec), "PII leaked into record"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=["us", "tr"])
    ap.add_argument("--index-pages", type=int, default=4)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-images", type=int, default=15)
    ap.add_argument("--workers", type=int, default=1)   # serial: Firecrawl rejects overflow
    args = ap.parse_args()

    urls = []
    for cc in args.countries:
        print(f"discovering {cc.upper()} listings")
        urls.extend(discover(cc, args.index_pages))
    urls = list(dict.fromkeys(urls))[:args.limit]
    print(f"{len(urls)} listing URLs to scrape")

    done = {"n": 0}

    def work(u):
        html = firecrawl(u)
        done["n"] += 1
        if done["n"] % 10 == 0:
            print(f"  scraped {done['n']}/{len(urls)}", flush=True)
        return parse(u, html, args.max_images) if html else None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = [r for r in pool.map(work, urls) if r]

    with open(OUT, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"kept {len(records)} tractor units -> {OUT}")
    print(f"total gallery images (capped at {args.max_images}/listing): "
          f"{sum(r['n_images'] for r in records)}")


if __name__ == "__main__":
    main()
