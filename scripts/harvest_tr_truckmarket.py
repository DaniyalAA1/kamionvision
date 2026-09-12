"""Harvest Ford Trucks Turkiye's OEM used network (truckmarket.com.tr).

Emits one JSON record per listing to data/metadata/sources/ with the
structured fields the site publishes plus the full-resolution (_Buyuk) gallery
URLs. Images are downloaded separately by download_images.py.

Access posture (verified 2026-09-12): no robots.txt published, plain
server-rendered HTML, no bot challenge, no auth. Rate-limited politely anyway.
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

BASE = "https://www.truckmarket.com.tr"
LIST_URL = f"{BASE}/arac-listesi"
OUT = P.listing_meta("tr_truckmarket")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8"}

# Turkish spec labels -> canonical English field names.
FIELD_MAP = {
    "Marka": "make", "Model": "model", "Araç Tipi": "body_type",
    "Model Yılı": "year", "Km": "km", "Vites Tipi": "transmission",
    "Renk": "color", "Kullanım Amacı": "usage_class", "5.Teker": "fifth_wheel_height_mm",
    "Yetki Belgesi": "authorisation_certificate", "Şehir": "city",
    "Çekiş Tipi": "drive_type", "Garanti": "warranty", "İlan Numarası": "ad_number",
}


def clean(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def discover():
    r = requests.get(LIST_URL, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return sorted({int(m) for m in re.findall(r"arac-detay/(\d+)", r.text)})


def parse(listing_id, htm):
    rec = {"listing_id": listing_id, "source": "truckmarket.com.tr",
           "source_type": "oem_dealer", "market": "TR", "country": "Turkey",
           "currency": "TRY", "url": f"{BASE}/arac-detay/{listing_id}"}

    # Header: "<h1>2021 /  FORD / F-MAX</h1>" then price and km spans.
    if (h1 := re.search(r"<h1>(.*?)</h1>", htm, re.S)):
        rec["title"] = re.sub(r"\s+", " ", clean(h1.group(1)))
    if (top := re.search(r'page-detail-top-title.*?<div class="text">(.*?)</div>', htm, re.S)):
        spans = [clean(s) for s in re.findall(r"<span>(.*?)</span>", top.group(1), re.S)]
        for s in spans:
            if "₺" in s or "20BA" in s:
                rec["price"] = int(re.sub(r"[^\d]", "", s) or 0) or None

    # Spec list: <ul class="list-type-col"><li><span>VALUE</span><p>LABEL</p></li>
    for blk in re.findall(r'<ul class="list-type-col">(.*?)</ul>', htm, re.S):
        for val, lab in re.findall(r"<li>\s*<span>(.*?)</span>\s*<p>(.*?)</p>", blk, re.S):
            if (key := FIELD_MAP.get(clean(lab))):
                rec[key] = clean(val)

    for k in ("year", "km", "fifth_wheel_height_mm"):
        if rec.get(k) is not None:
            digits = re.sub(r"[^\d]", "", str(rec[k]))
            rec[k] = int(digits) if digits else None
    if rec.get("warranty") is not None:
        rec["warranty"] = rec["warranty"].strip().lower() in ("evet", "var")

    # Only <a data-fancybox="gallery"> belongs to this listing; the page also
    # embeds a "Benzer Ilanlar" slider pointing at other stock IDs.
    gallery = re.findall(r'href="(https://[^"]*_Buyuk\.jpg)"\s+data-fancybox="gallery"', htm)
    seen, urls = set(), []
    for u in gallery:
        if u not in seen:
            seen.add(u)
            urls.append(u)
    rec["image_urls"] = urls
    rec["n_images"] = len(urls)
    if (sid := re.search(r"AracResim2El/(\d+)/", urls[0])) if urls else None:
        rec["stock_id"] = int(sid.group(1))
    return rec


def fetch(listing_id):
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}/arac-detay/{listing_id}", headers=HEADERS, timeout=45)
            if r.status_code == 200:
                return parse(listing_id, r.text)
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    print(f"  FAILED {listing_id}", file=sys.stderr)
    return None


def main():
    ids = discover()
    print(f"discovered {len(ids)} listings")
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = [r for r in pool.map(fetch, ids) if r]
    with open(OUT, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    imgs = sum(r["n_images"] for r in records)
    print(f"wrote {len(records)} listings -> {OUT}")
    print(f"total gallery images: {imgs}")


if __name__ == "__main__":
    main()
