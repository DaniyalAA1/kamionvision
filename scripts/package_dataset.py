"""Join, normalise and package the cleaned corpus into its delivered form.

Outputs, all inside the shareable `data/` bundle:
  data/DATASET_CARD.md          the bundle's entry point
  data/metadata/manifest.jsonl  one row per image, listing metadata joined in
  data/metadata/images.csv      flat image table
  data/metadata/listings.csv    one row per vehicle
  data/metadata/splits.json     train/val/test, grouped BY VEHICLE

Every stored path is relative to `data/`, so the bundle resolves wherever it is
unpacked; join it onto your own dataset root.

Splits are grouped by listing_id on purpose. A semi-tractor contributes 15-40
near-identical frames; splitting on images would put photos of the same truck on
both sides of the split and report a score that is mostly memorisation.

Prices are kept in their original currency and additionally converted to USD at
a single stated rate, because a Turkish price and a US price are not comparable
otherwise. The rate is stamped into the dataset card - at ~31% Turkish CPI it
has a short shelf life and should be recomputed rather than quoted later.
"""

import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
import paths as P
import argparse
import collections
import csv
import json
import os
import random

USD_TRY = 48.596          # 11 Sep 2026, per hackathon-plan.md appendix
FX_AS_OF = "2026-09-11"
SEED = 20260912

LISTING_SOURCES = {
    "tr_truckmarket": P.listing_meta("tr_truckmarket"),
    "us_selectrucks": P.listing_meta("us_selectrucks_sample"),
    "mascus": P.listing_meta("mascus_tractors"),
}

LISTING_FIELDS = [
    "listing_id", "source_key", "source", "source_type", "market", "country",
    "make", "model", "year", "km", "mileage_mi", "price", "currency", "price_usd",
    "city", "region", "dealer_name", "dealer_company", "body_type", "vin",
    "vpic_body_class", "vpic_gvwr", "vpic_drive_type", "vpic_engine",
    "transmission", "color", "drive_type", "usage_class", "fifth_wheel_height_mm",
    "spec_horsepower", "spec_engine_type", "spec_engine_manufacturer",
    "spec_sleeper_type", "spec_wheel_base", "damaged", "url", "n_images",
]

IMAGE_FIELDS = [
    "image_id", "listing_id", "source_key", "market", "country", "path", "variant",
    "image_index", "view", "view_conf", "is_whole_vehicle", "vehicle_class",
    "width", "height", "megapixels", "bytes", "phash",
    "blur_laplacian_var", "sharpness_tenengrad", "brightness", "contrast_rms",
    "dark_clipped_frac", "bright_clipped_frac", "colourfulness",
    "capture_quality", "quality_bucket", "clean_capture_quality",
    "exif_has_exif", "exif_has_gps", "exif_camera",
    "marketing_overlay", "dealer_watermark",
    "degradations", "severity", "clean_path",
]


def to_usd(price, currency):
    if not price:
        return None
    if currency in ("USD", "$", None):
        return round(float(price), 2)
    if currency in ("TRY", "TL", "₺"):
        return round(float(price) / USD_TRY, 2)
    return None


CARD = """# kamionvision truck-listing dataset

Photographs of **semi-tractors offered for sale**, with per-vehicle listing
metadata, from the United States and Türkiye. Built for image-first condition
and valuation work: the photos carry the signal, the typed fields are treated as
claims a seller might omit or misstate.

Generated {generated}. All source measurements taken on the same date.

## Scale

| | |
|---|---|
| Vehicles | **{vehicles}** |
| Images (total) | **{images_total}** |
| - original | {images_original} |
| - degraded twins | {images_degraded} |
| Images per vehicle | {ipv_min}-{ipv_max}, median {ipv_med} |

| Market | Vehicles | Images |
|---|---:|---:|
{market_table}

## Sources

| Source | Market | Type | Why it is here |
|---|---|---|---|
| **truckmarket.com.tr** | TR | Ford Trucks Türkiye OEM used network | Best per-vehicle photo coverage; the only accessible Turkish channel whose listings are confirmed Türkiye-domestic |
| **selectrucks.com** | US | Daimler Truck North America OEM used network | Carries VIN, so configuration is verifiable through NHTSA vPIC for free |
| **mascus.com** | US | Independent-dealer marketplace | The real-amateur layer: dealer-uploaded phone photos, wider quality spread, and units listed as damaged or inoperable |

Access posture was checked per source. SelecTrucks publishes `robots.txt:
Allow: /`; TruckMarket publishes no robots.txt and applies no bot challenge;
Mascus permits the crawled paths but rate-limits hard, so it is fetched through
Firecrawl. Sources the plan rules out - TruckPaper and Machinery Trader
(Terms of Use forbid scraping) and sahibinden.com (prohibited and defended) -
were not touched.

## What "semi-tractor only" means here

Vehicle type is established per listing before any image is downloaded, never
inferred from the photo:

- **TR** - the site's own `Araç Tipi` field must read `Çekici` (tractor). Drops
  road/construction bodies.
- **US SelecTrucks** - NHTSA vPIC `BodyClass` decoded from VIN must read
  `Truck-Tractor`. The raw inventory also contains cargo vans, school buses and
  trailers; all are excluded.
- **Mascus** - the `tractor-units` category.

No cars, pickups, vans, buses, trailers, tippers or mixers.

## Files

This folder is self-contained. Zip `data/` and it works anywhere.

```
data/
  DATASET_CARD.md                     you are here
  metadata/
    manifest.jsonl                    one row per image, listing metadata joined in
    images.csv                        same, flat
    listings.csv                      one row per vehicle
    splits.json                       train/val/test, grouped BY VEHICLE
    dataset_summary.json              the counts in this card
    cleaning_report.json              what was dropped and why
    manual_review_exclusions.json     per-image record of the visual review
    sources/                          per-source harvest records + pipeline intermediates
  images/
    tr_truckmarket/  us_selectrucks/  mascus/   originals, by source and listing
    degraded/                                   the seller-grade twins
```

**Every path stored in these files is relative to this folder**, e.g.
`images/mascus/C37A325B/002.jpg`. Join it onto wherever you unpacked the bundle:

```python
import json, pathlib
root = pathlib.Path("path/to/data")
rows = [json.loads(l) for l in open(root / "metadata/manifest.jsonl", encoding="utf-8")]
img = root / rows[0]["path"]
```

`images/` is gitignored in the source repository because it is several GB, but it
is part of the bundle when shared, and every script needed to rebuild it is
committed under `scripts/`.

## Cleaning

{candidates} candidate images in, **{kept} kept**, {dropped} dropped:

{drop_table}

Each surviving image carries a perceptual hash, measured dimensions, capture
metrics and a view tag.

- **Deduplication** - pHash, within and across listings, catching dealers who
  reuse the same frame on multiple units.
- **Content gate** - CLIP zero-shot removes frames that are not photographs of a
  vehicle: documents, dealer graphics, placeholders, and photographs of
  diagnostic laptop screens (common in the Turkish listings).
  It deliberately does *not* re-adjudicate car-vs-truck on close-ups, where CLIP
  is unreliable and the listing metadata is authoritative.
- **Privacy** - EXIF is read, recorded as flags, then stripped. No seller names,
  emails, phone numbers or messaging handles are stored from any source.

## Visual verification

Every one of the **{reviewed} original images was looked at**, not sampled. They
were tiled into 264 numbered 5x5 contact sheets and reviewed by 24 parallel
reviewers against one question: *is the main subject this truck, or a part of
it?* Reviewers flagged {flagged} cells. Each flagged cell was then re-opened
individually at full resolution before anything was deleted.

That second pass mattered — **half the flags were wrong**. Five real
Freightliner Cascadias were flagged as "placeholder graphics" because they
carried a dealer banner or a corner watermark, and a dealer decal photographed
on a truck's own bodywork was flagged as a "logo". Those were kept and
annotated (`marketing_overlay`, `dealer_watermark`) rather than discarded.

**{removed} images were removed**, all confirmed by eye: a "Photo Coming Soon"
dealer placeholder, a bare TruckMarket logo, three photographs of a diagnostic
laptop screen (two of which also contained a reflected human face), and a
Freightliner warranty marketing page. Their degraded twins were removed with
them.

**Zero images were removed for photographic quality.** Not one. Blur, darkness,
blown highlights, mud, glare, rain and bad framing are the reason this dataset
exists, and the review instructions forbade flagging them. See
`meta/manual_review_exclusions.json` for the per-image record.

## Photo reuse across listings

**{reuse_listings} listings publish a photo set that is byte-identical to another
listing's** ({reuse_clusters} clusters, largest {reuse_largest}), and
{dup_corpus} individual images are exact duplicates of an image on a different
listing. These are separate adverts with their own ad number, mileage and asking
price - a dealer photographed one unit and reused the pictures.

Deduplication keeps one vehicle per cluster. Retaining them would pair identical
pixels with conflicting labels, which is exactly the failure a photo-driven
valuation model must not learn. It also means **{lost_vehicles} downloaded
vehicles contribute no images** and are absent from the manifest; that is the
intended outcome, not a gap.

This is worth treating as a signal in its own right: it is the measurable form
of the problem this dataset exists for, where the listing text describes one
truck and the photographs come from another.

## Photo quality

The brief is a seller with a phone and no photography skills. Quality is
therefore **scored and labelled, never filtered on**:

`capture_quality` (0-1) combines blur (variance of Laplacian), exposure
deviation, highlight/shadow clipping, RMS contrast and resolution.
`quality_bucket` splits the corpus at its own 33rd/67th percentiles:

{quality_table}

Also per image: `blur_laplacian_var`, `sharpness_tenengrad`, `brightness`,
`contrast_rms`, `dark_clipped_frac`, `bright_clipped_frac`, `colourfulness`.

**Honest limitation.** Both OEM channels shoot in decent light on a prepped lot,
so their spread is narrower than a private seller's would be. That is why the
dataset ships two extra things: the Mascus layer, whose photography is genuinely
uneven, and a **degraded twin** for cleaned images - same truck, same view tag,
same metadata, with 2-4 stacked corruptions sampled from motion blur, defocus,
under/over-exposure, ISO noise, JPEG crush, downscaling, perspective skew, sun
glare, harsh shadow, mud spatter, rain and colour cast. Each twin records
exactly which corruptions were applied and at what severity, so clean/degraded
pairs are usable as supervision for an image-quality gate.

## View tags

Every image is tagged with the canonical view it shows:

{view_table}

`is_whole_vehicle` is true for the `exterior_*` tags. Filter on it to get only
photos framing the complete truck.

## Fields

Present for every vehicle: market, country, make, model, year, **km**, price,
currency, `price_usd`, city, body type, source URL.

Market-specific extras, because the two markets do not publish the same things:

- **TR** - transmission, colour, drive type, fifth-wheel height (mm), usage
  class (domestic vs international logistics), authorisation-certificate number.
- **US** - VIN plus vPIC decode (body class, GVWR class, drive type, engine),
  horsepower, engine type and manufacturer, sleeper type, wheelbase, dealer.
- **Mascus** - lat/long, damage flag, dealer company, ad create/modify dates.

`price_usd` converts TRY at **{usd_try} USD/TRY as of {fx_as_of}**. Turkish
inflation runs near 31%, so recompute rather than reuse this rate, and treat
prices as **asking** prices, not realised sale prices.

## Known limitations

1. **Asking prices, not sold prices.** No realised transaction price is
   included. Turkish asking prices in particular cluster on a dealer pricing
   table rather than on vehicle condition.
2. **OEM stock is reconditioned**, which independently compresses visible
   condition variance on the two OEM sources.
3. **No damage labels.** `damaged` exists only on Mascus and is the seller's own
   flag. Condition annotation is not included.
4. **View tags are zero-shot**, not human-verified. Treat them as a strong prior.
5. **Single snapshot.** One capture date; listings turn over.
6. **Turkish amateur layer is missing.** Mascus lists zero Türkiye-located
   tractor units, and the Turkish marketplaces that would supply private-seller
   photos are ToS-prohibited. The Turkish side is OEM-sourced, with degraded
   twins standing in for amateur capture.
"""


def write_card(outdir, summary, rows, listings, used):
    import datetime
    report_path = str(P.CLEANING_REPORT)
    rep = json.load(open(report_path, encoding="utf-8")) if os.path.exists(report_path) else {}

    reuse = rep.get("photo_set_reuse", {})
    mrev = rep.get("manual_visual_review", {})
    downloaded_vehicles = 0
    for src in ("tr_truckmarket", "us_selectrucks", "mascus"):
        idx = P.image_index(src)
        if os.path.exists(idx):
            downloaded_vehicles += len({json.loads(l)["listing_id"]
                                        for l in open(idx, encoding="utf-8")})
    lost_vehicles = max(0, downloaded_vehicles - summary["vehicles"])

    per_vehicle = collections.Counter(
        (r["source_key"], r["listing_id"]) for r in rows if r["variant"] == "original")
    counts = sorted(per_vehicle.values()) or [0]

    def table(d, head):
        if not d:
            return "_none_"
        total = sum(d.values()) or 1
        lines = [f"| {head} | count | share |", "|---|---:|---:|"]
        for k, v in sorted(d.items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {v:,} | {100 * v / total:.1f}% |")
        return "\n".join(lines)

    card = CARD.format(
        generated=datetime.date.today().isoformat(),
        vehicles=f"{summary['vehicles']:,}",
        images_total=f"{summary['images_total']:,}",
        images_original=f"{summary['images_original']:,}",
        images_degraded=f"{summary['images_degraded']:,}",
        ipv_min=counts[0], ipv_max=counts[-1], ipv_med=counts[len(counts) // 2],
        market_table="\n".join(
            f"| {m} | {summary['vehicles_by_market'].get(m, 0):,} | "
            f"{summary['by_market'].get(m, 0):,} |"
            for m in sorted(set(summary["by_market"]) | set(summary["vehicles_by_market"]))),
        candidates=f"{rep.get('candidates', 0):,}",
        kept=f"{rep.get('kept', 0):,}",
        dropped=f"{rep.get('dropped', 0):,}",
        drop_table=table(rep.get("drop_reasons", {}), "reason"),
        quality_table=table(summary.get("quality_buckets", {}), "bucket"),
        view_table=table(summary.get("views", {}), "view"),
        usd_try=USD_TRY, fx_as_of=FX_AS_OF,
        reuse_listings=reuse.get("listings_in_shared_clusters", 0),
        reuse_clusters=reuse.get("identical_photo_set_clusters", 0),
        reuse_largest=reuse.get("largest_cluster", 0),
        dup_corpus=f"{rep.get('drop_reasons', {}).get('drop_dup_corpus', 0):,}",
        lost_vehicles=lost_vehicles,
        reviewed=f"{mrev.get('images_reviewed', 0):,}",
        flagged=mrev.get("flagged_by_reviewers", 0),
        removed=mrev.get("confirmed_and_removed", 0),
    )
    with open(P.CARD, "w", encoding="utf-8") as fh:
        fh.write(card)
    print(f"wrote {P.CARD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default=str(P.CLEAN_INDEX))
    ap.add_argument("--degraded", default=str(P.DEGRADED_INDEX))
    ap.add_argument("--outdir", default=str(P.DATA))
    args = ap.parse_args()

    listings = {}
    for key, path in LISTING_SOURCES.items():
        if not os.path.exists(path):
            print(f"  !! missing {path}, skipping")
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            r["source_key"] = key
            if key == "tr_truckmarket":
                r["currency"] = "TRY"
            r["price_usd"] = to_usd(r.get("price"), r.get("currency"))
            listings[(key, str(r["listing_id"]))] = r
    print(f"listings loaded: {len(listings)}")

    images = []
    for line in open(args.clean, encoding="utf-8"):
        r = json.loads(line)
        r["variant"] = "original"
        images.append(r)
    n_clean = len(images)
    if os.path.exists(args.degraded):
        for line in open(args.degraded, encoding="utf-8"):
            images.append(json.loads(line))
    print(f"images: {n_clean} original + {len(images) - n_clean} degraded = {len(images)}")

    # Degraded twins are scored with the same metric functions but are not part
    # of the percentile fit, so their bucket is assigned from the thresholds the
    # cleaning pass established on the original corpus.
    thresholds = {}
    rep_path = str(P.CLEANING_REPORT)
    if os.path.exists(rep_path):
        thresholds = json.load(open(rep_path, encoding="utf-8")).get("quality_thresholds", {})
    q33, q67 = thresholds.get("poor<="), thresholds.get("good>")
    if q33 is not None:
        for im in images:
            if im.get("quality_bucket") or im.get("capture_quality") is None:
                continue
            s_ = im["capture_quality"]
            im["quality_bucket"] = "poor" if s_ <= q33 else ("fair" if s_ <= q67 else "good")

    rows, missing = [], 0
    for im in images:
        key = (im["source_key"], str(im["listing_id"]))
        lst = listings.get(key)
        if not lst:
            missing += 1
            continue
        row = {f: lst.get(f) for f in LISTING_FIELDS}
        row.update({f: im.get(f) for f in IMAGE_FIELDS})
        row["listing_id"] = str(im["listing_id"])
        row["source_key"] = im["source_key"]
        row["market"] = lst.get("market")
        row["country"] = lst.get("country")
        row["image_id"] = f"{im['source_key']}_{im['listing_id']}_{im['image_index']:03d}" + \
                          ("_deg" if im.get("variant") == "degraded" else "")
        if isinstance(row.get("degradations"), list):
            row["degradations"] = "|".join(row["degradations"])
        rows.append(row)
    if missing:
        print(f"  !! {missing} images had no matching listing record")

    P.ensure_dirs()
    # Splits grouped by vehicle, stratified by market.
    rng = random.Random(SEED)
    by_market = collections.defaultdict(list)
    for key, lst in listings.items():
        by_market[lst.get("market")].append(key)
    splits = {"train": [], "val": [], "test": []}
    for market, keys in by_market.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        splits["train"] += keys[:n_tr]
        splits["val"] += keys[n_tr:n_tr + n_va]
        splits["test"] += keys[n_tr + n_va:]
    split_of = {tuple(k): s for s, ks in splits.items() for k in ks}
    for r in rows:
        r["split"] = split_of.get((r["source_key"], r["listing_id"]), "train")
    # Written after the split assignment so manifest.jsonl and images.csv agree.
    with open(P.MANIFEST, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(P.SPLITS, "w", encoding="utf-8") as fh:
        json.dump({s: [list(k) for k in ks] for s, ks in splits.items()}, fh, indent=2)

    all_fields = LISTING_FIELDS + [f for f in IMAGE_FIELDS if f not in LISTING_FIELDS] + ["split"]
    with open(P.IMAGES_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    used = {(r["source_key"], r["listing_id"]) for r in rows}
    with open(P.LISTINGS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LISTING_FIELDS + ["split", "n_images_kept"],
                           extrasaction="ignore")
        w.writeheader()
        counts = collections.Counter((r["source_key"], r["listing_id"]) for r in rows
                                     if r.get("variant") == "original")
        for key, lst in sorted(listings.items()):
            if key not in used:
                continue
            row = {f: lst.get(f) for f in LISTING_FIELDS}
            row["listing_id"] = key[1]
            row["source_key"] = key[0]
            row["split"] = split_of.get(key, "train")
            row["n_images_kept"] = counts.get(key, 0)
            w.writerow(row)

    summary = {
        "images_total": len(rows),
        "images_original": sum(1 for r in rows if r["variant"] == "original"),
        "images_degraded": sum(1 for r in rows if r["variant"] == "degraded"),
        "vehicles": len(used),
        "by_market": dict(collections.Counter(r["market"] for r in rows)),
        "vehicles_by_market": dict(collections.Counter(
            listings[k].get("market") for k in used)),
        "by_source": dict(collections.Counter(r["source_key"] for r in rows)),
        "by_split_images": dict(collections.Counter(r["split"] for r in rows)),
        "views": dict(collections.Counter(r["view"] for r in rows if r.get("view")).most_common()),
        "quality_buckets": dict(collections.Counter(
            r["quality_bucket"] for r in rows if r.get("quality_bucket"))),
        "fx": {"usd_try": USD_TRY, "as_of": FX_AS_OF},
    }
    with open(P.SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    write_card(args.outdir, summary, rows, listings, used)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
