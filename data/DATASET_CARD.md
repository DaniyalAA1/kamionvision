# kamionvision truck-listing dataset

Photographs of **semi-tractors offered for sale**, with per-vehicle listing
metadata, from the United States and Türkiye. Built for image-first condition
and valuation work: the photos carry the signal, the typed fields are treated as
claims a seller might omit or misstate.

Generated 2026-09-12. All source measurements taken on the same date.

## Scale

| | |
|---|---|
| Vehicles | **200** |
| Images (total) | **7,458** |
| - original | 3,729 |
| - degraded twins | 3,729 |
| Images per vehicle | 14-22, median 19 |

| Market | Vehicles | Images |
|---|---:|---:|
| TR | 84 | 3,358 |
| US | 116 | 4,100 |

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

7,623 candidate images in, **6,575 kept**, 1,048 dropped:

| reason | count | share |
|---|---:|---:|
| `drop_dup_corpus` | 826 | 78.8% |
| `drop_content_reject_screen` | 120 | 11.5% |
| `drop_content_reject_document` | 83 | 7.9% |
| `drop_dup_listing` | 8 | 0.8% |
| `drop_manual_review_not_a_truck` | 6 | 0.6% |
| `drop_content_reject_graphic` | 4 | 0.4% |
| `drop_content_reject_placeholder` | 1 | 0.1% |

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

Every one of the **6,581 original images was looked at**, not sampled. They
were tiled into 264 numbered 5x5 contact sheets and reviewed by 24 parallel
reviewers against one question: *is the main subject this truck, or a part of
it?* Reviewers flagged 12 cells. Each flagged cell was then re-opened
individually at full resolution before anything was deleted.

That second pass mattered — **half the flags were wrong**. Five real
Freightliner Cascadias were flagged as "placeholder graphics" because they
carried a dealer banner or a corner watermark, and a dealer decal photographed
on a truck's own bodywork was flagged as a "logo". Those were kept and
annotated (`marketing_overlay`, `dealer_watermark`) rather than discarded.

**6 images were removed**, all confirmed by eye: a "Photo Coming Soon"
dealer placeholder, a bare TruckMarket logo, three photographs of a diagnostic
laptop screen (two of which also contained a reflected human face), and a
Freightliner warranty marketing page. Their degraded twins were removed with
them.

**Zero images were removed for photographic quality.** Not one. Blur, darkness,
blown highlights, mud, glare, rain and bad framing are the reason this dataset
exists, and the review instructions forbade flagging them. See
`meta/manual_review_exclusions.json` for the per-image record.

## Photo reuse across listings

**0 listings publish a photo set that is byte-identical to another
listing's** (0 clusters, largest 0), and
826 individual images are exact duplicates of an image on a different
listing. These are separate adverts with their own ad number, mileage and asking
price - a dealer photographed one unit and reused the pictures.

Deduplication keeps one vehicle per cluster. Retaining them would pair identical
pixels with conflicting labels, which is exactly the failure a photo-driven
valuation model must not learn. It also means **245 downloaded
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

| bucket | count | share |
|---|---:|---:|
| `poor` | 3,801 | 51.0% |
| `fair` | 1,886 | 25.3% |
| `good` | 1,771 | 23.7% |

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

| view | count | share |
|---|---:|---:|
| `exterior_front_34` | 1,326 | 17.8% |
| `interior_cab` | 1,288 | 17.3% |
| `tire_wheel` | 976 | 13.1% |
| `exterior_front` | 914 | 12.3% |
| `dashboard_odometer` | 812 | 10.9% |
| `chassis_undercarriage` | 640 | 8.6% |
| `engine_bay` | 572 | 7.7% |
| `exterior_side` | 402 | 5.4% |
| `fifth_wheel` | 252 | 3.4% |
| `damage_detail` | 170 | 2.3% |
| `exterior_rear` | 106 | 1.4% |

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

`price_usd` converts TRY at **48.596 USD/TRY as of 2026-09-11**. Turkish
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
