# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hackathon entry for the Kamion (YC) sponsor challenge: appraise a used semi-tractor **from photos
alone** — a price range, a condition assessment citing what's visible, and an explicit "I can't tell,
send me a shot of the tires" refusal path. Judged live on photos the team has never seen.

The repo currently contains **only the dataset pipeline** (`scripts/`). The appraisal system itself
(gate → evidence → price → report) is not built yet; `hackathon-plan.md` is its spec.

## The two docs, in order of authority

| File | Role |
|---|---|
| `kamion-truck-appraisal-brief.md` | The sponsor's brief. Ground truth for what gets judged. |
| `hackathon-plan.md` | Weekend scope, derived from the brief. **The body is the plan.** |

`hackathon-plan.md` is ~100 KB because the old `blueprint.md` was merged into it as "Appendix:
Long-Term Product Blueprint (Not Weekend Scope)" (line ~136 onward). The appendix **predates the
scoping decision and contradicts the plan body** in two ways that matter: it narrows the demo to Ford
F-MAX only (the brief says judges bring *unseen* photos, so single-brand is a liability) and it builds
a parallel US/Cascadia track (Kamion is Türkiye-only). The body drops both deliberately — don't
resurrect them from the appendix. Its §9e access-posture table and §9c measured market asymmetry are
still the best reference in the repo.

The US corpus on disk was collected under the older two-market scope. It is real and usable, but its
presence is not a reason to steer the build back toward a US demo.

## Environment

No `requirements.txt` or `pyproject.toml` — dependencies live in `.venv` (uv, CPython 3.12), installed
ad hoc. **Run everything from the repo root**; every path inside `scripts/` is relative.

```bash
.venv/bin/python scripts/<name>.py
uv pip install --python .venv/bin/python <pkg>     # how deps got in
```

In use: torch 2.14 (MPS on this machine), open_clip_torch, opencv-python-headless, albumentations,
imagehash, pillow, requests, numpy, pandas. `harvest_mascus.py` shells out to the **`firecrawl` CLI**
(homebrew), not a Python package.

No test suite, no linter, no CI. Scripts self-verify by printing counts; `clean_dataset.py` also
writes `data/meta/cleaning_report.json`.

## Dataset pipeline

```
harvest_*.py        → data/meta/<source_key>.jsonl              listing records + image URLs
sample_us.py        → data/meta/us_selectrucks_sample.jsonl     US only: 961 → 150, proportional by year
download_images.py  → data/images/<source_key>/<listing_id>/NNN.jpg
                      + data/meta/<source_key>_images.jsonl
clean_dataset.py    → data/meta/images_clean.jsonl + cleaning_report.json
degrade_images.py   → data/images/degraded/… + data/meta/images_degraded.jsonl
package_dataset.py  → data/{manifest.jsonl,images.csv,listings.csv,splits.json,DATASET_CARD.md}
```

```bash
.venv/bin/python scripts/harvest_tr_truckmarket.py
.venv/bin/python scripts/harvest_us_selectrucks.py
.venv/bin/python scripts/harvest_mascus.py --countries us --index-pages 4 --limit 100
.venv/bin/python scripts/sample_us.py
.venv/bin/python scripts/download_images.py data/meta/tr_truckmarket.jsonl tr_truckmarket --tractor-only
.venv/bin/python scripts/download_images.py data/meta/us_selectrucks_sample.jsonl us_selectrucks
.venv/bin/python scripts/download_images.py data/meta/mascus_tractors.jsonl mascus
.venv/bin/python scripts/clean_dataset.py --sources tr_truckmarket us_selectrucks mascus
.venv/bin/python scripts/degrade_images.py
.venv/bin/python scripts/package_dataset.py
```

Harvest and download are resumable (downloads skip files already on disk); cleaning and packaging
rewrite their outputs wholesale.

**`source_key` is the join key through all five stages** and must match in four places: the
`download_images.py` subdir argument, the `data/meta/{key}_images.jsonl` filename, `clean_dataset.py
--sources`, and the `LISTING_SOURCES` dict in `package_dataset.py`. Records are keyed
`(source_key, listing_id)` — listing IDs collide across sources.

## State as of the last run

| | |
|---|---|
| TR TruckMarket | 218 listings → 208 tractors → 3,932 images ✅ |
| US SelecTrucks | 961 listings → 150 sampled → 2,447 images ✅ |
| Mascus | 91 listings (all US, **none priced**) → 1,244 images ✅ |
| `clean_dataset.py` | all three sources: 7,623 candidates → 6,581 kept |
| `degrade_images.py` | 6,581 twins, 1:1 with the clean set ✅ |
| `package_dataset.py` | run: `manifest.jsonl`, `images.csv`, `listings.csv`, `splits.json`, `DATASET_CARD.md` ✅ |

Joined corpus: 408 vehicles (173 TR / 235 US), 6,581 original + 6,581 degraded images, 317 with an
asking price. `data/images/` and `.venv/` are gitignored; `data/meta/`, the manifest, CSVs, splits and
card are tracked. Images are ~3.7 GB.

## Invariants the scripts encode — don't break them

- **Vehicle type comes from listing metadata, never from the photo.** TR: the site's `Araç Tipi` field
  must read `Çekici`. US: NHTSA vPIC `BodyClass == Truck-Tractor`, decoded from VIN. Mascus: the
  `tractor-units` category. The CLIP pass may reject frames that are not photographs of a vehicle at
  all, but `vehicle_class` (truck/car/trailer_only) is **advisory — nothing is dropped on it**. As a
  rejection rule it removed 342 genuine tractors from 6,379 images: a mud-covered F-MAX rear
  three-quarter scored `trailer_only` at 0.71, a clean Cascadia side profile scored `car` at 0.78.
- **Photo quality is scored, never filtered on.** The brief's input is a seller with a phone;
  `capture_quality` / `quality_bucket` are labels. Only integrity, dedup and content failures drop an
  image. The degraded-twin generator exists because both OEM sources shoot on a prepped lot and
  under-represent that input.
- **Splits are grouped by vehicle** (`listing_id`), not by image. One tractor contributes 15–40
  near-identical frames; splitting on images leaks them across the boundary and scores memorisation.
- **No seller PII.** `harvest_mascus.py` asserts against `PII_PREFIXES`; `clean_dataset.py` records
  EXIF as flags, then strips it.
- **Prices are asking prices, not realised sale prices** — state that out loud in any demo or output.
  TRY→USD uses one stamped rate (`USD_TRY` / `FX_AS_OF` in `package_dataset.py`); at ~31% Turkish CPI
  it goes stale fast, so recompute rather than quote it.
- **Ruled-out sources are a decision, not an oversight.** sahibinden.com, arabam.com, TruckPaper,
  Machinery Trader and Copart are excluded on ToS or hard blocks (`hackathon-plan.md` appendix §9e).
  Don't add scrapers for them.

## Gotchas already paid for

- **SelecTrucks detail URLs require a trailing slash** — without one the site 301s to an empty body.
  Its own `/api/v1/inventory/search/` ignores every pagination parameter and returns the same 28 items
  forever, so `sitemap.xml` is the only complete enumeration.
- **Mascus 403s hard** on direct parallel `requests`, and the block outlasts a back-off — hence the
  `firecrawl` CLI, run serially (`--workers 1`; Firecrawl rejects parallel overflow rather than
  queueing it). Its `imagesLargeUrls` are watermarked tiles; clean originals come from
  `imagesSmallUrls` with `/medium/` swapped to `/large/`. Mascus lists **zero Türkiye-located tractor
  units**, so it can only ever contribute US vehicles.
- **CLIP must be `ViT-B-32-quickgelu`** with `pretrained="openai"`, and similarities must be multiplied
  by the trained `logit_scale` before softmax. Plain `ViT-B-32` silently degrades accuracy, and
  unscaled cosine similarities (~0.15–0.35) softmax to near-uniform, so argmax picks noise.
- **TruckMarket detail pages embed a "Benzer İlanlar" slider** pointing at other stock IDs; only
  `<a data-fancybox="gallery">` hrefs belong to the listing being parsed.
