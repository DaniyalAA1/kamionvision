# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hackathon entry for the Kamion (YC) sponsor challenge: appraise a used semi-tractor **from photos
alone** — a price range, a condition assessment citing what's visible, and an explicit "I can't tell,
send me a shot of the tires" refusal path. Judged live on photos the team has never seen.

Two halves. `scripts/` is the dataset pipeline that produced the corpus; `app/` is the appraisal
system built on it (gate → evidence → price → report, a CLI and a streamed web screen).
`hackathon-plan.md` was its spec and the body of it is built.

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

In use: torch 2.14 (MPS on this machine), open_clip_torch, ultralytics, opencv-python-headless,
albumentations, imagehash, pillow, requests, numpy, pandas, scikit-learn, fastapi, uvicorn, and the
three vision SDKs (openai, anthropic, cursor-sdk). `harvest_mascus.py` shells out to the
**`firecrawl` CLI** (homebrew), not a Python package.

Credentials load from `.env` at the repo root via `app/config.py` (see `.env.example`). Nothing
reads a key at import time, so `app.cli doctor` works with none set and tells you what is missing.

No linter, no CI. Dataset scripts self-verify by printing counts; `clean_dataset.py` also writes
`data/metadata/cleaning_report.json`.

Two test layers for `app/`:

```bash
.venv/bin/python -m unittest discover -s tests    # 28 offline tests, ~0.5s, no API calls
.venv/bin/python -m app.demo                      # 7 end-to-end cases, ~3 min, spends vision calls
```

Run the offline suite on every edit; it covers JSON extraction, evidence-to-photo binding, the
price interval and the capture thresholds — the things that have actually broken. `app.demo` is
the real check but costs a vision call per case, so keep it for before a commit that matters.

## The appraisal system (`app/`)

```
app/
  config.py      paths, credentials, model ids, FX rate - everything tunable
  schema.py      the data contract; an Appraisal serialises to JSON whole
  vision.py      lazily-loaded YOLOv8n + CLIP, pinned to MPS
  gate.py        stage 1: refuse / re-ask / pass, deterministic, ~1s
  evidence.py    stage 2: one structured vision call over a closed component enum
  vlm/           backends: openai (GPT-5.6), cursor (Agent SDK), anthropic
  pricing/       stage 3: features, ridge fit + calibration (train.py), estimate
  report.py      stage 4: terminal card
  pipeline.py    wires them together, emits a timed trace
  cli.py         doctor | appraise | serve | demo
  server.py      FastAPI + SSE, so the gate verdict paints before the VLM returns
  web/           the demo screen
  calibrate_gate.py   derives models/gate_thresholds.json from the corpus
```

```bash
.venv/bin/python -m app.cli doctor                       # backends, models, calibration
.venv/bin/python -m app.cli appraise <folder> --year 2021 --km 164374
.venv/bin/python -m app.cli serve                        # demo screen on :8000
.venv/bin/python -m app.demo --build                     # rebuild demo/ fixtures
.venv/bin/python -m app.vlm.bench                        # compare backends on the real task
.venv/bin/python -m app.pricing.train                    # refit + recalibrate the price model
.venv/bin/python -m app.calibrate_gate                   # recompute gate thresholds (~2 min)
```

### Invariants the appraisal system encodes — don't break these either

- **Truck detection is set-level, never per-photo.** A tire close-up contains no truck-shaped
  object and is still a photo of the truck. Refusing per-photo would reject genuine listings;
  measured false refusal of the set-level rule is 1 of 200 real vehicles.
- **A truck box only counts when it is the subject** — ≥12% of the frame and no smaller than a
  competing vehicle box. Without that, a van parked behind a motorcycle (truck 0.87 over 5% of
  frame, car over 24%) passed as a truck listing.
- **A YOLO label that disagrees with CLIP does not disqualify a frame.** YOLO calls a truck
  dashboard filling 98% of the frame a "train" at 0.56; refusing on that is the worst error the
  gate can make.
- **Every condition claim carries a `photo_id` that was actually sent.** `evidence.parse` drops
  the ones that don't and records a warning. The VLM sees sequential ids; the parser maps them
  back to gate ids. Never bypass that mapping.
- **The VLM never sees or emits a price; the regression never sees the photos.** That separation
  is the answer to "is it more than a thin wrapper".
- **Price folds are grouped by (market, brand, model, year, price).** 87 of 155 priced listings
  collapse into 19 identical-spec groups; row splits score memorisation.
- **The interval is built from out-of-fold residuals, never in-sample ones.** In-sample residuals
  after a ridge fit on ~18 groups gave an "80%" band that covered 70%.
- **Two bands, and they are not interchangeable.** The measured 80.3% coverage belongs to the
  comparable-*asking* band. The condition-adjusted band is that estimate moved by the photos and
  carries no such guarantee — never label it with the measured number.
- **Measured numbers and assumed ones are labelled differently in the output.** Interval coverage,
  gate false-refusal and the 1.65× unseen-brand widening are measured. The 1.12×-per-missing-view
  widening and the severity weights are stated assumptions and say so.

## Dataset pipeline

```
harvest_*.py        → data/metadata/sources/<source_key>.jsonl              listing records + image URLs
sample_us.py        → data/metadata/sources/us_selectrucks_sample.jsonl     US only: 961 → 150, proportional by year
download_images.py  → data/images/<source_key>/<listing_id>/NNN.jpg
                      + data/metadata/sources/<source_key>_images.jsonl
clean_dataset.py    → data/metadata/sources/images_clean.jsonl + cleaning_report.json
degrade_images.py   → data/images/degraded/… + data/metadata/sources/images_degraded.jsonl
package_dataset.py  → data/{manifest.jsonl,images.csv,listings.csv,splits.json,DATASET_CARD.md}
```

```bash
.venv/bin/python scripts/harvest_tr_truckmarket.py
.venv/bin/python scripts/harvest_us_selectrucks.py
.venv/bin/python scripts/harvest_mascus.py --countries us --index-pages 4 --limit 100
.venv/bin/python scripts/sample_us.py
.venv/bin/python scripts/download_images.py data/metadata/sources/tr_truckmarket.jsonl tr_truckmarket --tractor-only
.venv/bin/python scripts/download_images.py data/metadata/sources/us_selectrucks_sample.jsonl us_selectrucks
.venv/bin/python scripts/download_images.py data/metadata/sources/mascus_tractors.jsonl mascus
.venv/bin/python scripts/clean_dataset.py --sources tr_truckmarket us_selectrucks mascus
.venv/bin/python scripts/degrade_images.py
.venv/bin/python scripts/package_dataset.py
```

Harvest and download are resumable (downloads skip files already on disk); cleaning and packaging
rewrite their outputs wholesale.

**`source_key` is the join key through all five stages** and must match in four places: the
`download_images.py` subdir argument, the `data/metadata/sources/{key}_images.jsonl` filename, `clean_dataset.py
--sources`, and the `LISTING_SOURCES` dict in `package_dataset.py`. Records are keyed
`(source_key, listing_id)` — listing IDs collide across sources.

## State as of the last run

The corpus was filtered after the counts in `cleaning_report.json`: `filter_dataset.py` drops photos
whose perceptual hash appears under more than one listing, then keeps the **top half of vehicles by
photo count within each source** (a vehicle with four photos is not a usable appraisal example).
That is why the live numbers below are smaller than the cleaning report's.

| | |
|---|---|
| Vehicles | 200 — 84 TR, 116 US |
| Images | 3,729 original + 3,729 degraded twins, median 19/vehicle (14–22) |
| Priced | 155 — all 84 TR, 71 of the US; Mascus is price-on-request throughout |
| TR brands | **78 of 84 are Ford**, 6 MAN — the binding limitation, see README source vetting |
| Price model | `tr_only`, R² 0.84, median error 4.2%, 80% band covers 80.3% |
| Gate thresholds | calibrated on all 7,458 images; 1 of 200 vehicles false-refused |

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
