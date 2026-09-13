# kamionvision

Appraises a used semi-tractor **from photos alone**: a price range grounded in
real comparable listings, a condition assessment where every claim cites the
photo it came from, and an explicit "I can't tell from these, send me a shot of
the tires" path. Built for the Kamion (YC) sponsor challenge — Türkiye's largest
freight platform — and judged live on photos the team has never seen.

- [`kamion-truck-appraisal-brief.md`](kamion-truck-appraisal-brief.md) — the sponsor's brief: what the system has to do and how it will be judged.
- [`hackathon-plan.md`](hackathon-plan.md) — the build plan scoped to that brief. Its [appendix](hackathon-plan.md#appendix-long-term-product-blueprint-not-weekend-scope) carries the long-form product blueprint as reference material, not weekend scope.
- [`data/DATASET_CARD.md`](data/DATASET_CARD.md) — the shareable dataset bundle: scale, sources, cleaning, quality distribution, fields and limitations.
- [`docs/2026-09-13-session-improvements.md`](docs/2026-09-13-session-improvements.md) — structured logging, pricing enhancements, and UI layout containment fixes.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python openai anthropic cursor-sdk ultralytics \
    open_clip_torch torch torchvision opencv-python-headless scikit-learn pandas \
    numpy pillow imagehash albumentations fastapi "uvicorn[standard]" python-multipart

cp .env.example .env          # add one vision API key
.venv/bin/python -m app.cli doctor          # backends, models, calibration
.venv/bin/python -m app.demo --build        # build the six rehearsed fixtures
.venv/bin/python -m app.cli serve           # the demo screen on :8000
```

Appraise a folder from the terminal:

```bash
.venv/bin/python -m app.cli appraise demo/tr_clean --year 2021 --km 164374 --asking 2550000
.venv/bin/python -m app.demo                        # run all eight rehearsed cases
.venv/bin/python -m unittest discover -s tests      # 52 offline tests, 0.5s, no API calls
```

**Demo insurance.** Any appraisal can be frozen into one self-contained HTML
file — stylesheet, script and every photo inlined — that opens with no server,
no network and no API key:

```bash
.venv/bin/python -m app.cli appraise demo/tr_clean --html report.html
.venv/bin/python -m app.demo --export demo_reports/   # all eight, plus an index
```

That is the backup the plan's checklist asks for, and unlike a screen recording
you can still click through it when a judge asks a question.

## How it works

```
photos ─▶ 1. Gate ─▶ 1b. Heads ─▶ 2. Evidence ─▶ 2b. Reconcile ─▶ 3. Price ─▶ 4. Report
          reject/    trained on   one VLM call   heads correct    comps +     range +
          re-ask     the corpus   fixed schema   the VLM, aloud   new-price   citations
          ~1s                                                     anchor
```

The ordering is the argument. The cheap deterministic stage runs first and can
stop the pipeline before a token is spent. The vision model runs second and is
allowed to *describe* but never to price. The regression runs third, on evidence
it did not author. A single VLM call returning a number would be faster, and is
the thing the brief says will not win.

**1. Gate** (`app/gate.py`) — YOLOv8n COCO truck detection, Laplacian-variance
blur and exposure checks, and CLIP zero-shot view tagging. It has three distinct
outcomes, because refusing and re-asking are not the same answer:

| Outcome | When | What the user gets |
|---|---|---|
| `pass` | a truck is the subject of at least one frame, required views present | full appraisal |
| `ask_more` | real truck, thin coverage | appraisal with a widened band, plus the exact photos to take |
| `refuse` | not a truck, or nothing legible | no price, and the reason |

Truck detection is **set-level, never per-photo**: a close-up of a tire has no
truck-shaped object in it and is still a photo of the truck. A truck box only
counts when it is the *subject* — at least 12% of the frame and no smaller than
a competing vehicle in the same frame.

**1b. Perception heads** (`app/perception/`) — three linear heads trained on the
corpus, learning on the CLIP embedding the gate already computes. They exist
because the gate can only *measure* a photo and the VLM can only *describe* one;
neither can say "this frame is too corrupted for the claim you are about to
read". The degradation head can, and its labels are real: 3,729 exactly-paired
clean/degraded images where the applied transform is known. Every fold is
grouped by vehicle.

The view head is the interesting one. Its labels are CLIP zero-shot
pseudo-labels, so accuracy against them only measures agreement with a noisy
teacher. But a degraded twin inherits its original's label, so we can ask a
better question — does the head hold its answer when the photo is muddied? It
does, on **75.8%** of twins against the teacher's own **69.6%**. That is a real
robustness gain from a teacher that is otherwise the ceiling.

**2. Evidence** (`app/evidence.py`) — one structured call, all photos at once,
into a fixed schema over a closed enum of 33 heavy-vehicle components. Every
issue must carry a `photo_id` of a photo that was actually sent; ones that cannot
are dropped rather than trusted. Anything not visible becomes a coverage gap,
which becomes a request for another photo, rather than a confident guess. The
photo set is selected view-first, so thirty frames of the same tire cost one slot.

It also answers two questions the detector cannot, and either one stops the
pricing stage:

  * **Is this one vehicle?** Sellers pad listings with photos of a tidier truck,
    and a condition report averaged over two vehicles is confidently wrong about
    both. The model compares plates, paint, trim, badges, wheels and cab
    generation across the set.
  * **Is it a tractor unit?** COCO calls a rigid, a tipper and a tractor all
    "truck". Every comparable in the corpus is a tractor unit, established from
    listing metadata, so a rigid has nothing here to be priced against.

Vocabulary is the point: steer versus drive tires, tread shoulder wear and
cupping, fifth wheel plate scoring, frame rail corrosion, air bag sag, AdBlue
tank, DPF, bunk and bolster wear. A generic dent/scratch/paint taxonomy is the
tell that a car tool was pointed at a truck.

**2b. Reconciliation** (`app/reconcile.py`) — `Issue.confidence` is a number the
vision model writes about its own claim, and it is multiplied straight into the
price. Nothing else in the pipeline was in a position to disagree with it. Now a
fine-detail claim resting on a frame the degradation head scores past its
calibrated cutoff gets downgraded, and the downgrade is written down. On the
phone-photo case:

```
2 correction(s) applied to the vision model's findings:
  · steer_tires: fine detail claimed from a photo scored 0.38 for degradation
    (low_light_noise, awkward_angle) [photo 8]
      confidence 0.98 → confidence 0.61
```

Nothing is deleted. A downgraded finding is still shown, still cites its photo,
and carries the reason — the same posture as `fell_back_from`: changing your
mind is allowed, doing it quietly is not. The identity head is forbidden from
disputing a brand it was never trained on, because the corpus contains no Scania
and a head that has never seen one will still name a class, confidently.

**3. Price** (`app/pricing/`) — a hedonic ridge on `log(price) ~ log1p(age) +
log(km) + brand + market + euro6` fit on the harvested asking prices, plus the
comparable listings it was priced against and a bounded condition adjustment.
The model never sees the photos; the VLM never sees a price.

If the seller's asking price is supplied, it is judged against the comparable
band — the measured one — and reported as in line with, above, or below the
market, with the gap to the condition-adjusted estimate alongside. That is the
question a Kamion buyer actually opens the app with.

A **second pricing route** (`app/pricing/anchor.py`) covers what comparables
cannot. The fit is 84 Turkish listings and 78 are Ford, so an unseen make
previously got only a 1.85× widening — and the worst holdout needed 3.25× at 75%
median error. Judges bring unseen photos, so that was the likeliest way this
fails in the room.

Published new price × a fitted retention curve needs no same-brand comparable to
exist. The curve turns out to be a *better* model of the corpus than the hedonic
fit — R² **0.936** out-of-fold against 0.842 — because a published price carries
brand, segment and specification that three brand dummies cannot. Anchoring on
today's price for the current model keeps both sides of `price / new_price`
current, which sidesteps ~31% Turkish CPI rather than trying to model it.

The two routes combine by inverse variance, so neither is privileged; inflating
the hedonic variance by the measured unknown-brand factor is what hands the
decision to the anchor. The blend may claw a widening back but is floored at
1.0×, because the measured 80.3% coverage belongs to the unwidened band.

Reference prices are hand-curated and stamped like `USD_TRY`, each row carrying
a source URL, a date and a `source_type` — Ford Trucks Türkiye and MAN Türkiye
official lists, Mercedes from trade press and widened 1.35× for it. Scania
publishes its list only as a JPG, so that row is a deliberate null: OCR-ing a
price off a scan to put a number in a truck valuation is not a risk worth
taking. A brand with no row falls back to the existing widening, as Freightliner
and Scania both correctly do.

**4. Report** (`app/report.py`) — the same `Appraisal` object rendered as a
terminal card or streamed to the web screen. Every condition line names its
photo by filename, the price is always a range, and the measured coverage of
that range is printed beside it.

Phone photos work: HEIC/HEIF is registered at import (`pillow-heif`) and EXIF
rotation is applied everywhere. Without that, a folder of iPhone photos decodes
to nothing and the gate refuses the whole set on capture quality — the worst
possible way to fail in front of someone holding the photos.

## What is measured, and what is assumed

Judges ask which numbers are real. These are, and the code that produced each one
is named:

| Quantity | Value | How it was measured |
|---|---|---|
| 80% band coverage | **80.3%** over 958 held-out evaluations | `app/pricing/train.py`, spec-grouped splits, band built from out-of-fold residuals |
| Point accuracy | R² 0.84, median error 4.2% | same, out-of-fold |
| Gate false-refusal on real trucks | **1 of 200** vehicles (3 of their degraded twins) | `app/calibrate_gate.py`, over all 3,729 originals and 3,729 twins |
| Capture-quality floors | drop 2.3% of originals, 14.4% of twins | quantiles of the degraded-twin population, not round numbers |
| Condition adjustment cap | ±9.4% | one out-of-fold residual standard deviation — the price variation age, km, brand and market do *not* explain |
| Unseen-brand widening | 1.85× | **within-market** leave-one-brand-out over 7 holdouts across two markets |
| Degradation head | AUC **0.985** degraded-vs-original, severity R² 0.695 | `app/perception/train.py`, vehicle-grouped 5-fold, against known synthetic transforms |
| View head stability | **0.758** vs 0.696 zero-shot | agreement between a photo and its degraded twin, held out |
| Identity head | **79.5%** brand vs 39.0% majority | vehicle-grouped, one row per vehicle |
| Fine-detail cutoff | flags 5% of clean photos, catches **97%** of degraded | a quantile of predicted severity on clean photos, so the fixed quantity is a false-flag rate |
| Retention curve | R² **0.936**, median error 3.1% | `app/pricing/train.py`, spec-grouped, out-of-fold |
| Anchor on an unseen make | band 1.85×→**1.00×**, coverage 0.17→**0.83**, error 16.3%→**3.7%** | TR:MAN held out of the fit entirely and priced as an unknown — **n=6**, which is small and stated everywhere the number is |

Three numbers are **stated assumptions, not measurements**, and the report says so
where it uses them: the 1.12× band widening per missing canonical view, the
severity/impact weights that convert findings into a condition score, and the
1.35× widening applied to a trade-press list price against an OEM-official one.

**A learned image→price model was measured, and does not exist on this corpus.**
The obvious next move is to let the photos predict the part of the price that age
and kilometres cannot explain, and fit the condition weights instead of stating
them. `scripts/probe_residual_signal.py` asks whether that signal is there at all,
under nested vehicle-grouped CV against a permuted-target null:

| | residual sd | R² | null 95th pct | p |
|---|---|---|---|---|
| TR | 9.9% | −0.222 | −0.005 | 1.00 |
| US | 22.1% | −0.038 | +0.017 | 0.57 |
| pooled | 18.9% | −0.046 | −0.004 | 0.93 |

Six pooling variants, all negative. The same pipeline passes its positive
controls — brand 76.5% against a 39% majority, market 100%, year R² +0.23 — so
this is the data, not the code. The blocker is the *target*, not the perceiver:
Turkish asking prices are **23 distinct values across 84 listings**, 70% of them
round to ₺100k, and ₺2,800,000 alone covers 26 of them. That is a dealer's price
table, not a market. Both OEM sources also sell reconditioned stock, which
compresses visible condition variance on purpose.

So the condition weights stay hand-set and stay labelled an assumption. The gate
that established this cost one script and saved 155 vision calls on a fit with
nothing to learn. What the probe *does* bound is a linear read of CLIP
embeddings; it is not proof that no perceiver could do better, and the honest
next step is a corpus with real transaction prices rather than asking prices.

Three things that went the other way, and were fixed because they were measured:

- The 80% band first covered **70%**. It was being built from in-sample residuals,
  which are far tighter than real prediction error after a ridge fit on ~18 groups.
- Price folds grouped by row scored memorisation: 87 of the 155 priced listings
  collapse into 19 identical-spec groups (one dealer lists the same 2022 F-MAX
  twenty-six times). Folds are grouped by spec — the same reason the image splits
  are grouped by vehicle.
- Unseen-brand widening measured by dropping the brand columns came out at 1.05×,
  which is meaningless on a corpus that is 93% Ford. Leave-one-brand-out across
  markets then reported a **442% median error** for Ford — because holding Ford
  out removes Türkiye, so it was measuring the border, not the brand. Done
  within a market it says **1.85×**.
- A 2021 F-MAX priced at **₺118,000,000**. Moving the model target to native
  currency left `estimate` still multiplying by the TRY rate, and every relative
  assertion in the test suite sailed past a 48× error because both numbers were
  wrong by the same factor. There is now an absolute magnitude test.

## Vision backends

Three ship behind one interface (`app/vlm/`). The default is pinned to
**Cursor SDK / GPT-5.6 Sol**, with low reasoning effort and tools disabled.
No automatic provider fallback is allowed; other backends require an explicit
`KAMION_VLM_BACKEND` override. Each Cursor response must report the requested
model and a finished run before its evidence is accepted.

| Backend | Model | Structured output |
|---|---|---|
| `openai` | GPT-5.6 (`luna` / `sol` / `terra` — no bare `gpt-5.6` exists) | Responses API `json_schema`, strict |
| `cursor` | Cursor Agent SDK, native `SDKImage` upload | prompt-enforced, validated by the parser |
| `anthropic` | Claude Opus 5 | `output_config.format` |

`python -m app.vlm.bench` compares them on the real evidence task — latency,
findings returned, share of findings using heavy-vehicle vocabulary, schema
violations — which is how the default was chosen rather than guessed.

Cursor was authenticated and live photo inference verified on 2026-09-12.
The SDK uses a user API key from [API & SSH Keys](https://cursor.com/dashboard/api),
not the CLI session token. Store it as `CURSOR_API_KEY` in the git-ignored `.env`
(file permissions `600`); never commit credentials. SDK runs use Cursor's
[existing pricing and request pools](https://cursor.com/docs/sdk/python#usage-and-billing).
Use `python -m app.cli doctor` to check current authentication; authentication
alone is not proof of a completed inference.

## The rehearsed cases

"Does it know its limits" is its own line in the rubric, so the refusal and
re-ask paths are tested cases with expected outcomes, not something to improvise
on stage. `python -m app.demo` runs all six and fails loudly if any changes
behaviour.

Declared year/km/make for each case are filled in from the real listing by
`--build`, so a rehearsed case is never priced against a spec that was invented.
The one deliberately falsified input in the set is `odometer_lie`, and it is
falsified because the brief says sellers "sometimes lie" and the odometer
cross-check needs a lie to catch.

| Case | Expected |
|---|---|
| `tr_clean` — complete Turkish dealer set | priced, asking price judged |
| `tr_phone` — degraded twins: blur, underexposure, mud | priced, some frames dropped |
| `odometer_lie` — mileage typed into the form contradicts the dash | priced, conflict surfaced, band widened |
| `closeups_only` — tire and cab only, never the whole truck | condition notes, **no price**, asks for the shot |
| `mixed_vehicles` — three different rigids in one listing | condition notes, **no price**, names what differs |
| `not_a_truck` — motorcycle, parked car, empty room | refused before any vision call |
| `dark_blur` — unreadably dark and blurry | refused on capture quality |
| `unseen_brand` — an American conventional in the Turkish market | priced, band widened 1.65×, says why |

Four of the eight produce no price. That is the point: the brief scores "does it
know its limits" on its own line, and each of those four fails for a *different*
reason with a *different* thing to do about it.

## The dataset

Photographs of semi-tractors offered for sale in the **United States** and
**Türkiye**, with per-vehicle metadata (year, make, model, kilometres, price,
location) and multiple images per vehicle. Tractor units only — no cars, vans,
buses, trailers, tippers or mixers. Vehicle type is established from listing
metadata before any image is downloaded, never inferred from the photo.

| | Türkiye | United States | Total |
|---|---:|---:|---:|
| Vehicles | 84 | 116 | **200** |
| Images, original | 1,679 | 2,050 | **3,729** |
| Images, degraded twins | 1,679 | 2,050 | **3,729** |
| Vehicles with an asking price | 84 | 71 | 155 |

Median 19 photos per vehicle (range 14–22). 7,623 candidate images in, 6,575
survived cleaning, and a final coverage pass kept the **top half of vehicles by
photo count within each source** — because a truck photographed four times is not
a usable appraisal example. Cleaning dropped 826 corpus-wide perceptual-hash
duplicates, 208 non-vehicle frames (documents, diagnostic-laptop screens, dealer
graphics), 8 within-listing duplicates, and 6 removed by full visual review.

| Source | Market | Vehicles | Images | What it contributes |
|---|---|---:|---:|---|
| `truckmarket.com.tr` | TR | 84 | 1,679 | Ford Trucks Türkiye's OEM used network — the best per-vehicle photo coverage available, and confirmed Türkiye-domestic stock |
| `selectrucks.com` | US | 71 | 1,385 | Daimler Truck North America's OEM used network — carries VIN, so configuration is verifiable free through NHTSA vPIC |
| `mascus.com` | US | 45 | 665 | The amateur layer: independent dealers uploading their own phone photos, so capture quality actually varies |

Every image carries a view tag, capture-quality metrics and a perceptual hash.
Because the brief is "a seller with a phone and no photography skills", quality is
**scored and labelled rather than filtered on**: `capture_quality` (0–1) combines
blur, exposure deviation, highlight/shadow clipping, contrast and resolution, and
`quality_bucket` splits the corpus at its own 33rd/67th percentiles.

Both OEM channels shoot in decent light on a prepped lot, so every cleaned image
also gets a **degraded twin** — same truck, same view tag, same metadata, with 2–4
stacked corruptions sampled from motion blur, defocus, under/over-exposure, ISO
noise, JPEG crush, downscaling, perspective skew, sun glare, harsh shadow, mud
spatter, rain and colour cast. Each twin records exactly which corruptions were
applied, at what severity, and its own recomputed capture-quality score alongside
the original's. That pairing is the point: it is supervision for an image-quality
gate that no amount of scraping produces on its own.

## Reproducing it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python requests pillow imagehash numpy pandas \
    tqdm lxml beautifulsoup4 opencv-python-headless albumentations open_clip_torch \
    torch torchvision

# 1. harvest listing metadata
.venv/bin/python scripts/harvest_tr_truckmarket.py
.venv/bin/python scripts/harvest_us_selectrucks.py
.venv/bin/python scripts/sample_us.py
.venv/bin/python scripts/harvest_mascus.py --countries us --limit 100 --max-images 15

# 2. download images
.venv/bin/python scripts/download_images.py data/metadata/sources/tr_truckmarket.jsonl tr_truckmarket --tractor-only
.venv/bin/python scripts/download_images.py data/metadata/sources/us_selectrucks_sample.jsonl us_selectrucks
.venv/bin/python scripts/download_images.py data/metadata/sources/mascus_tractors.jsonl mascus

# 3. clean, tag and score
.venv/bin/python scripts/clean_dataset.py --sources tr_truckmarket us_selectrucks mascus

# 4. generate degraded twins (parallel; --workers defaults to CPU count)
.venv/bin/python scripts/degrade_images.py

# 5. package: manifest, CSVs, vehicle-grouped splits, dataset card
.venv/bin/python scripts/package_dataset.py
```

Run everything from the repo root — every path in `scripts/` is relative to it. The
`source_key` (`tr_truckmarket`, `us_selectrucks`, `mascus`) is the join key through
all five stages and has to match in each: the `download_images.py` subdirectory
argument, `clean_dataset.py --sources`, and `LISTING_SOURCES` in
`package_dataset.py`. Harvesting and downloading are resumable; cleaning and
packaging rewrite their outputs.

`scripts/harvest_mascus.py` fetches through the [Firecrawl](https://firecrawl.dev)
CLI (`firecrawl` on `PATH`, authenticated) and runs serially; Mascus 403s hard on
direct parallel crawling and the block outlasts a back-off. The other harvesters use
plain HTTP. `clean_dataset.py` runs CLIP on MPS/CPU and is the slow step.

Images land in `data/images/` and are **gitignored** — several GB, fully reproducible
from these scripts. The manifests, metadata, cleaning report and dataset card are
committed.

## Layout

`data/` is a self-contained bundle — zip it and it works anywhere, because every
path stored inside it is relative to that folder.

```
data/                                 <- share this
  DATASET_CARD.md                     entry point: scale, sources, cleaning, limits
  metadata/
    manifest.jsonl                    one row per image, listing metadata joined in
    images.csv                        same, flat
    listings.csv                      one row per vehicle
    splits.json                       train/val/test, grouped BY VEHICLE
    dataset_summary.json              headline counts
    cleaning_report.json              what was dropped and why
    manual_review_exclusions.json     per-image record of the visual review
    sources/                          per-source harvest records + intermediates
  images/                             (gitignored: several GB, rebuildable)
    tr_truckmarket/ us_selectrucks/ mascus/    originals
    degraded/                                  seller-grade twins

scripts/
  paths.py                    the layout above, in one place
  harvest_tr_truckmarket.py   Ford Trucks Türkiye OEM used network
  harvest_us_selectrucks.py   Daimler Truck North America OEM used network (+ NHTSA vPIC)
  harvest_mascus.py           independent-dealer marketplace, via Firecrawl
  sample_us.py                stratified US subset (Turkey-heavy corpus)
  download_images.py          resumable image fetch, tractor-only filter
  clean_dataset.py            integrity, dedup, content gate, view tags, quality metrics
  build_contact_sheets.py     tile every image for visual review
  apply_manual_review.py      apply the review verdicts
  degrade_images.py           paired "seller with a phone" variants
  package_dataset.py          manifest, CSVs, splits, dataset card
  cache_embeddings.py         CLIP embedding per image, what the heads train on
  probe_residual_signal.py    the measurement that killed the learned price head
  freeze_story.py             turns one real appraisal into the landing page's
                              scroll story - photos, story.json and the markup

data/reference/
  new_prices_tr.json          hand-curated new-truck list prices, each row cited
  story_appraisal.json        the appraisal the landing page is a record of

models/
  gate_thresholds.json        capture floors, from quantiles of the degraded twins
  price_model.json            hedonic ridge + calibration + the retention curve
  perception.json             the three trained heads
  yolov8n.pt                  stock COCO detector, vendored, never fine-tuned
```

Reading it from anywhere:

```python
import json, pathlib
root = pathlib.Path("path/to/data")
rows = [json.loads(l) for l in open(root / "metadata/manifest.jsonl", encoding="utf-8")]
img  = root / rows[0]["path"]          # e.g. images/mascus/C37A325B/002.jpg
```

Train/val/test splits are grouped **by vehicle**, not by image: one tractor
contributes up to 22 near-identical frames, and splitting on images would put photos
of the same truck on both sides of the boundary and report a score that is mostly
memorisation.

## Known limitations

- **Asking prices, not realised sale prices.** Turkish asking prices in particular
  cluster on a dealer pricing table rather than on vehicle condition.
- **Mascus carries no prices.** All 91 of its listings are "price on request", so it
  contributes photographic variety and no price labels.
- **OEM stock is reconditioned**, which independently compresses visible condition
  variance on the two OEM sources.
- **No damage labels.** Condition annotation is not included; `damaged` exists only
  as a Mascus field and is unset across the harvested set. This is why the
  condition weights are stated rather than fitted, and why the image→price probe
  above has nothing to learn from.
- **The degradation head was trained on synthetic corruption.** The 14 transforms
  in `scripts/degrade_images.py` are a stand-in for a seller with a phone, not a
  sample of one. It scores 0.985 AUC against the corruptions it was shown; on a
  judge's genuine bad photo the failure mode is unknown, and the calibrated 5%
  false-flag rate is a rate on *clean corpus* photos, not on real phone shots.
- **The new-price reference covers four brands and one market.** Ford, MAN and
  Mercedes have prices; Scania's is a deliberate null. A Volvo, DAF, Iveco or
  Renault falls back to the 1.85× unseen-brand widening, and every non-Turkish
  market falls back entirely.
- **The anchor's payoff is measured on n=6.** Turkish stock is two makes, so
  holding out Ford leaves six MAN rows to fit on — there is no larger honest
  holdout available on this corpus.
- **View tags are zero-shot**, not human-verified — treat them as a strong prior.
  `vehicle_class` is advisory only and nothing is dropped on it; a forced
  truck/car/trailer choice proved unreliable on real vehicle photos.
- **Single snapshot**, one capture date. Listings turn over, and TRY→USD uses one
  stamped rate — at ~31% Turkish CPI, recompute rather than reuse it.
- **The Turkish side is OEM-sourced.** Mascus lists zero Türkiye-located tractor
  units and the Turkish marketplaces that would supply private-seller photos are
  ToS-prohibited, so degraded twins stand in for amateur capture.
- **The Turkish comparables are 93% Ford** (78 of 84, the rest MAN). The price
  model prices an Actros, a Scania or a DAF off the market average with a 1.65×
  wider band and says so on the card. That is not an oversight — see the source
  vetting below. It is a property of what is reachable.
- **No damage-history field.** Turkish listings state `hasar kaydı`, which would
  have let the condition adjustment be calibrated against a measured price effect
  instead of capped at the residual standard deviation. It is not in the harvested
  fields, so the cap is principled but the per-finding weights are not measured.

## Source vetting: why the Turkish side is one brand

The concentration above is the binding constraint, so the multi-brand sources
`hackathon-plan.md` nominated were checked. All three turn out to carry **no
Türkiye-located tractor units**, which is the same reason Mascus contributes
only US vehicles:

| Source | Verdict | Evidence |
|---|---|---|
| `autoline.com.tr` | rejected | Turkish-language interface over pan-European stock. 25 listings on the `cntTR` category page: Poland 5, Finland 3, Netherlands 3, Slovakia 3, Belgium 2, Norway 2, and one each from six more — **Türkiye 0**. Three different country-filter URL forms all returned the same pan-European set. |
| Mercedes-Benz TruckStore Türkiye | rejected | The `.com.tr` domain does not resolve; the Turkish storefront is `truckstore.com/TR/`. Its search API (`proxyprod.tso-aws.com/tsoApp/widget/truck/search/en_TR`) accepts `{"country":"TR"}`, but that sets the storefront currency, not the vehicle location: the `center` facet offers exactly one option, **"Europe" (1,584 vehicles)**, and sampled results are located in Romania. |
| MAN TopUsed | rejected | No Türkiye locale. `mantopused.com` and `man.com.tr` both redirect away; the TopUsed homepage contains no Turkey markers. |

So the accessible Turkish sources reduce to Ford Trucks' own OEM channel, because
the Turkish marketplaces that would carry the other makes (sahibinden.com,
arabam.com) are ToS-prohibited and technically defended. The brand concentration
is a property of what is legitimately reachable, not a collection shortcut — and
the 1.65× unseen-brand widening exists precisely because of it.

### The European experiment, and why it did not ship

TruckStore's European stock *is* freely reachable — `scripts/harvest_truckstore_eu.py`
pulls **1,056 tractor units** with dealer-stated Euro norm, engine, body,
kilometres and ex-VAT price, across Germany, Spain, Czechia, Romania, Portugal,
Italy, Austria and the Netherlands. Unlike the US conventionals these are the
same product as the Turkish stock, so the question was whether depreciation and
mileage slopes transfer across the border. Measured on held-out **Turkish**
listings:

| Training set | R² on TR | Median error | 80% band |
|---|---:|---:|---:|
| **`tr_only`** (84 listings) | **0.842** | **6.8%** | ±16% |
| `pooled_tr_eu` (1,140) | 0.701 | 9.4% | ±21% |
| `pooled_tr_us` (155) | 0.654 | 10.0% | ±26% |
| `pooled_tr_us_eu` (1,211) | 0.614 | 10.9% | ±22% |

Pooling loses. Fourteen times the data makes Turkish predictions worse, so
`tr_only` ships and the European rows are kept only as a measurement instrument.

It also does not fix the brand problem, because TruckStore is Mercedes' own OEM
channel: **95% of its stock is Mercedes-Benz**, against 93% Ford in Türkiye. That
makes brand very nearly collinear with market, which is precisely why the
unseen-brand widening has to be measured *within* a market.

That measurement is the most demo-relevant number in the repo, because the brief
hands you photos of a truck you have never seen:

| Held-out make | n | 80% band covered | Widening needed | Median error |
|---|---:|---:|---:|---:|
| US Kenworth | 5 | 0.80 | 1.00× | 5.2% |
| EU DAF | 6 | 0.67 | 1.55× | 10.0% |
| US Freightliner | 58 | 0.69 | 1.10× | 18.9% |
| EU Mercedes-Benz | 1,005 | 0.51 | 1.85× | 21.4% |
| EU MAN | 16 | 0.38 | 1.85× | 27.1% |
| EU Volvo | 15 | 0.00 | 3.25× | **75.1%** |
| EU Scania | 9 | 0.11 | 3.65× | **50.7%** |

A make the model has never fitted can be mispriced by half. 1.85× is the median,
and the report says out loud when it is being applied.

## Provenance and ethics

- Access posture was checked per source before harvesting; sources whose Terms
  of Use forbid scraping (TruckPaper, Machinery Trader) or that are explicitly
  prohibited and technically defended (sahibinden.com, arabam.com) were **not**
  touched.
- Listing location is read from a per-listing country field, never inferred from
  page language or currency.
- No seller PII is stored from any source. EXIF — including GPS — is recorded as
  flags and then stripped.

## Visual verification

Every original image in the corpus was visually checked, not sampled: tiled into
264 numbered contact sheets, reviewed in parallel, and every flagged cell
re-opened at full resolution before deletion. Half the flags turned out to be
wrong — real trucks wearing a dealer banner — and were kept and annotated
instead. Six images were removed (placeholder graphic, bare logo, three
diagnostic-laptop screens, one marketing page).

**No image was ever removed for being badly shot.** Blur, darkness, blown
highlights, mud and bad framing are the point.

```bash
.venv/bin/python scripts/build_contact_sheets.py <out_dir>   # tile for review
.venv/bin/python scripts/apply_manual_review.py              # apply the verdicts
```

## License plates and vehicle history

With `KAMION_HISTORY_ENABLED=1` (off by default to preserve demo behavior),
the app reads plates from individual detected vehicle crops and can match them
against an authorized records snapshot configured with `KAMION_HISTORY_DB`.
The result includes source-attributed history and explicit price reasoning.
Only VIN-confirmed history of the selected truck affects its valuation; background
vehicles are isolated. Missing history never means accident-free. Live Turkish
SBM/TRAMER access is **not connected**; see [setup and limits](docs/vehicle-history.md).
