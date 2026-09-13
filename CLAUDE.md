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
albumentations, imagehash, pillow, requests, numpy, pandas, scikit-learn, fastapi, uvicorn,
rapidocr-onnxruntime + onnxruntime (the offline odometer OCR), and the three vision SDKs (openai,
anthropic, cursor-sdk). `harvest_mascus.py` shells out to the **`firecrawl` CLI** (homebrew), not a
Python package.

Credentials load from `.env` at the repo root via `app/config.py` (see `.env.example`). Nothing
reads a key at import time, so `app.cli doctor` works with none set and tells you what is missing.

No linter, no CI. Dataset scripts self-verify by printing counts; `clean_dataset.py` also writes
`data/metadata/cleaning_report.json`.

Two test layers for `app/`:

```bash
.venv/bin/python -m unittest discover -s tests    # 496 offline tests, ~5s, no API calls
.venv/bin/python -m app.demo                      # 8 end-to-end cases, spends vision calls
```

Run the offline suite on every edit; it covers JSON extraction, evidence-to-photo binding, the
price interval, the capture thresholds, the `on_step`/`on_gate`/`on_photo` callback contracts,
subject-box selection, the crop rule, near-duplicate merging, the gallery API, the nested
`/static/` route and the drawing's zone vocabulary — the things that have actually broken.
`app.demo` is the real check but now costs ~51 vision calls per case rather than one, so keep it
for before a commit that matters.

## The appraisal system (`app/`)

```
app/
  config.py      paths, credentials, model ids, FX rate - everything tunable
  schema.py      the data contract; an Appraisal serialises to JSON whole
  vision.py      lazily-loaded YOLOv8n + CLIP, pinned to MPS; the view prompts are
                 an ensemble, several templates per class, max-pooled
  subject.py     which vehicle in the frame is the one being sold, set-level
  condition.py   the one rollup the grade AND the price are both derived from
  gate.py        stage 1: refuse / re-ask / pass, deterministic, ~1s
  perception/    stage 1b: three heads trained on the corpus (heads.py, train.py)
  evidence/      stage 2: three vision passes over a closed component enum
                 prompts.py  the enums, the schemas, the per-view question bank
                 passes.py   identity() / closeup() / synthesize(), and the crop
                 stage.py    orchestration, concurrency, the on_photo stream
  odometer.py    an offline OCR (RapidOCR) that reads the dashboard mileage; reconcile's fourth rule
  reconcile.py   stage 2b: cross-checks the VLM against the heads and the OCR read, records every change
  modelspec.py   what this repo knows about a specific MODEL, not about trucks;
                 reads data/reference/models_tr.json, price-guarded on load
  identity.py    stage 2c: the four identity witnesses adjudicated into one
                 verdict, and the single place an identity widening is applied
  vin.py         chassis-plate VIN: check digit, model year, and the WMI brand
  gallery.py     the unified truck grid: 200 corpus vehicles + the rehearsed cases
  vlm/           backends: openai (GPT-5.6), cursor (Agent SDK), anthropic
  pricing/       stage 3: features, ridge fit + calibration (train.py), estimate
                 anchor.py: second route - published new price x fitted retention
  report.py      stage 4: terminal card
  pipeline.py    wires them together, emits a timed trace
  cli.py         doctor | appraise | serve | demo
  server.py      FastAPI + SSE, so the gate verdict paints before the VLM returns
  calibrate_gate.py   derives models/gate_thresholds.json from the corpus
  web/           the demo screen - static ES modules, no build step
    index.html          gallery -> run -> result, plus the working disclosure
    app.js              entry: wiring, the verdict, the dollar band, the frozen export
    js/dom.js           helpers; the Motion wrapper and the rAF value tween
    js/net.js           health, gallery, upload, the SSE stream
    js/gallery.js       the 200-truck grid and its filters, all client-side
    js/run.js           the run: the dark stage, the frame, the progress count
    js/reasoning.js     the rail: one blur-in card per finished vision call
    js/elevation.js     the truck drawing's zone state machine
    js/frames.js        contact strip, the ONE subject box, grid, lightbox
    js/band.js          the two price bands drawn apart, in lira; the dollar
                        equivalent sits under the price
    js/panels.js        the result blocks and the "how I worked this out" body
    landing.html        `/` - the hero, Kip, and the scroll story between the
                        `<!-- story:start -->` / `<!-- story:end -->` markers
    js/landing.js       Kip, the speech bubble, the cruise toggle
    js/story/           the scroll story: scroll.js (one shared reader),
                        deck.js (the six-card deck), boxes.js, reveal.js,
                        index.js (wires acts to scroll progress)
    styles/             tokens, base, gallery, run, reasoning, result, elevation,
                        landing, story
    assets/             tractor-elevation.svg, self-hosted woff2 + OFL
    assets/story/       the frozen run's six hero photos + story.json
    vendor/motion.min.js  Motion 13.2.0, MIT, vendored - never a CDN
```

```bash
.venv/bin/python -m app.cli doctor                       # backends, models, calibration
.venv/bin/python -m app.cli appraise <folder> --year 2021 --km 164374
.venv/bin/python -m app.cli serve                        # demo screen on :8000
.venv/bin/python -m app.demo --build                     # rebuild demo/ fixtures
.venv/bin/python -m app.demo --export demo_reports/      # freeze every case as offline HTML
.venv/bin/python -m app.cli appraise <folder> --html r.html   # one case, offline, self-contained
.venv/bin/python -m app.vlm.bench                        # compare backends on the real task
.venv/bin/python -m app.pricing.train                    # refit + recalibrate the price model
.venv/bin/python -m app.calibrate_gate                   # recompute gate thresholds (~2 min)
.venv/bin/python scripts/cache_embeddings.py             # CLIP embedding cache (~2 min, once)
.venv/bin/python -m app.perception.train                 # refit the three perception heads
.venv/bin/python scripts/probe_residual_signal.py        # is there image signal in price? (it says no)
.venv/bin/python scripts/freeze_story.py data/reference/story_appraisal.json
.venv/bin/python scripts/freeze_story.py data/reference/story_appraisal.json --check
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
- **Every condition claim carries a `photo_id` that was actually sent.** Since the close-up pass
  is handed exactly one photograph per call, the binding is structural rather than something the
  model has to remember - `parse_closeup` takes the id from the caller and never reads one out of
  the response. The identity pass still sees the whole set, so it keeps the old index mapping.
- **Evidence is five passes, and three of them exist because of what the others cannot see.**
  `identity()` answers make / model / body type / `same_vehicle` - do not collapse it into the
  synthesis, because a text-only pass cannot notice that photo 9 is a different truck, and
  `pipeline.pricing_blocker` and the `mixed_vehicles` case both rest on that answer. It is now
  sampled `IDENTITY_SAMPLES` times on ONE backend and gets its OWN photo selection
  (`select_identity_photos`) and its own resolution: it was the single most load-bearing call in
  the run - make picks the brand column, model picks the anchor row, body type can stop pricing -
  and it was one draw from an unmeasured distribution while every close-up got three.
  `badge()` is one full-resolution call on a crop of the grille and door, and it is deliberately
  blind to what `identity()` concluded. `closeup()` is one call per photo, sampled
  `CLOSEUP_SAMPLES` times.
  `synthesize()` is text-only. `calibrate()` is text-only and runs **between `merge_duplicates`
  and `condition.rollup`**, and neither side of that is negotiable: after the merge because the
  pre-merge list still holds three paraphrases of one worn drive tire, and before the rollup
  because the rollup is where severity becomes both the grade and the price - run it first and
  pass D's work is computed and then discarded. It may lower a severity freely, raise it by at
  most one level, and raise to `major` only with corroboration; a refused raise cannot come back
  through `price_impact`, because the price multiplies the two.
- **The four severity words have a written meaning, and the close-up is told the distance.**
  `prompts.SEVERITY_RUBRIC` defines cosmetic / minor / moderate / major on repair effort - a
  workshop morning, a component replacement - never on lira, so teaching the model what a severity
  means never puts a currency figure in front of it (`config.REPAIR_BANDS` holds the lira reading
  for the README only, and a test asserts it never reaches a prompt). Every anchor is a visually
  checkable state, "tread level with the wear bars" and not "below 1.6mm", because a state claim
  survives JPEG crush and a measurement claim does not. The close-up gets the truck's age and a km
  **band** - never the figure, which it would echo into `odometer_km` and destroy reconcile's
  fourth rule and the rehearsed `odometer_lie` case. Before this, a 2021 F-MAX at 400,000 km and
  one at 80,000 km received byte-identical prompts and there was no rubric anywhere in the repo.
- **The view tag is measured now, and it was worse than anyone had checked.** The figure this repo
  used to quote, 0.758, is twin *stability*; the trained head is fitted on CLIP's own zero-shot
  output and inherits its errors. Against 90 hand-labelled frames (60 dev, 30 held out), one
  template per class scored 83.3%/86.7% on whole-vehicle-vs-component and called **18.9%/15.8% of
  genuine component close-ups an exterior view**. That is not cosmetic: such a frame is handed the
  exterior checklist, counts toward `has_whole_vehicle`, and is exempt from every view-gated rule.
  Three or four templates per class, **max-pooled** before the softmax, takes it to 91.7%/96.7%
  and 10.8%/0.0%. Max and not mean - the templates are alternative phrasings and averaging the
  best against worse ones dilutes it. Asking whole-vs-part directly as its own binary was tried
  and was worse (76.7%, dangerous error doubled). A margin on top scored better on dev and worse
  on held-out and was dropped; that is what the split was for.
- **Splitting `dashboard_odometer` into a dashboard class and a `steering_wheel` class was
  measured and rejected.** `data/reference/steering_wheel_card.md`,
  `scripts/probe_steering_wheel_view.py`, 160 hand labels in
  `data/reference/steering_wheel_labels.jsonl` (107 dev / 53 held out, grouped by listing). The
  class is real in the schema - `evidence.COMPONENTS` has tracked `steering_wheel_controls` and
  `dashboard_instruments` separately all along - but it is not readable off a phone photo as a
  *view*. A truck cab puts the wheel directly in front of the cluster, so **2 of 48 randomly
  sampled interior frames (4.2%) show the wheel alone and 19 of 48 (39.6%) show the wheel and the
  cluster together**; a hard single-label class has to throw one of them away. Best of six
  candidate prompt banks, chosen on dev: **40.7% precision on held-out** over the 27 frames it
  claims, while genuine dashboard frames keeping `dashboard_odometer` fall from **69.2% to
  48.7%**. Corpus-wide it claims 102 frames of which only 54 came from `dashboard_odometer` (36
  came from `damage_detail`), and **3 of 200 vehicles lose `dashboard_odometer` entirely** - a
  `coverage.required` view, against a measured gate false refusal of 1 in 200 today. The trade is
  monotone across all six banks: recall the wheel well and you shred the dashboard class, leave
  the other classes alone and you recall 23-41% of the wheels. Don't re-open it without beating
  those numbers. Two things the probe found on the way, both still true and neither fixed by a
  split: the shipped 11-way taxonomy sends **45% of genuine wheel and column-stalk frames to
  `damage_detail`**, and `VIEW_ZONES` over-lighting is bounded because it only promotes a zone to
  *established* and `interior_cab` already lists `steering_wheel_controls` too - so the lever for
  that is the finding-level components, not a coarser upstream class.
- **`ClipTagger.score_bank` is how a candidate taxonomy gets measured.** `grouped_bank` and
  `pooled_softmax` are methods rather than closures inside `__init__` precisely so a probe can
  score an arbitrary prompt-bank dict through the *same* max-pooled, logit-scaled path `tag`
  uses - a second copy of that arithmetic is how a candidate gets measured as better than it is.
  A test asserts `index_reduce` appears exactly once in `vision.py` and that `tag` still goes
  through `pooled_softmax`.
- **The per-view question bank is the depth.** `prompts.VIEW_QUESTIONS` asks a tire close-up about
  tread across the ribs, cupping, sidewall cracking, DOT dates and brand match across the axle.
  The single call it replaced said "at most 12 issues" and "a `per_photo` entry ONLY for photos
  that are illegible" - it was instructed to skim, and it did. A test asserts every id in
  `vision.VIEW_LABELS` has its own checklist; falling back to the generic four is the old failure.
- **One defect seen in three frames is one finding.** Sixteen independent calls each describe the
  same worn drive tire. `merge_duplicates` takes the synthesis pass's own pairing first, then runs
  a deterministic content-word overlap pass within a single component behind it, because the model
  missed three paraphrases of one shoulder-worn tire on the first real run and they read as three
  major findings. Corroboration lands on `Issue.also_seen_in` - never dropped, and "seen in three
  photos" is a better consumer signal than a confidence decimal.
- **A close-up call that fails costs one photo, not the appraisal.** The `PhotoFinding` carries an
  `error` and the run continues; every call failing raises. Same posture as `fell_back_from`.
  A failed synthesis degrades to `_fallback_synthesis`, which groups the findings that already
  exist through `COMPONENT_SUMMARY` - losing that call should cost the prose, not the work.
- **The VLM never sees or emits a price; the regression never sees the photos.** That separation
  is the answer to "is it more than a thin wrapper".
- **Price folds are grouped by (market, brand, model, year, price).** 87 of 155 priced listings
  collapse into 19 identical-spec groups; row splits score memorisation.
- **The interval is built from out-of-fold residuals, never in-sample ones.** In-sample residuals
  after a ridge fit on ~18 groups gave an "80%" band that covered 70%.
- **The model target is log price in each market's OWN currency.** No FX inside the fit: a
  conversion is an additive constant in log space and the market dummy absorbs it exactly.
  `estimate()` is the only place a rate is applied. Getting this wrong once produced a 2021 F-MAX
  at ₺118,000,000, and every *relative* test passed because both numbers were off by the same 48×
  — hence the absolute magnitude test in `tests/test_offline.py`.
- **Leave-one-brand-out is done WITHIN a market, never across.** Brand is nearly collinear with
  market here (TR 93% Ford, EU 95% Mercedes), so a cross-market holdout measures the border and
  reports nonsense — 442% median error for Ford.
- **Pooling markets was measured and rejected.** `pooled_tr_eu` R²=0.70 and `pooled_tr_us` R²=0.65
  against `tr_only` R²=0.84 on held-out Turkish listings. 14x the data makes it worse. The
  European rows stay as a measurement instrument, not as training data. Don't re-pool without
  re-running `app.pricing.train` and beating 0.84.
- **`log_new_price` is the only column carrying the MODEL, and its gain is mostly model identity —
  say that, don't oversell it.** Before it, the model name reached the fit nowhere: it was in the
  fold key and the anchor lookup and in no term, so an F-MAX and a Cargo-derived `TRUCKS` were the
  same truck to the regression. A per-model dummy is the wrong fix (23 distinct prices across 84
  listings, and it learns nothing about a model it has never seen — the judges' case). The column
  is TR-only, missing gets the training mean plus an indicator, and it is **clipped to the
  6.85M–8.67M TRY span it was identified on**, because the coefficient is ≈+2.4 per log unit and
  extrapolating a slope fitted on a 0.24-log lever turns a dearer reference row into a 90% uplift.
  The permutation test is in the artifact and is honest: across rows p=0.000, but **across models
  the true assignment ranks 2nd of 24 (p=0.083, floor 0.042)** — with four models that test cannot
  return significance, and the best wrong assignment scores 0.9508 against the true 0.9506. Read
  it as "model identity was missing and now is not", not as "published prices carry segment". The
  generalisation evidence is split: TR:MAN|TGS held out entirely goes 23.0%→4.4% median error;
  FORD|TRUCKS held out goes 24.69%→24.69% because the clip costs exactly that gain. The clip stays.
- **The blend's two routes are no longer independent** — the same reference figure feeds the column
  and the anchor. The `max(1.0, ...)` floor in `estimate()` is what stops that becoming a band
  narrower than the measured one, and a test over five makes asserts it never does.
- **A VIN's position-10 year is a North American convention, and applying it to a European VIN
  invents a year.** Real DAF VINs from trucks built 2019–2025 decode to **1994**, and
  `pricing/model.py` falls back to `vin_year` — so a 2023 DAF was priced as a 1994 truck *and* its
  seller accused of lying about the year. `vin.model_year` now requires the North American region
  AND `_check_vin` requires the check digit: both, because roughly 1 European VIN in 11 passes the
  check digit by chance, so that test alone leaks ~9% of them.
- **Two bands, and they are not interchangeable.** The measured 80.4% coverage belongs to the
  comparable-*asking* band. The condition-adjusted band is that estimate moved by the photos and
  carries no such guarantee — never label it with the measured number.
- **Measured numbers and assumed ones are labelled differently in the output.** Measured:
  interval coverage, gate false-refusal, the 1.85× unseen-brand widening, the price model's
  residual sigma (0.0551, out-of-fold), the focus threshold, the subject area floor, and the
  view-framing accuracy against 90 hand-labelled frames. Assumed and labelled so wherever they
  surface: the 1.12×-per-missing-view widening, **the choice to cap condition at exactly one
  residual sigma**, and the severity × impact weight tables, the per-subsystem decay, the family
  weights and the coverage thresholds - this corpus has no condition ground truth, so none of
  them can be fitted. `ConditionAdjustment` carries `cap_basis` (measured) and `weights_basis`
  (assumed) as separate strings. Both were previously wrong in opposite directions:
  `pricing/model.py` called the 1σ choice a measurement and `web/js/panels.js` called the
  measured sigma an assumption.
- **Two questions the detector cannot answer stop the pricing stage** (`pipeline.pricing_blocker`,
  pure and unit-tested): is this one vehicle, and is it a tractor unit? COCO calls a rigid, a
  tipper and a tractor all "truck", and every comparable in the corpus is a tractor unit. A low
  body-type confidence does NOT block — "I don't know what this is" is not "I know it's a rigid".
- **A backend pin prefers, it does not restrict.** `vlm.resolve_chain` puts the pinned backend
  first and keeps the rest as fallbacks, and `evidence.run` walks the whole chain. A fallback is
  recorded on `EvidenceReport.fell_back_from` and shown in the trace, the card and the UI — falling
  back is allowed, doing it quietly is not.
- **The measured numbers stay in the UI; they just stop being the first thing read.** Everything
  model-shaped - R squared, out-of-fold coverage, the driver table in log space, backend ids,
  stage timings, parse warnings - lives inside the `How I worked this out` disclosure on the
  result screen. The measured-versus-assumed labelling travels there with the figures it
  qualifies, and the measured 80.4% stays welded to the comparable-asking band. What stays in the
  open, in words: the band, the condition, the findings with their photos, and the backend
  fallback note - falling back is allowed, doing it quietly is not, and a disclosure nobody opens
  would be quiet.
- **One market, one gallery.** The Turkiye/United States selector is gone and every appraisal
  asks for `market=TR`. The price model is `tr_only`, so the dropdown let someone ask a Turkish
  fit for a number in dollars; an American truck now prices in lira, takes the unseen-brand
  widening and says why, which is a rehearsed and measured path. `app/gallery.py` serves all 200
  corpus vehicles in one grid with the eight rehearsed cases pinned in front of them, and the
  filters are client-side over rows already in memory.
- **The screen never reaches the network at run time.** Fonts are self-hosted
  woff2 under `app/web/assets/fonts/`, Motion is vendored in `app/web/vendor/`.
  A demo that depends on venue wifi for its typography is a demo that can fail
  in the room. A test asserts every `/static/` reference in `index.html`
  resolves; keep it that way rather than adding a CDN link.
- **`pipeline.appraise`'s `on_step` takes exactly two arguments** (step, detail).
  `cli.py` and `demo.py` both pass two-parameter callbacks, so adding a third
  positional argument breaks every CLI appraise - and a `lambda *a` in a test
  will hide it. Anything a caller needs beyond the string goes through
  `on_gate`, `on_photo`, or a new callback, not by widening this one.
- **`on_photo` fires once per photo, out of order.** The close-ups run concurrently, so they
  finish in whatever order the provider answers. The rail prepends rather than sorts, because the
  arrival order is the one honest thing it has to say. The progress figure is a count of finished
  calls; the old asymptotic bar existed because nothing could know when a single call would return.
- **`on_gate` fires once, before the refusal return.** It hands the finished
  `GateReport` over about a second in so the screen can paint the real
  detections, view coverage and frames during the ~50 s vision call instead of
  hiding data it already has. It must fire on the refusal path too: on a
  not-a-truck refusal that frame *is* the explanation.
- **The drawing's zones are the real vocabularies, and tests pin them.**
  `app/web/assets/tractor-elevation.svg` tags 30 zones with `data-component`
  values from `evidence.COMPONENTS`; `paint_finish`, `corrosion` and
  `fluid_leaks` have no geometry and alias a panel they are observed on.
  `VIEW_ZONES` in `js/elevation.js` must cover exactly `vision.VIEW_LABELS`.
  A typo silently stops a finding from ever lighting anything up, so
  `ElevationDrawing` in the offline suite checks both directions.
- **A detection box only gets the refusal colour when the *set* was refused for
  not being a truck.** Truck detection is set-level; colouring a box red
  because one frame looks odd would contradict the gate.
- **One box, and the gate picks it.** `gate.pick_subject` decides which vehicle is being sold and
  records it on `PhotoCheck.subject_box`; the screen draws that one and dims the rest, behind a
  `show everything it detected` toggle. Deciding it in the browser meant the box a viewer saw and
  the pixels the model read were only coincidentally the same truck. The rule is two-phase. Per
  frame, every vehicle box scores on label prior × confidence × √area × centrality × edge relief:
  holding the centre is a **1.35× bonus, not an outright win**, so it loses to a rival roughly
  three times its size but still beats a whole tractor parked to the left - on
  `demo/tr_clean/000.jpg` the subject ran off the top of the frame and measured 15% of the area
  against 23% for that tractor, and it still wins by 1.22×. Across the set, candidate crops are
  embedded with the CLIP model the gate already runs and the vehicle that **recurs** becomes the
  subject; only frames where the answer is already obvious may seed that prototype, so a
  dealer-lot frame with two comparable trucks gets no vote on who the subject is. The outright
  override was the bug: any 0.26-confidence box straddling the centre pixel eliminated a
  0.95-confidence box filling a third of the frame.
- **Three rules decide the subject without consulting the view tag, because the tag is the thing
  most likely to be wrong.** `SUBJECT_MIN_AREA_FRAC` (0.04, measured against the whole-vehicle
  distribution) - no box that small is the vehicle being sold in any view. `FOCUS_SCENERY_RATIO`
  (0.55, measured at a 5% false-suppression budget) - Laplacian variance inside the box over the
  frame around it, because the subject is what the photographer focused on and the yard behind it
  is not. And the part-view rule, which does use the tag. The first two are what still hold when
  the classifier calls a dashboard `exterior_front`, which it does on roughly 7% of component
  close-ups even after the prompt ensemble.
- **`TRUCK_PART_VIEWS` and `CLOSE_UP_VIEWS` are different sets on purpose.** `damage_detail` is
  in the second and not the first. For the gate's refusal ladder it may not vouch that a set is
  genuine truck close-ups - it is the taxonomy's catch-all and it matched a parked motorcycle at
  0.42. For cropping it is unambiguously a close-up of one part. The costs run opposite ways:
  wrongly calling a frame a close-up costs one uncropped frame, and the whole frame is what a
  close-up call wants anyway; wrongly calling a close-up an exterior costs a confident,
  photo-cited description of a lorry forty metres behind it.
- **When another vehicle is in frame, the close-up call gets the crop.** `evidence.wants_crop` is
  true only when a subject box exists, covers under 70% of the frame, and something else competes
  - a lone truck is already the crop and a tire close-up has no box at all. It measures the
  **padded** rect, not the raw box: `CROP_PAD` is 8% per side, ×1.35 on area, so a subject at 60%
  used to pass the 70% test and yield a crop covering 81% while the prompt said other vehicles had
  been excluded. A confident close-up is never cropped at all - the crop can only remove the
  component the checklist is about. Measured on the rehearsed Ford set after this change: 2 of 16
  frames cropped, both genuine exterior views.
- **The image embedding never reaches the price fit; only the condition multiplier does.**
  `scripts/probe_residual_signal.py` measured whether a CLIP embedding explains the out-of-fold
  price residual, under nested vehicle-grouped CV against a permuted-target null: TR R² −0.222,
  US −0.038, pooled −0.046, all inside the null (p ≥ 0.57), across six pooling variants. The
  positive controls pass on the same pipeline (brand 76.5% vs 39% majority, market 100%), so the
  negative is the data, not the code. **Do not fit a learned image→price head on this corpus
  without re-running that probe and beating its null.** The blocker is the target, not the
  perceiver: TR has 23 distinct prices across 84 listings and both OEM sources sell reconditioned
  stock.
- **The heads refine; they never gate.** A missing `models/perception.json` degrades to the
  shipped behaviour rather than erroring, and `reconcile` deliberately does NOT clear
  `gate.blocks_pricing` — the two conditions that set it are decided before `missing_views` exists,
  so no coverage restore legitimately answers them.
- **A downgraded finding is still shown.** `reconcile.apply` lowers `Issue.confidence` and records
  a `Correction` carrying the before, the after and the reason; it never deletes. Same posture as
  `EvidenceReport.fell_back_from`: changing your mind is allowed, doing it quietly is not.
- **The odometer is read a second time, by OCR, and reconciled in the open.** `app/odometer.py`
  (RapidOCR, bundled ONNX, offline) reads the dashboard mileage independently of the VLM; `reconcile`'s
  fourth rule supplies it when the VLM missed one (`odometer_recovered`, which then feeds the price
  model a km it would otherwise lack, via the existing "read off the dashboard" widening — not a new
  one), and when the two disagree past 2% it resolves the disagreement by *how much OCR is trusted*,
  not by always deferring to the VLM. A read below the module's `MIN_CONFIDENCE` legibility floor
  abstains entirely (a wrong six-figure mileage is worse than none). A confident read that only one
  frame supports, or that another frame contradicts by backing the VLM, only widens the band 1.25×
  and shows both figures (`odometer_conflict`, keeping the VLM figure — one frame could be the
  misread). But a confident read corroborated across `ODOMETER_CORROBORATION_MIN` (2) dashboards,
  when the VLM's figure appears on none, **overrides** the priced figure (`odometer_overridden`, no
  widening — a mileage read the same off several frames is more certain, not less); because the price
  model reads `evidence.vehicle.odometer_km`, this also settles pricing's own stated-vs-odometer
  `km_conflict` when it rested on the same vision misread. Mileage is a first-class hedonic term,
  unlike condition, so a checkable second reading of it is worth more than any damage signal. Soft
  dependency: absent RapidOCR the rule no-ops, and it runs whether or not `perception.json` is
  present, since it needs only the dashboard frames and the VLM's own read.
- **The identity head may not dispute a brand it was never trained on.** The corpus has no Scania;
  a head that has never seen one still names a class, at high confidence. `reconcile` only raises
  an identity conflict when the VLM's make is in the head's own class list. A test pins this.
  `app/identity.py` applies the same test to the head's vote, and it is easy to get backwards:
  gating on the head's OWN output lets the artefact straight through, because the class it names
  is by construction one it was trained on.
- **One widening per fact.** `reconcile` records a correction when the head disputes the badge and
  another when the chassis-plate WMI does; `app/identity.py` adjudicates all four witnesses and
  owns the multiplier. Each per-rule widening is right alone and wrong together - 1.25 × 1.20 ×
  1.25 is a 1.87× band for one disagreement. `identity.merge_widening` drops the superseded ones.
  The corrections stay: the audit trail is the point, and nothing there deletes one.
- **A witness that cannot have an opinion gets no vote, and absence is never conflict.** A WMI
  missing from `data/reference/wmi.json`, or present but unverified, means unknown - it must never
  dispute a badge the vision model can plainly read. Same posture as the head's class list.
- **The badge pass is blind to what pass A concluded.** `prompts.badge_prompt` takes `make` and
  `model` and deliberately ignores both; a test asserts the string is identical with and without
  them. Naming either makes the pass a paraphrase of pass A rather than a second witness, and its
  entire value is independence. Its crop is the upper cab band, which on a dealer lot contains the
  windscreen - so transcription is restricted to what is moulded, pressed or scripted on the
  truck, which serves the no-price rule and the no-seller-PII rule at once.
- **The generation is read off the bodywork, never off the declared year.** The year is the thing
  the generation read exists to check, so a year-derived generation agrees by construction and the
  cross-check measures nothing. `generation` is not a schema enum - the legal list belongs to a
  (brand, model) pair and the schema is built before the model is known - so the closed list
  travels in the prompt text instead.
- **A generation with no visual marker is dropped, not shipped.** Its only job is to be read off a
  photograph; offering an ungroundable one in the closed list hands the model a choice it cannot
  justify from anything it can see. One DAF and two Volvo generations were dropped on this rule.
- **`data/reference/models_tr.json` is stamped and cited like the price reference, and it may not
  contain money.** `modelspec.assert_priceless` walks it on load, because this file is read into
  the prompt that assigns severity and the natural way to write a weak point is "cracks, and a
  replacement is expensive". Scoped to `models` and skipping maintainer-only keys. Both narrowings
  it needed are the same lesson: **"Euro 6" is an emission standard and "worth noting" is not a
  valuation** - a guard that fires on the most common phrases in its own subject matter gets
  switched off, and then the real leak goes through. Pinned from both directions.
- **`normalise_model` is the half of the pair that was missing.** `normalise_brand` has existed
  since the first fit, but `anchor.lookup` matched the model exactly, so "F Max", "FMAX" and
  "F-MAX 500" all missed the F-MAX row and fell through to the brand default. An unknown model
  passes through uppercased rather than becoming None - returning None loses a model the anchor
  could still match.
- **The view head's number to quote is twin agreement, not accuracy.** Its labels are CLIP
  zero-shot pseudo-labels, so accuracy against them measures agreement with a noisy teacher. What
  is real is stability: a degraded twin inherits its original's label, so 0.758 vs the teacher's
  own 0.696 is a measured robustness gain. Never report the 0.772 teacher-agreement as accuracy.
- **The anchor may claw back a widening; it may never narrow below the measured band.**
  `estimate()` floors the blended band factor at 1.0 because the 80.4% coverage belongs to the
  unwidened hedonic interval. Measured payoff, TR:MAN held out (n=6, and say the n): band
  1.85×→1.00×, coverage 0.17→0.83, median error 16.3%→3.7%.
- **`data/reference/new_prices_tr.json` is hand-curated and stamped, like `USD_TRY`.** Every row
  carries a source URL, a date and a `source_type`; `anchor.py` widens a `trade_press` row 1.35×
  against an `oem_official` one, and `app.cli doctor` warns past 120 days. A brand with no row
  simply falls back to the existing widening — Freightliner and Scania both do, correctly.
- **HEIC is registered in `config.py` at import.** iPhones shoot it by default and the brief is
  "a seller with a phone". `config.IMAGE_SUFFIXES` is the single source of truth; the CLI folder
  walk and the web upload filter both read it, and a test asserts they agree.
- **Nothing on the landing page's scroll story is authored prose about the truck.** Its whole
  argument is that this is not a thin wrapper, and a page that argued that with copywriting would
  be self-refuting. Every sentence comes out of one run that actually happened
  (`data/reference/story_appraisal.json`, 2021 Ford F-MAX, 21 photos, 231 s, cursor/gpt-5.6-sol),
  and `scripts/freeze_story.py` is the only thing allowed to put it on the page. Edit the region
  between the `story:start` / `story:end` markers by hand and `tests/test_story.py` fails, because
  it regenerates the region and diffs it. Re-freeze rather than retype. The two claims it reshapes
  rather than quotes - capitalising `subject_evidence`, dropping the grade `grade_reason` restates -
  each have their own test asserting no substance was lost.
- **The static markup is the fallback, not a second implementation.** `freeze_story.py` writes the
  whole section into `landing.html` as a readable document; `js/story/index.js` only adds
  `.is-driven`, and every pinned, absolutely-positioned or transformed rule in `story.css` is scoped
  to it. No-JS, `prefers-reduced-motion: reduce` and anything under 820px therefore all land on the
  *same* page. A test walks the stylesheet asserting `position: sticky` never escapes that scope, so
  a rule written one level too high does not silently stack six photographs on top of each other.
- **The measured 80.4% is asserted to be in the comparable-asking row and nowhere else.** It is the
  same invariant as everywhere else in the repo, but the landing page is where getting it wrong is
  worst: it would put a measured guarantee on the photo-adjusted band in the largest type on the
  site. `tests/test_story.py` reads the two `band-row`s and checks the figure appears in one and not
  the other, and that the adjusted row carries its disclaimer.
- **`vector-effect: non-scaling-stroke` takes the dash pattern out of user space too.** The subject
  box draws itself with `stroke-dashoffset`, and a perimeter computed in the viewBox's 0-1 units
  became 2.6 *screen pixels*, so the box rendered dotted rather than drawing. The perimeter is
  measured in the frame's `offsetWidth`/`offsetHeight`, which is the space the dash is really in.
- **The odometer OCR is re-read at freeze time, because `reconcile` is silent when it agrees.** That
  silence is correct in the pipeline and useless on a page whose point is the agreement, so
  `freeze_story.py` calls `odometer.read` itself and records all three figures. A test re-runs it
  against the shipped frame. RapidOCR is a soft dependency everywhere else; here its absence drops
  act 4's OCR line rather than asserting a reading nobody took.

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
| Price model | `tr_only`, R² 0.95, median error 3.6%, 80% band covers 80.4% at ±8% |
| — without `log_new_price` | R² 0.84, median error 4.2%, covers 80.3% at ±16% — same 84 rows, same folds, one column fewer; carried in the artifact as `calibration.without_new_price_column` so the gain is readable, not asserted |
| New-price anchor | retention curve R² 0.94, median error 3.1%; 5 cited reference rows |
| Model spec card | 12 models, 12 cited, 25 generations, 47 weak points; 55 sources |
| VIN manufacturer table | 32 WMIs, 32 verified against vPIC or a published recall VIN list |
| Perception heads | degradation AUC 0.985, view stability 0.758 vs 0.696, brand 79.5% vs 39% |
| Gate thresholds | calibrated on all 7,458 images; 1 of 200 vehicles false-refused |
| EU comparables | 1,056 TruckStore tractor units (95% Mercedes) — measurement only, not training |

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
