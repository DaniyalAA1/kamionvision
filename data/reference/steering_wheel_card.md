# A separate `steering_wheel` view class: measured, and not recommended

## Why this exists

`app/vision.py`'s `VIEW_PROMPTS["dashboard_odometer"]` is one class doing three
jobs — an odometer readout, a gauge cluster, and a steering wheel — and one of
its own templates says so out loud: *"a steering wheel and dash panel
photographed from inside"*. That coarseness has two visible consequences:

- `VIEW_ZONES["dashboard_odometer"]` in `app/web/js/elevation.js` establishes
  `dashboard_instruments`, `warning_lights` **and** `steering_wheel_controls`
  the instant any one dashboard-tagged frame exists, whichever of the three it
  actually shows.
- `dashboard_odometer` is one of only three entries in
  `models/gate_thresholds.json`'s `coverage.required`, so whatever wins that
  label earns the coverage credit.

So the question was asked properly: would splitting the class in two label these
frames better? It was measured before anything was wired, because the split
touches five files and a taxonomy rename is not free (`app/vision.py`,
`app/subject.py`, `app/evidence/prompts.py`, `app/web/js/elevation.js`,
`models/gate_thresholds.json`).

**The answer is no, and the numbers are not close.** `scripts/probe_steering_wheel_view.py`
reproduces every figure below.

## The labels

160 frames, hand-labelled in two passes, in `steering_wheel_labels.jsonl`.

| label | meaning | n |
|---|---|---|
| `steering` | the wheel is the subject: rim, hub, spokes, spoke controls or a column stalk fill the frame and no cluster is legible or centred | 33 |
| `gauges` | the instrument cluster / odometer is the subject, centred and legible | 15 |
| `dash_general` | the centre stack, OR a driver's-station frame where the wheel **and** the cluster are both fully assessable and neither dominates | 70 |
| `not_interior_dash` | bunk, seats, floor, storage, a decal, a chassis, an engine bay — nothing to do with a dash | 42 |

Two passes, because the class is rare and a random sample cannot measure it:

- **pass A, 48 frames sampled at random** (stratified by source and by the
  existing CLIP tag) from `dashboard_odometer` and `interior_cab`. This measures
  the *base rate*.
- **pass B, 112 further frames retrieved** — every frame corpus-wide that a
  candidate `steering_wheel` class actually wins. Retrieval inflates the
  positive rate by construction; what it buys is the new class's **precision**,
  which is the expensive error.

`dev` is 107 frames and `test` is 53. The split is **grouped by listing** and
only then balanced across labels, so no truck straddles it. Doing that the other
way round — stratify by label, group by listing *within* each label — looks right
and is not: a truck carrying both a `steering` frame and a `gauges` frame gets
assigned twice, independently, and lands in dev for one and test for the other.
The first version of this split leaked 9 listings that way, and a test in
`tests/test_offline.py` caught it. Candidate templates were chosen on `dev` only.

## Provenance, stated plainly

Labelled by Claude Opus 5 in a single session, by eye, from 400px contact-sheet
tiles, with the borderline frames re-read at full resolution. **One rater, not a
panel**, so there is no inter-rater figure to quote and none should be invented.

The tie-break, written down before pass B and applied consistently: *which
component could an inspector actually assess from this frame?* If the answer is
"both the wheel and the cluster", the frame is `dash_general`. That is not
fence-sitting — it is the finding.

Six frames were labelled in **both** passes as an accident of sampling, which
gives a one-rater self-consistency check: **5 of 6 agreed**. The one that
flipped, `tr_truckmarket_14409_008`, flipped on exactly the wheel-versus-cluster
boundary and was resolved to `dash_general` by the rule above — the Ford wheel
fills ~60% of the frame *and* the cluster is centred above it with 473,793 km
legible. The same frame, the same rater, the same rule, twice, two answers. That
is itself evidence the boundary is not crisp.

## What was measured

### The base rate kills it before the classifier is involved

Of the 48 frames sampled at random from the interior classes:

| | n | share |
|---|---|---|
| `steering` — the wheel alone is the subject | 2 | **4.2%** |
| `gauges` | 14 | 29.2% |
| `dash_general` — wheel and cluster together, or the centre stack | 19 | 39.6% |
| `not_interior_dash` | 13 | 27.1% |

A truck cab puts the wheel directly in front of the cluster, so a photograph
taken from the driver's seat contains both. The single most common real seller
photo of a driver's station is not a steering-wheel photo *or* an odometer photo
— it is both at once, and a hard single-label class has to throw one away.

### The split, best candidate, chosen on dev

Six candidate template banks were tried, so the split got its best shot rather
than a strawman: the obvious phrasing, a cab-scoped one, an explicitly
contrastive one ("not a road wheel"), one naming the column stalks as their own
template, a wheel-only one, and a single-template one. `v5_stalks_named` won on
dev — naming the stalks anchors the class inside the cab and is what stops it
grabbing engine pulleys. Held out:

| | dev | test |
|---|---|---|
| recall of genuine `steering` frames | 95.5% | 100.0% |
| **precision of the new class**, over its own claimed set | 45.7% (46 claimed) | **40.7% (27 claimed)** |
| genuine dashboard frames keeping `dashboard_odometer` | 53.2% | **48.7%** |
| non-interior frames stolen by the new class | 1/28 | 3/14 |

Against the lumped taxonomy as shipped, on the same frames: dashboard frames
keep their label **68.4% / 69.2%**. So the split **destroys one in five genuine
dashboard labels on held-out data — 69.2% down to 48.7% — to buy a class that is
wrong three times in five.**

The other five candidates are worse on both counts, and the trade is monotone:
the banks that recall the wheel well (`v1`, `v2`, `v3`, at 95-100%) are the ones
that shred dashboard retention to 45-52% and steal 9-19 of 28 non-interior
frames; the banks that keep their hands off the other classes (`v4`, `v6`)
recall only 23-41% of the wheels. There is no setting of this dial that is not a
loss.

Note what this failure is *not*. It is not the margin-requirement failure mode
from the view-framing work, where a choice looked good on 60 dev frames and
worse on 30 held-out. `v5_stalks_named` degrades in the same direction on both
halves. The split fails consistently, which is a cleaner no-go than a flip.

### What the new class actually grabs, corpus-wide

Across all 3,729 original frames, `v5_stalks_named` claims 102 (2.7%). Only **54
of those 102 come from `dashboard_odometer`**. The rest:

    damage_detail=36  interior_cab=7  exterior_side=2  exterior_rear=1
    fifth_wheel=1  chassis_undercarriage=1

More than a third of what the new class claims is a frame the taxonomy already
had somewhere else, and the frames it wrongly claims on held-out include the
rear of a tractor with its mudflaps, a road wheel lying on the ground beside a
stripped chassis, and an under-dash fuel filter and brake booster. The
less-anchored candidates are far worse: `v1_plain` claims 137 frames including
an engine accessory drive, an X15 engine bay, a power-steering box on the
chassis, a tire tread close-up and a fuel filler cap. CLIP is matching *round
mechanical object* at least as readily as *steering wheel*.

### The number that decides it: vehicles that lose a required view

Coverage in this repo is set-level, so the metric that matters is not per-frame
accuracy — it is how many **vehicles** lose a view that
`gate_thresholds.json` requires. A vehicle that loses one is asked for a
photograph its seller already sent.

| required view | vehicles with it, shipped | with the split | **lose it entirely** |
|---|---|---|---|
| `dashboard_odometer` | 166 | 163 | **3** |
| `tire_wheel` | 150 | 150 | 0 |
| `exterior_front_34` | 180 | 180 | 0 |

Three of 200 vehicles regress into a re-ask they do not deserve — `256243`,
`256436` and `257468` each had exactly one dashboard frame and the new class
takes it. Measured false refusal of the gate today is **1 of 200**. This change
would add three more, in exchange for a class that is right 40.7% of the time.
(`v3_contrastive`, the runner-up, costs 4 vehicles their dashboard view *and* one
its `tire_wheel`.)

## What the split does NOT fix, and what would

Two things worth recording, because they are real and the split was the wrong
instrument for both.

**The lumped taxonomy already mislabels these frames, just differently.** Of the
genuine `steering` frames, the shipped 11-way taxonomy calls only about half
`dashboard_odometer` — **45% land on `damage_detail`** (dev 45%, test 45%). So a
column-stalk close-up today gets the `damage_detail` question bank. That is a
pre-existing gap in the lumped taxonomy; the split "fixes" it by breaking
something worse.

**The elevation over-lighting has a cheaper cure than a taxonomy change.**
`VIEW_ZONES` only promotes a zone from *phantom* to *established* — "this area
was photographed" — and findings promote it again with a severity. So the cost
of the coarse label is bounded: it over-states coverage, never a defect. And
`interior_cab` **already** lists `steering_wheel_controls` too, so that zone's
established state does not hinge on `dashboard_odometer` alone. If the
over-lighting is worth fixing, the lever is the finding-level components that
`app/evidence/prompts.py` *already* separates — `steering_wheel_controls` versus
`dashboard_instruments` — not a coarser upstream class that CLIP cannot resolve.

## Reproducing

    .venv/bin/python scripts/cache_embeddings.py          # once, ~2 min
    .venv/bin/python scripts/probe_steering_wheel_view.py
