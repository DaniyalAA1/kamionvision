# Backend Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the appraisal backend more accurate by using signals it already computes, completing the measurement harness that can tell a real gain from noise, and then adding specialist readers (body type, odometer, VIN, new-price anchors) that move the price or the refusal without violating the corpus limits.

**Architecture:** Keep the existing four-stage pipeline (`gate → perception → evidence → price`). Do not add an image→price head, do not re-pool markets, and do not scrape ToS-blocked sources. Each track produces a testable backend change that either (a) refuses/re-asks more honestly, (b) grounds a number the VLM currently self-reports, or (c) turns an assumed widening into a measured one.

**Tech Stack:** Python 3.12, existing `.venv` (torch/MPS, open_clip, ultralytics, RapidOCR, scikit-learn, FastAPI). Offline tests via `unittest`. Vision-call eval via `eval/`. No new ML frameworks.

**Spec:** This plan *is* the research spec. Findings below were measured against `app/`, `eval/`, `panel/`, `data/DATASET_CARD.md`, and a grouped-CV probe of unused listing fields on 2026-09-13.

## Global Constraints

- The VLM never sees or emits a price; the regression never sees the photos.
- Truck detection is set-level, never per-photo. Measured false-refusal (1 of 200) must not regress.
- Do not fit a learned image→price head unless `scripts/probe_residual_signal.py` is re-run and beats its permuted-target null (TR R² −0.222, p = 1.00).
- Do not re-pool TR+EU or TR+US without beating `tr_only` held-out R² 0.84.
- Do not scrape sahibinden.com, arabam.com, TruckPaper, Machinery Trader, or Copart.
- Do not resurrect a US/Cascadia demo track or a Ford-F-MAX-only model.
- Condition weights stay labelled **assumed** until `scripts/probe_condition_residual.py` passes its four pre-registered tests.
- `pipeline.appraise`'s `on_step` takes exactly two arguments.
- Run `.venv/bin/python -m unittest discover -s tests` after every task. No API calls in that suite.
- Every new widening, cap, or weight that is not fitted is labelled assumed wherever it surfaces.

## What this plan deliberately does not do

These were already measured in-repo and would make accuracy *worse* or unfalsifiable:

| Idea | Why not |
|---|---|
| CLIP embedding → price residual | `scripts/probe_residual_signal.py`: TR R² −0.222 inside the null |
| Pool EU/US listings into the TR fit | `pooled_tr_eu` R² 0.70 vs `tr_only` 0.84 |
| Fit severity×impact weights on asking prices | 23 distinct TR prices / 84 listings; OEM stock is reconditioned; no damage labels |
| Fine-tune a damage YOLO without labels | `damaged` is null in 100% of 7,458 rows |
| Treat view-head teacher-agreement (0.772) as accuracy | Labels are CLIP pseudo-labels; quote twin stability (0.758) instead |
| OCR Scania's published JPG price list into `new_prices_tr.json` | Deliberate null in that file; provenance risk |

---

## Research: concrete backend gaps

The appraisal backend is `app/` (pipeline, gate, perception, evidence, reconcile, pricing, odometer, FastAPI). The JS demo screen is out of scope.

### Signals already computed, never used for a decision

1. **`subject.clusters` is counted and mentioned in prose, then ignored.** `app/subject.py` clusters winning subject-crop CLIP embeddings at `CLUSTER_COSINE = 0.75`. The comment says this is "Reported, never enforced: `same_vehicle` is the VLM's call." Mixed listings therefore spend ~50 vision calls before `pricing_blocker` can stop. `GateReport` does not even store `clusters`.
2. **`cab_type` and `axle_config` are read by pass A and never enter the price model.** Identity schema requires them (`app/evidence/prompts.py`). Pricing features are still `log1p(age) + log(km) + brand + euro6`.
3. **TR listings carry `usage_class` at 100% fill** (65 domestic / 17 international / 2 construction) and it is unused. A spec-grouped 5-fold ridge on 2026-09-13: adding usage dummies moved OOF R² **0.834 → 0.847** (Δ +0.013). International listings sit **−8.8% log-price** versus the shipped hedonic. This is a one-shot result and must beat a permutation null before shipping. `fifth_wheel_height_mm` made OOF R² *worse* (−0.003) and must not be added as a continuous term.
4. **90 hand-labelled whole-vs-part frames exist** (`data/reference/view_framing_labels.jsonl`) and are used only to justify CLIP prompt templates. The trained view head is still fitted on CLIP's own pseudo-labels, so it cannot correct the error the 90 labels measured (dashboard tagged `exterior_front`).
5. **`scripts/benchmark_detector.py` exists; `data/metadata/detector_benchmark.json` does not.** The gate still ships `yolov8n.pt` at 640. The 1-of-200 false-refusal is a recall failure a larger COCO weight was explicitly written to fix.

### Measurement harness is designed and mostly unimplemented

`eval/` has six suites. Only `twin_fp` has `run`/`score`. Stubs: `demo_gate`, `retest`, `distribution`, `monotonic`, `panel`. `retest`'s headline metric (`multiplier_sd_log`) is the number that would replace the **assumed** `condition_read` widening in `models/price_model.json`. Until it runs, condition-band geometry is a guess.

`panel/` can produce condition labels the corpus lacks. `scripts/probe_condition_residual.py` is the pre-registered test that would let weights stop being assumed. No panel store has been filled.

### Specialist readers that would actually move the number

Kilometres are a first-class hedonic term; condition is capped at one residual σ (~9.4%). OCR on the dashboard already exists. The next readers with the same shape:

- Odometer: `km` only. No `mi`/`miles`. No crop of the cluster before RapidOCR. View-tag misses skip OCR unless the VLM named that photo.
- No chassis-plate VIN path (`hackathon-plan.md` Block 6, never built). Year from a VIN would replace a typed year and shrink the "seller-stated" widening.
- No DOT-date / warning-lamp OCR. Those stay VLM self-reports.
- New-price table: Ford, MAN, Mercedes (trade-press), Scania null. Volvo/DAF/Iveco/Renault/Renault Trucks fall through to 1.85× unseen-brand widening. Anchor payoff was measured on **n=6**.

### Server (FastAPI) reliability, not model accuracy

`app/server.py`: one daemon thread per appraisal, no concurrency cap, YOLO/CLIP on MPS from overlapping threads, sessions under `/tmp` never expired. A second live run can crash the first. This is backend correctness under demo load.

### Subject retrieval is still weak on a secondary metric

`data/metadata/subject_eval.json`: the shipped algorithm beat the old one on the bugs that mattered (background-truck reject 0.35 → 0.86, distractor resist 0.48 → 0.68, whole-vehicle miss 0.031 → 0.009) but same-listing retrieval P@1 is **0.22**. Do not chase P@1 if it regresses those rates. Detector upgrade is the lever that file itself names.

---

## Approaches

**A — Use what is already computed (recommended first).** Enforce CLIP cluster / body-type / whole-vs-part / usage_class only after a false-positive budget is measured on the 200 known-single tractors. Cheap, offline-testable, no new labels.

**B — Finish the instruments, then change prompts.** Implement `eval` stubs + a panel pilot, then only keep prompt/weight changes that beat `--diff` noise. Highest long-run accuracy, costs vision calls.

**C — Specialist stack (VIN, miles, DOT, bigger YOLO, more MSRP rows).** Expands capability for unseen judge photos. Each piece is independently shippable.

Ship **A, then the cheap subset of C, then B**. Do not wait on a panel to enforce mixed-vehicle clustering; do wait on a panel to retune severity weights.

---

## File map

| File | Responsibility |
|---|---|
| `app/schema.py` | `GateReport.subject_clusters`; optional `body_type_tag` on `PhotoCheck` / gate |
| `app/subject.py` | Keep clustering; expose a calibrated mixed-vehicle verdict |
| `app/vision.py` | Zero-shot body-type prompt bank (tractor / rigid / other), max-pooled like views |
| `app/gate.py` | Record clusters + body tag; do not change the 1-in-200 refusal ladder |
| `app/pipeline.py` | `pricing_blocker` grows two pure reasons: mixed CLIP clusters, confident rigid |
| `app/perception/train.py` + `heads.py` | Binary whole-vs-part head trained on the 90 labels |
| `app/reconcile.py` | Prefer the binary whole-vs-part head over CLIP when they disagree |
| `app/pricing/features.py` + `train.py` | Optional `usage_international` column, permutation-gated |
| `app/odometer.py` | `mi`/`miles`, cluster crop |
| `app/vin.py` (new) | Chassis-plate checksum, year, WMI; abstain-heavy |
| `data/reference/new_prices_tr.json` | Add cited Volvo/DAF/Iveco/Renault rows when a typed source exists |
| `eval/suites/*.py` | Implement stubs in the order below |
| `app/server.py` | Appraisal semaphore + session TTL |
| `tests/test_offline.py`, `tests/test_layers.py` | Pure tests for every new blocker and parser |

---

### Task 1: Calibrate CLIP mixed-vehicle clustering (do not enforce yet)

**Files:**
- Create: `scripts/calibrate_mixed_vehicle.py`
- Modify: `app/schema.py` (add `subject_clusters: int = 0` on `GateReport`)
- Modify: `app/gate.py` (copy `identity.clusters` onto the report)
- Test: `tests/test_offline.py`

**Interfaces:**
- Consumes: `subject.ground` → `SubjectIdentity.clusters`
- Produces: `GateReport.subject_clusters: int`; script writes `models/mixed_vehicle_thresholds.json` with `{ "min_clusters_to_flag": int, "seed_frames_min": int, "false_positive_rate": float, "n_vehicles": 200, "basis": "measured on known-single corpus vehicles, whole-vehicle seeds only" }`

Whole-vehicle crops of one F-MAX from front vs rear can land in two CLIP clusters. Enforcing `clusters > 1` uncalibrated will false-refuse honest listings. Restrict clustering to **seed whole-vehicle frames** (`WHOLE_VEHICLE_VIEWS`, the same seeds `ground()` already uses), then measure the distribution on all 200 known-single vehicles.

- [ ] **Step 1: Write the failing test that GateReport carries cluster count**

```python
def test_gate_report_records_subject_clusters(self):
    from app.schema import GateReport
    g = GateReport(subject_clusters=2)
    self.assertEqual(g.to_dict()["subject_clusters"], 2)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_offline.PricingBlockers` after adding the method next to the existing `GateReport` / schema assertions in `tests/test_offline.py`.
Expected: FAIL (`subject_clusters` missing from `GateReport`).

- [ ] **Step 3: Add the field and copy it in `gate.run`**

```python
# app/schema.py GateReport
subject_clusters: int = 0

# app/gate.py run(), next to subject_consistency:
report.subject_clusters = identity.clusters
```

- [ ] **Step 4: Restrict `_clusters` to whole-vehicle seed winners**

Add `subject.cluster_seed_winners(winners, checks) -> int` that only embeds winners whose `check.view in WHOLE_VEHICLE_VIEWS` and who were in `identity.seeded_from`. Keep `_clusters` for the prose sentence. Unit-test with two fake embeddings at cosine 0.5 vs 0.95.

- [ ] **Step 5: Write `scripts/calibrate_mixed_vehicle.py`**

Walk `images.csv` originals, grouped by `(source_key, listing_id)`. For each of 200 vehicles, run `gate.inspect` (or reuse cached embeddings if present). Record `cluster_seed_winners`. Print the histogram and the smallest threshold whose false-positive rate on this known-single set is ≤ 0.5% (at most 1 of 200). Write the JSON artifact. Also run `demo/rigid_truck` / `demo` mixed case and print its cluster count — that is the true-positive check, not a training input.

- [ ] **Step 6: Run offline tests**

Run: `.venv/bin/python -m unittest discover -s tests`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/schema.py app/gate.py app/subject.py tests/test_offline.py scripts/calibrate_mixed_vehicle.py models/mixed_vehicle_thresholds.json
git commit -m "$(cat <<'EOF'
Record set-level CLIP vehicle clusters and calibrate a mixed-listing threshold.

The cluster count was computed and then ignored; measuring its false-positive
rate on 200 known-single tractors is required before it can block pricing.
EOF
)"
```

---

### Task 2: Enforce the calibrated mixed-vehicle block in `pricing_blocker`

**Files:**
- Modify: `app/pipeline.py`
- Modify: `app/subject.py` (load threshold from the artifact; fallback = do not block)
- Test: `tests/test_offline.py` (existing `pricing_blocker` tests)

**Interfaces:**
- Consumes: `GateReport.subject_clusters`, `models/mixed_vehicle_thresholds.json`
- Produces: `pricing_blocker(ev, gate=None) -> (headline, reason) | None` with a new reason when `gate.subject_clusters >= min_clusters_to_flag`

Do **not** put this on the gate refusal ladder. A dealer-lot set with two similar F-MAX tractors is still photos of trucks; the honest answer is "I will describe condition, I will not price a mixed set." That is `need_more_photos`, same as today's VLM `same_vehicle=False` path. The CLIP block must fire **before** evidence if possible, so a mixed set does not spend 48 close-up calls.

Placement: after `on_gate`, if `subject_clusters` meets the threshold, skip evidence entirely and return `need_more_photos` with the existing mixed-set request string. If the artifact is missing, skip (heads-refine posture: absent artifact degrades to shipped behaviour).

- [ ] **Step 1: Write the failing tests**

```python
def test_clip_clusters_block_pricing_when_calibrated(self):
    from app.pipeline import pricing_blocker
    from app.schema import GateReport, EvidenceReport
    gate = GateReport(subject_clusters=3)
    # Force the threshold in the test rather than reading disk.
    blocked = pricing_blocker(self.make(), gate=gate, min_clusters=2)
    self.assertIsNotNone(blocked)
    self.assertIn("same truck", blocked[0].lower())

def test_clip_clusters_do_not_block_below_threshold(self):
    from app.pipeline import pricing_blocker
    from app.schema import GateReport
    gate = GateReport(subject_clusters=1)
    self.assertIsNone(pricing_blocker(self.make(), gate=gate, min_clusters=2))

def test_missing_threshold_does_not_block(self):
    from app.pipeline import pricing_blocker
    from app.schema import GateReport
    gate = GateReport(subject_clusters=9)
    self.assertIsNone(pricing_blocker(self.make(), gate=gate, min_clusters=None))
```

Extend existing `pricing_blocker` tests so `gate` and `min_clusters` default to `None` and current call sites keep passing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_offline.PricingBlockers`
Expected: FAIL on unexpected keyword / no block

- [ ] **Step 3: Implement**

```python
def pricing_blocker(ev, gate=None, min_clusters=None) -> tuple[str, str] | None:
    if min_clusters is None and gate is not None:
        min_clusters = _mixed_cluster_threshold()  # None if artifact missing
    if gate is not None and min_clusters and gate.subject_clusters >= min_clusters:
        return ("These photos look like more than one truck. Send one set of the vehicle you are selling.",
                "CLIP appearance clusters among whole-vehicle frames split this set, "
                "so there is nothing coherent to price.")
    # existing same_vehicle / body_type rules unchanged
```

In `appraise`, if `pricing_blocker(None, gate=gate)` hits after the gate, return before `evidence.run`. Still fire `on_gate`. Do not set `REFUSE_NOT_A_TRUCK`.

- [ ] **Step 4: Run offline tests**

Run: `.venv/bin/python -m unittest discover -s tests`
Expected: PASS. Existing mixed_vehicles demo still expects `need_more_photos` (VLM path remains as fallback when CLIP does not fire).

- [ ] **Step 5: Commit**

```bash
git add app/pipeline.py tests/test_offline.py
git commit -m "$(cat <<'EOF'
Block pricing when calibrated CLIP clusters show mixed vehicles.

A mixed set used to spend the full evidence fan-out before same_vehicle
could stop it; the cluster count was already on the gate path.
EOF
)"
```

---

### Task 3: Zero-shot tractor vs rigid body tag at the gate

**Files:**
- Modify: `app/vision.py` (`BODY_PROMPTS`, max-pooled, same logit_scale as views)
- Modify: `app/schema.py` (`GateReport.body_tag`, `body_tag_conf`)
- Modify: `app/gate.py` (fill from CLIP on whole-vehicle frames only)
- Modify: `app/pipeline.py` (`pricing_blocker` when tag is rigid at high confidence)
- Test: `tests/test_offline.py`

**Interfaces:**
- Consumes: CLIP image features already computed in `ClipTagger.tag`
- Produces: `GateReport.body_tag: str` in `{tractor_unit, rigid, other, unknown}`; `body_tag_conf: float`

COCO cannot tell a rigid from a çekici. Today the VLM identity pass is the only body-type signal, and `pricing_blocker` only trusts it at confidence ≥ 0.55. A gate-time CLIP tag lets a rigid set skip the VLM the same way mixed CLIP clusters should.

Corpus is 100% tractor units — there are no rigid negatives to train on. Zero-shot only. True-positive check: `demo/rigid_truck`. False-positive budget: 0 of 200 corpus vehicles tagged `rigid` at the confidence used to block. If that budget cannot be met, **do not block**; only record the tag for the report.

Prompt ensemble, max-pooled, then softmax over three classes. Several templates per class, same lesson as views.

```python
BODY_PROMPTS = {
    "tractor_unit": [
        "a semi truck tractor unit with a fifth wheel coupling and no cargo box",
        "a cab-over lorry tractor, chassis ending at the fifth wheel",
        "a truck-tractor designed to pull a semi trailer",
    ],
    "rigid": [
        "a rigid box truck with a cargo body behind the cab",
        "a straight truck with an integrated box van body",
        "a dump truck or tipper with a cargo bed, not a tractor unit",
    ],
    "other": [
        "a passenger car, van or motorcycle, not a heavy truck",
    ],
}
```

Pool per-frame tags by taking the **median confidence among whole-vehicle usable frames**. Ignore close-ups.

- [ ] **Step 1: Failing tests for parse/block rules**

```python
def test_confident_rigid_tag_blocks_pricing(self):
    from app.pipeline import pricing_blocker
    from app.schema import GateReport
    gate = GateReport(body_tag="rigid", body_tag_conf=0.80)
    blocked = pricing_blocker(self.make(body_type=None, confidence=0.0), gate=gate)
    self.assertIsNotNone(blocked)
    self.assertIn("rigid", blocked[0].lower())

def test_low_confidence_rigid_tag_does_not_block(self):
    from app.pipeline import pricing_blocker
    from app.schema import GateReport
    gate = GateReport(body_tag="rigid", body_tag_conf=0.20)
    self.assertIsNone(pricing_blocker(self.make(body_type=None, confidence=0.0), gate=gate))

def test_tractor_tag_does_not_block(self):
    from app.pipeline import pricing_blocker
    from app.schema import GateReport
    gate = GateReport(body_tag="tractor_unit", body_tag_conf=0.99)
    self.assertIsNone(pricing_blocker(self.make(), gate=gate))
```

Pin `BODY_TYPE_GATE_CONF` in `pipeline.py` next to `BODY_TYPE_CONFIDENCE = 0.55`. Choose the gate threshold from the 200-vehicle sweep in the Task 3 script (below), not by guessing 0.80.

- [ ] **Step 2: Implement prompts + tagger + blocker**

Extend `ClipTagger.tag` to also return `body` / `body_conf` using the same features matrix (one extra text bank, no second image encode). `gate.inspect` averages over whole-vehicle frames.

- [ ] **Step 3: Measure false-positive rate**

Add a function to `scripts/calibrate_mixed_vehicle.py` or a tiny `scripts/calibrate_body_tag.py`: tag all 200 vehicles + `demo/rigid_truck`. Print confusion. Only set `BODY_TYPE_GATE_CONF` to a value with 0 false rigids on the 200, and a hit on the rigid demo. If no such threshold exists, ship the tag as display-only and leave `pricing_blocker` unchanged. Write the outcome into `models/gate_thresholds.json` under `body_tag`.

- [ ] **Step 4: Offline tests + commit**

Run: `.venv/bin/python -m unittest discover -s tests`

```bash
git commit -m "$(cat <<'EOF'
Tag tractor vs rigid at the gate with CLIP and block only at a measured threshold.

COCO cannot separate a box truck from a tractor unit; the identity pass was
the first place the pipeline could tell, and it cost the full evidence call.
EOF
)"
```

---

### Task 4: Whole-vs-part head trained on the 90 hand labels

**Files:**
- Modify: `app/perception/train.py` (new binary head)
- Modify: `app/perception/heads.py` (score it)
- Modify: `app/reconcile.py` (`_restore_coverage` and a new `_correct_whole_vehicle` that cannot invent a whole-vehicle view from a part frame)
- Modify: `app/gate.py` optionally: if binary head says `part` at high confidence, do not count the frame toward `has_whole_vehicle`
- Test: `tests/test_layers.py`

**Interfaces:**
- Consumes: `data/reference/view_framing_labels.jsonl` (fields: index, split, label `whole`|`part`) plus cached CLIP embeddings
- Produces: `perception.json` key `framing: { mean, scale, coef, intercept, classes: [part, whole], held_out_acc, dangerous_error }`
- `PhotoPerception.framing: str`, `framing_conf: float`

The pipeline acts on **whole vs part**, not on `exterior_front` vs `exterior_front_34`. The 90 labels are exactly that distinction. The 11-class head cannot unlearn CLIP's mistakes because CLIP is its teacher.

Dangerous error = a `part` frame counted as whole-vehicle (counts toward `has_whole_vehicle`, gets the exterior checklist, skips part-view crop protection). Held-out 30 frames already showed CLIP ensemble at 0% dangerous error; the head is to make that robust under degradation and to override the 11-class tag when they disagree.

- [ ] **Step 1: Failing test — a part-tagged frame does not satisfy has_whole_vehicle**

```python
def test_part_framing_cannot_create_whole_vehicle_coverage(self):
    # Build a GateReport whose only "exterior_front" photo is framing=part
    # after reconcile.apply, missing_views still includes a whole-vehicle shot
    # and blocks_pricing stays True if it was True.
```

Follow the existing reconcile coverage tests in `tests/test_layers.py`.

- [ ] **Step 2: Train the binary head**

In `perception/train.py`, load the 90 labels, join embeddings by the contact-sheet provenance documented in `view_framing_card.md`. Fit logistic regression on `dev` (indices 0–59), report accuracy and dangerous error on `test` (60–89). Refuse to ship if held-out dangerous error > CLIP ensemble's (0.0 on those 30). This head is **in addition to** the 11-class view head, which still routes question banks.

- [ ] **Step 3: Reconcile rule**

If `framing == "part"` at conf ≥ 0.70 and `gate` view is in `WHOLE_VEHICLE_VIEWS`, rewrite that photo's view to `damage_detail` if unknown, or leave the 11-class view but **drop it from `has_whole_vehicle`**. Record a `Correction(kind="framing_override", ...)`. Never the reverse (do not promote a close-up to whole-vehicle from this head without a second source) — the 90-label sample over-weights exteriors and a false promote is the costly error.

- [ ] **Step 4: Retrain artifact, offline tests, commit**

Run: `.venv/bin/python -m app.perception.train`
Run: `.venv/bin/python -m unittest discover -s tests`

```bash
git commit -m "$(cat <<'EOF'
Train a whole-versus-part head on the 90 hand labels and stop counting close-ups as exteriors.

The 11-class view head is still taught by CLIP, so it cannot correct the
dashboard-as-exterior failure the labels were collected to measure.
EOF
)"
```

---

### Task 5: Usage-class price feature, permutation-gated

**Files:**
- Modify: `app/pricing/features.py`, `app/pricing/train.py`, `app/pricing/model.py`
- Create: extend `scripts/probe_condition_residual.py` pattern OR add `scripts/probe_usage_class.py` (throwaway, nothing in `app/` imports it)
- Test: `tests/test_offline.py` (vector width / unknown usage is zero)

**Interfaces:**
- Consumes: `listings.csv` column `usage_class`
- Produces: design column `usage_international` ∈ {0, 1}; construction collapsed into domestic (n=2). Live appraisals without the field get 0 (domestic/unknown), which is the majority class, and the band does **not** claim an international discount it did not observe.

Pre-registered ship rule (write this at the top of the probe script before running):

1. Spec-grouped nested CV R² with the dummy beats the permutation null's 95th percentile (N_PERM ≥ 60).
2. Coefficient on international is **negative** (the residual probe said −8.8%; a positive sign is a bug, not a discovery).
3. Held-out 80% band coverage does not drop versus the shipped model.
4. Leave-one-brand-out within TR does not worsen MAN coverage.

If any fail, do not add the column. Report the negative in the probe output and stop.

Live photos from judges will not have TruckMarket's `Kullanım Amacı`. Do not infer usage from cab_type in this task. Inference is a later task and needs its own null.

- [ ] **Step 1: Write the probe script and run it**

Copy the GroupKFold + permute pattern from `scripts/probe_residual_signal.py`. Print R², null 95th, sign of the coefficient, coverage delta.

- [ ] **Step 2: If the four rules pass, add the column with tests**

```python
def test_unknown_usage_is_zero_not_international(self):
    from app.pricing.features import row_features, to_vector, design_columns
    feats = row_features(2021, 200000, "FORD", "TR")
    self.assertEqual(feats.get("usage_international", 0.0), 0.0)

def test_international_flag_is_one(self):
    feats = row_features(2021, 200000, "FORD", "TR", usage_class="Yurt Dışı Lojistik")
    self.assertEqual(feats["usage_international"], 1.0)
```

- [ ] **Step 3: Refit**

Run: `.venv/bin/python -m app.pricing.train`
Confirm printed OOF R² / coverage / median APE versus the numbers in README (0.84 / 80.3% / 4.2%). If coverage drops, revert the column.

- [ ] **Step 4: Commit only if the ship rule passed**

```bash
git commit -m "$(cat <<'EOF'
Add an international-usage dummy to the TR hedonic after it beat a permutation null.

Domestic vs international logistics is filled on every Turkish listing and
moved grouped-CV R²; construction (n=2) is collapsed rather than given a column.
EOF
)"
```

---

### Task 6: Odometer — miles, cluster crop, view-miss recall

**Files:**
- Modify: `app/odometer.py`
- Modify: `app/reconcile.py` (`_check_odometer` candidate selection)
- Test: `tests/test_layers.py` (existing odometer tests)

**Interfaces:**
- Consumes: dashboard photo path
- Produces: `OdometerRead.km` still in kilometres (convert miles × 1.60934), `rule` may be `joined_miles` | `adjacent_miles` | `joined` | `adjacent`

Today `_JOINED` is `^0?(\d{4,7})km$`. A US or mixed cluster with `242780mi` abstains. Judges may still bring a km cluster; miles is for completeness and for US corpus dashboards already on disk.

Also: crop to the lower-middle third (instrument cluster) before OCR when the frame is a wide cab shot tagged `dashboard_odometer` with a subject box. RapidOCR on a full 4K dash-plus-windscreen is where trip meters and `km/h` win.

Candidate expansion in reconcile: if no `dashboard_odometer` tag, also try usable photos whose CLIP `view` softmax's second class is dashboard, or whose `shows` from a prior run is dashboard. For this task, cheaper rule: try top-3 capture-quality frames whose view_conf for dashboard would have been ≥ 0.25 if you expose `view_p` — if that requires threading probabilities through PhotoCheck, skip and only add miles + crop.

- [ ] **Step 1: Failing tests**

```python
def test_joined_miles_converts_to_km(self):
    from app import odometer as O
    # Unit-test the selector, not RapidOCR: feed fake tokens.
    km = O._select([("242780mi", 0.9, [0, 0, 10, 10])])
    self.assertEqual(km, round(242780 * 1.60934))

def test_kmh_still_rejected(self):
    km = O._select([("90km/h", 0.99, [0, 0, 10, 10])])
    self.assertIsNone(km)

def test_trip_decimal_still_rejected(self):
    km = O._select([("976.6km", 0.99, [0, 0, 10, 10])])
    self.assertIsNone(km)
```

Refactor `read()` so `_select(tokens) -> OdometerRead` is pure.

- [ ] **Step 2: Implement miles regex and conversion; keep KM_CEILING in km**

- [ ] **Step 3: Optional cluster crop** — if `subject_box` exists and covers < 50% of the frame, OCR the padded box; on abstain, retry the full frame. Test that a tiny 40px crop is not used (`CROP_MIN_SIDE_PX` analogue, 80px).

- [ ] **Step 4: Offline tests, commit**

```bash
git commit -m "$(cat <<'EOF'
Read mile odometers and OCR the cluster crop before the full dashboard frame.

The selector already abstains rather than emit a wrong six-figure mileage;
miles and a tighter crop extend that without changing the conflict rule.
EOF
)"
```

---

### Task 7: Chassis-plate VIN (optional input, never required)

**Files:**
- Create: `app/vin.py`
- Modify: `app/vision.py` or `app/gate.py` — add view prompts for `chassis_plate` only if it does not steal probability from `dashboard_odometer`. Safer: do not add an 12th view class. Run VIN OCR on frames whose CLIP content is keep and whose YOLO finds no large truck box (plate close-up) plus any frame the identity pass already called a plate in `badges_seen`.
- Modify: `app/reconcile.py` — fifth rule: `vin_year` vs declared year
- Modify: `app/pipeline.py` — if VIN year disagrees with declared year past 1 year, widen 1.15× (assumed) and surface both; if no declared year, supply VIN year as the priced year with the existing "read off the plate" provenance, same as odometer_recovered.
- Test: `tests/test_layers.py`

**Interfaces:**
- `vin.read(path) -> VinRead(vin: str | None, year: int | None, wmi: str, check_ok: bool, confidence: float, reason: str)`
- Check digit: transliterate, weights 8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2, sum mod 11, remainder 10 → X. **Warn on failure, do not reject** (European trucks can fail).
- Characters I, O, Q never appear; map OCR `I`→`1`, `O`→`0`, `Q`→`0`.
- Position 10 → model year, 30-year cycle; if position 7 is a letter, 2010+ cycle.
- Do **not** call NHTSA vPIC. Positions 4–8 are manufacturer-specific and useless for a Turkish Actros.

Abstain unless the cleaned string is 17 chars after substitution. A wrong VIN year is worse than none.

- [ ] **Step 1: Pure checksum tests (no OCR)**

```python
def test_check_digit_accepts_a_vin_from_the_us_corpus(self):
    import pandas as pd
    from app.config import LISTINGS_CSV
    from app.vin import check_digit_ok
    vin = (pd.read_csv(LISTINGS_CSV)["vin"].dropna().astype(str)
             .loc[lambda s: s.str.len() == 17].iloc[0])
    self.assertTrue(check_digit_ok(vin), vin)

def test_ioq_substituted_before_length_check(self):
    from app.vin import normalise
    self.assertEqual(normalise("IOQ"), "100")
```

`listings.csv` has 71 US VINs; the test reads one from disk rather than inventing a passing string.

- [ ] **Step 2: Implement `normalise`, `check_digit_ok`, `model_year`, `read` using RapidOCR**

- [ ] **Step 3: Reconcile rule mirrored on odometer_*** (`vin_recovered`, `vin_conflict`)

- [ ] **Step 4: Offline tests, commit**

```bash
git commit -m "$(cat <<'EOF'
Read a chassis-plate VIN when one is photographed and reconcile year in the open.

The plate is optional; a missing shot leaves the pipeline unchanged, and a
failed European check digit warns rather than rejects.
EOF
)"
```

---

### Task 8: New-price anchors for the makes judges actually bring

**Files:**
- Modify: `data/reference/new_prices_tr.json`
- Modify: nothing in `anchor.py` if rows follow the existing schema (`brand`, `model`, `list_price`, `source_url`, `source_type`, `as_of`, `source_note`)
- Test: existing anchor tests in `tests/test_layers.py`

**Rule:** Only add a row with a **typed** (HTML/PDF text) official or trade-press figure, dated. Do not OCR Scania's JPG. Brands to search, in order: Volvo FH, DAF XF, Iveco S-Way, Renault T. Mercedes already exists as `trade_press` and already widens 1.35×.

If no typed Turkish list price exists, leave the brand on 1.85× widening and document the search in `source_note` as a null row (Scania pattern).

- [ ] **Step 1: For each brand, fetch the candidate URL, extract the tractor-unit (çekici) list price, stamp `as_of`**

- [ ] **Step 2: Add rows; run doctor**

Run: `.venv/bin/python -m app.cli doctor`
Expected: new rows appear; age warning only if `as_of` > 120 days.

- [ ] **Step 3: Offline tests, commit**

```bash
git commit -m "$(cat <<'EOF'
Extend the Turkish new-price table with cited tractor-unit list prices.

Unseen-brand widening stays the fallback; an OEM or trade-press row lets the
retention curve take over instead of the 1.85× ignorance band.
EOF
)"
```

---

### Task 9: Run the detector weight benchmark and ship only if constraints hold

**Files:**
- Run: `scripts/benchmark_detector.py` (weights: yolov8n/m, yolo11m/l × 640/960)
- Modify: `app/vision.py` (`yolo()` default path) only if stage 2 wins
- Produce: `data/metadata/detector_benchmark.json`
- Test: `tests/test_offline.py` subject tests using `demo/tr_clean/000.jpg` must still pick the foreground tractor

Pre-registered ship rule (already in the script docstring): maximise same-listing retrieval P@1 subject to (a) part_view_subject_rate no higher than incumbent, (b) vehicles_with_no_truck_box no higher than incumbent (currently 1/200), (c) duplicate_vehicle_pairs_per_frame no higher.

Larger COCO weights will still call a cab-forward tractor a `bus`. That is `subject.py`'s job, not a reason to reject the weight.

- [ ] **Step 1: Run stage 1 subsample, then stage 2 on the shortlist**

- [ ] **Step 2: If an arm wins, vendor the `.pt` into `models/`, point `yolo()` at it, re-run `app.calibrate_gate`**

- [ ] **Step 3: Re-run `scripts/eval_subject.py --algorithm new` and refuse the switch if background_truck_rejected_rate falls below 0.85 or whole_vehicle_miss_rate rises above 0.015**

- [ ] **Step 4: Commit artifact + calibration + code together**

```bash
git commit -m "$(cat <<'EOF'
Ship the COCO detector weight that improved subject-crop retrieval without raising false refusals.

yolov8n was never compared; the benchmark script already stated the constraints
that keep the 1-in-200 false-refusal and the background-truck fix intact.
EOF
)"
```

If no arm wins, commit only `detector_benchmark.json` so the negative is on disk.

---

### Task 10: FastAPI — one appraisal at a time, expire sessions

**Files:**
- Modify: `app/server.py`
- Test: `tests/test_offline.py` if gallery/server tests exist; otherwise a small `tests/test_server.py` using FastAPI `TestClient`

**Interfaces:**
- `APPRAISAL_LOCK = threading.Semaphore(1)` (or 1 GPU-bound + queue). A second `/api/appraise` waits or returns 429 with `Retry-After`.
- Session directories older than 6 hours deleted on startup and after each successful result.
- `GET /api/health` includes `rapidocr: bool`, `perception: bool`, `appraisals_in_flight: int`.

YOLO/CLIP on MPS are not safe to overlap. Two demo clicks is enough to deadlock the room.

- [ ] **Step 1: Test that a second appraise while one is in-flight returns 429 or is serialised**

Use TestClient plus a fake `pipeline.appraise` that blocks on an Event.

- [ ] **Step 2: Implement semaphore + TTL**

- [ ] **Step 3: Offline tests, commit**

```bash
git commit -m "$(cat <<'EOF'
Serialise live appraisals and expire upload sessions.

Overlapping YOLO/CLIP threads on MPS and unbounded /tmp sessions were demo
failure modes, not product features.
EOF
)"
```

---

### Task 11: Implement `eval` stubs in payoff order

Order: `demo_gate` → `retest` → `distribution` → `monotonic` → `panel`.

**Files:** `eval/suites/demo_gate.py`, `app/demo.py` (`run_case` extract), `eval/suites/retest.py`, others as designed in their docstrings.

`demo_gate` first: extract `run_case(case, backend) -> dict` from `app/demo.py` so the suite and the CLI cannot drift. Add the ninth case `tr_clean_degraded` (twins of the same listing as `tr_clean`) as the docstring specifies.

`retest` second: its `multiplier_sd_log` becomes `models/price_model.json` `widening.condition_read` with `basis: measured`. Until then that widening is assumed.

Each suite's metric list is already in its module docstring — implement those metrics, do not invent new ones. Match findings with `twin_fp.cluster_observations`.

- [ ] **Step 1: `run_case` extraction + `demo_gate.run/score` + ninth fixture**

- [ ] **Step 2: `retest` photo tier then vehicle tier; write measured widening only if N meets the docstring**

- [ ] **Step 3: `distribution` + `monotonic` sharing one cached sweep**

- [ ] **Step 4: `panel` suite scoring against `panel/store.py` once a pilot exists (Task 12)**

Run: `.venv/bin/python -m unittest discover -s tests` (eval tests already in `tests/test_eval.py`)

```bash
git commit -m "$(cat <<'EOF'
Implement the demo_gate eval suite and extract a shared run_case helper.

The eight rehearsed cases were a print-and-hope runner; the suite is what
makes a grade regression fail before a vision-call experiment is trusted.
EOF
)"
```

(Separate commits per suite.)

---

### Task 12: Panel pilot, then `probe_condition_residual`

**Files:** `panel/cli.py` (already works), `scripts/probe_condition_residual.py` (already written), panel store under whatever path `panel/store.py` uses.

Run a **pilot** (`--pilot`) with 3 panelists on the small set `panel/select.py` defines. Do not change `SEVERITY_WEIGHT` / `IMPACT_WEIGHT` unless all four pre-registered rules in `probe_condition_residual.py` pass. If they fail, paste the script's drafted negative into README and leave the weights assumed. That is still a backend accuracy improvement: the severity *scale* gets a measured agreement number even when the *weights* stay assumed.

- [ ] **Step 1: `python -m panel.cli prompts --pilot --out <dir>` and run panelists independently**

- [ ] **Step 2: ingest + report; confirm `rubric_sha` matches current `prompts.py`**

- [ ] **Step 3: `python scripts/probe_condition_residual.py`; ship or record the negative**

Do not commit API transcripts that contain photos of identifiable people or dealer PII.

---

## Suggested execution order

```
Task 1-2  mixed-vehicle CLIP          # accuracy of refusal, saves ~50 VLM calls
Task 3    tractor vs rigid CLIP       # same
Task 4    whole-vs-part head          # stops the dangerous view error
Task 10   server semaphore            # demo correctness, half a day
Task 6    odometer miles + crop       # km is a first-class price term
Task 5    usage_class (if null beaten)
Task 8    new-price rows
Task 9    detector benchmark          # hours of MPS; do overnight
Task 7    VIN                         # optional input
Task 11   eval stubs                  # unlocks everything else
Task 12   panel                       # only path to fitted condition weights
```

Tasks 1–4 and 10 do not spend vision-API money. Tasks 11–12 do.

## Self-review

- Spec coverage: mixed CLIP, body tag, framing head, usage_class, odometer, VIN, anchors, detector, server, eval, panel — each has a task. Unused `cab_type`/`axle_config` in the hedonic is intentionally deferred (no listing fill on TR; do not invent a mapping from prose). DOT-date OCR deferred (consumable, not a hedonic term).
- No TBD placeholders in task steps.
- `pricing_blocker` signature is extended in Task 2 and reused in Task 3; both tasks pass `gate=` and keep old call sites working.
- Ship rules that can fail (usage_class, detector, body-tag threshold, condition weights) say what to do on failure: record the negative, do not ship the change.
