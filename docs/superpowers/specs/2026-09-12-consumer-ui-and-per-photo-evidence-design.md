# Consumer surface + per-photo evidence

Date: 2026-09-12
Status: approved, implementing

## Why

Five complaints, from the person who has to demo this:

1. The market split (Türkiye / United States) is noise. One unified gallery of trucks instead.
2. The screen reads like a lab notebook. It should read like a product.
3. Nothing shows while the vision call runs except a fake progress bar.
4. The detection boxes light up every truck in a lot photo instead of the one being sold.
5. Per-photo analysis is shallow and misses detail.

(5) is the substantive one, and its cause is in the prompt rather than the model.
`_SCHEMA_TEMPLATE` rule 7 reads *"at most 12 issues … a `per_photo` entry ONLY for
photos that are illegible or that carry a finding"*. The system is instructed to skim.

## Decisions taken

| Question | Decision |
|---|---|
| Where does the measured/assumed material go? | Behind one "How I worked this out" disclosure. Still in the UI, so the CLAUDE.md honesty invariants hold. |
| Boxes | Subject box only by default, `show all detections` toggle. |
| Depth | Three passes: set-level identity, per-photo fan-out, text-only synthesis. |
| Gallery | All 200 corpus vehicles, rehearsed cases pinned, filters on make / year / km / photo count. |

## 1. Evidence becomes three passes

```
Pass A  identity     every selected photo, short output      ~12 s
Pass B  close-up     one call per photo, 5 concurrent        ~28 s
Pass C  synthesis    text only, no images                    ~5 s
```

Pass A keeps `same_vehicle` honest: one call still sees every frame at once, which a
text-only synthesis could never do. It is what the `mixed_vehicles` case and
`pipeline.pricing_blocker` rest on.

Pass B is the depth. Each call carries a **per-view question bank** — a tire close-up is
asked about tread across ribs versus shoulder, cupping, feathering, sidewall cracking,
DOT date, retread evidence, brand match across the axle and rim corrosion; a dashboard is
asked to read the odometer digit by digit and name every lit warning lamp. One generic
"describe this truck" prompt is what produced the shallow output.

Pass C merges. 14 photos x ~6 observations is ~80 raw findings and the same worn steer
tire appears in three frames; `Issue` gains `also_seen_in: list[int]` so corroboration
becomes one finding with three photos rather than three findings.

### Invariants this strengthens

- **Every claim carries a photo_id that was actually sent.** In pass B the call *is* one
  photo, so binding is by construction rather than by the model remembering to cite.
- **The VLM never sees or emits a price.** Unchanged — no pass is given a number.

### Module layout

`app/evidence.py` (468 lines) would reach ~900. It becomes a package:

```
app/evidence/__init__.py   public surface, unchanged import sites
app/evidence/prompts.py    COMPONENTS, both JSON schemas, the per-view question bank
app/evidence/passes.py     identity() / closeup() / synthesize()
app/evidence/stage.py      orchestration, concurrency, the on_photo callback
```

## 2. Subject box

Subject selection moves from the browser (`frames.js` picked the largest vehicle box)
into `gate.py`, scored `area_frac x centrality`, recorded as `PhotoCheck.subject_box`.
The UI and the vision call then agree on which truck is the subject.

When a photo has a subject box **and** a competing vehicle box, pass B sends the padded
crop rather than the full frame. That is the direct fix for a dealer-lot photo where the
model currently averages five trucks and reports about none of them.

The refusal colour rule is untouched: a box only reads red when the *set* was refused as
not-a-truck.

## 3. Streaming

`on_step` stays at exactly two arguments — `cli.py` and `demo.py` both pass two-parameter
callbacks and CLAUDE.md pins it. A new sibling callback carries the per-photo stream:

```
pipeline.appraise(..., on_step=, on_gate=, on_photo=)
     -> server.py pushes SSE {"type": "photo", ...}
     -> js/reasoning.js blurs a card in on the right rail
```

Cards arrive out of order, because the calls are concurrent. The rail replaces the single
narration line and the asymptotic progress bar: `7 of 14 frames read` is real.

## 4. Gallery

`GET /api/trucks` joins `listings.csv` and `images.csv` into 200 cards — cover photo
(preferring a whole-vehicle frame), make, model, year, km, photo count, quality mix.
Rehearsed cases pin to the front. Selecting a card copies the folder into a session
exactly as an upload does, so one code path runs.

Filters are client-side over 200 rows: make (10 chips, case-normalised from the 15 raw
spellings), year range, km band, photo count and capture quality.

The market selector is removed; everything prices in TRY. An American truck takes the
unseen-brand widening and says so in words.

## 5. Consumer surface

Default view: price band, condition in words, findings with their photos, what is still
needed, the truck drawing. Everything else moves into one "How I worked this out"
disclosure — R2, coverage, drivers table, FX stamp, model ids, trace timings, parse
warnings.

Numbers become words. `capture_quality 0.82` reads "sharp"; `confidence 0.87` reads
"likely, seen in 3 photos".

The backend fallback note stays on the default view in plain language. CLAUDE.md permits
falling back; it does not permit doing it quietly. Measured-versus-assumed labelling
travels into the disclosure alongside the figures it qualifies, and the measured 80.3%
stays welded to the comparable-asking band.

## 6. Cost

One appraisal goes from 1 vision call to ~16, so `python -m app.demo` goes from 8 calls to
~128. The offline suite stays free. `MAX_EVIDENCE_PHOTOS` rises 14 -> 16 because photos
are now parallel calls rather than one longer call.

## 7. Tests

- `ParseEvidence` splits into set-level and close-up parsers.
- New: subject-box selection with two trucks in frame; the crop rule; `on_photo` fires
  once per photo and a single failed call degrades rather than killing the appraisal;
  the gallery API shape.
- `StaticAssets` and `FrozenExport` pick up new files automatically, provided new modules
  keep the `from './js/x.js'` convention that `export.py:_module_url` rewrites.
