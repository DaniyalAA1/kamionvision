# Brainstorm — tools beyond one look at the photos

Ideas, ranked. **Not decisions** — a chosen one becomes a `docs/decisions/` entry. Every idea is
scored against the brief's four judging questions: *does it work* (live, unseen photos), *is it
sensible* (defensible to a truck buyer), *does it know its limits*, *is it interesting*.

Opened 2026-09-12. Sources for the Türkiye facts are at the bottom.

## The frame that decides the ranking

1. **Kamion is Türkiye-only.** Carfax, Kelley Blue Book, NHTSA recalls and US VIN decoding are
   US products. What matters is the Turkish equivalent, and two real ones exist (below).
2. **The brief's sharpest line is "sellers leave things out and sometimes lie."** The strongest
   additions *cross-check* a seller claim against an independent record. Adding more data sources
   matters less.
3. **Anything that needs a network call at demo time is a demo risk** (`CLAUDE.md`: the screen
   never reaches the network at run time). Official lookups should produce an *actionable next
   step* or parse a record the seller supplies, not call an API live on stage.
4. **No measurement, no claim.** This repo's edge is that every number is measured. An idea with
   no way to score it gets demoted until it has one.

## What already exists (so we don't re-propose it)

- Odometer read twice (vision model + offline OCR) and reconciled — `app/odometer.py`, `reconcile.py`.
- Tire DOT date codes and every lit warning lamp are already on the close-up checklist —
  `evidence/prompts.py` `VIEW_QUESTIONS`.
- Per-photo evidence: one vision call per photo with a view-specific checklist; duplicates merged.
- Number plate is used only as a same-vehicle cue in the identity pass. **No VIN or plate
  extraction exists.**
- Price: hedonic comparables + new-price anchor, measured (decision 0001).

## Ranked

### 1. Read the VIN and plate — as cross-checks and a handoff, not a lookup ★ recommended first

- **What:** OCR the chassis number (door-jamb/frame plate, windscreen) and the number plate.
  - **VIN decode (offline):** the first 3 characters (WMI, the manufacturer code) give the
    manufacturer; check that against the badge the vision model read. Flag a mismatch.
  - **Year check:** the 10th character is a model-year code, which is mandatory in North America
    but optional under the EU standard — verify per make on the corpus before trusting it. Where
    it holds, "seller says 2021, VIN says 2019" is exactly the lie the brief describes.
  - **Plate:** validate the Turkish format, and confirm the same plate appears across photos. This
    strengthens the `mixed_vehicles` refusal with a hard signal.
  - **Handoff:** generate the exact TRAMER query ("text `HASAR S <VIN>` to 5664") as the buyer's
    next step.
- **Judging fit:** sensible + knows-its-limits + interesting. Offline, so no demo network risk.
- **Measure first:** how many of the 200 corpus vehicles have a legible VIN or plate at all?
  Dealers often blur plates. If it is under ~15%, this is a "when present" bonus, not a pillar.
- **Privacy:** plate and VIN identify a vehicle, and the corpus rule is no seller PII. Mask them in
  the UI, never persist them, and never send them to a third party from the app.

### 2. Seller-supplied official records → mileage timeline and damage total

- **TÜVTÜRK inspection history** (official odometer at every inspection, visible to the owner on
  e-Devlet). Heavy trucks are inspected often, so the timeline is dense.
  - Accept a screenshot/PDF upload and parse it (vision call or OCR).
  - Plot km over time. Flag a rollback against the photographed odometer.
  - Feed the latest official km into pricing as a *documented* reading, which is tighter than the
    photo read.
  - Mileage is a first-class price term; condition is not. So this is worth more than any damage
    signal.
- **TRAMER reply** (the 5664 SMS answer, or the e-Devlet damage page): parse dated repair amounts.
  Sum them, flag major claims, and widen the band or add findings.
- **Judging fit:** the most "more thoughtful than the obvious approach" item on the list. It turns
  "trust us, we looked at photos" into "here is the state's own record, checked against the photos."
- **Risk:** the judges bring photos, not records. Demo it as an optional second input on one
  rehearsed case. The photo-only path must stay complete without it.
- **Measure:** needs a few real records. Ask Kamion's founder at the event whether Kamion already
  sees these; a platform could integrate SBM (the insurance information agency) officially.

### 3. Deeper damage finding than one look

- **Tiled zoom:** re-send high-res crops of tire/tread and frame-rail regions. One downscaled frame
  loses tread depth and hairline cracks.
- **Two independent readers:** a second vision backend reads the same close-ups. Disagreement
  lowers confidence through the existing `reconcile` correction path, and agreement becomes
  "seen by both." The backend chain already exists.
- **Blocker:** there is no ground-truth damage label in the corpus, so "more detailed" can't be
  shown to be "more correct." **Hand-label ~30 close-ups first** (tire wear, rust, body damage),
  then score one look vs tiled vs two-reader. Without that, this is the thin-wrapper trap in a
  nicer coat.
- **Cost:** 2× vision calls on an appraisal that already spends ~18.

### 4. Use make / model / year for more than price

- Already used: brand column, new-price row, Euro norm from year.
- **Worth doing:** model-generation-specific checklist items (known wear points per generation)
  added to `VIEW_QUESTIONS` — but only from a cited source, or it is invented expertise.
- **Not worth it:** recall lookups (the free recall API is US-only) and spec databases (no free
  Turkish one).

### 5. "Kelley Blue Book" pricing — deprioritise

- No KBB exists for Turkish heavy trucks. The big listing sites (sahibinden, arabam) are ruled out
  on ToS. The hedonic + anchor model *is* this system's blue book.
- **The real pricing weakness is the corpus, not the method:** 78 of 84 Turkish listings are Ford.
  Legitimate, ToS-clean used-stock sources for other makes (manufacturer used-truck programmes)
  would do more than any external price guide. So would refreshing the 219-day-old new-price rows.

## Where each idea plugs in (from the knowledge graph)

`graphify-out/GRAPH_REPORT.md` shows the extension point is already built. `reconcile.apply()`
(`app/reconcile.py:76`, called from `pipeline.appraise()`) runs four cross-check rules:
`_check_odometer`, `_check_identity`, `_downgrade_unsupported`, `_restore_coverage`. Each one
records a `Correction` rather than silently overwriting.

| Idea | Where it goes | Shape of the change |
|---|---|---|
| VIN make/year check | new `_check_vin` in `reconcile.py`; VIN read like `odometer.py` | Correction on `VehicleRead` year/make; widening via `price_from_evidence` provenance |
| Plate consistency | new `_check_plate`; feeds `pipeline.pricing_blocker` | hard signal for `mixed_vehicles` |
| TÜVTÜRK km record | new optional input → `_check_odometer` gains a third reading | documented km outranks both photo reads |
| TRAMER damage total | parsed record → `EvidenceReport.issues` with a `source` of "insurance record" | adjusts via existing `condition_adjustment` |
| Two-reader damage | second backend over `evidence.closeup()` → `_downgrade_unsupported` | agreement/disagreement moves `Issue.confidence` |

`GateReport` (37 links) and `EvidenceReport` (36) are the most-connected objects in the graph, so
new data should land as fields there or as a sibling report `reconcile` reads. Don't add a new
pipeline stage the UI must learn about.

## Not on the list, and arguably above all of it

- **The live demo cannot run on this machine today: there is no `.env` vision key.** Fix before
  anything else.
- **Run `app.demo` on a fresh set of unseen photos** (phone shots of any truck) and time it. "Does
  it work, without falling over" is criterion one.

## Sources

- SBM 5664 SMS queries: https://www.sbm.org.tr/tr/sms-sorgulamalari
- TRAMER by plate or VIN, 60 TL: https://turkehliyet.com/tr/blog/tramer-kaydi-sorgulama-hasar-gecmisi-sms
- TÜVTÜRK inspection km via e-Devlet: https://www.hurriyet.com.tr/egitim/km-sorgulama-nasil-yapilir-araba-ve-arac-km-ogrenme-ptt-sms-tuvturk-ve-e-devlet-42101150
