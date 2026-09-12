# Hackathon Build Plan — "What's This Truck Worth?"

Scoped directly to `kamion-truck-appraisal-brief.md`: a weekend build, 4-minute demo + 3-minute Q&A.
`blueprint.md` is the long-term research doc (multi-week product build, US+Türkiye, full
observability/pricing stack) — treat it as a reference for ideas, not the weekend plan. Nothing here
needs it read first.

> **Open question — settle this before Block 1.** An earlier draft of this plan read the brief as
> "judged live on photos we've never seen." Current read is that it's a proof of concept, with us
> supplying the demo inputs. The two readings imply different priorities, so re-read the brief and
> confirm. If judges *do* bring their own photos, restore the robustness items flagged
> **[unseen-photo only]** below to top priority.

## What actually gets judged

| Criterion (from the brief) | What it means for the build |
|---|---|
| Does the approach hold up | Is the pricing grounded in real comparable listings, with a measured error rate — not a model's opinion. |
| Is it sensible to someone who knows trucks | Condition notes must cite something visible (a tire, a panel), not generic filler. Truck vocabulary, not car vocabulary. |
| Does it know its limits | Refuse or ask for more photos on bad/non-truck input. Also a near-certain Q&A question regardless of scoring. |
| Is it interesting | More than a single VLM call — some deterministic structure the judges can see. |

**Kamion is Türkiye-only** (per the brief: "Türkiye's largest freight platform"). Drop the US/Cascadia
track entirely for this build — it has no bearing on what's being judged. Also drop the single-brand
narrowing (`blueprint.md` §9 scoped to Ford F-MAX only): even when we pick the demo trucks, a comps
model fit on one brand can't separate brand effects from age and mileage effects, so collect across at
least 3–4 Turkish brands. **[unseen-photo only]** If judges bring photos, this moves from "better model"
to "don't fall over on stage."

## Architecture (buildable in a weekend, still not "a thin wrapper")

```
photos (+ optional year/km/make) ──▶ 1. Gate  ──▶ 2. Evidence  ──▶ 3. Price  ──▶ 4. Report
                                       (reject/     (condition       (comp lookup  (range +
                                       re-ask)      vector)          + adjustment)  citations)
                                                         ▲
                                          optional: chassis-plate photo ──▶ 2b. VIN
```

**On the VIN module (2b).** Viable *because* we supply the demo photos, so we can guarantee a
chassis-plate shot exists. **[unseen-photo only]** If judges bring their own photos there will be no
chassis plate in them, the feature never fires, and it should be cut entirely. Build it last, see Block 6.

1. **Gate (deterministic, cheap, no training).**
   - Is-it-a-truck check: a pretrained COCO object detector (YOLOv8n via `ultralytics`, zero fine-tuning)
     has a `truck` class out of the box — reject/flag if no truck detected above a confidence floor.
     This alone kills the "confidently prices a motorcycle" failure mode called out in the brief.
   - Blur/exposure check: Laplacian variance for blur, mean/stddev brightness for over/under-exposure —
     a few lines, no model needed. Below threshold → "photo too blurry, please retake."
   - Coverage check: ask for a minimum count of photos and, if time allows, a cheap CLIP zero-shot
     classifier over canonical views (front, side, tire close-up, cab interior) so "10 photos of the
     same tire" doesn't pass as complete. Cut this first if short on time — least essential of the three.

2. **Evidence (VLM, grounded by prompt structure, not fine-tuned specialists).**
   A weekend has no time to train detectors (`blueprint.md` §3's specialist stack is a multi-week
   investment). Substitute: a single structured-output call to a vision-capable LLM, given *all* photos
   at once, forced into a fixed JSON schema — this is what makes it "not a thin wrapper": the schema is
   the product, not the raw model output.
   ```json
   {
     "per_photo": [{"photo_id": 0, "view": "front_3q", "issues": [
        {"type": "dent", "location": "front bumper", "severity": "slight", "confidence": 0.7}
     ]}],
     "condition_summary": {"tires": "...", "rust": "...", "body": "...", "cab_interior": "..."},
     "coverage_gaps": ["no odometer photo", "no tire close-ups"],
     "confidence": 0.0-1.0
   }
   ```
   Force every claim to name a `photo_id` — this is the cheap version of `blueprint.md`'s
   pixel→claim evidence chain (§5), and it's exactly what makes the output checkable by "someone who
   knows trucks."

3. **Price (a real regression, not vibes).**
   Fit `log(price) ~ year + log(km) + brand + euro_norm` (plain linear regression or a small GBM) on
   the harvested listings — year+km alone already explains most of the variance (`blueprint.md` §9c
   found R²=0.79 on Turkish asking price). Two cheap additions worth the extra scrape columns:
   - `log(km)`, not raw `km` — Turkish heavy-truck listings routinely show 1M+ km, and a linear term
     overweights that tail.
   - `euro_norm` — Euro 5 vs Euro 6 is a step function in this segment, not a gradient, and most
     listings state it. Likely the largest single R² gain available beyond year/km/brand. If a listing
     omits it, derive it from year: group the harvested listings by year and find where the standard
     flips (expect somewhere around 2014–2016), then mark the field as inferred rather than observed.

   Apply the VLM's condition summary as a bounded multiplicative adjustment (e.g. ±15% based on issue
   count/severity, capped so it can't dominate year/km). **Calibrate that cap instead of guessing it:**
   Turkish listings state damage history (`hasar kaydı var/yok`), so scrape it as a binary and let the
   regression tell you what a damage record actually costs in this segment. That converts the ±15% from
   a made-up number into a measured one, which is exactly the kind of thing that survives Q&A. Output
   the regression's prediction interval (or just ± the model's residual std) as the range — this
   satisfies "a range rather than a single number" without needing conformal prediction (`blueprint.md`
   §4's CQR is the funded-product version of this).

4. **Report.** Render the JSON as a short human-readable card: price range, per-issue bullets each
   citing a photo, and the coverage gaps as explicit "ask for X" prompts. This is the demo screen.

## Data collection (do this first — it's the actual bottleneck)

The brief says "a few hundred [listings], an hour of work." Reuse the access research already done in
`blueprint.md` §9e rather than re-discovering it: TruckMarket (Ford Trucks TR) is already vetted as
scrapable (no bot-blocking, plain HTML) — but pull from **more than one brand's OEM channel** this time,
since the demo needs to survive an unfamiliar brand:

| Source | Brand(s) | Status (per `blueprint.md` §9b/§9e) |
|---|---|---|
| `truckmarket.com.tr` | Ford Trucks (F-MAX, F-LINE) + some MAN/Iveco/Mercedes/Scania stock | Vetted — plain HTML, no robots.txt block |
| Mercedes-Benz TruckStore TR | Mercedes Actros etc. | Public listing pages noted, not yet scraped |
| MAN TopUsed | MAN TGX/TGS | Public listing pages noted, not yet scraped |
| `autoline.com.tr` | Mercedes, MAN, DAF, Scania, Volvo, Iveco, Renault, Ford, plus Chinese entrants | Not yet vetted — worth 20 min to check. Listings expose km, Euro norm, power, axle config and suspension as structured fields, which is exactly what §3's extra regression columns need. **Prices are quoted in EUR with a ₺ conversion**, so convert to a single currency before pooling with the OEM channels |

Grab year, price, km, brand/model, `euro_norm`, damage history (`hasar kaydı`), and photos for ~200–400
listings across these sources. One currency note: a single-day scrape in ₺ is internally consistent, so
inflation is not a within-sample problem — but the moment you pool a EUR-quoted source with ₺-quoted
ones, convert everything at the scrape-date rate and pick one modelling currency. Don't chase
`blueprint.md`'s realized-sale-price problem (§9d.3, needs a dealer agreement) — for a hackathon, asking
price is fine as a rough training target; just say so out loud in the demo ("trained on asking price as
a proxy, not realized sale price — the honest caveat, not a hidden one").

## Build order

**Block 1 — data + price model.** Scrape the sources (grab `euro_norm` and `hasar kaydı` while you're in
there — they're free at scrape time and expensive to backfill); fit `log(price) ~ year + log(km) + brand
+ euro_norm`; sanity-check residuals. This alone is a demoable, if boring, baseline.

**Block 1b — calibrate the interval (~30 min, do not skip).** Hold out 20% of listings and check what
fraction of actual prices land inside the predicted range. Report it: "our 80% interval contained the
real asking price 78% of the time across 60 held-out trucks." Everyone shows a range; almost nobody
shows theirs is correct. On a proof-of-concept judging, where the question is whether the *approach* is
sound rather than whether it survives a surprise photo, this is the strongest single piece of evidence
you can put on screen.

**Block 2 — gate.** Wire up YOLO truck-detection + blur/exposure check. Test it on a couple of deliberately
bad inputs (a phone photo of something that isn't a truck, a dark blurry shot) — this is a guaranteed
Q&A question ("what happens if I show it a motorcycle") so have it working and rehearsed. Keep this even
under the proof-of-concept reading: the question gets asked either way, and the answer is cheap.

**Block 3 — VLM evidence call.** Write the fixed JSON schema prompt; test on 10–15 held-out listings.
Use **truck vocabulary, not car vocabulary** — fifth wheel and coupling wear, air suspension bags,
chassis corrosion, and tires by position (steer vs drive). A generic dent/scratch/paint taxonomy is the
tell that a car tool was pointed at a truck, and it's what "sensible to someone who knows trucks" is
scoring. **[unseen-photo only]** Also pull those held-out listings from a source you didn't train on, to
catch photo-style overfitting.

**Block 4 — glue + report UI.** Wire gate → VLM → price adjustment → report card. Minimal front end is
fine — a script that takes a folder of photos and prints/renders the report is enough for a 4-minute demo.

**Block 5 — rehearse the failure modes.** Explicitly demo: (a) a normal case, (b) a deliberately bad/blurry
input triggering refusal, (c) a photo set missing a canonical view triggering a re-ask. The brief scores
"does it know its limits" as its own line item — don't leave it implicit, show it.

**Block 6 — VIN module (optional, ~2h, only if 1–5 are done and tested).** Include a chassis-plate photo
in the demo set and extract the number from it. Achievable without any paid decoder:
- Clean the read: 17 characters, and the letters I, O and Q never appear in a VIN — an `I` or `O` in the
  output is a misread of 1 or 0.
- Validate the check digit at position 9 (transliterate, weight 8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2, sum,
  mod 11, remainder 10 is written X). Mandated in North America and used fairly consistently elsewhere,
  so **warn on failure, don't reject** — a legitimate European truck can fail it.
- First three characters give manufacturer and country of origin. Position 10 gives model year on a
  30-year cycle; if position 7 is a letter it's the 2010+ cycle, if a digit the 1980–2009 one.
- Do **not** add a VIN decode library for specs. NHTSA vPIC covers US-market vehicles only and returns
  nothing useful for a Turkish Actros or MAN, and positions 4–8 are manufacturer-specific with no
  public map for European truck makers.

Demo it as an **optional input that visibly improves the estimate**: show the range before and after,
with the band narrowing once the year is confirmed from the VIN rather than user-entered. The pipeline
must behave identically when no chassis-plate photo is supplied.

**Block 4 — glue + report UI.** Wire gate → VLM → price adjustment → report card. Minimal front end is
fine — a script that takes a folder of photos and prints/renders the report is enough for a 4-minute demo.

**Block 5 — rehearse the failure modes.** Explicitly demo: (a) a normal case, (b) a deliberately bad/blurry
input triggering refusal, (c) a photo set missing a canonical view triggering a re-ask. The brief scores
"does it know its limits" as its own line item — don't leave it implicit, show it.

## Cut list — deliberately not building this weekend

Everything below is real and in `blueprint.md`, but is multi-week scope, not weekend scope:

- Fine-tuned specialist detectors/segmenters (YOLO/RT-DETR/SAM2 trained on truck damage) — use the
  pretrained-COCO truck-class check instead; it's enough to satisfy the gate requirement.
- Conformalized Quantile Regression — a plain residual-based interval is enough for "a range."
- Bayesian hierarchical / residual-correction pricing, internal-transaction posterior — no internal
  transactions exist yet; this only matters once Kamion has closed sales to learn from.
- Full multi-agent DAG (LangGraph/Temporal/Ray), OTel + Langfuse observability — nice for a funded
  product's audit trail, not gradable in a 4-minute demo. A printed JSON trace of the pipeline steps
  covers "is it more than a thin wrapper" just as well for this purpose.
- US market entirely — Kamion has no US operations per the brief; drop it for this build.
- EU AI Act / GDPR-KVKK compliance work, TAM/SAM/SOM — not evaluated by hackathon judging criteria.

## Judging-criteria self-check before submitting

- [ ] Interval calibration measured and stated as a number, not a claim
- [ ] Every condition claim in the report cites a specific photo
- [ ] Condition vocabulary is truck-specific, not generic car damage terms
- [ ] Price is shown as a range, not a point estimate
- [ ] Tried a non-truck photo and a garbage/blurry photo — both handled gracefully, not silently priced
- [ ] Said out loud that training target is asking price, not realized sale price
- [ ] Can explain in one sentence why this isn't "a thin wrapper around a vision API"
- [ ] Demo run end-to-end three times on the actual presentation machine
- [ ] Backup screen recording exists in case the live run fails
- [ ] Screen recording + repo link ready for `hack@kamion.co`
- [ ] **[unseen-photo only]** Ran the demo on photos from a source not used in training/dev, at least
      once, and it didn't fall over
