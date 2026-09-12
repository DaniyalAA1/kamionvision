# Hackathon Build Plan — "What's This Truck Worth?"

Scoped directly to `kamion-truck-appraisal-brief.md`: a weekend build, judged live on unseen photos,
4-minute demo + 3-minute Q&A. `blueprint.md` is the long-term research doc (multi-week product build,
US+Türkiye, full observability/pricing stack) — treat it as a reference for ideas, not the weekend plan.
Nothing here needs it read first.

## What actually gets judged

| Criterion (from the brief) | What it means for the build |
|---|---|
| Does it work live, on photos we've never seen | No overfitting to one brand/channel's photo style. Test on photos from a *different* source than you trained/demo'd on. |
| Is it sensible to someone who knows trucks | Condition notes must cite something visible (a tire, a panel), not generic filler. |
| Does it know its limits | Refuse or ask for more photos on bad/non-truck input — a scored feature, not a fallback. |
| Is it interesting | More than a single VLM call — some deterministic structure the judges can see. |

**Kamion is Türkiye-only** (per the brief: "Türkiye's largest freight platform"). Drop the US/Cascadia
track entirely for this build — it has no bearing on what's being judged. Also drop the single-brand
narrowing (`blueprint.md` §9 scoped to Ford F-MAX only): the brief explicitly says judges bring
**unseen** photos, brand unstated, so a demo that only handles F-MAX is a real liability. Collect across
at least 3–4 Turkish brands.

## Architecture (buildable in a weekend, still not "a thin wrapper")

```
photos (+ optional year/km/make) ──▶ 1. Gate  ──▶ 2. Evidence  ──▶ 3. Price  ──▶ 4. Report
                                       (reject/     (condition       (comp lookup  (range +
                                       re-ask)      vector)          + adjustment)  citations)
```

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
   Fit `log(price) ~ year + km + brand` (plain linear regression or a small GBM) on the harvested
   listings — this alone explains most of the variance (`blueprint.md` §9c found R²=0.79 on Turkish
   asking price from year+km alone). Apply the VLM's condition summary as a bounded multiplicative
   adjustment (e.g. ±15% based on issue count/severity, capped so it can't dominate year/km). Output
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

Grab year, price (₺), km, brand/model, and photos for ~200–400 listings across these three. Don't chase
`blueprint.md`'s realized-sale-price problem (§9d.3, needs a dealer agreement) — for a hackathon, asking
price is fine as a rough training target; just say so out loud in the demo ("trained on asking price as
a proxy, not realized sale price — the honest caveat, not a hidden one").

## Build order

**Block 1 — data + price model.** Scrape the three sources; fit `log(price) ~ year + km + brand`; sanity
-check residuals. This alone is a demoable, if boring, baseline.

**Block 2 — gate.** Wire up YOLO truck-detection + blur/exposure check. Test it on a couple of deliberately
bad inputs (a phone photo of something that isn't a truck, a dark blurry shot) — this is a guaranteed
Q&A question ("what happens if I show it a motorcycle") so have it working and rehearsed.

**Block 3 — VLM evidence call.** Write the fixed JSON schema prompt; test on 10–15 held-out listings from
sources you *didn't* train the price model's brand encoding on, to catch overfitting to one photo style
before judges do it for you live.

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

- [ ] Ran the demo on photos from a source not used in training/dev, at least once, and it didn't fall over
- [ ] Tried a non-truck photo and a garbage/blurry photo — both handled gracefully, not silently priced
- [ ] Every condition claim in the report cites a specific photo
- [ ] Price is shown as a range, not a point estimate
- [ ] Can explain in one sentence why this isn't "a thin wrapper around a vision API"
- [ ] Screen recording + repo link ready for `hack@kamion.co`
