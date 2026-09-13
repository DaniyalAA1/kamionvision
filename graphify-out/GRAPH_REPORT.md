# Graph Report - pricing-served-blend  (2026-09-12)

## Corpus Check
- 79 files · ~83,541 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1053 nodes · 2252 edges · 57 communities (45 shown, 12 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 113 edges (avg confidence: 0.91)
- Token cost: 171,006 input · 0 output

## Community Hubs (Navigation)
- Pricing Training & Anchor
- Per-Photo Evidence Passes
- Schema Contracts & Layer Tests
- Pipeline, Estimate & Reconcile
- CLI, Demo & Report Rendering
- Server & Truck Gallery
- Gate & Vision Models
- Hackathon Plan & Brief
- Perception Heads
- Web App Shell & Price Band
- Frames & Reasoning Rail
- Live Run Stream UI
- Gallery Filters UI
- Subject Box Selection Tests
- Pricing Tests
- Dataset Path Utilities
- Truck Elevation Drawing
- Odometer OCR
- Result Panels UI
- Offline Test Misc
- Pricing Blockers
- Vision Response Parsing Tests
- Cursor Backend
- Config & Image Encoding
- Anthropic/OpenAI Backend Calls
- Image-Price Residual Probe
- Landing Page Tests
- Anchor-in-Estimate Tests
- Backend Registry & Chain
- Dataset Cleaning
- Backend Chain Tests
- VLM Backend Interface
- EU TruckStore Harvest
- Anchor Unit Tests
- Elevation Zone Tests
- Gallery Tests
- Near-Duplicate Merge Tests
- Merge Duplicates Tests
- Page Routing Tests
- Mascus Harvest
- TR TruckMarket Harvest
- Fan-Out Failure Tests
- Static Asset Tests
- Market Pooling Decisions
- Photo Degradation Twins
- Duplicate Photo Filter
- US SelecTrucks Harvest
- JSON Extraction Tests
- JSON Schema Tests
- Dataset Packaging
- Image Download
- Orphan Image Pruning
- Demo Cases & Core Invariants
- No State Only in Chat
- Train Rewrites Model Trap

## God Nodes (most connected - your core abstractions)
1. `GateReport` - 37 edges
2. `EvidenceReport` - 36 edges
3. `el()` - 31 edges
4. `VLMError` - 23 edges
5. `Issue` - 22 edges
6. `_Dict` - 20 edges
7. `reduced()` - 20 edges
8. `run()` - 19 edges
9. `appraise()` - 19 edges
10. `Appraisal` - 19 edges

## Surprising Connections (you probably didn't know these)
- `Pipeline Stage Ordering (gate->evidence->reconcile->price->report)` --references--> `pricing_blocker()`  [EXTRACTED]
  CLAUDE.md → app/pipeline.py
- `README Pipeline Overview (Gate->Heads->Evidence->Reconcile->Price->Report)` --references--> `pricing_blocker()`  [EXTRACTED]
  README.md → app/pipeline.py
- `appraise()` --calls--> `server.py (SSE endpoint pushing photo events)`  [EXTRACTED]
  app/pipeline.py → docs/superpowers/specs/2026-09-12-consumer-ui-and-per-photo-evidence-design.md
- `Per-photo SSE streaming (on_photo callback)` --implements--> `appraise()`  [EXTRACTED]
  docs/superpowers/specs/2026-09-12-consumer-ui-and-per-photo-evidence-design.md → app/pipeline.py
- `Decision 0001: Serve a Measured Blend` --references--> `estimate()`  [EXTRACTED]
  docs/decisions/0001-serve-a-measured-blend.md → app/pricing/model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Gate -> Evidence -> Price -> Report appraisal pipeline** — hackathon_plan_gate, hackathon_plan_evidence, hackathon_plan_price, hackathon_plan_report [EXTRACTED 1.00]
- **Identity -> close-up -> synthesis evidence passes with UI streaming** — spec_three_pass_evidence, app_evidence_module, app_pipeline_appraise, js_reasoning_js [EXTRACTED 0.90]
- **Turkish truck listing data sources used for price model training** — truckmarket_com_tr, autoline_com_tr, mercedes_truckstore_tr, man_topused, data_dataset_card [INFERRED 0.85]
- **Served Blend Measurement and Governance** — docs_decisions_0001_serve_a_measured_blend, app_pricing_train_fit_blend, app_pricing_model_estimate, claude_measured_blend_invariant, docs_playbook_served_price_not_measured_price [INFERRED 0.85]
- **Cross-Market Pooling Rejected on Measurement** — claude_pooled_markets_rejected, readme_european_pooling_experiment, claude_leave_one_brand_out_within_market, readme_source_vetting_turkish_one_brand [INFERRED 0.85]
- **Docs Rehydration / Continuity Chain** — docs_continuity_rehydration_order, docs_status_current_state, claude_kamionvision_appraisal_system, docs_decisions_0001_serve_a_measured_blend, docs_playbook_served_price_not_measured_price [EXTRACTED 1.00]

## Communities (57 total, 12 thin omitted)

### Community 0 - "Pricing Training & Anchor"
Cohesion: 0.05
Nodes (70): euro_norm_for_year(), blend(), estimate(), lookup(), _predict(), Price a truck from what it cost new, for when no comparable exists. The hedonic…, The anchor's own log-space error, widened for a weaker source., Inverse-variance combine in log space -> (mu, sd, anchor's weight). Nothing… (+62 more)

### Community 1 - "Per-Photo Evidence Passes"
Cohesion: 0.06
Nodes (61): Stage 2 - every condition claim bound to the photo it was seen in. What stops…, _clamp(), closeup(), _confidence(), extract_json(), identity(), identity_context(), merge_duplicates() (+53 more)

### Community 2 - "Schema Contracts & Layer Tests"
Cohesion: 0.07
Nodes (25): EvidenceReport, GateReport, Issue, PhotoPerception, What the trained heads make of one photo. Distinct from PhotoCheck on purpose:…, One defect, bound to the photo it was seen in. `photo_id` is mandatory by…, What the photos say the truck is, independent of what the seller typed., VehicleRead (+17 more)

### Community 3 - "Pipeline, Estimate & Reconcile"
Cohesion: 0.06
Nodes (50): Any, appraise(), listings(), price_model(), DataFrame, gate -> evidence -> price -> report, with a trace the judges can see. The…, `on_step(step, detail)` takes exactly two arguments and always will - `cli.py`…, Stage 3 - what the truck is worth, from comparable listings. (+42 more)

### Community 4 - "CLI, Demo & Report Rendering"
Cohesion: 0.07
Nodes (43): cmd_appraise(), cmd_demo(), cmd_doctor(), cmd_serve(), main(), Command line entry point. .venv/bin/python -m app.cli doctor .venv/bin/python…, build_fixtures(), _commons_get() (+35 more)

### Community 5 - "Server & Truck Gallery"
Cohesion: 0.07
Nodes (49): CASES with declared values filled in from the real listings. `--build` records…, resolved_cases(), cards(), _case_cover(), _demo_cards(), facets(), _load(), photo_paths() (+41 more)

### Community 6 - "Gate & Vision Models"
Cohesion: 0.07
Nodes (40): main(), _pct(), Derive the gate's reject thresholds from the corpus instead of guessing them.…, capture_metrics(), _capture_verdict(), inspect(), pick_subject(), ndarray (+32 more)

### Community 7 - "Hackathon Plan & Brief"
Cohesion: 0.05
Nodes (48): app/evidence package (prompts.py, passes.py, stage.py), gate.py (subject box scoring, PhotoCheck.subject_box), app.js (module script driving app shell), KamionVision app shell (index.html), Intake section (dropzone, seller fields, gallery), Result section (verdict, findings, working disclosure), Run section (inspection studio, live progress rail), KamionVision landing page (landing.html) (+40 more)

### Community 8 - "Perception Heads"
Cohesion: 0.09
Nodes (37): _apply(), available(), model(), PerceptionModel, photo_matrix(), ndarray, Inference for the trained perception heads. Three small linear heads sit…, PhotoChecks -> the (n, 512 + len(SCALARS)) matrix the heads expect. (+29 more)

### Community 9 - "Web App Shell & Price Band"
Cohesion: 0.15
Nodes (26): bandNote(), declaredParams(), dz, fail(), fillDeclared(), loadGallery(), loadHealth(), normalise() (+18 more)

### Community 10 - "Frames & Reasoning Rail"
Cohesion: 0.15
Nodes (26): animate(), reduced(), titleise(), viewName(), buildStrip(), CAPTURE_WORD, checkFor(), checks (+18 more)

### Community 11 - "Live Run Stream UI"
Cohesion: 0.20
Nodes (24): startOver(), reset(), setCoverage(), markCell(), setSource(), showFrame(), smokingGun(), note() (+16 more)

### Community 12 - "Gallery Filters UI"
Cohesion: 0.15
Nodes (21): anyPicked(), build(), buildFilters(), buildTools(), CAPTURE, cards, chip(), EXPECT_LABEL (+13 more)

### Community 13 - "Subject Box Selection Tests"
Cohesion: 0.17
Nodes (6): Detection, Which truck in the frame is the one being sold. Decided in the gate rather than…, The case that sent the box to the wrong truck on a real frame. On…, When the close-up call gets the crop instead of the whole frame., SubjectBox, SubjectCrop

### Community 14 - "Pricing Tests"
Cohesion: 0.14
Nodes (4): AskingPriceVerdict, Pricing, Order-of-magnitude guard, in absolute terms. The ratio test above is internally…, The comparable band for a listing IN the corpus should contain it.

### Community 15 - "Dataset Path Utilities"
Cohesion: 0.10
Nodes (12): Apply the human-in-the-loop visual review to the cleaned manifests. Every…, Tile every original image into numbered contact sheets for visual review. Each…, image_index(), listing_meta(), Canonical layout of the dataset bundle. `data/` is designed to be shared on its…, Harvested listing records for one source., Downloaded-image index for one source., Absolute or repo-relative path -> path relative to the bundle root. (+4 more)

### Community 16 - "Truck Elevation Drawing"
Cohesion: 0.16
Nodes (17): NS, svg(), svgText(), SYMBOL, timers(), tween(), ALIAS, drawRevisionMarks() (+9 more)

### Community 17 - "Odometer OCR"
Cohesion: 0.19
Nodes (16): Candidate, _demo(), _engine(), _norm(), OdometerRead, Path, Pretrained-OCR odometer reader - a specialist that grounds the mileage. This is…, Read the odometer from a single dashboard photo. Returns an `OdometerRead`;… (+8 more)

### Community 18 - "Result Panels UI"
Cohesion: 0.24
Nodes (17): el(), fixed(), kkm(), photoOrdinal(), asksAndGaps(), bullet(), identity(), IMPACT_WORD (+9 more)

### Community 19 - "Offline Test Misc"
Cohesion: 0.11
Nodes (7): BrandPalette, CaptureMetrics, ImageFormats, ParseSynthesis, Offline regression tests. No API calls, no network, ~2 s. `python -m app.demo`…, The landing page and the appraisal screen are one product. `styles/landing.css`…, A file the folder walk collects must not be rejected by the upload.

### Community 20 - "Pricing Blockers"
Cohesion: 0.21
Nodes (9): pricing_blocker(), Reasons the comparables cannot honestly price what the photos show. Returns…, Pipeline Stage Ordering (gate->evidence->reconcile->price->report), README Pipeline Overview (Gate->Heads->Evidence->Reconcile->Price->Report), PricingBlockers, Conditions where the comparables cannot honestly price what is shown., A missing read is not evidence of the wrong vehicle., I don't know what this is' is different from 'I know it's a rigid'. (+1 more)

### Community 21 - "Vision Response Parsing Tests"
Cohesion: 0.18
Nodes (4): ParseCloseup, ParseIdentity, Pass A answers what the truck is and whether it is one truck., Pass B is one call per photo, which is what makes the citation structural.

### Community 22 - "Cursor Backend"
Cohesion: 0.21
Nodes (7): BackendStatus, _classify(), CursorBackend, Path, Exception, CursorInference, Cursor routing and inference-result verification, without network calls.

### Community 23 - "Config & Image Encoding"
Cohesion: 0.21
Nodes (10): _load_dotenv(), Runtime configuration: paths, credentials, model ids, market constants.…, Read REPO/.env into os.environ without clobbering real env vars. Deliberately…, Freeze an appraisal into a single self-contained HTML file. Demo insurance.…, Anthropic Messages API backend. Two things here are load-bearing and were both…, encode_b64(), encode_jpeg(), Path (+2 more)

### Community 24 - "Anthropic/OpenAI Backend Calls"
Cohesion: 0.24
Nodes (5): AnthropicBackend, Path, VLMError, VLMResponse, Path

### Community 25 - "Image-Price Residual Probe"
Cohesion: 0.27
Nodes (12): main(), nested_r2(), probe(), DataFrame, ndarray, r2(), Step 0: is there any image signal in the price residual at all? The learned…, Out-of-fold residual of the shipped spec model, per listing. (+4 more)

### Community 26 - "Landing Page Tests"
Cohesion: 0.18
Nodes (4): HTMLParser, LandingTests, Page, Offline landing-page contract checks (no model calls).

### Community 28 - "Backend Registry & Chain"
Cohesion: 0.33
Nodes (10): available_names(), _load(), make(), probe_all(), Vision-model backends behind one interface. Three implementations ship:…, Status of every backend, in resolution order. Never raises., Every usable backend, best first. `evidence.stage` walks this rather than…, The backend an appraisal should use, or raise with every reason listed. (+2 more)

### Community 29 - "Dataset Cleaning"
Cohesion: 0.29
Nodes (9): capture_metrics(), exif_summary(), load_clip(), main(), quality_score(), Clean, tag and score every harvested image. Runs six passes over the raw…, Photographic quality signals. All cheap, all classical, no model., 0-1 capture quality. Deliberately penalises the phone-photo failure modes. (+1 more)

### Community 30 - "Backend Chain Tests"
Cohesion: 0.20
Nodes (3): BackendChain, Ordering logic only, against stub backends. Deliberately does not probe the…, A pin must not disable failover - the demo has to survive an outage.

### Community 31 - "VLM Backend Interface"
Cohesion: 0.31
Nodes (5): Interface every backend implements. `json_schema` is honoured natively where…, VLMBackend, register(), OpenAIBackend, OpenAI GPT-5.6 backend, via the Responses API. Uses `text.format = json_schema`…

### Community 32 - "EU TruckStore Harvest"
Cohesion: 0.33
Nodes (8): RuntimeError, fetch_page(), main(), parse_int(), parse_make_model(), parse_year(), Harvest Mercedes-Benz TruckStore's European tractor-unit stock. Why a European…, `dateOfRegistration` is "M/YYYY"; the month is not used anywhere.

### Community 39 - "Mascus Harvest"
Cohesion: 0.38
Nodes (5): discover(), firecrawl(), main(), Harvest Mascus tractor-unit listings for the real-amateur layer. Mascus is the…, Scrape one URL through the Firecrawl CLI, returning its text output. Retries…

### Community 40 - "TR TruckMarket Harvest"
Cohesion: 0.48
Nodes (6): clean(), discover(), fetch(), main(), parse(), Harvest Ford Trucks Turkiye's OEM used network (truckmarket.com.tr). Emits one…

### Community 41 - "Fan-Out Failure Tests"
Cohesion: 0.48
Nodes (3): _Client, FanOutFailure, One lost frame is not a lost appraisal; every lost frame is.

### Community 43 - "Market Pooling Decisions"
Cohesion: 0.40
Nodes (6): Leave-One-Brand-Out Done Within a Market, Never Across, Pooling Markets Was Measured and Rejected, Trap: A Diagnostic Can Slice the Wrong Columns and Still Print a Plausible Number, The Dataset (200 vehicles, TR+US), The European Experiment, and Why It Did Not Ship, Source Vetting: Why the Turkish Side Is One Brand

### Community 44 - "Photo Degradation Twins"
Cohesion: 0.53
Nodes (5): _apply(), build_ops(), main(), Generate paired "seller with a phone" variants of the clean listing photos. The…, Apply one precomputed recipe. Runs in a worker process.

### Community 45 - "Duplicate Photo Filter"
Cohesion: 0.47
Nodes (5): _hash(), main(), Narrow the corpus to photos that are unambiguously tied to one truck, then keep…, pHashes that appear under more than one listing, across every download., shared_photo_hashes()

### Community 46 - "US SelecTrucks Harvest"
Cohesion: 0.53
Nodes (5): detail(), discover(), main(), Harvest SelecTrucks (Daimler Truck North America's OEM used network).…, vpic_enrich()

### Community 49 - "Dataset Packaging"
Cohesion: 0.60
Nodes (4): main(), Join, normalise and package the cleaned corpus into its delivered form.…, to_usd(), write_card()

### Community 50 - "Image Download"
Cohesion: 0.67
Nodes (3): grab(), main(), Download listing images to data/images/<source>/<listing_id>/. Resumable: skips…

### Community 51 - "Orphan Image Pruning"
Cohesion: 0.67
Nodes (3): prune(), Delete image files that no manifest row references. The cleaning pass drops…, referenced_paths()

### Community 52 - "Demo Cases & Core Invariants"
Cohesion: 0.67
Nodes (3): Odometer Read Twice, Reconciled in the Open, Truck Detection Is Set-Level, Never Per-Photo, The Rehearsed Cases (eight demo cases)

## Ambiguous Edges - Review These
- `"Thin wrapper" disqualifier` → `Interactive demo playground (guided hotspot inspection)`  [AMBIGUOUS]
  app/web/landing.html · relation: conceptually_related_to

## Knowledge Gaps
- **45 isolated node(s):** `dz`, `NS`, `SYMBOL`, `VIEW_ZONES`, `ALIAS` (+40 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `"Thin wrapper" disqualifier` and `Interactive demo playground (guided hotspot inspection)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `VLMError` connect `Anthropic/OpenAI Backend Calls` to `EU TruckStore Harvest`, `Fan-Out Failure Tests`, `Offline Test Misc`, `Cursor Backend`, `Config & Image Encoding`, `Backend Registry & Chain`, `Backend Chain Tests`, `VLM Backend Interface`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `appraise()` connect `Pipeline, Estimate & Reconcile` to `CLI, Demo & Report Rendering`, `Server & Truck Gallery`, `Gate & Vision Models`, `Hackathon Plan & Brief`, `Perception Heads`, `Pricing Blockers`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `GateReport` (e.g. with `Odometer` and `Reconciliation`) actually correct?**
  _`GateReport` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `EvidenceReport` (e.g. with `condition_lines()` and `Odometer`) actually correct?**
  _`EvidenceReport` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `VLMError` (e.g. with `AnthropicBackend` and `CursorBackend`) actually correct?**
  _`VLMError` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `Issue` (e.g. with `Reconciliation` and `FallbackSynthesis`) actually correct?**
  _`Issue` has 6 INFERRED edges - model-reasoned connections that need verification._