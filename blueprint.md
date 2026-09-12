# Vision-First Multi-Agent Truck Appraisal System for Kamion: A Technical Blueprint

## TL;DR
- **Build a hybrid architecture**: specialist vision models (YOLO/RT-DETR detectors, SAM 2 segmentation, DINOv2 embeddings, a dedicated no-reference image-quality gate) produce a structured, auditable **condition vector**, and a VLM (Claude/Gemini/Qwen-VL) reasons over that evidence — never a VLM alone, which hallucinates damage and cannot localize. A two-stage pricing engine (public-data prior + internal-transaction posterior via Bayesian hierarchical / residual-correction modeling) converts the condition vector into a price *range* with conformal-calibrated confidence tied to photo quality.
- **The single biggest constraint is data**: there is **no large, commercially-licensed public dataset of commercial-truck damage**. The only genuine one (DS4E "Truck Damage Detection," 4,962 images, CC BY 4.0) is small and single-author. Car-damage sets (CarDD 4,000 imgs; VehiDE 13,945 imgs) are non-commercial (Flickr/Shutterstock encumbered) and passenger-car only. Kamion **must bootstrap a proprietary truck/trailer/van dataset** and can only use public sets for transfer-learning R&D and the quality gate.
- **"Perfect observability" is achievable** by combining OpenTelemetry GenAI semantic conventions + a self-hosted trace backend (Langfuse or Arize Phoenix) with a domain-specific **evidence-grounding layer**: every price delta chains back pixel → bounding-box/mask → detected defect → condition score → price adjustment, with per-claim confidence, exact model/prompt versions, and replayability.
- **Make/model/configuration is a gate, not a covariate** (§3a, §4a). A US Cascadia is a 6×4 conventional with a hood; a Turkish F-MAX is a 4×2 cab-over without one. They share no part geometry, no damage priors and no repair-cost table. An identity stage must resolve make → model → generation → configuration *before* any damage model runs, and it must work from pixels alone in Türkiye, where no free VIN decoder exists.
- **The two markets are structurally inverted, and this is measurable** (§7, §9). Live inventory sampled 2026-09-12: US listings carry VINs, engine/HP/sleeper/wheelbase specs and ~15 photos; Turkish listings carry 35 photos at 1440×1080 and no VIN at all. US asking price leaves a **±22.6%** residual after year+mileage — the space a condition model can win. Turkish asking price leaves **±5.0%**, almost all of it quantization (20 distinct prices across 147 listings). **Learn vision where the pixels are (Türkiye); learn price where the labels are (US).**
- **Size the software, not the trucks.** The US+TR used-heavy-truck market moves ~$20–23B/yr in vehicle value, but an assessment-only product addresses ≈**$143M TAM / $22M SAM / $1.8–2.2M 3-year SOM** (§8). Per assessed truck the same appraisal is worth ~**13× more** as a transaction take-rate than as a report (~$600 vs ~$45; ACV's auction+assurance ARPU is **$554/unit**) — **build the appraisal as the wedge, monetise the transaction.**

## Key Findings

1. **Data is the gating problem, not modeling.** Public truck-damage data barely exists. Kamion's competitive moat is its own labeled transaction+photo corpus. Plan for a data-collection program from day one (guided capture at driver onboarding + expert labeling).
2. **VLMs alone are insufficient but excellent reasoners.** Published evidence (Hogale et al., arXiv 2608.02470, "Grounding Agentic VLMs with Dedicated Segmentation for Fine-Grained Vehicle Damage Assessment"): a state-of-the-art VLM (Qwen-VL-2B, 4-bit GPTQ) "achieves strong semantic classification accuracy (87.3%) on this task but is systematically ungrounded at the spatial level: it hallucinates damage in reflective regions, misses elongated scratches entirely" — while the paper's dedicated segmentation approach far outperforms VLM localization. Use specialists for detection/localization, VLM for synthesis and narrative.
3. **A dedicated image-quality gate is mandatory and cheap.** Modern NR-IQA (MUSIQ, MANIQA, CLIP-IQA, Q-Align) is available off-the-shelf via `pyiqa`/IQA-PyTorch. But generic IQA answers "is this photo technically good," not "does this photo show the part I need" — Kamion needs a separate coverage/completeness classifier.
4. **Market-agnostic pricing is best done as prior+posterior.** Train a hedonic GBM on public comparables (currency-normalized, region-indexed, log-price target), then learn a residual correction / Bayesian hierarchical layer that up-weights Kamion's internal sales as they accumulate. Monotonic constraints keep mileage subordinate to the vision-derived condition delta.
5. **Uncertainty must be first-class.** Conformalized Quantile Regression (CQR) gives calibrated price intervals with finite-sample coverage guarantees; interval width should widen automatically when photo quality/coverage is poor, up to a refusal-to-quote threshold.
6. **Regulatory exposure is real.** A vehicle valuation tool is not automatically EU AI Act "high-risk," but photos containing plates/faces/GPS trigger GDPR and Turkey's KVKK. Turkish marketplace data (sahibinden, arabam) is largely off-limits for scraping — confirmed in testing: sahibinden returns an error page and arabam.com serves a Cloudflare managed challenge on every request, including `robots.txt`.
7. **Identity must gate the vision stack.** Make/model/generation/configuration determines which part segmenter is valid, which damage priors apply, and which repair-cost table converts a defect into money. Getting the model wrong produces a confident, silently wrong appraisal — the worst available failure mode. See §3a.
8. **A usable pair of demo datasets exists today and was measured, not assumed.** US: **385** Freightliner Cascadia MY2019–2022 on SelecTrucks (`robots.txt: Allow: /`, JSON inventory API, full VIN + spec per listing, full-res 2500×1875 images, free NHTSA vPIC enrichment verified on live VINs). TR: **218** listings on Ford Trucks' TruckMarket, **168 F-MAX**, **32–38 photos each at 1440×1080** — ~7,600 images available right now. The overlap window is **MY2020–2022**, not 2019–2022: the Turkish source has zero MY2019 listings. See §9.
9. **Turkish asking prices are not a valid training target.** 20 distinct price points across 147 listings; 62% of listings sit on the five most common prices; 76% are divisible by ₺100,000; year+mileage alone give R²=0.791 with a ±5.0% residual. Turkish dealers price off a rule-of-thumb table, not off condition. **Obtaining 300–500 realized Turkish sale prices with matching photos is the gate on the entire Turkish pricing claim** and nothing substitutes for it.
10. **Nominal TRY hides real depreciation.** Fitted on the same live inventory: −21.8%/yr in USD vs −10.9%/yr in nominal TRY, against ~31% CPI. Train on deflated real prices with an as-of date; emit nominal TRY. A model fitted on nominal TRY will conclude Turkish trucks barely depreciate and will over-value old stock.
11. **Mileage means different things in the two markets.** At the same model years, US Cascadias carry **2.69×** the odometer of Turkish F-MAXes (726,447 km vs 270,000 km median) at 1.24× the price — $96 vs $210 of value per 1,000 km driven. A shared mileage coefficient is wrong in both directions.

## Details

### 1. DATASET RESEARCH

#### 1a/1b. Vehicle & commercial-vehicle damage datasets

| Dataset | Size | Classes | License | URL | Notes |
|---|---|---|---|---|---|
| **CarDD** | 4,000 hi-res imgs, >9,000 instances | 6: dent, scratch, crack, glass shatter, lamp broken, tire flat | Non-commercial research only (Flickr/Shutterstock copyright) | arxiv.org/abs/2211.00945; cardd-ustc.github.io; HF mirror harpreetsahota/CarDD | Largest well-curated academic car-damage set; supports classification, detection, instance seg, salient object detection. Avg resolution 684,231 px. Split 70.4/20.2/9.4%. |
| **VehiDE (VeHIDE)** | 13,945 imgs, >32,000 instances | 8: broken glass, broken lights, scratch, lost parts, dents, torn, punctured, non-damaged | CC BY-NC 4.0 paper; images under Flickr/Shutterstock licenses — **non-commercial only** | tandfonline.com/doi/full/10.1080/24751839.2024.2367387 (DOI 10.1080/24751839.2024.2367387); Kaggle: hendrichscullen/vehide-dataset | Larger than CarDD but legally unusable for a commercial product. Cars only. Train/val ≈ 11,621/2,324. |
| **DS4E Truck Damage Detection** | 4,962 imgs | 8 (messy taxonomy): chassis_damage, front_end_damage, front_end_corner_damage, rear_light, side_step, truck_damage, etc. | **CC BY 4.0 (commercial OK)** | universe.roboflow.com/ds4e/truck-damage-detection | ★ **Only genuine commercial-truck damage set.** European trucks (MAN TGX, Volvo FM, DAF, Actros; filenames "SCHADEOBJECT"). Small, single-author; vet quality. |
| Damaged Vehicle Images (Sammy) | 5,072 imgs | 7: crack_and_hole, {slight/medium/severe}_deformation, {slight/severe}_scratch, windshield_damage | CC BY 4.0 | universe.roboflow.com/sammy/damaged-vehicle-images | Cars; usable commercially for transfer learning. |
| vehicle-damage-detection-hhxfj | 6,416 imgs | 7 incl. crack_and_hole, deformation grades, windshield_damage | CC BY 4.0 | universe.roboflow.com/damage-detection-d25qu/vehicle-damage-detection-hhxfj | Cars; instance seg. |
| Car Accidents & Deformation (M-Arslan) | 1K–10K | YOLO deformation classes | CC BY-NC 4.0 | HF: M-ArslanArshad/Car_Accidents_and_deformation_dataset | Non-commercial. |

**Truck/trailer detection (not damage):** trailers-detection (930 imgs, Roboflow signals-rlxbl), Truck Trailer Detection VBeta (35 imgs), Martin/trailer (200 imgs), axle-classification (~7.46k imgs, vehicle-type only). All low relevance for condition — they detect presence/type, not defects.

**Bottom line (1a/1b):** No academic-grade, large, commercially-licensed commercial-truck damage dataset exists. This is the central gap.

#### 1c. Imperfect / low-quality image datasets (for the quality gate)

| Dataset | Size | What it captures | Notes |
|---|---|---|---|
| **KonIQ-10k** | 10,073 imgs, 1.2M ratings from 1,459 crowd workers | In-the-wild authentic distortions (brightness, colorfulness, contrast, sharpness, noise) | doi 10.18419/darus-2435; arxiv 1910.06180. Deep model KonCept512 reaches 0.921 SROCC. The standard NR-IQA training set. |
| **SPAQ** | 11,125 imgs, 66 smartphones | Real smartphone distortions (noise, blur, exposure) + scene tags | Lab-collected MOS. Most representative of driver phone photos. |
| **LIVE-in-the-Wild (CLIVE)** | 1,162 imgs | Authentic mobile-capture distortions | Classic cross-dataset benchmark. |
| **KADID-10k** | 81 refs → 10,125 distorted | 25 synthetic distortion types × 5 levels | Controlled distortion taxonomy. |
| **BID** | 586 imgs | Realistic blur (out-of-focus, simple/complex motion) | Blur-specific. |
| **TID2013 / PIPAL / BIQ2021** | — | Synthetic & GAN distortions | For robustness/generalization eval. |

**Synthetic degradation pipelines (manufacture imperfection from clean truck photos):** Real-ESRGAN high-order degradation (blur → resize → noise → JPEG, randomized order; generalized Gaussian + motion/defocus kernels, Poisson/Gaussian noise, DiffJPEG) — see Wang et al., ICCVW 2021; BSRGAN random-shuffle degradation; and KAIR. Use **Albumentations/imgaug** for controllable augmentations (motion blur, rain, glare, occlusion, over/under-exposure, compression). This is the recommended way to build a labeled "bad photo" training set cheaply, since Kamion's own clean truck photos can be degraded to order.

#### 1d. Marketplace listing sources (photos + price + mileage + year + make/model)

| Source | Official API? | Scraping/ToS posture | Relevance |
|---|---|---|---|
| **Marketcheck** | ✅ Yes — Cars API (5B+ listings since 2015; ~40M+ active + 800M+ historical records), VIN decode, Price API (predicted price + comps), **used-heavy-equipment Inventory Search endpoint**, Cached Images endpoint | Licensed data product; subscription + usage billing ($1,000 usage threshold triggers early billing) | ★ Best turnkey source for a public-price prior (US/Canada/UK). Legally clean. |
| **sahibinden.com** (TR) | Restricted account-only "API/veri indirme"; no open API | **Prohibited**; Cloudflare + login wall + captcha + IP bans | Turkish trucks; hardest. KVKK applies to seller PII. |
| **arabam.com** (TR) | None public | Gray-area; check ToS; no sanctioned program | Turkish trucks; more accessible than sahibinden. |
| **Mascus** | None public (3rd-party scrapers: Apify, Piloterr) | Unofficial only | ★ Largest EU heavy-truck/trailer pool — "more than 400,000 listings of used heavy machinery and trucks" with "over 3,500,000 visits from buyers every month" (Mascus official app listing). Pursue data partnership. |
| **mobile.de** (commercial) | ✅ Official Seller-API + Search-API (dealer-scoped, own inventory; incl. in dealer package) | Market-wide scraping not sanctioned | Strong DE commercial-truck data via partnership. |
| **Truck1** (.eu) | ✅ JSON/API dealer **import** feed (inbound) | Check ToS | EU trucks; partnership channel. |
| **TruckPaper / Machinery Trader** | No open API | **ToS explicitly forbids** robots/scrapers/data-mining (TruckPaper Terms of Use) | US heavy trucks; scrape-prohibited. Apify actors exist but violate ToS. |
| **Autoline (.info)** | Inbound dealer feeds only | No data-out API | EU commercial vehicles. |
| **Ritchie Bros / IronPlanet** | Auction results (no open API) | 3rd-party scrapers | ★ Auction "sold" prices = real transaction ground truth (better than asking prices). |
| **Kaggle used-vehicle sets** | e.g., US Used Cars (3M rows), Cars for Sale (20k w/ condition+accident+options), cars.com set (4,009) | Open (CC) | Tabular price/mileage/year; mostly **no images** or cars only. Good for pricing-model prototyping, not vision. |

**Bottom line (1d):** Use **Marketcheck** as the legally-clean public-price prior (has a heavy-equipment endpoint + images), pursue **Mascus / mobile.de / Truck1** data partnerships for European commercial-truck comps, and treat **auction sold-prices (Ritchie Bros/IronPlanet)** as the gold transaction signal. Avoid scraping TruckPaper/Machinery Trader (ToS-prohibited) and sahibinden (prohibited + technically defended).

#### 1e. Tire / wheel / undercarriage datasets

- **TyreNet** (Mendeley Data, 32b5vfj6tc): 1,698 tyre images (866 defective / 832 good), from six car/bike service stations + two showrooms, expert-annotated. Good real-world lighting variety.
- **foduucom/Tyre-Quality-Classification-AI** (HF): defective vs good classifier + dataset.
- **Roboflow tire-defect** use-case sets: classes crack, bulge, tread wear, cut, puncture (RF-DETR deployable).
- **Tire X-ray defect sets** (academic, e.g., a Springer study with 120,000 images, 100k good/20k defective) — manufacturing QA, not roadside tread wear; limited relevance.
- **Gap:** No public dataset for **truck tire tread-depth grading per axle** or **retread/recap detection** — Kamion must collect this. A vehicle-inspection patent (US 12541840) reports ML ROC-AUCs of 0.76 (uneven tread wear), 0.84 (damaged wheels), 0.94 (oversized tires), 0.78 (mismatched tires), 0.76 (rotted/cracked tires), 0.95 (aftermarket wheels), indicating feasibility of image-based tire condition models.

#### 1f. Odometer / dashboard OCR datasets

- **TRODO** (Data in Brief 38 (2021) 107321, doi 10.1016/j.dib.2021.107321): 2,389 annotated odometer images (analog+digital; reduced from 2,613 raw), varied resolution/illumination/vehicle type, bounding boxes + digit labels, CVAT-annotated, CC BY. **Built in Turkey (Eskişehir Technical University), images from the Marketyo delivery network** — directly relevant to Kamion's market.
- **Motorcycle Odometer dataset** (IEEE DataPort): 5,860 images, analog+digital, standardized 320×160, varied lighting/angles (Sept 2025).
- **umutkavakli/odometer-mileage-extraction** (GitHub): 2,389 imgs, YOLOv8 + OCR pipeline (also Turkish author).
- Industry benchmark: **Cognexa OdoCap** reports "a reading accuracy of 99.9%" on mixed-quality smartphone dash images plus "car make and model accuracy at 98% (trained on 25 different car makes and models – mostly European)"; American Family Insurance published an SSD/Faster-RCNN odometer pipeline (Frontiers).
- **Recommendation:** Fine-tune a small detector (odometer localization) + digit OCR on TRODO, augment with Kamion's own dash photos. Add odometer-fraud heuristics (EXIF timestamp/GPS cross-check, digit-plausibility vs age).

#### 1g. Concrete recommendation — which datasets to combine

**Recommended training/eval corpus (~35–40k images before Kamion's own data):**
1. **DS4E Truck Damage** (4,962, CC BY 4.0) — the only commercial-usable truck-damage seed; base for truck detectors.
2. **Damaged Vehicle Images (Sammy)** + **hhxfj** (~11,500, CC BY 4.0) — commercial-usable car-damage for transfer learning of dent/scratch/deformation/glass classes.
3. **KonIQ-10k + SPAQ** (~21,000) — train/calibrate the NR-IQA gate; SPAQ especially mirrors phone captures.
4. **TyreNet + Roboflow tire-defect** (~2,500) — tire condition head.
5. **TRODO** (2,389) — odometer OCR (Turkish origin).
6. **CarDD / VehiDE** — **R&D and architecture benchmarking only** (non-commercial license); do NOT ship models trained on them in production.

**Stitching:** Normalize all to a unified COCO schema; harmonize damage taxonomy to a canonical set (dent, scratch, crack, corrosion/rust, glass damage, lamp damage, missing/torn part, deformation-severity {slight/medium/severe}); use synthetic degradation (Real-ESRGAN/Albumentations) to create paired clean/degraded versions for the quality gate and for robustness. Reserve a **held-out golden set labeled by Kamion's expert appraisers** for final evaluation.

> **For the narrowed, measured two-market demo corpus — the specific US and Turkish listing sources, live counts, per-listing fields, photo counts and image resolutions — see §9.** The public academic datasets above are for pretraining the detectors and the quality gate; §9 is the paired market data the valuation demo actually runs on.

**Remaining gaps Kamion must fill with proprietary collection:** (a) commercial-truck & trailer damage at scale; (b) trailer-type-specific defects (reefer units, dry van floors, flatbed decks, tanker shells, tipper bodies); (c) truck tire tread-depth per axle + retread detection; (d) cab-interior wear; (e) chassis/undercarriage corrosion; (f) **Turkish market price+photo+mileage transaction records** (the pricing ground truth).

### 2. THE IMAGE-QUALITY GATE

**Two distinct checks — do not conflate them:**

**(A) Generic technical quality (NR-IQA).** Off-the-shelf via `pyiqa` (IQA-PyTorch by Chaofeng Chen):
- **Classical (fast, no GPU):** BRISQUE, NIQE, PIQE — cheap first-pass filters but weak on authentic distortions (BRISQUE SROCC ~0.015–0.087 on GAN-distortion PIPAL; near-useless there).
- **Learned NR-IQA:** MUSIQ (multi-scale transformer), MANIQA (0.667–0.704 SROCC on the hard NTIRE2022 GAN set — far above classical), HyperIQA, CLIP-IQA/CLIP-IQA+, TOPIQ, and LLM-based **Q-Align** (ICML 2024; strongest on blurs, color, noise, brightness). MM-IQA reports SRCC 0.647–0.830 across KonIQ/CLIVE/KADID/TID2013/BIQ2021.
- **Recommendation:** Run a fast classical gate (BRISQUE/NIQE) to reject obvious garbage, then MANIQA or CLIP-IQA+ (fine-tuned on SPAQ + Kamion phone photos) for the graded score. Q-Align optionally for a human-readable quality rationale in the audit trail.

**(B) Task-specific sufficiency ("does this show the part I need").** This is more important than aesthetic quality and has **no off-the-shelf model** — Kamion builds it:
- **Coverage/completeness classifier**: a viewpoint/part classifier that verifies presence of required canonical views — front 3/4, rear 3/4, both sides, cab interior, engine bay, odometer, each tire/axle, chassis/undercarriage, trailer floor/deck, VIN plate, reefer unit (if applicable).
- **Part-segmentation** (SAM 2 + a fine-tuned truck-part segmenter) confirms the target panel is actually visible and unoccluded.
- A photo can score high on NR-IQA yet be useless (a sharp close-up of a tire when you needed the cab) — the coverage agent flags this.

**Metadata & fraud heuristics:**
- **EXIF/metadata:** resolution floor, capture timestamp vs submission time, GPS presence, device model; flag screenshots (no camera EXIF) and downloaded stock images.
- **Duplicate / re-use detection:** perceptual hashing (pHash) + **CLIP/DINOv2 embedding similarity** against prior submissions and known catalog/stock imagery — catches re-posted or stolen photos.
- **AI-generated / tampered image detection:** classifier for synthetic images + copy-move/inpainting forensics to catch photoshopped-out damage.
- **Stock/catalog detection:** reverse-image similarity to OEM press photos.

**Response to insufficient input (graceful degradation):**
1. Widen the price confidence interval (conformal — see §4).
2. Issue **targeted re-shoot requests** ("retake left-front tire, too blurry"; "cab interior missing").
3. Below a coverage/quality threshold, **refuse to quote** and route to human — never emit a confident number from bad input.

### 3. VISION-FIRST CONDITION ASSESSMENT ARCHITECTURE

**Three approaches compared:**

| Approach | Accuracy | Latency | Cost | Auditability | Verdict |
|---|---|---|---|---|---|
| (i) End-to-end VLM with rubric prompt | Good at semantic/holistic judgment (Qwen-VL-2B 87.3% semantic damage class); **poor localization, hallucinates damage in reflections, misses scratches** | Med (1–5s/img) | Med-High | Weak (no pixel grounding) | Not alone |
| (ii) Specialist CNN/ViT detectors+segmenters (YOLOv8/v11, RT-DETR/RF-DETR, SAM 2, DINOv2) → condition vector | High localization precision; deterministic | Low (10s–100s ms/img on GPU) | Low | Strong (boxes/masks) | Core evidence layer |
| (iii) **Hybrid: specialists produce evidence, VLM reasons over it** | Best of both; VLM grounded in detections + masks | Med | Med | **Strongest** | ★ **Recommended** |

**Recommendation: Hybrid (iii).** Precedent: the "Grounding Agentic VLMs with Dedicated Segmentation" work (arXiv 2608.02470) implements exactly this with a 7-node LangGraph pipeline where a segmentation model (TinyDamage) grounds VLM generation, precisely because VLMs alone hallucinate. Specialists detect and localize; the VLM synthesizes the condition narrative and resolves ambiguity, always citing the specialist evidence.

**The structured, auditable CONDITION VECTOR** (per vehicle):
- Per-panel damage severity (hood, doors, fenders, bumpers, cab, roof, trailer walls/floor/deck): {none, slight, medium, severe} + type {dent, scratch, crack, deformation}
- Rust/corrosion grade per region (chassis, undercarriage, cab mounts)
- Tire tread grade per axle + retread flag + mismatch flag
- Glass condition (windshield crack/chip, side/rear)
- Lighting condition (headlamp/taillamp/indicator functional/broken)
- Interior wear grade (seats, dash, controls)
- Paint condition (fade, oxidation, repaint mismatch)
- Modification/aftermarket detection
- Accident-indicator detection (panel-gap misalignment, weld/repair signatures, frame indicators)
- Each entry carries: confidence, source photo ID, bounding box/mask, model+version.

**Mapping to industry-recognized grades:**
- **NAAA Vehicle Condition Grading Scale** (0–5, 5 = excellent): grades interior, exterior, mechanical, frame/underbody, tires. **AutoGrade™** (Manheim algorithm, endorsed by NAAA, distributed via AASC) converts damage line-items to a 0–5 score, explicitly combining with mileage/year/make/model for valuation — this is the closest existing "damage → grade → value" rubric to replicate. ADESA publishes a comparable 0–5 scale.
- **Truck-specific:** ATA used-truck condition guidelines; Ritchie Bros inspection reports; DEKRA/TÜV commercial-vehicle inspection standards; FMCSA/DOT annual inspection criteria; **European PTI (periodic technical inspection) roadworthiness** standards. Map Kamion's condition vector to a 0–5 AutoGrade-style scale AND to a PTI pass/advisory/fail schema so grades are recognizable to European buyers.
- Note NAAA/AutoGrade explicitly separate condition grade from mileage (mileage applied at valuation, not grading) — mirror this separation, which also keeps vision dominant over mileage in the pricing stage.

**Trailers handled separately (distinct value drivers & inspection points):**
- **Reefer (refrigerated):** reefer unit make/model, **engine hours** (not just km), cooling performance indicators, insulation/panel integrity, floor drainage. Engine hours are a primary value driver.
- **Dry van:** floor condition (wood/aluminum), wall/roof integrity, door seals, interior damage.
- **Flatbed:** deck condition, tie-down points, frame straightness.
- **Tanker:** shell corrosion, baffle/valve condition, certification/test dates (critical for hazmat).
- **Tipper:** body wear, hydraulic ram condition, liner state.


#### 3a. VEHICLE IDENTITY & CONFIGURATION — the gate in front of every damage model

Make, model, generation and configuration are **not** just pricing covariates bolted on at the end. They
determine which vision models are valid at all. This is a distinct pipeline stage that runs before any
damage agent.

**Why it cannot be deferred to the pricing stage:**

1. **Part geometry does not transfer across architectures.** A Freightliner Cascadia is a 6×4 conventional
   with a long hood, aero side extenders, chassis fairings and a 72" raised-roof sleeper. A Ford F-MAX is a
   4×2 cab-over with a flat front face and a tilting cab and no hood at all. "Front bumper", "hood",
   "side fairing" and "sleeper panel" are not the same objects — and axle/tyre counting differs (10 wheels
   vs 6). A single global part segmenter will hallucinate parts that do not exist on the vehicle in front of it.
2. **Damage priors are model-specific.** Where corrosion starts, which panels crack, which mounts sag, and
   what "normal wear at 400,000 km" looks like are all functions of the specific model and generation.
3. **Repair cost — and therefore the price delta — is brand-, model- and market-specific.** The same 15 cm
   dent is a different number of dollars on an Actros in İstanbul and a Cascadia in Houston, because parts
   availability, parts price and labour rate all differ. **One condition vector, many cost tables.**
4. **Configuration inside a model often matters more than the model.** Cascadia 116 vs 126 BBC; day cab vs
   72" raised-roof sleeper; DD13 vs DD15; 6×4 vs 4×2; 10-speed manual vs DT12 automated; air ride vs spring.
   F-MAX: 4×2 vs 6×2; 5th-wheel height 960 / 1100 / 1200 mm; Ecotorq vs Ecotorq GEN2; domestic vs
   international-haul spec. These swing value more than a moderate cosmetic defect does.

**The Identity Agent (new pipeline stage, runs before the damage fan-out):**

| Input path | Method | Availability |
|---|---|---|
| VIN present | **NHTSA vPIC** decode — free, no key, no rate limit, commercial use permitted; returns make, model, year, body class, GVWR class, engine model, displacement, drive type, plant | **US only.** Verified working on live Cascadia VINs (§9a) |
| VIN present, non-US | VIN WMI/VDS parse + OEM lookup tables built per brand | Partial; **Türkiye has no free public decoder** |
| No VIN | Photo-based identification: badge/grille OCR, cab-architecture classifier (conventional vs COE), fine-grained model classifier, axle/wheel counting from side views, silhouette matching against an OEM reference embedding bank | **The Turkish default.** Must work from pixels alone |
| Registration document | OCR of the Turkish *ruhsat* (or equivalent) for make, model, year, engine, axle config | Requires the seller to photograph it; add it to the guided-capture checklist (§2B) |

**Output — the identity vector (every downstream stage is conditioned on it):**

```
{ make, model, generation, market,
  cab_architecture: conventional | cab_over,
  cab_type: day_cab | sleeper, sleeper_type, sleeper_size_in,
  axle_config: 4x2 | 6x2 | 6x4, wheel_count,
  engine_family, displacement_l, horsepower, emissions_tier: EPA10|GHG14|GHG21|EuroV|EuroVI,
  transmission_type, transmission_speeds, suspension_type,
  wheelbase, fifth_wheel_height_mm, usage_class,
  identity_confidence, identity_source: vin|ruhsat|photo }
```

**It gates three things:**
- **Which part-segmentation head runs** — architecture-specific, selected by `cab_architecture` + `model`.
- **Which damage→cost table applies** — keyed on `(make, model, market)`.
- **Which hedonic price prior is loaded** — see §4a.

**Identity is also a fraud check.** If the VIN decodes to a conventional tractor and the cab-architecture
classifier says cab-over, the listing is mislabelled, VIN-swapped, or using stock photos of a different
vehicle. Route to the Fraud Agent (§5) rather than proceeding. Same for a declared model year that is
inconsistent with the generation-level visual classifier.

**Failure mode to design for:** `identity_confidence` below threshold must widen the price interval and,
below a floor, refuse to quote — exactly like the photo-quality gate. **An appraisal of the wrong model is
worse than no appraisal**, and it is silent unless identity is explicitly scored.

### 4. PRICING ENGINE (market-agnostic, vision-dominant, mileage-aware)

**Two-stage model:**
- **Stage A — Base/reference price:** `f(year, make, model, configuration, axle layout, mileage, engine hours, regional market index, currency)`. Model: gradient-boosted trees (LightGBM/CatBoost/XGBoost) or a probabilistic tabular model (ProbSAINT reported strong calibrated MAPE on used-car pricing, arXiv 2403.03812). Target = **log price**. Trained on public comparables (Marketcheck + partner feeds).
- **Stage B — Vision-derived condition adjustment:** the condition vector → a multiplier/additive delta. **This delta is the primary output of the vision system** and the dominant driver of the final number relative to a clean-condition baseline.

**Market-agnostic design:**
- Currency normalization to a base (e.g., EUR) via daily FX; keep local currency as output.
- Regional price-index features (country/region hedonic offsets) instead of hardcoded per-country models.
- Log-price target stabilizes variance across price tiers and markets.
- Transfer learning: pretrain on high-volume markets (EU via Mascus/mobile.de), fine-tune with few samples per new market; hierarchical pooling shares strength across markets.

**Heavily weighting internal Kamion data over public comps:**
- **Bayesian hierarchical model:** public data forms the prior; Kamion's internal sales form the posterior. As internal N grows, posterior dominates automatically.
- **Residual-correction (recommended to start):** train Stage A on public data, then train a second model on Kamion's internal transactions to predict the *residual* (internal_actual − public_prediction). Final = public_pred + learned_correction. Simple, robust with small internal N, and cleanly separates "market prior" from "Kamion reality."
- **Sample weighting:** weight internal transactions 5–20× public comps; increase as internal data accumulates.
- **Online/continual learning:** periodic retrain as transactions close; monitor drift.

**Mileage handling (kept subordinate to vision):**
- Distinct depreciation curves: heavy tractors (often valued on total km + engine hours + rebuild history) vs light commercial (km-dominant); mileage×age interaction terms.
- Engine hours for reefers/PTO units as a separate feature.
- **Odometer-fraud detection:** OCR reading cross-checked vs age, service history, EXIF/GPS; implausible readings flagged.
- **Keeping vision dominant:** apply **monotonic constraints** in the GBM (price non-increasing in mileage) and **feature-importance / contribution caps** so mileage cannot exceed the condition delta's influence; alternatively bound Stage-B condition delta to be the larger-magnitude adjustment.

**Uncertainty quantification:**
- **Conformalized Quantile Regression (CQR)** (Romano et al., arXiv 1905.03222; via the MAPIE library) — calibrated intervals with finite-sample coverage, interval width adapts to input. LLM-based quantile regression (Mistral-7B-Quantile, arXiv 2506.06657) also demonstrated tight intervals on used-car data (RCIW 0.2).
- Interval width should be an explicit function of photo quality + coverage from §2 — poor photos → wider intervals → possible refusal.
- Ensembles for epistemic uncertainty; quantile heads (LightGBM `objective=quantile`) for the base.

**Evaluation metrics (automated valuation):**
- **MAPE, MdAPE** (median APE, robust to outliers).
- **PA10 / PA20** — % of predictions within 10%/20% of actual (industry-standard for AVMs).
- **Prediction-interval coverage** (does the 80/90% interval contain truth at the stated rate).
- Benchmark reference: used-car AVMs typically target MdAPE in the high single digits / low teens and PA10 well above 50%; heavy trucks are noisier (thinner comps) so expect wider intervals — set internal targets against Kamion's own appraiser variance rather than car benchmarks.


#### 4a. BRAND / MODEL / CONFIGURATION IN THE PRICE MODEL

**Hierarchical, not flat.** Encode identity as a nested hierarchy and pool partially at each level, so a
thin model (MAN TGA in Türkiye, n≈1) borrows strength from its brand and its segment instead of either
being dropped or overfitted:

```
market → cab_architecture → brand → model → generation → configuration(trim)
```

- Random/hierarchical intercepts at brand and model level in the Bayesian layer; target encoding with
  shrinkage for the GBM. Both degrade gracefully to the parent level when a cell is thin.
- **Brand residual offsets must be learned, not assumed.** Brand-level value retention differs materially
  within each market and is the kind of folk knowledge that is usually wrong. Fit it; do not hardcode it.
- **Configuration features carry more signal than model identity in a mature market.** On the US demo source,
  sleeper type/size, wheelbase, engine family (DD13 vs DD15) and horsepower are all published per listing and
  are strong hedonic covariates. On the Turkish source none of these exist — the available config features
  are drive type, 5th-wheel height and usage class. **The two markets therefore need different feature sets
  behind the same interface**, which is precisely what the hierarchical encoding accommodates.

**Market-specific handling that the hierarchy does not solve on its own:**

| Issue | Handling |
|---|---|
| **Odometer units and intensity** | Normalise to km. Do **not** share a mileage coefficient — measured median odometer at the same model years is **2.69× higher** in the US (§9c). Fit market-specific depreciation curves and mileage×age interactions |
| **TRY inflation** | Fit on **CPI-deflated or FX-indexed real prices** with an explicit as-of date; emit nominal TRY. Nominal TRY depreciation measures −10.9%/yr against −21.8%/yr in USD — an artefact, not a market fact (§9c) |
| **Price staleness** | Carry listing first-seen/last-seen timestamps and treat price age as a feature. At ~31% CPI a six-month-old Turkish asking price is a different number today |
| **Quantized Turkish asking prices** | 20 distinct price points across 147 listings; ±5.0% residual after year+km. **Do not use Turkish asking price as a training target or an evaluation label** — use realized transactions (§9e) |
| **Turkey's closed used market** | Used commercial-vehicle imports require Ministry permission, so Turkish prices are supply-constrained and decoupled from EU comparables. **Do not use Mascus/mobile.de EU comps as a Turkish prior** without an explicitly fitted country offset |
| **Emissions tier** | EPA10/GHG14/GHG21 (+CARB) vs Euro V/VI are not interchangeable regulatory buckets; encode as a market-scoped categorical, not a shared ordinal |

**Keeping vision dominant, restated with brand in the picture:** the monotonic constraints and contribution
caps in §4 apply to mileage *and* to the identity block. Make/model/configuration set the **clean-condition
baseline**; the vision condition delta moves the number away from it. If brand/model dummies are absorbing
more contribution than the condition delta on a typical appraisal, the model has learned to price a catalogue
rather than a truck — monitor this ratio explicitly as a guardrail metric.

### 5. MULTI-AGENT DESIGN & "PERFECT OBSERVABILITY"

**Agent topology (supervisor/worker with parallel fan-out + critic loop):**
1. **Intake/Triage Agent** — ingests photo set, dedups, extracts EXIF, routes.
2. **Image Quality Agent** — NR-IQA gate (§2A) per photo.
3. **Coverage/Completeness Agent** — canonical-view checklist (§2B); triggers re-shoot requests.
4. **Vehicle Identity Agent (§3a)** — resolves make → model → generation → configuration from VIN (vPIC in the US), registration document, or pixels alone (Türkiye). **Gates which part segmenter, damage priors, cost table and price prior every downstream agent uses**, and flags VIN-vs-visual mismatches to the Fraud Agent. Runs before the damage fan-out; low `identity_confidence` widens the interval or refuses to quote.
5. **Damage Assessment Agents (per region, parallel)** — detectors+segmenters over front/rear/sides/cab/trailer, **selected by the identity vector**; fan-out over photos via Ray.
6. **Tire/Undercarriage Agent** — tread/axle/corrosion; axle count taken from the identity vector (6×4 vs 4×2).
7. **Odometer/OCR Agent** — mileage + fraud check; unit inferred from market (mi vs km).
8. **Fraud Agent** — stock/AI-generated/tamper/duplicate detection + identity-mismatch cases from agent 4.
9. **Comparables/Retrieval Agent** — vector search over comps (year/make/model/config + image-embedding similarity), scoped to market.
10. **Pricing Agent** — two-stage model + CQR interval, on deflated real prices where the market requires it (§4a).
11. **Adjudicator/Critic Agent** — reconciles disagreements (e.g., VLM claims damage a detector missed), runs a debate/critic loop, decides confidence, can demand re-analysis.
12. **Report Agent** — assembles the human-readable "why this price" report with annotated images.

**Orchestration pattern:** DAG with parallel fan-out over photos and specialist agents, converging at the Adjudicator; dynamic re-shoot loop back to intake; critic loop only when specialists and VLM disagree (saves cost).

**Framework recommendation:**

| Framework | Strengths | Fit |
|---|---|---|
| **LangGraph** | Graph/state-machine, durable execution, human-in-the-loop, checkpointing, v1.0 (late 2025), OTel support (2026); precedent in vehicle-damage agentic pipelines | ★ **Primary orchestrator** |
| **Temporal** | Durable, retryable long-running workflows | ★ Wrap the DAG for reliability at scale |
| **Ray** | Parallel GPU vision inference | ★ Specialist inference layer |
| Claude Agent SDK / OpenAI Agents SDK | Clean single-vendor agents, tool use | Use for the VLM reasoning nodes (model-swappable) |
| CrewAI / AutoGen (AG2) | Role-based / debate loops | AutoGen-style debate optional for the critic loop |

**Recommended stack:** LangGraph (orchestration) + Temporal (durability) + Ray (parallel inference) + model-agnostic VLM nodes (Claude/Gemini/Qwen-VL swappable).

**Observability — first-class deliverable:**

| Tool | Trace granularity | Multimodal/image artifacts | Cost tracking | Eval integration | Self-host | License |
|---|---|---|---|---|---|---|
| **Langfuse** | Nested spans, strong | Yes (media) | Strong | Online evals + human annotation queues | ✅ Full (self-host free, unlimited events at infra cost; ~<$500/mo for 2–3M traces on ClickHouse) | MIT |
| **Arize Phoenix** | OTel/OpenInference-native, dataframe evals | Yes | Yes | Notebook/pipeline evals, drift/RAG | ✅ | Elastic 2.0 |
| **LangSmith** | Best-in-class for LangChain/LangGraph, breakpoints, LangGraph Studio | Yes | Yes | Strong | Enterprise-only | Closed |
| W&B Weave | Experiment-tracking lineage | Yes | Yes | Good | Partial | — |
| Braintrust | Eval-first regression | Yes | Yes | ★ Evals | — | Closed |
| Helicone | Gateway/proxy logging | Limited | Yes | Basic | ✅ | Apache 2.0 |
| Datadog LLM Obs | APM correlation, **native OTel GenAI SemConv (v1.37+)** | Yes | Yes | Experiments SDK | SaaS | Closed |

**Recommendation:** **Langfuse self-hosted** (MIT, data residency for EU/Turkey GDPR/KVKK, multimodal artifact logging, cost tracking) as the primary trace store, instrumented via **OpenTelemetry GenAI semantic conventions** (`gen_ai.*` spans: `invoke_agent`, `chat`, `execute_tool`; attributes `gen_ai.request.model`, `gen_ai.usage.input_tokens`/`output_tokens`, `gen_ai.output.messages`) so Kamion is never locked in and can dual-export to Datadog for infra correlation. Arize Phoenix is the alternative if drift-detection/eval rigor is prioritized.

**Beyond generic tracing — the auditable appraisal trail (the "perfect observability" the user wants):**
- **Per-photo evidence linking:** every claim carries the bounding box/segmentation mask on the exact photo that produced it.
- **Per-claim confidence** and the exact prompt/model/version used for each decision.
- **The full causal chain rendered explicitly:** pixel → detected defect (box/mask) → condition-vector entry (score) → price delta → final number. This is a reasoning-provenance graph (cf. OTel GenAI + PROV-style provenance, arXiv 2603.21692), not just spans.
- **Replayability:** any past appraisal reconstructable with the same inputs, model versions, and prompts.
- **Human-readable "why this price" report** with annotated images, per-defect deltas, and comparables used.
- **Evidence-grounded / citation-style explanations:** each sentence in the report cites the photo+box it derives from (as in grounded-VLM work). Add **Grad-CAM / attention-rollout** saliency overlays for detector decisions and **counterfactuals** ("if front-left tire were new: +€X"; "if no cab dent: +€Y").

**Reviewer UI / human-in-the-loop:** side-by-side photo + condition vector + editable fields; reviewer can override any claim or the final price; overrides are captured as **labeled training data** (active-learning loop) and as ground truth for the pricing residual-correction model. LangGraph's human-in-the-loop checkpointing supports pause/inspect/modify/resume.

### 6. IMPLEMENTATION BLUEPRINT

**End-to-end system:**
- **Ingestion:** mobile app with **guided photo flow** (AR overlays for required angles, on-device blur/coverage pre-checks) + upload API.
- **Storage:** object store (S3-compatible) for photos; metadata DB (Postgres) for appraisals/condition vectors; **vector store** (pgvector/Qdrant) for comparables (image + tabular embeddings).
- **Inference layer:** Ray Serve for specialists (YOLO/RT-DETR, SAM 2, IQA, OCR, tire); VLM via API (Claude/Gemini) or self-hosted Qwen-VL/InternVL.
- **Orchestration:** LangGraph + Temporal.
- **Observability:** Langfuse + OTel.
- **Serving API + Reviewer dashboard.**

**Guided capture UX (raises quality at the source):** on-device NR-IQA (lightweight model) rejects blurry frames before upload; AR skeleton overlays for front-3/4, sides, tires, odometer, VIN; real-time coverage checklist. This is the single highest-ROI investment — every downstream uncertainty shrinks when input quality rises, reducing re-shoot loops and refusals.

**Phased roadmap:**
- **Weeks 1–2:** IQA gate (pyiqa off-the-shelf) + coverage classifier (bootstrap labels) + odometer OCR (TRODO fine-tune) + guided-capture prototype + data-collection pipeline live. **Harvest the §9 demo corpus: 218 TR listings / ~7,600 images, and 385 US Cascadia records enriched through NHTSA vPIC.** Pricing v0 = Marketcheck/partner comps + LightGBM on public tabular data (no vision yet). **Open the commercial conversation for realized Turkish sale prices on day one — it has the longest lead time of anything in this plan (§9d.3).**
- **Weeks 3–6:** **Vehicle Identity Agent (§3a) first** — cab-architecture classifier, make/model resolution, VIN-vs-visual cross-check — because every damage model is conditioned on it. Then damage specialists (fine-tune on DS4E + CC-BY car sets + early Kamion labels); condition vector v1; VLM reasoning node; LangGraph DAG; Langfuse observability; residual-correction pricing on first internal transactions. **Expert-label the 300-vehicle golden set (150 TR + 150 US) and publish inter-rater κ before any accuracy claim.**
- **Weeks 7–12:** trailer-specific heads; tire/axle grading; fraud agent; CQR intervals + refusal thresholds; adjudicator/critic loop; reviewer UI with override→label loop; golden-set eval + A/B vs human appraisers. **Run the cross-market transfer test (§9f step 7): TR-trained condition head scored on US photos and vice versa. That result — not aggregate MdAPE — is the evidence of market-agnosticism.**

**Build vs buy:** Off-the-shelf = NR-IQA (pyiqa), SAM 2, YOLO/RT-DETR backbones, OCR, Marketcheck comps, Langfuse. Train = coverage classifier, truck/trailer damage detectors, tire grading, condition→grade mapper, pricing residual model.

**VLM inference cost per appraisal at scale:** with the hybrid design, the VLM runs a bounded number of reasoning calls (not per-pixel). Estimate ~5–15 VLM calls per appraisal (adjudication + report), each with a few images — on the order of low tens of US cents to ~US$1–2 per appraisal at current flagship-VLM multimodal pricing; specialists run on owned GPUs at near-zero marginal cost. Batching, caching (Helicone/gateway), and using smaller VLMs (Qwen-VL) for routine cases cut this further.

**Key risks & failure modes:**
- **Adversarial/fraudulent photos** (stock, AI-generated, tampered) → Fraud Agent + perceptual-hash/embedding checks.
- **Distribution shift** across markets/brands/trailer types → hierarchical pricing + drift monitoring + continual learning.
- **Hallucinated damage** → grounding specialists + adjudicator; never let VLM assert un-localized damage.
- **Underrepresented trailer types** → targeted data collection; refuse-to-quote when out of distribution.
- **Liability of inaccurate valuation** → always output a *range* with confidence + disclaimers; human review above value thresholds.
- **Regulatory:** EU AI Act — a valuation tool is likely **not** Annex III high-risk per se (not creditworthiness/essential-service scoring), but confirm classification; high-risk obligations were phased to Dec 2027 (Annex III) / Aug 2028 (Annex I) after the 2026 Digital Omnibus (Reg. (EU) 2026/1744). **GDPR + Turkey KVKK** apply to photos containing plates/faces/GPS → auto-blur plates/faces, strip/secure GPS, data-minimization, retention limits.

**Evaluation plan:**
- **Golden set:** ~1,000–2,000 vehicles labeled by ≥2 expert appraisers (condition vector + price); measure **inter-rater agreement** (Cohen's/Fleiss' κ for grades, MdAPE between appraisers for price) to establish the human ceiling.
- Report MAPE/MdAPE/PA10/PA20 + interval coverage on the golden set.
- **A/B vs human appraisers:** blind comparison of AI vs appraiser estimates against realized sale prices; track where AI beats/loses and feed back.
- Continuous eval on closed transactions; alert on drift.

### 7. MARKET RESEARCH — UNITED STATES vs TÜRKİYE

The two markets look superficially similar (heavy tractor units hauling freight) and are structurally
different in almost every way that matters to a vision-based appraisal system. Building "one model,
two markets" without encoding these differences will fail silently — the model will look calibrated
on aggregate metrics while being wrong in each market for opposite reasons.

#### 7a. Structural comparison

| Dimension | United States | Türkiye |
|---|---|---|
| **Tractor architecture** | Conventional long-nose; BBC 116" / 126" platforms | Cab-over-engine (COE), EU-style |
| **Typical drive config** | 6×4 (three axles, 10 wheels) | 4×2 (two axles, 6 wheels); 6×2 minority |
| **Cab/sleeper taxonomy** | Day Cab, Mid-Roof XT, Raised Roof Sleeper, Raised Roof Condo, 48" XT, 60" XT, 72" Raised, 82" Ultra-High; sleeper size 34–96 in | COE integrated cab; differentiated instead by 5th-wheel height (960 / 1100 / 1200 mm) and usage class |
| **Odometer unit** | miles | kilometres |
| **Emissions regime** | EPA 2010 / GHG Phase 1 (2014) / Phase 2 (2021) + CARB overlays | Euro VI (EU-aligned; Euro 6 adopted for LDV new types in 2016) |
| **Max GVW** | 80,000 lb (~36.3 t), 5-axle | 40–44 t, EU-aligned |
| **Heavy-truck parc** | ~3.9–4.06M Class 8 in operation | ~1.04M "kamyon" (3.0% of 34,746,396 registered vehicles, Jul 2026); ~850k trucks cited in freight-market sources |
| **New heavy sales (2025)** | 240,249 Class 8 (2024); 416,467 medium+heavy retailed by franchised dealers in 2025 | 38,957 HCV total — **17,321 çekici (tractors)**, 14,850 trucks ≥16 t, 6,786 trucks <16 t |
| **Brand leader** | Daimler Truck / Freightliner — 39.6% North America share (2025); Cascadia >1M units sold lifetime | Ford Trucks — domestic HCV leader; F-MAX is the most-chosen model in its segment (TAİD) |
| **Fleet age** | (not sourced here) | **HCV average age 18.1 years (2025)**, up from 17.4 (2023) — OSS/Frost & Sullivan |
| **Carrier fragmentation** | ~2.09M FMCSA registrants; ~580,000 own/lease ≥1 tractor | Extreme — no logistics firm holds >1% share; ~300k trucks take spot loads directly; SME fleets ~95% of market |
| **Used-vehicle import** | Open | **Restricted** — used commercial/construction vehicle imports need Ministry permission; the domestic used market is effectively closed and supply-constrained, so prices decouple from EU comparables |
| **Periodic inspection** | DOT annual inspection + CVSA roadside | **TÜVTÜRK annual** for kamyon/çekici/tanker/otobüs from year 1; ₺4,446 fee (2026); ~11.3M vehicles inspected/yr across all classes |
| **Free VIN decode** | **NHTSA vPIC** — no key, no rate limit, commercial use permitted; returns make/model/year/body class/GVWR class/engine/displacement/drive type/plant | **None.** No public national decoder; identity must come from the ruhsat (registration document) or from the photo itself |
| **Published price guides** | J.D. Power Commercial Truck Guidelines (>1M wholesale+retail transactions analysed monthly); Black Book Medium & Heavy Duty Truck Values ($362.98/yr desktop+mobile, tiers to $714.99); Price Digests Truck Blue Book; Sandhills | **No equivalent HCV guidebook.** Passenger-car valuation tools exist (otodegeri, otoendeks, ViaCars, DAT, smartIQ) but not heavy commercial |
| **Public realized-sale prices** | Yes — Ritchie Bros Price Results, IronPlanet auction results, GovDeals/GovPlanet public-agency surplus | **No public source.** Transaction prices live only inside dealer/OEM systems |
| **Currency & inflation** | USD; low inflation | TRY; CPI 30.89% y/y (Dec 2025); USD/TRY 48.596 (11 Sep 2026). Vehicle price index +15.4% Jun-2025→Jun-2026 — **below** inflation, so real prices are falling while nominal prices rise |
| **Aftermarket size** | (not sourced here) | HCV after-sales market **$2.66B (2025)**, >3.5 t incl. trucks, tractors, buses, trailers |

#### 7b. The five differences that change the system design

1. **Geometry is different, so the vision stack cannot be shared naively.** A US Cascadia is a 6×4 conventional
   with a long hood, aero side extenders, chassis fairings and a 72" raised-roof sleeper. A Turkish F-MAX is a
   4×2 cab-over with a flat front face, a tilting cab, and no hood at all. Part-segmentation classes ("hood",
   "front bumper", "side fairing", "sleeper panel") do not map one-to-one. Axle/tyre counting differs (10 wheels
   vs 6). **The part segmenter must be conditioned on the resolved make/model/generation (§3a), not applied
   globally.**

2. **Mileage means something different.** Measured on live inventory (§9), at the same model years US Cascadias
   carry **2.69× the odometer** of Turkish F-MAXes (median 726,447 km vs 270,000 km) while selling for only
   1.24× the price. Price per 1,000 km driven is $96 in the US and $210 in Türkiye. A single mileage
   coefficient — or a single "high mileage" threshold — is wrong in both markets. Depreciation curves, and
   the point at which mileage starts dominating condition, must be market-specific.

3. **Nominal TRY prices hide real depreciation.** Fitted on the same live inventory, log-price against age gives
   **−21.8%/yr in the US (USD)** but only **−10.9%/yr in Türkiye (nominal TRY)**. With ~31% CPI, the *real*
   Turkish depreciation is far steeper than the nominal curve suggests. Any model trained on nominal TRY will
   learn that Turkish trucks barely depreciate, and will then over-value old stock. **Train on CPI-deflated
   or FX-indexed real prices with an explicit date index; output nominal TRY.**

4. **Türkiye has no price ground truth, and its asking prices are nearly information-free.** In the US sample,
   year + mileage explain R²=0.429 of log price, leaving a **±22.6% residual band** — that residual is exactly
   the space a condition model can win. In the Turkish sample the same regression gives R²=0.791 with a
   **±5.0% residual band**, because Turkish asking prices are quantized: only **20 distinct price points across
   147 listings**, with 62% of listings sitting on the five most common prices and 76% divisible by ₺100,000.
   Turkish dealers are pricing off a year/km rule of thumb, not off condition. **There is nothing for a
   condition model to learn from Turkish asking prices. Turkish supervision must come from realized
   transactions** — Kamion's own closed deals, or dealer sale records obtained by agreement.

5. **Data access is inverted.** The US gives you metadata cheaply (VIN + free vPIC decode + engine/HP/sleeper/
   wheelbase in the listing) but relatively few photos. Türkiye gives you photos in abundance (32–38 per
   listing at 1440×1080) but no VIN, no engine spec, no horsepower. **So: learn the identity/condition
   vision models where the pixels are (Türkiye), and learn the price model where the labels are (US) —
   then transfer.** This is the core of the two-market demo in §9.

#### 7c. Competitive landscape

| Category | Players | Relevance to Kamion |
|---|---|---|
| **Passenger-car AI inspection** | Tractable (~$380.5M cumulative funding; $191M raised 2025), Ravin AI (smartphone + fixed-camera, used by insurers/dealers/rental/OEM remarketing), Monk (ACV), Click-Ins, Inspektlabs, Bdeo, WeProov, Pave | Proven the business model and the buyer. None is truck-native; all are car-shaped taxonomies |
| **Drive-through scanning hardware** | UVeye (Helios undercarriage; markets a Class 6–8 + bus product), ProovStation (130+ booths, 13 countries), DeGould | Hardware capex, fixed-site. **Cannot serve a fragmented Turkish market of single-truck owner-operators** — this is Kamion's opening |
| **US valuation incumbents** | J.D. Power Commercial Truck Guidelines, Black Book M/HD, Price Digests Truck Blue Book, Sandhills/FleetEvaluator, ACT Research | Own the *price prior*; none of them look at your photos. Partner or replicate — do not fight |
| **Marketplace with condition layer** | ACV Auctions — Auction & Assurance ARPU **$554/unit** (Q2 2026, +6% y/y), ~210,000 vehicles/quarter | The closest analogue for "condition report as the product". Cars, US, wholesale. Shows a condition layer can carry ~$500 ARPU when bundled with the transaction |
| **Türkiye** | No heavy-commercial AI appraisal player found. Used-HCV channels are OEM-run (Ford TruckMarket, Mercedes TruckStore, MAN TopUsed) plus sahibinden/arabam classifieds and independent galeri | **Open field.** Kamion's 6,000+ onboarded drivers and Turkish dealer relationships are the wedge |

**Pricing anchors for willingness-to-pay:** US pre-purchase inspection $100–450; in-shop DOT inspection $80–200;
Black Book M/HD subscription $362.98/yr/seat; AI-inspection SaaS $2,000–6,000/month per fixed lane; ACV
auction+assurance ARPU $554/unit; TÜVTÜRK statutory HCV inspection ₺4,446 (~$91) in 2026.

### 8. MARKET SIZING — TAM / SAM / SOM

**The distinction that matters:** the *used-truck market* and the *revenue a truck-appraisal product can earn*
are two different numbers roughly two orders of magnitude apart (~$20B of vehicle value against ~$143M of software spend). Sizing the first and calling it TAM is the most
common error in this category. Both are given below, clearly separated.

#### 8a. Market context (transaction value — NOT addressable revenue)

| | United States | Türkiye |
|---|---|---|
| Used heavy-truck transactions/yr | ~250,000–265,000 Class 8 (all channels: retail, wholesale, auction) | ~235,000 kamyon transfers (2.1% of 11,213,405 total 2025 used-vehicle transfers) |
| Average used price | $60,986 (Jul 2026, ACT); 2-yr sleeper $105,310, 5-yr $53,370, 10-yr $26,890 (Mar 2026) | Measured median for certified F-MAX MY2020–22: ₺2.75M ≈ $56,589. Blended across an 18.1-yr-old fleet: assume $20,000–30,000 |
| **Gross transaction value** | **≈ $15–16B/yr** | **≈ $4.7–7.1B/yr** |

That ~$20–23B of combined annual GMV is the pool the product *influences*. It is not the pool it *bills against*.

#### 8b. Addressable revenue — assumptions

Every number below is an explicit, editable assumption. Change the assumption, not the conclusion.

| ID | Assumption | Value | Basis |
|---|---|---|---|
| **US** | | | |
| A1 | Used Class 8 transactions/yr | 260,000 | ACT: 250k (2020), 265k (2021) |
| A2 | Paid assessment events per transaction | 2.2 | Seller listing + buyer/wholesale bid + lender or insurer touch |
| A3 | Blended ASP per assessment | $45 | Below the $100–450 human PPI floor; above per-API-call comparables |
| A4 | Lease/rental return condition events/yr | 300,000 | Penske + Ryder + regional lessors, order-of-magnitude |
| A5 | Fleets with ≥6 power units | 58,000 | ~10% of the ~580,000 carriers owning/leasing ≥1 tractor |
| A6 | Fleet subscription ASP | $1,500/yr | ~4× Black Book M/HD seat ($362.98/yr), multi-seat + API |
| **TR** | | | |
| B1 | Used heavy-truck transfers/yr | 235,000 | 2.1% × 11,213,405 (TÜİK 2025) |
| B2 | Paid assessment events per transaction | 1.6 | Thinner intermediary chain than US |
| B3 | Blended ASP per assessment | $18 | ~20% of the ₺4,446 statutory TÜVTÜRK inspection fee |
| B4 | Fleets with ≥5 power units | 25,000 | Fragmented market; SME fleets ~95% share |
| B5 | Fleet subscription ASP | $400/yr | TRY purchasing-power adjusted |
| **Reach** | | | |
| C1 | US share of transactions in API-integratable channels | 25% | OEM used networks + top-200 independents + 2 major lessors + digital marketplaces |
| C2 | TR share of transfers in reachable channels | 20% | Ford TruckMarket + Mercedes TruckStore + MAN TopUsed + organised galeri + Kamion's own base |
| C3 | 3-year capture of SAM | 8–10% | Standard early-stage B2B capture for a new category with no incumbent |

#### 8c. TAM — every heavy-truck condition event in both markets, served once, at market-anchored prices

| Component | Formula | US | Türkiye |
|---|---|---|---|
| Transaction assessments | A1×A2×A3 / B1×B2×B3 | $25.7M | $6.8M |
| Lease/rental return assessments | A4×A3 | $13.5M | — (negligible lessor base) |
| Fleet / remarketing subscriptions | A5×A6 / B4×B5 | $87.0M | $10.0M |
| **TAM** | | **$126.2M** | **$16.8M** |

**Combined TAM ≈ $143M/yr.**

This is the honest headline: **an assessment-only product in US + Türkiye heavy trucks addresses roughly
$143M of annual software spend — about 0.7% of the $20B+ of truck value it touches.** A resale-assessment
product is a high-leverage, modest-revenue business unless it also captures the transaction (§8f).

#### 8d. SAM — channels that will actually integrate within 3 years

| Component | US | Türkiye |
|---|---|---|
| Transaction assessments in reachable channels | 65,000 × 2.2 × $45 = **$6.4M** | 47,000 × 1.6 × $18 = **$1.4M** |
| Lessor return assessments (2 majors ≈ 40% of A4) | 120,000 × $45 = **$5.4M** | — |
| Fleet subscriptions (mid/large only) | 5,000 × $1,500 = **$7.5M** | 3,000 × $400 = **$1.2M** |
| **SAM** | **$19.3M** | **$2.6M** |

**Combined SAM ≈ $21.9M/yr.**

#### 8e. SOM — bottom-up, 3 years, Türkiye-first

| | Scope | Assessments | Subs | ARR |
|---|---|---|---|---|
| **Year 1** | 1 Turkish OEM used channel (TruckMarket-scale) + Kamion's own 6,000-driver base at 15% assess rate + design partners | ~3,400 | 20 | **~$70k** |
| **Year 2** | + Mercedes TruckStore TR + ~300 organised galeri + US pilot across 2 SelecTrucks regions | 7,000 TR ($126k) + 6,000 US ($270k) | 150 TR ($60k) + 60 US ($90k) | **~$0.55M** |
| **Year 3** | ~9.6% of combined SAM | 12,000 TR ($216k) + 22,000 US ($990k) | 900 TR ($360k) + 350 US ($525k) | **$2.09M** |

**3-year SOM (assessment product only): $1.8–2.2M ARR**, of which the Year-3 bottom-up case lands at **$2.09M**
on **34,000 assessments** and 1,250 subscribed fleets.

#### 8f. The alternative shape — and why it changes the answer by 5×

If Kamion uses the appraisal to *own the transaction* rather than to sell reports, the revenue model becomes
a take rate on remarketed trucks instead of a per-report fee. **Per assessed truck, that is ~$600 (1.5% of a
$40,000 sale) against ~$45 for a report — a 13× difference in unit economics.** At the Year-3 volume above
(34,000 assessments), if half convert to Kamion-facilitated sales:

> 17,000 × $40,000 × 1.5% ≈ **$10.2M/yr**, against **$2.1M** for the report-only model — on identical
> assessment volume.

ACV Auctions is the proof point: **$554 auction+assurance ARPU per unit** — almost exactly the $600 figure
above, and an order of magnitude above what a standalone condition report commands. The report is worth far
more as the trust layer under a transaction than as a document sold on its own.

**Recommendation: build the appraisal as the wedge and price it near cost; monetise the transaction.**
That also solves the Türkiye data problem in §7b(4) — owning the transaction is the only way to obtain
realized Turkish sale prices, which is the single hardest input in this whole system.

#### 8g. Sensitivity

| Swing | Effect on combined TAM |
|---|---|
| A3/B3 ASP ±50% | ±$23M (±16%) |
| A6/B6 fleet subscription ASP ±50% | ±$48M (±34%) — **the dominant lever** |
| A5/B4 addressable fleet count ±50% | ±$48M (±34%) |
| A1/B1 transaction volume ±20% | ±$6.5M (±5%) |

The TAM is driven far more by *how many fleets will pay a subscription* than by *how many trucks change hands*.
Validate A5/A6 and B4/B5 first — they are the least-grounded assumptions here and they carry two-thirds of
the number.

### 9. THE NARROWED TWO-MARKET DEMO CORPUS

Goal: one paired dataset, US + Türkiye, small enough to build in weeks, specific enough that a valuation
result is interpretable, and structurally different enough that working on both is real evidence of
market-agnosticism rather than a coincidence.

**Scoping decisions (and why):**
- **Tractor units only.** Mixing tractors, tippers and box trucks makes both condition scoring and price
  comparison uninterpretable — different value drivers, different inspection points, different buyers.
- **One dominant domestic model per market.** Holding make/model fixed isolates the condition signal from
  brand/model residual effects, which is exactly what a first demo needs to prove.
- **The OEM-run used channel in each country.** Symmetric channel type (manufacturer-certified used
  network), symmetric incentives, and both publish photos, price, mileage and year.
- **Model years 2020–2022.** This is the *overlap* window — not 2019–2022 as first scoped. The Turkish source
  has **zero MY2019 listings**, so any 2019 comparison would be US-only.

#### 9a. Demo A — United States: Freightliner Cascadia, MY2020–2022, tractor

| | |
|---|---|
| **Source** | SelecTrucks — Daimler Truck North America's OEM used network, 40 selling centres across 21 US states + 2 Canadian provinces |
| **Access** | `robots.txt` = `User-agent: * / Allow: /`. JSON inventory API at `POST /api/v1/inventory/search/`, array-valued filter params, `X-CSRF-TOKEN-SELT` header + session cookie required, `count` capped at 100. Detail pages at `/truck/{year}/{MAKE}/{truckId}/{dealerId}/{stock}`; `sitemap.xml` enumerates them (1,084 URLs) |
| **Live volume (measured 2026-09-12)** | **385** Cascadia MY2019–2022; **333** with a listed price (52 are "Call For Price"); **322** priced in the MY2020–22 demo window |
| **Price** | $24,000 – $144,995; p25 $53,950, **median $69,900**, p75 $79,900 |
| **Mileage** | 1,316 – 924,769 mi (**median 450,537 mi = 726,447 km**) |
| **Per-listing fields** | VIN, stock #, year, make, model, horsepower (410/450/455), engine type (**DD13 / DD15**), engine manufacturer, engine brake, sleeper size (in), sleeper type (Raised Roof Sleeper / Raised Roof Condo / Daycab / Mid-Roof / XT), **wheelbase (174–229 in)**, fuel type, colour, condition, dealer, price, mileage |
| **Photos** | 0–20 per listing (median ~15 over n=6 sampled). **Two of six sampled listings had zero photos** — a real coverage gap to plan around. Served via imgix (`selectrucks.imgix.net/{dealerId}/{file}?w=&h=`); page default 500×380 but **full-res 2500×1875** is retrievable by dropping the size params |
| **VIN enrichment** | **NHTSA vPIC**, free, no key, no rate limit, commercial use permitted. Verified on two live VINs (`3AKJHHDR4NSNA2397`, `3AKJHLDV7MSMG0922`) — both decoded clean, returning Make/Model/ModelYear, BodyClass=**Truck-Tractor**, GVWR=**Class 8**, EngineModel (Detroit DD15 14.8L / DD13 12.8L), DriveType=**6×4**, brake system, plant (Saltillo, Mexico) |
| **Filter vocabulary exposed** | classes 3–8; sleeper sizes 0/34/40/48/54/58/60/62/63/64/68/70/72/72ES/73/76/78/80/82/96; engine mfrs Cummins/Detroit/Paccar/Volvo/Mack/International/Mercedes/Isuzu; transmissions 6–18 speed + Automatic/Manual; suspensions 4-Bag Air Ride/Air Ride/Airliner/Chalmers/Haulmaax/Spring/Tuftrac; HP 0–600; mileage 0–2,002,277; years 1993–2027 |

#### 9b. Demo B — Türkiye: Ford Trucks F-MAX 4×2, MY2020–2022, tractor

| | |
|---|---|
| **Source** | TruckMarket (`truckmarket.com.tr`) — Ford Trucks Türkiye's OEM used network |
| **Access** | No `robots.txt` published (path returns the site's 404 page). List at `/arac-listesi`, detail at `/arac-detay/{id}`. Plain server-rendered HTML, no bot challenge, no auth |
| **Live volume (measured 2026-09-12)** | **218** listings total — 168 Ford **F-MAX**, 33 Ford Trucks (F-LINE etc.), 7 MAN TGS, 4 Iveco Stralis, 3 Mercedes Actros, 1 Scania R, 1 MAN TGA, 1 Mercedes Arocs. **147** F-MAX priced in the MY2020–22 demo window |
| **Price** | ₺1,800,000 – ₺2,950,000; **median ₺2,750,000 ≈ $56,589** at USD/TRY 48.596 |
| **Mileage** | 41,173 – 605,000 km (**median 270,000 km**) |
| **Per-listing fields** | year, brand, model, price ₺, km, **city**, transmission (Otomatik), colour, **drive type (4×2)**, **5th-wheel height (960 / 1100 / 1200 mm)**, **usage class (Yurt İçi / Yurt Dışı Lojistik = domestic / international logistics)**, warranty flag, authorisation-certificate number (yetki belgesi), ad number |
| **Photos** | **32–38 per listing (mean 35.1, n=12)** — two sizes, `_Orta` and `_Buyuk`; `_Buyuk` measured at **1440×1080**, ~278 KB. CDN path `truckmarket.b2el.net/B2ELResim/AracResim2El/{stockId}/{uuid}_Buyuk.jpg`. **218 listings × ~35 ≈ 7,600 images available today** |
| **Provenance check** | Listing cities are İstanbul, Denizli, Kayseri, Hatay — **all Turkish**, sold through Ford Trucks TR dealers. This resolves the "Turkish-language page may list a non-Turkish truck" risk *for this source*. It does **not** resolve it for sahibinden/arabam/Mascus, where a TR-language page routinely lists EU-located stock; for those, location must be verified per listing, not inferred from language or currency |
| **Missing vs US** | **No VIN. No engine spec. No horsepower. No wheelbase.** And no free Turkish VIN/registration decoder exists — identity must be recovered from the ruhsat or from the photographs themselves |

#### 9c. The measured asymmetry — the reason this pair is worth building

All figures measured on the live MY2020–2022 inventory of both sources on 2026-09-12.

| | US — Cascadia (SelecTrucks) | TR — F-MAX (TruckMarket) |
|---|---|---|
| Priced listings in window | 322 | 147 |
| Photos per listing | 0–20 (median ~15) | 32–38 (mean 35) |
| Max image resolution | 2500×1875 | 1440×1080 |
| VIN / free decode | ✅ vPIC | ❌ none |
| Engine, HP, sleeper, wheelbase | ✅ | ❌ |
| Market-specific config fields | sleeper type/size, suspension | 5th-wheel height, usage class |
| Median odometer | 726,447 km | 270,000 km (**2.69× lower**) |
| Median price | $69,900 | $56,589 (**1.24× lower**) |
| Price per 1,000 km | $96.22 | $209.59 |
| Distinct price points | 98 / 333 (**29%**) | 20 / 147 (**14%**) |
| Share on 5 most common prices | 25.5% | **61.9%** |
| Prices divisible by ₺100k / $10k | 1.2% | **76.2%** |
| log-price ~ age slope | **−21.8%/yr** (USD) | **−10.9%/yr** (nominal TRY) |
| R² of log-price ~ year + km | 0.429 | 0.791 |
| **Residual price spread after year+km** | **±22.6%** | **±5.0%** |

*Price-granularity rows (distinct points, top-5 share, round-number share) are computed over the full priced US set,
n=333 MY2019–22, against n=147 TR MY2020–22; every other row is the MY2020–22 window on both sides.
Regression rows use n=322 US / n=147 TR, a 3-parameter OLS of log price on age and mileage.*

**Read it this way:**
- The US asking price still contains ~±23% of variance that year and mileage do not explain. **That residual
  is the target.** If the vision condition model cannot compress it, it is not adding value.
- The Turkish asking price contains almost none — ±5%, most of which is quantization artefact (one price,
  ₺2,800,000, covers 50 of 147 listings). **A Turkish demo scored against asking price would produce a
  flatteringly low MdAPE that means nothing.** It would be measuring the dealer's pricing table, not the truck.

#### 9d. What this implies for the build

1. **Train the vision stack on the Turkish photos.** 7,600 images at 1440×1080, 35 views per vehicle,
   one model — the best per-vehicle coverage available in either market, and free of the per-listing
   photo gaps that affect the US source.
2. **Train and validate the pricing head on the US data.** It is the only side of the pair whose public
   price label carries condition signal, and VIN→vPIC gives clean configuration features for free.
3. **Do not score the Turkish demo against asking price.** Score it against (a) expert appraiser grades
   for the condition vector, and (b) realized sale prices obtained from Ford Trucks TR dealers or Kamion's
   own closed transactions. **Securing ~300–500 realized Turkish sale prices with matching photos is the
   single highest-value data-acquisition task in this project** — it is the gate on the whole Turkish
   pricing claim, and nothing else substitutes for it.
4. **Build the identity gate first (§3a).** The two vehicles share no part geometry. Without make/model/
   generation resolution upstream, one segmenter will hallucinate a hood on a cab-over.
5. **Deflate before you fit.** Fit the Turkish price model on CPI-deflated or FX-indexed real prices with an
   explicit as-of date; emit nominal TRY at inference. The −10.9%/yr nominal curve is an artefact of inflation.
6. **Stamp listing freshness.** A page being reachable today does not establish when its asking price was set.
   Record first-seen and last-seen timestamps per listing and treat price age as a feature and as a filter;
   a Turkish price set six months ago is, at ~31% CPI, a materially different number today.

#### 9e. Filling the realized-price gap

| Market | Source | Access posture | What it gives |
|---|---|---|---|
| US | **GovDeals** (Liquidity Services) | `robots.txt` permissive, `Crawl-delay: 5`, **public sitemaps** (`files.lqdt1.com/zstcntr-sitemap/…`); $903M FY2025 sales, 15,000–25,000 live items | Public-agency surplus trucks with **realized hammer prices** and photos. Skews municipal (dumps, vac trucks) rather than sleeper tractors — use for price-label supervision, not for Cascadia comps |
| US | **Ritchie Bros Price Results / IronPlanet** | Free tool behind registration; `rbauction.com` blocks unauthenticated datacentre IPs at the edge | Historic realized prices across RB auctions, IronPlanet, Marketplace-E and Mascus — the strongest public sold-price archive for heavy trucks |
| US | **Marketcheck** | Licensed, paid | Legally clean listings + cached images + used-heavy-equipment endpoint |
| US | ~~TruckPaper / Machinery Trader~~ | `robots.txt` permits listing paths **but the Terms of Use explicitly forbid robots, scrapers and data mining** — ToS governs, not robots.txt | Do not scrape |
| US | ~~CommercialTruckTrader~~ | `robots.txt` permissive, but returns **HTTP 403** to non-browser clients | Effectively closed |
| US | ~~Copart~~ | Incapsula challenge, HTTP 403 | Effectively closed |
| TR | **Ford Trucks TR dealers / Kamion closed deals** | Requires a commercial agreement | **The only realistic path to Turkish realized prices.** Target 300–500 records with photos |
| TR | Mercedes-Benz **TruckStore** TR, MAN **TopUsed** | Public listing pages; partnership needed for sale data | Second and third Turkish OEM channels — widen brand coverage beyond Ford |
| TR | **Mascus** | `robots.txt` permissive (blocks only a few query patterns), publishes sitemaps | Pan-EU heavy-truck pool. **Location must be verified per listing** — a Turkish-language page is not evidence of a Turkish truck |
| TR | ~~sahibinden.com~~ | Prohibited by ToS; login wall, Cloudflare, captcha, IP bans; blocked in testing | Do not use |
| TR | ~~arabam.com~~ | Cloudflare managed challenge on every request incl. `robots.txt`; no sanctioned programme | Do not use |

#### 9f. Concrete build order for the demo

| Step | Output | Check that it worked |
|---|---|---|
| 1 | Harvest TR: 218 listings → ~7,600 images at `_Buyuk`, plus the full structured record per listing | ≥95% of listings yield ≥30 usable images; city field populated for all |
| 2 | Harvest US: 385 Cascadia records via the inventory API; enrich every VIN through vPIC; pull full-res imgix originals | ≥90% of VINs decode clean; log the zero-photo rate explicitly |
| 3 | Build the identity gate: VIN-first for US, photo-first (badge/grille/silhouette) for TR; cross-check VIN against visual class | Cab-over vs conventional classified at ≥99%; make/model top-1 ≥95% on the two-class demo |
| 4 | Expert-label a golden set: 150 TR + 150 US vehicles, ≥2 appraisers, full condition vector | Report inter-rater κ per condition field — this is the human ceiling all later numbers are measured against |
| 5 | Fit the US pricing head on year + km + config, then add the vision condition delta | Baseline residual is ±22.6%; the condition delta has to beat it. Report MdAPE, PA10, PA20 and 80/90% interval coverage |
| 6 | Obtain 300–500 realized TR sale prices with photos; fit the TR head on deflated real prices | Until this exists, publish **no** Turkish price-accuracy claim — only condition-grade agreement against appraisers |
| 7 | Cross-market transfer test: TR-trained condition head, applied to US photos, scored against US appraiser grades (and vice versa) | This, not aggregate MdAPE, is the actual evidence of market-agnosticism |

**Deliberate exclusions for v1:** trailers (separate value drivers, §3), <16 t trucks, tippers/mixers/construction
bodies, and all non-tractor configurations. Add MAN TGX and Mercedes Actros on the Turkish side only after the
single-model demo closes — that is the first test of whether the brand/model conditioning in §3a and §4a works.

## Recommendations

1. **Start the proprietary data flywheel immediately** (Weeks 1–2). The public-data gap for commercial-truck damage means data collection — via guided capture at driver onboarding + expert labeling — is the critical path. Everything else is available off-the-shelf. **Benchmark to change course:** once you have ≥5,000 internally-labeled truck images per vehicle class, shift damage detectors from transfer-learning on car sets to training on your own data.
2. **Ship pricing v0 on public comps (Marketcheck) + tabular GBM before vision is ready**, then layer the vision condition-delta and internal residual-correction. **Threshold:** switch from public-prior-dominant to internal-posterior-dominant weighting once you have ~500–1,000 closed internal transactions per major segment.
3. **Adopt the hybrid architecture** (specialists → VLM) from the start; do not prototype on VLM-only, which will hallucinate and set false expectations.
4. **Instrument observability on day one** with OTel GenAI conventions + self-hosted Langfuse; build the evidence-grounding (box→claim→delta→price) as a core schema, not an afterthought — it is an explicit user requirement and a liability shield.
5. **Make the quality gate + refusal thresholds non-negotiable.** A confident price from a bad photo is the worst failure mode. Widen intervals, request re-shoots, refuse below threshold.
6. **Pursue data partnerships** with Mascus (largest EU heavy-truck pool), mobile.de (official dealer API), and Truck1 (JSON import feed); use Ritchie Bros/IronPlanet **sold** prices as transaction ground truth. **Do not scrape** TruckPaper/Machinery Trader (ToS-prohibited) or sahibinden (prohibited + defended).
7. **Handle plates/faces/GPS from the first upload** (auto-blur, GPS strip) to stay clean under GDPR/KVKK.
8. **Resolve identity before you assess damage** (§3a). Build the cab-architecture + make/model classifier in Weeks 3–6, ahead of the damage specialists. Two vehicles that share no part geometry cannot share a part segmenter, and an appraisal of the wrong model is worse than no appraisal because it fails silently. **Guardrail metric:** if brand/model features absorb more price contribution than the vision condition delta on a typical appraisal, the model is pricing a catalogue, not a truck.
9. **Run the two-market demo on Freightliner Cascadia (US, SelecTrucks) × Ford F-MAX (TR, TruckMarket), MY2020–2022, tractors only** (§9). Both sources are live, accessible and measured. **MY2020–2022, not 2019–2022** — the Turkish source has zero MY2019 listings, so a 2019 comparison would be US-only.
10. **Get realized Turkish sale prices before making any Turkish price claim.** Turkish asking prices resolve to 20 distinct values across 147 listings with a ±5.0% residual after year and mileage; they encode a dealer pricing table, not vehicle condition. Target **300–500 closed sales with matching photos** from Ford Trucks TR dealers or Kamion's own transactions. Until they exist, publish only condition-grade agreement against appraisers for Türkiye — never a price accuracy number.
11. **Deflate Turkish prices before fitting, and stamp every listing with a date.** Nominal TRY depreciation measures −10.9%/yr against −21.8%/yr in USD at ~31% CPI. Fit on real (CPI-deflated or FX-indexed) prices with an explicit as-of date; emit nominal TRY. Record first-seen/last-seen per listing and treat price age as both a feature and a filter.
12. **Verify listing location per record, never by language or currency.** TruckMarket listings are confirmed Türkiye-domestic (İstanbul, Denizli, Kayseri, Hatay), but on Mascus, sahibinden and arabam a Turkish-language page routinely lists EU-located stock. A comparable is only Turkish if the record says so.
13. **Price the appraisal as a wedge, not as the product** (§8f). An assessment-only business in US+TR heavy trucks addresses ~$143M TAM and yields a ~$2M three-year SOM. Per assessed truck, a 1.5% take rate on a $40,000 sale is ~$600 against ~$45 for a report — **13×** — and ACV's auction+assurance ARPU of $554/unit says that is the right order of magnitude. **Validate assumptions A5/A6 and B4/B5 (fleet count × subscription ASP) first: they are the least-grounded inputs and they carry two-thirds of the TAM.**
14. **Treat a fixed-site scanner incumbent as a non-threat in Türkiye and a real one in the US.** UVeye and ProovStation need the truck to drive through a booth. That is viable for US fleet yards and dealer lanes; it cannot serve a Turkish market where SME fleets hold ~95% share and the average heavy vehicle is 18.1 years old and nowhere near an inspection lane. **Smartphone-first is not a compromise in Türkiye — it is the only reachable form factor.**

## Caveats
- **No commercial-truck damage dataset at scale exists publicly** — the biggest uncertainty; Kamion's success depends on its own data program, and timelines assume that program starts immediately.
- **CarDD and VehiDE are non-commercial-licensed** (Flickr/Shutterstock) — usable for R&D/benchmarking only, not shippable models. This is a hard legal constraint.
- **Benchmark scores cited** (e.g., Qwen-VL 87.3% semantic; MANIQA SROCC; Cognexa 99.9% odometer) come from academic/vendor settings and different domains (cars, general images); real truck performance will differ and must be validated on Kamion's golden set.
- **VLM cost estimates** are order-of-magnitude at current pricing and will shift with model choice, image count, and provider pricing.
- **EU AI Act classification** for a valuation tool is not definitively settled here; obtain a formal legal classification before EU deployment.
- **Industry AVM benchmarks (MdAPE, PA10)** are largely from passenger-car markets; heavy-truck valuation is thinner and noisier, so calibrate expectations to appraiser-variance, not car-AVM numbers.
- **All §7/§9 market measurements are a single snapshot taken 2026-09-12 from one OEM-certified used channel per country** (SelecTrucks, TruckMarket). They are *asking* prices, not realized ones. OEM-certified stock is reconditioned to a standard, which independently compresses condition variance — so part of the narrow Turkish price spread is genuine channel homogeneity and part is pricing behaviour. The two effects are not separated here and both would need a second, independent Turkish source (Mercedes TruckStore TR, MAN TopUsed) to disentangle.
- **The MY2019 and MY2020 cells are thin** (US n=11 and n=32 respectively). Per-year medians in §9 for those years are indicative only; the MY2021/2022 cells (n=125 / n=216) carry the weight.
- **Photo-count figures are sampled, not censused** — 12 Turkish detail pages and 6 US detail pages. The US zero-photo rate (2 of 6) is a small-sample observation flagged for measurement in §9f step 2, not an established rate.
- **The ±22.6% / ±5.0% residual figures come from a 3-parameter OLS on log price against age and mileage.** They bound what year+mileage alone explain; they are not a claim about what a full hedonic model with configuration features would leave. The US figure in particular will shrink once sleeper type, engine family and wheelbase are included — the point is the *contrast* between the two markets, which is large enough to survive that.
- **All §8 TAM/SAM/SOM figures are modelled from the explicit assumption table in §8b, not measured.** The transaction-volume inputs (A1, B1) are the best-grounded; the fleet-subscription inputs (A5, A6, B4, B5) are the weakest and drive ~two-thirds of the TAM. Turkish transfer counts derive from applying a 2.1% truck share to the 11,213,405 total 2025 used-vehicle transfers, and TÜİK's "kamyon" category spans all goods vehicles over 3,500 kg — it is broader than the tractor segment this product initially targets.
- **USD/TRY 48.596 (11 Sep 2026) is used throughout for TRY→USD conversion.** With ~31% CPI and a managed depreciation path, every Turkish dollar figure here has a short shelf life and should be recomputed at the as-of date rather than quoted from this document.
- **Site access postures were tested from a single datacentre IP and are subject to change.** SelecTrucks and TruckMarket responded; rbauction.com, copart.com and commercialtrucktrader.com returned 403, and arabam.com returned a Cloudflare challenge — those blocks may be IP-reputation artefacts rather than policy. **In every case the Terms of Use govern, not `robots.txt`:** TruckPaper's robots.txt permits listing paths while its ToS explicitly forbids scraping. Re-verify posture and obtain written permission before any production harvest.
- Some dataset URLs (Roboflow single-author sets) may change or disappear; mirror any dataset you depend on.