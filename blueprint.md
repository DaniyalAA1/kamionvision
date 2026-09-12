# Vision-First Multi-Agent Truck Appraisal System for Kamion: A Technical Blueprint

## TL;DR
- **Build a hybrid architecture**: specialist vision models (YOLO/RT-DETR detectors, SAM 2 segmentation, DINOv2 embeddings, a dedicated no-reference image-quality gate) produce a structured, auditable **condition vector**, and a VLM (Claude/Gemini/Qwen-VL) reasons over that evidence — never a VLM alone, which hallucinates damage and cannot localize. A two-stage pricing engine (public-data prior + internal-transaction posterior via Bayesian hierarchical / residual-correction modeling) converts the condition vector into a price *range* with conformal-calibrated confidence tied to photo quality.
- **The single biggest constraint is data**: there is **no large, commercially-licensed public dataset of commercial-truck damage**. The only genuine one (DS4E "Truck Damage Detection," 4,962 images, CC BY 4.0) is small and single-author. Car-damage sets (CarDD 4,000 imgs; VehiDE 13,945 imgs) are non-commercial (Flickr/Shutterstock encumbered) and passenger-car only. Kamion **must bootstrap a proprietary truck/trailer/van dataset** and can only use public sets for transfer-learning R&D and the quality gate.
- **"Perfect observability" is achievable** by combining OpenTelemetry GenAI semantic conventions + a self-hosted trace backend (Langfuse or Arize Phoenix) with a domain-specific **evidence-grounding layer**: every price delta chains back pixel → bounding-box/mask → detected defect → condition score → price adjustment, with per-claim confidence, exact model/prompt versions, and replayability.

## Key Findings

1. **Data is the gating problem, not modeling.** Public truck-damage data barely exists. Kamion's competitive moat is its own labeled transaction+photo corpus. Plan for a data-collection program from day one (guided capture at driver onboarding + expert labeling).
2. **VLMs alone are insufficient but excellent reasoners.** Published evidence (Hogale et al., arXiv 2608.02470, "Grounding Agentic VLMs with Dedicated Segmentation for Fine-Grained Vehicle Damage Assessment"): a state-of-the-art VLM (Qwen-VL-2B, 4-bit GPTQ) "achieves strong semantic classification accuracy (87.3%) on this task but is systematically ungrounded at the spatial level: it hallucinates damage in reflective regions, misses elongated scratches entirely" — while the paper's dedicated segmentation approach far outperforms VLM localization. Use specialists for detection/localization, VLM for synthesis and narrative.
3. **A dedicated image-quality gate is mandatory and cheap.** Modern NR-IQA (MUSIQ, MANIQA, CLIP-IQA, Q-Align) is available off-the-shelf via `pyiqa`/IQA-PyTorch. But generic IQA answers "is this photo technically good," not "does this photo show the part I need" — Kamion needs a separate coverage/completeness classifier.
4. **Market-agnostic pricing is best done as prior+posterior.** Train a hedonic GBM on public comparables (currency-normalized, region-indexed, log-price target), then learn a residual correction / Bayesian hierarchical layer that up-weights Kamion's internal sales as they accumulate. Monotonic constraints keep mileage subordinate to the vision-derived condition delta.
5. **Uncertainty must be first-class.** Conformalized Quantile Regression (CQR) gives calibrated price intervals with finite-sample coverage guarantees; interval width should widen automatically when photo quality/coverage is poor, up to a refusal-to-quote threshold.
6. **Regulatory exposure is real.** A vehicle valuation tool is not automatically EU AI Act "high-risk," but photos containing plates/faces/GPS trigger GDPR and Turkey's KVKK. Turkish marketplace data (sahibinden, arabam) is largely off-limits for scraping.

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

### 5. MULTI-AGENT DESIGN & "PERFECT OBSERVABILITY"

**Agent topology (supervisor/worker with parallel fan-out + critic loop):**
1. **Intake/Triage Agent** — ingests photo set, dedups, extracts EXIF, routes.
2. **Image Quality Agent** — NR-IQA gate (§2A) per photo.
3. **Coverage/Completeness Agent** — canonical-view checklist (§2B); triggers re-shoot requests.
4. **Damage Assessment Agents (per region, parallel)** — detectors+segmenters over front/rear/sides/cab/trailer; fan-out over photos via Ray.
5. **Tire/Undercarriage Agent** — tread/axle/corrosion.
6. **Odometer/OCR Agent** — mileage + fraud check.
7. **Fraud Agent** — stock/AI-generated/tamper/duplicate detection.
8. **Comparables/Retrieval Agent** — vector search over comps (year/make/model/config + image-embedding similarity).
9. **Pricing Agent** — two-stage model + CQR interval.
10. **Adjudicator/Critic Agent** — reconciles disagreements (e.g., VLM claims damage a detector missed), runs a debate/critic loop, decides confidence, can demand re-analysis.
11. **Report Agent** — assembles the human-readable "why this price" report with annotated images.

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
- **Weeks 1–2:** IQA gate (pyiqa off-the-shelf) + coverage classifier (bootstrap labels) + odometer OCR (TRODO fine-tune) + guided-capture prototype + data-collection pipeline live. Pricing v0 = Marketcheck/partner comps + LightGBM on public tabular data (no vision yet).
- **Weeks 3–6:** damage specialists (fine-tune on DS4E + CC-BY car sets + early Kamion labels); condition vector v1; VLM reasoning node; LangGraph DAG; Langfuse observability; residual-correction pricing on first internal transactions.
- **Weeks 7–12:** trailer-specific heads; tire/axle grading; fraud agent; CQR intervals + refusal thresholds; adjudicator/critic loop; reviewer UI with override→label loop; golden-set eval + A/B vs human appraisers.

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

## Recommendations

1. **Start the proprietary data flywheel immediately** (Weeks 1–2). The public-data gap for commercial-truck damage means data collection — via guided capture at driver onboarding + expert labeling — is the critical path. Everything else is available off-the-shelf. **Benchmark to change course:** once you have ≥5,000 internally-labeled truck images per vehicle class, shift damage detectors from transfer-learning on car sets to training on your own data.
2. **Ship pricing v0 on public comps (Marketcheck) + tabular GBM before vision is ready**, then layer the vision condition-delta and internal residual-correction. **Threshold:** switch from public-prior-dominant to internal-posterior-dominant weighting once you have ~500–1,000 closed internal transactions per major segment.
3. **Adopt the hybrid architecture** (specialists → VLM) from the start; do not prototype on VLM-only, which will hallucinate and set false expectations.
4. **Instrument observability on day one** with OTel GenAI conventions + self-hosted Langfuse; build the evidence-grounding (box→claim→delta→price) as a core schema, not an afterthought — it is an explicit user requirement and a liability shield.
5. **Make the quality gate + refusal thresholds non-negotiable.** A confident price from a bad photo is the worst failure mode. Widen intervals, request re-shoots, refuse below threshold.
6. **Pursue data partnerships** with Mascus (largest EU heavy-truck pool), mobile.de (official dealer API), and Truck1 (JSON import feed); use Ritchie Bros/IronPlanet **sold** prices as transaction ground truth. **Do not scrape** TruckPaper/Machinery Trader (ToS-prohibited) or sahibinden (prohibited + defended).
7. **Handle plates/faces/GPS from the first upload** (auto-blur, GPS strip) to stay clean under GDPR/KVKK.

## Caveats
- **No commercial-truck damage dataset at scale exists publicly** — the biggest uncertainty; Kamion's success depends on its own data program, and timelines assume that program starts immediately.
- **CarDD and VehiDE are non-commercial-licensed** (Flickr/Shutterstock) — usable for R&D/benchmarking only, not shippable models. This is a hard legal constraint.
- **Benchmark scores cited** (e.g., Qwen-VL 87.3% semantic; MANIQA SROCC; Cognexa 99.9% odometer) come from academic/vendor settings and different domains (cars, general images); real truck performance will differ and must be validated on Kamion's golden set.
- **VLM cost estimates** are order-of-magnitude at current pricing and will shift with model choice, image count, and provider pricing.
- **EU AI Act classification** for a valuation tool is not definitively settled here; obtain a formal legal classification before EU deployment.
- **Industry AVM benchmarks (MdAPE, PA10)** are largely from passenger-car markets; heavy-truck valuation is thinner and noisier, so calibrate expectations to appraiser-variance, not car-AVM numbers.
- Some dataset URLs (Roboflow single-author sets) may change or disappear; mirror any dataset you depend on.