"""Runtime configuration: paths, credentials, model ids, market constants.

Everything tunable lives here so the demo can be re-pointed at a different
backend or a re-fitted price model without touching pipeline code.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
META = DATA / "metadata"
IMAGES = DATA / "images"
MODELS = REPO / "models"
WEB = Path(__file__).resolve().parent / "web"

LISTINGS_CSV = META / "listings.csv"
IMAGES_CSV = META / "images.csv"
PRICE_MODEL = MODELS / "price_model.json"
GATE_THRESHOLDS = MODELS / "gate_thresholds.json"


def _load_dotenv() -> None:
    """Read REPO/.env into os.environ without clobbering real env vars.

    Deliberately hand-rolled: the only consumer is this file, and a
    dependency that reads a five-line KEY=VALUE file is not worth the
    install on a machine that has to work offline on demo day.
    """
    env_file = REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


# --- image formats --------------------------------------------------------
# Registered here, once, at import: iPhones shoot HEIC by default and the brief
# is explicitly "a seller with a phone". Without this, a folder of iPhone photos
# is collected, fails to decode, and every frame is marked unreadable - the gate
# then refuses the whole set on capture quality, which is the worst possible way
# to fail in front of someone holding the photos.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:  # pragma: no cover - the fallback is a clear error, not a crash
    HEIF_SUPPORT = False

# Single source of truth: the CLI folder walk and the web upload filter must
# agree, or one accepts a file the other rejects.
IMAGE_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif",
    *({".heic", ".heif", ".avif"} if HEIF_SUPPORT else set()),
})


# --- vision backend -------------------------------------------------------
# Pin Cursor by default: an unavailable Cursor account must fail visibly rather
# than silently spending through another provider. Explicit overrides remain
# available for deliberate benchmarking.
BACKEND_CHAIN = ("cursor", "openai", "anthropic")
BACKEND_OVERRIDE = os.environ.get("KAMION_VLM_BACKEND", "cursor").strip().lower() or "cursor"

CURSOR_API_KEY = os.environ.get("CURSOR_API_KEY", "").strip()
CURSOR_MODEL = os.environ.get("KAMION_CURSOR_MODEL", "gpt-5.6-sol").strip()
CURSOR_EFFORT = os.environ.get("KAMION_CURSOR_EFFORT", "low").strip()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
# The GPT-5.6 family ships as luna / sol / terra - there is no bare
# `gpt-5.6`. See `python -m app.vlm.bench` for how this default was chosen.
OPENAI_MODEL = os.environ.get("KAMION_OPENAI_MODEL", "gpt-5.6-sol").strip()
OPENAI_EFFORT = os.environ.get("KAMION_OPENAI_EFFORT", "low").strip()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL = os.environ.get("KAMION_ANTHROPIC_MODEL", "claude-opus-5").strip()
# Thinking is on by default on Opus 5. Effort is the latency dial: this is a
# live 4-minute demo, and the evidence call is a perception task rather than a
# reasoning one, so it does not need the default "high".
ANTHROPIC_EFFORT = os.environ.get("KAMION_ANTHROPIC_EFFORT", "low").strip()

# Timeout in seconds for an individual VLM completion call. Protects against
# hangs from stalled network connections or unresponsive provider agents.
VLM_TIMEOUT_SECONDS = int(os.environ.get("KAMION_VLM_TIMEOUT", "90"))

# Physical salvage and scrap value floor for a heavy semi-tractor in Turkey (TRY).
# Heavy commercial tractors retain significant intrinsic value in powertrain, axles,
# alloy fuel tanks, chassis steel, and recyclable materials even at high age/mileage.
SALVAGE_VALUE_TRY = float(os.environ.get("KAMION_SALVAGE_FLOOR_TRY", "250000.0"))
SALVAGE_VALUE_USD = float(os.environ.get("KAMION_SALVAGE_FLOOR_USD", "7500.0"))

# Photos sent to the vision model per appraisal. A phone-toting seller
# uploads 15-40 near-identical frames; the evidence stage gets a view-diverse
# subset instead, which is both cheaper and measurably less repetitive.
#
# Raised from 14 when evidence became a fan-out. Under the old single call each
# extra photo made one request longer and blunter; now each is its own call in
# a bounded pool, so the cost of one more frame is one more parallel call
# rather than a thinner share of the same attention.
MAX_EVIDENCE_PHOTOS = int(os.environ.get("KAMION_MAX_EVIDENCE_PHOTOS", "16"))
EVIDENCE_MAX_TOKENS = 16000

# --- the evidence fan-out -------------------------------------------------
# Pass A sees every photo and answers identity only, so it is short. Pass B is
# one photo per call and needs room for a full checklist. Pass C sees no images
# at all - it is rolling up text that already exists.
IDENTITY_MAX_TOKENS = int(os.environ.get("KAMION_IDENTITY_MAX_TOKENS", "4000"))
CLOSEUP_MAX_TOKENS = int(os.environ.get("KAMION_CLOSEUP_MAX_TOKENS", "4000"))
SYNTHESIS_MAX_TOKENS = int(os.environ.get("KAMION_SYNTHESIS_MAX_TOKENS", "6000"))
# Close-up calls in flight at once. Measured on a 16-photo set: median call
# 12.4 s, so five meant four waves and an 84 s run. Eight halves the waves
# without tripping a provider; the whole point of the pool is that
# wall-clock stays roughly where the single call was while the depth goes up.
#
# Raised 8 -> 16 when each photo became CLOSEUP_SAMPLES calls instead of one.
# 48 calls at 8-way is six waves; at 16-way it is three.
EVIDENCE_CONCURRENCY = int(os.environ.get("KAMION_EVIDENCE_CONCURRENCY", "16"))
# Long edge the photos are downscaled to before upload. 1024 keeps tread
# blocks and rust pitting legible while holding a 14-photo call near 20k
# input tokens.
EVIDENCE_IMAGE_LONG_EDGE = 1024

# --- self-consistency -----------------------------------------------------
# Each photo is read this many times and the samples are combined. One read
# was a single draw from a distribution nobody had measured: the severity a
# finding got depended on a sample, and that sample was multiplied straight
# into a price. Three is the smallest k that supports both a majority and a
# median. It is a config value rather than a constant so `eval.suites.retest`
# can settle it with a number instead of an assertion.
CLOSEUP_SAMPLES = int(os.environ.get("KAMION_CLOSEUP_SAMPLES", "3"))
# Samples that must report a defect before it counts as corroborated. Below
# this the finding is still shown and still cited - it just cannot be
# promoted to `major` on one vote.
CLOSEUP_QUORUM = int(os.environ.get("KAMION_CLOSEUP_QUORUM", "2"))

# --- identity, sampled the same way ---------------------------------------
# Pass A was a single call, and it is the most load-bearing call in the run:
# `make` picks the brand column in the price model, `model` picks the anchor
# row, `body_type` can stop the pricing stage outright and `same_vehicle` can
# stop the valuation. Every close-up photo has been read three times with a
# measured agreement rate since the fan-out landed; the call that decides what
# the truck IS was still one draw from an unmeasured distribution.
#
# Three, for the same reason as CLOSEUP_SAMPLES: the smallest k supporting both
# a majority and a median. Two extra calls against a ~51-call run.
IDENTITY_SAMPLES = int(os.environ.get("KAMION_IDENTITY_SAMPLES", "3"))
# Samples that must name the same make before it counts as agreed. Below this
# the read still stands - it is the band that widens, never the answer that
# gets deleted.
IDENTITY_QUORUM = int(os.environ.get("KAMION_IDENTITY_QUORUM", "2"))

# Pass A gets its own photo selection and its own resolution. `select_photos`
# round-robins on VIEW_PRIORITY, which leads with a tire close-up - correct for
# pass B, wrong for A. Identity lives in the front three-quarter, the side
# profile and the badge; a tire close-up contributes nothing to make, model or
# axle count and costs a slot. A side profile is also the ONLY view that can
# honestly settle 4x2 against 6x2.
IDENTITY_PHOTOS = int(os.environ.get("KAMION_IDENTITY_PHOTOS", "8"))
# Higher than EVIDENCE_IMAGE_LONG_EDGE: a model badge is small in frame and
# 1024 px across a whole tractor leaves "F-MAX 500" a few pixels tall.
IDENTITY_IMAGE_LONG_EDGE = int(os.environ.get("KAMION_IDENTITY_LONG_EDGE", "1536"))

# --- the badge read -------------------------------------------------------
# A second, independent identity measurement, the same posture as the odometer
# OCR and the chassis-plate VIN: one full-resolution call on a crop of the
# grille and door region, rather than trusting a badge read off a downscaled
# 8-photo montage. `badges_seen` has been collected since the first fan-out and
# was only ever displayed - nothing checked it corroborated make and model.
BADGE_READ = os.environ.get("KAMION_BADGE_READ", "1").strip() not in ("0", "false", "")
BADGE_MAX_TOKENS = int(os.environ.get("KAMION_BADGE_MAX_TOKENS", "2000"))
BADGE_EFFORT = os.environ.get("KAMION_BADGE_EFFORT", "low").strip()
# Fraction of the frame height the badge band occupies, measured down from the
# top of the subject box. The grille badge and the door model script both sit
# in the upper half of a cab-over tractor.
BADGE_CROP_TOP = float(os.environ.get("KAMION_BADGE_CROP_TOP", "0.10"))
BADGE_CROP_BOTTOM = float(os.environ.get("KAMION_BADGE_CROP_BOTTOM", "0.75"))

# --- identity widenings ---------------------------------------------------
# All three ASSUMED, and labelled so wherever they surface. This corpus has no
# ground truth for a misidentification, so none of them can be fitted - the
# same footing as the 1.12x-per-missing-view figure, and deliberately smaller
# than the measured 1.85x unknown-brand widening, which is a different and
# larger claim.
WMI_CONFLICT_WIDENING = float(os.environ.get("KAMION_WMI_CONFLICT_WIDENING", "1.20"))
IDENTITY_DISPUTED_WIDENING = float(os.environ.get("KAMION_IDENTITY_DISPUTED_WIDENING", "1.25"))
IDENTITY_UNKNOWN_WIDENING = float(os.environ.get("KAMION_IDENTITY_UNKNOWN_WIDENING", "1.35"))

# --- the model spec card --------------------------------------------------
# data/reference/models_tr.json, read by app/modelspec.py. Soft: absent, every
# accessor returns empty and the prompts read exactly as they did before.
# Days before app.cli doctor calls the card stale, matching new_prices_tr.json.
MODEL_SPEC_STALE_DAYS = int(os.environ.get("KAMION_MODEL_SPEC_STALE_DAYS", "180"))
# Pass D: the set-aware severity calibration, text-only, after the merge.
CALIBRATION_MAX_TOKENS = int(os.environ.get("KAMION_CALIBRATION_MAX_TOKENS", "8000"))

# --- reasoning effort, per pass -------------------------------------------
# Not one dial. Pass B was pinned at "low" because it was framed as pure
# perception, and that was true until it was given a rubric to apply. Reading
# a tread block off a photograph is perception; deciding whether that tread
# block is "moderate" against a written standard is deduction.
#
# These defaults are a hypothesis, not a measurement, and the hypothesis can
# fail in a specific way: more reasoning budget on a task whose failure mode
# is over-reporting may produce MORE findings rather than better-calibrated
# ones. `eval.suites.retest` sweeps low/medium/high and settles it.
IDENTITY_EFFORT = os.environ.get("KAMION_IDENTITY_EFFORT", "medium").strip()
CLOSEUP_EFFORT = os.environ.get("KAMION_CLOSEUP_EFFORT", "high").strip()
SYNTHESIS_EFFORT = os.environ.get("KAMION_SYNTHESIS_EFFORT", "high").strip()
CALIBRATION_EFFORT = os.environ.get("KAMION_CALIBRATION_EFFORT", "high").strip()


# --- market constants -----------------------------------------------------
# Stamped, not fetched: a demo must produce the same number twice. Turkish
# CPI runs ~30%/yr, so re-stamp this rather than quoting it months later.
USD_TRY = 48.596
FX_AS_OF = "2026-09-11"

# Turkish tractor-unit distance per year, as (q25, median, q75) over the 84 TR
# listings in the corpus. Used to tell a close-up call whether the wear in
# front of it is ahead of or behind what the odometer predicts - a judgement
# it could not make at all before, because it was never told the distance.
#
# Measured (n=84), but measured on DEALER STOCK OFFERED FOR SALE, which is not
# the same population as trucks in service. Label it that way wherever it
# surfaces.
KM_PER_YEAR_TR = (48_981, 61_901, 71_267)
KM_PER_YEAR_AS_OF = "2026-09-12"

# What each severity level costs to put right, in lira. NEVER goes in a
# prompt - the rubric anchors severity on repair EFFORT ("a workshop
# morning", "a component replacement") precisely so that teaching the model
# what a severity means never puts a currency figure in front of it, and the
# "the VLM never sees or emits a price" invariant stays whole. This table is
# for the README and the panel card only, and a test asserts it never appears
# in a prompt template.
#
# Guessed, and inflation-sensitive at ~31% CPI. Stamped like USD_TRY.
REPAIR_BANDS = {
    "cosmetic": (0, 5_000),
    "minor": (5_000, 25_000),
    "moderate": (25_000, 120_000),
    "major": (120_000, None),
}
REPAIR_BANDS_AS_OF = "2026-09-12"

# Euro emission standard is a step function in this segment, not a gradient.
# Türkiye mandated Euro 6 for new heavy-vehicle type approvals from 2016, so
# registration year is a usable proxy when the seller does not state it.
# Always surfaced as `inferred`, never as observed.
EURO_NORM_BY_YEAR = ((2016, "Euro 6"), (2009, "Euro 5"), (2006, "Euro 4"))


def euro_norm_for_year(year: int | None) -> str | None:
    if not year:
        return None
    for cutoff, norm in EURO_NORM_BY_YEAR:
        if year >= cutoff:
            return norm
    return "Euro 3"
