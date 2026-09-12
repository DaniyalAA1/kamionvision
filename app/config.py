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


# --- vision backend -------------------------------------------------------
# The chain is tried in order and the first available backend wins, unless
# KAMION_VLM_BACKEND pins one. OpenAI leads because GPT-5.6 is the model this
# build is meant to run on; Cursor is the SDK integration and takes over as
# soon as it has a key and a settled account; Anthropic is the last fallback
# so a live demo never dies on one provider being unreachable.
BACKEND_CHAIN = ("openai", "cursor", "anthropic")
BACKEND_OVERRIDE = os.environ.get("KAMION_VLM_BACKEND", "").strip().lower() or None

CURSOR_API_KEY = os.environ.get("CURSOR_API_KEY", "").strip()
CURSOR_MODEL = os.environ.get("KAMION_CURSOR_MODEL", "claude-4.5-sonnet").strip()

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

# Photos sent to the vision model per appraisal. A phone-toting seller
# uploads 15-40 near-identical frames; the evidence call gets a view-diverse
# subset instead, which is both cheaper and measurably less repetitive.
MAX_EVIDENCE_PHOTOS = int(os.environ.get("KAMION_MAX_EVIDENCE_PHOTOS", "14"))
EVIDENCE_MAX_TOKENS = 16000
# Long edge the photos are downscaled to before upload. 1024 keeps tread
# blocks and rust pitting legible while holding a 14-photo call near 20k
# input tokens.
EVIDENCE_IMAGE_LONG_EDGE = 1024


# --- market constants -----------------------------------------------------
# Stamped, not fetched: a demo must produce the same number twice. Turkish
# CPI runs ~30%/yr, so re-stamp this rather than quoting it months later.
USD_TRY = 48.596
FX_AS_OF = "2026-09-11"

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
