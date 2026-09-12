"""Canonical layout of the dataset bundle.

`data/` is designed to be shared on its own: zip it, hand it over, and it
resolves without the repo around it. Two rules make that work.

  1. Everything the bundle needs lives inside `data/` - the card, the manifests,
     the per-source provenance, and the images.
  2. Every path stored in a manifest is relative to `data/`, never to the repo
     root. A consumer joins it onto wherever they unpacked the bundle.

Scripts should import these constants rather than hardcoding strings, so the
layout can move again without a nine-file search-and-replace.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --- the shareable bundle -------------------------------------------------
DATA = REPO / "data"
IMAGES = DATA / "images"                 # gitignored; rebuildable from scripts/
META = DATA / "metadata"
SOURCES = META / "sources"               # per-source harvest + pipeline intermediates

CARD = DATA / "DATASET_CARD.md"

# --- delivered tables -----------------------------------------------------
MANIFEST = META / "manifest.jsonl"
IMAGES_CSV = META / "images.csv"
LISTINGS_CSV = META / "listings.csv"
SPLITS = META / "splits.json"
SUMMARY = META / "dataset_summary.json"
CLEANING_REPORT = META / "cleaning_report.json"
REVIEW = META / "manual_review_exclusions.json"

# --- pipeline intermediates ----------------------------------------------
CLEAN_INDEX = SOURCES / "images_clean.jsonl"
DEGRADED_INDEX = SOURCES / "images_degraded.jsonl"

DEGRADED_DIR = IMAGES / "degraded"


def listing_meta(source_key):
    """Harvested listing records for one source."""
    return SOURCES / f"{source_key}.jsonl"


def image_index(source_key):
    """Downloaded-image index for one source."""
    return SOURCES / f"{source_key}_images.jsonl"


def to_bundle(path):
    """Absolute or repo-relative path -> path relative to the bundle root."""
    p = Path(path)
    if not p.is_absolute():
        p = (REPO / p) if str(p).startswith("data/") else (DATA / p)
    return str(p.resolve().relative_to(DATA.resolve()))


def resolve(bundle_relative):
    """Bundle-relative path -> absolute path on this machine."""
    return DATA / bundle_relative


def ensure_dirs():
    for d in (DATA, IMAGES, META, SOURCES):
        d.mkdir(parents=True, exist_ok=True)
