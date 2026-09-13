"""Stage 2 - every condition claim bound to the photo it was seen in.

What stops this from being "a thin wrapper that sends photos to a vision API":

  * The model never emits a price, or prose. It fills a fixed structure.
  * `component` is a closed enum of heavy-vehicle parts. A tool that offers
    dent/scratch/paint is a car tool pointed at a truck, and that is exactly
    what "sensible to someone who knows trucks" is scoring.
  * Every issue names a photo. Since the close-up pass is given exactly one
    photograph per call, that binding is structural rather than something the
    model is asked to remember.
  * Anything not visible goes to `cannot_tell` and becomes a request for
    another photo, instead of becoming a confident guess.
  * Photos are selected view-first, so 30 frames of the same tire cost one
    slot, not thirty - and pass A gets its own selection, because the views
    that settle what a truck IS are not the views that show what is wrong
    with it.
  * Identity is read more than once. `make` picks the brand column in the
    price model and `same_vehicle` can stop the valuation; both are now
    decided by a majority of sampled reads with a measured agreement rate,
    and the badge is transcribed a second time from its own crop.

The stage was one call over every photo until it became five passes; see
`stage.py` for the orchestration and `prompts.py` for why the single call was
never going to look at any one frame properly.
"""
from __future__ import annotations

from .passes import (best_badge_frame, extract_json, merge_duplicates,
                     parse_badge, parse_closeup, parse_identity, parse_synthesis,
                     wants_crop, write_badge_crop, write_subject_crop)
from .prompts import (COMPONENT_SUMMARY, COMPONENTS, GRADES, IMPACTS, SEVERITIES,
                      SUMMARY_KEYS, SYSTEM, VIEW_QUESTIONS, closeup_schema,
                      questions_for, synthesis_schema)
from .stage import (IDENTITY_VIEW_PRIORITY, VIEW_PRIORITY, run,
                    select_identity_photos, select_photos)

__all__ = [
    "COMPONENTS", "COMPONENT_SUMMARY", "GRADES", "IDENTITY_VIEW_PRIORITY",
    "IMPACTS", "SEVERITIES", "SUMMARY_KEYS", "SYSTEM", "VIEW_PRIORITY",
    "VIEW_QUESTIONS", "best_badge_frame", "closeup_schema", "extract_json",
    "merge_duplicates", "parse_badge", "parse_closeup", "parse_identity",
    "parse_synthesis", "questions_for", "run", "select_identity_photos",
    "select_photos", "synthesis_schema", "wants_crop", "write_badge_crop",
    "write_subject_crop",
]
