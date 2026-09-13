"""The panelist instrument: one prompt, one response schema, one rubric hash.

Why this exists
---------------
The backend exaggerates truck condition and there is nothing in the corpus to
measure that against. `damaged` is null in 100% of 7,458 rows, no Turkish
hasar / tramer / boyalı / değişen / ekspertiz field appears anywhere in the
metadata, and `data/DATASET_CARD.md` records the absence as known limitation 3.
Nobody has inspected these trucks.

So the reference is graded by a panel of Claude Opus 5 agents under the rubric
in `app/evidence/prompts.py`. That is legitimate only because of a structural
asymmetry, and the asymmetry runs in the panel's favour, not the pipeline's:

  the panel sees EVERY photograph of the vehicle at once, plus the registration
  year and the odometer reading. The production close-up call sees ONE
  photograph and is never told the distance.

The panel is therefore not a ceiling the pipeline could reach by being smarter
at the same task. It is a better-informed reader, and part of any gap between
them is an information gap by construction. Say that wherever a panel number is
quoted; `panel/card.py` writes it into the artifact card.

What is embedded, and why verbatim
----------------------------------
The panel grades under the SAME words the production call is given - the
severity rubric, the per-component anchors, the expected-wear row for this
truck's distance, and the six worked examples. If the panel graded under
different words it would be measuring a different rubric, and the gap between
panel and pipeline would be a wording difference rather than a reading
difference.

Which makes `rubric_sha` load-bearing: a panel label is only valid for the
rubric version it was graded under. Every stored row carries it. Change a
comma in `SEVERITY_RUBRIC` and the hash moves, and the rows that no longer
match are re-scorable (all panelist answers are kept) but not directly
comparable.

One thing is deliberately NOT embedded: `config.REPAIR_BANDS`, the lira reading
of the four severity words. The rubric is anchored on repair EFFORT precisely so
that teaching a model what a severity means never puts a currency figure in
front of it. A test asserts it never reaches this prompt either.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.evidence import prompts as P

SCHEMA_VERSION = "condition_panel/1"

# The ordinal scale the panel answers on. "none" is the bottom of the severity
# continuum, not a separate category: a component that is visible and in the
# state its age predicts is a real reading and the most common one.
SEVERITY_SCALE = ["none"] + list(P.SEVERITIES)   # none < cosmetic < minor < moderate < major
NOT_VISIBLE = "not_visible"
# not_visible is NOT on that scale - it is the absence of a reading, and
# `agreement.py` treats it as missing rather than as a fifth low level.
COMPONENT_STATES = SEVERITY_SCALE + [NOT_VISIBLE]

SEVERITY_INDEX = {level: i for i, level in enumerate(SEVERITY_SCALE)}

GRADES = list(P.GRADES)


# --- the rubric, recomposed from the production constants ------------------
# `COMPONENT_ANCHORS[view]` is the family blocks for that view joined on a
# blank line, and no block contains a blank line, so the split is exact. It is
# checked here rather than assumed: a rubric that silently loses a family would
# produce a panel that grades components it was never shown the levels for.

def family_blocks() -> dict[str, str]:
    """Every distinct `COMPONENT_ANCHORS` block, once, in family order."""
    seen: dict[str, str] = {}
    for view, families in P.VIEW_FAMILIES.items():
        parts = P.COMPONENT_ANCHORS[view].split("\n\n")
        if len(parts) != len(families):
            raise AssertionError(
                f"COMPONENT_ANCHORS[{view!r}] does not split into "
                f"{len(families)} family blocks - the composition rule in "
                "prompts.py changed and panel/protocol.py must be re-read.")
        for family, block in zip(families, parts):
            if seen.setdefault(family, block) != block:
                raise AssertionError(
                    f"anchor block for family {family!r} differs between views")
    missing = set(P.ANCHOR_COMPONENTS) - set(seen)
    if missing:
        raise AssertionError(f"no anchor block recovered for {sorted(missing)}")
    return {family: seen[family] for family in P.ANCHOR_COMPONENTS}


def wear_rows(km: int | None) -> tuple[str | None, dict[str, str]]:
    """The `EXPECTED_WEAR` row per family for this truck's distance band."""
    band = P.wear_band(km)
    if band is None:
        return None, {}
    rows: dict[str, str] = {}
    for view, families in P.VIEW_FAMILIES.items():
        for family, text in zip(families, P.EXPECTED_WEAR[view][band]):
            rows.setdefault(family, text)
    return band, {family: rows[family] for family in P.ANCHOR_COMPONENTS}


def rubric_text() -> str:
    """The exact bytes the hash covers: the four constants, canonicalised."""
    return "\n\n".join([
        "## SEVERITY_RUBRIC",
        P.SEVERITY_RUBRIC,
        "## COMPONENT_ANCHORS",
        json.dumps(P.COMPONENT_ANCHORS, sort_keys=True, ensure_ascii=False),
        "## EXPECTED_WEAR",
        json.dumps(P.EXPECTED_WEAR, sort_keys=True, ensure_ascii=False,
                   default=list),
        "## WORKED_EXAMPLES",
        P.WORKED_EXAMPLES,
    ])


def rubric_sha() -> str:
    return hashlib.sha256(rubric_text().encode("utf-8")).hexdigest()


RUBRIC_SHA = rubric_sha()


# --- the response schema ---------------------------------------------------

def response_schema() -> dict:
    state = {"type": "string", "enum": COMPONENT_STATES}
    photos = {"type": "array", "items": {"type": "integer"}}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["components", "findings", "grade", "grade_reason",
                     "confidence", "not_assessable"],
        "properties": {
            "components": {
                "type": "object",
                "additionalProperties": False,
                "required": list(P.COMPONENTS),
                "properties": {c: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["severity", "note", "photos"],
                    "properties": {"severity": state,
                                   "note": {"type": "string"},
                                   "photos": photos},
                } for c in P.COMPONENTS},
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["component", "severity", "note", "photos"],
                    "properties": {
                        "component": {"type": "string", "enum": list(P.COMPONENTS)},
                        "severity": {"type": "string", "enum": list(P.SEVERITIES)},
                        "note": {"type": "string"},
                        "photos": photos,
                    },
                },
            },
            "grade": {"type": "string", "enum": GRADES},
            "grade_reason": {"type": "string"},
            "confidence": {"type": "number"},
            "not_assessable": {"type": "array", "items": {"type": "string"}},
        },
    }


def validate_response(obj: object, *, n_photos: int) -> list[str]:
    """Every way a panelist answer can be unusable, named. Empty list is clean."""
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["response is not a JSON object"]

    comps = obj.get("components")
    if not isinstance(comps, dict):
        errors.append("components missing or not an object")
    else:
        missing = [c for c in P.COMPONENTS if c not in comps]
        unknown = [c for c in comps if c not in P.COMPONENTS]
        if missing:
            errors.append(f"components missing {len(missing)}: {missing[:5]}")
        if unknown:
            errors.append(f"components has unknown ids: {unknown[:5]}")
        for cid, entry in comps.items():
            if cid not in P.COMPONENTS:
                continue
            if not isinstance(entry, dict):
                errors.append(f"components.{cid} is not an object")
                continue
            sev = entry.get("severity")
            if sev not in COMPONENT_STATES:
                errors.append(f"components.{cid}.severity {sev!r} not in vocabulary")
            errors += _photo_errors(entry.get("photos"), n_photos, f"components.{cid}")
            if sev in P.SEVERITIES and not entry.get("photos"):
                errors.append(f"components.{cid} is {sev} but cites no photo")

    findings = obj.get("findings")
    if not isinstance(findings, list):
        errors.append("findings missing or not a list")
    else:
        for i, f in enumerate(findings):
            if not isinstance(f, dict):
                errors.append(f"findings[{i}] is not an object")
                continue
            if f.get("component") not in P.COMPONENTS:
                errors.append(f"findings[{i}].component {f.get('component')!r} unknown")
            if f.get("severity") not in P.SEVERITIES:
                errors.append(f"findings[{i}].severity {f.get('severity')!r} not a level")
            errors += _photo_errors(f.get("photos"), n_photos, f"findings[{i}]")
            if not f.get("photos"):
                errors.append(f"findings[{i}] cites no photo")

    if obj.get("grade") not in GRADES:
        errors.append(f"grade {obj.get('grade')!r} not in {GRADES}")
    if not isinstance(obj.get("grade_reason"), str) or not obj.get("grade_reason"):
        errors.append("grade_reason missing")
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not 0 <= conf <= 1:
        errors.append(f"confidence {conf!r} not a number in 0..1")
    if not isinstance(obj.get("not_assessable"), list):
        errors.append("not_assessable missing or not a list")
    return errors


def _photo_errors(photos: object, n_photos: int, where: str) -> list[str]:
    if photos is None:
        return [f"{where}.photos missing"]
    if not isinstance(photos, list):
        return [f"{where}.photos is not a list"]
    bad = [p for p in photos
           if not isinstance(p, int) or isinstance(p, bool) or not 0 <= p < n_photos]
    return [f"{where}.photos cites {bad[:5]} outside 0..{n_photos - 1}"] if bad else []


def severity_vector(response: dict) -> dict[str, str]:
    """component -> state, for the agreement pass. Absent reads as not_visible."""
    comps = response.get("components") or {}
    out = {}
    for c in P.COMPONENTS:
        entry = comps.get(c)
        sev = entry.get("severity") if isinstance(entry, dict) else None
        out[c] = sev if sev in COMPONENT_STATES else NOT_VISIBLE
    return out


# --- the prompt ------------------------------------------------------------
# Rule 1 is the whole reason this instrument exists. The reference cannot be
# built by a reader that is itself inflated, so "none" is stated as the
# expected common answer before anything else is said.

RULES = """\
1. "none" means the component is visible and in the state its age and distance
   predict. That is the most common answer and you should expect to write it
   more often than any severity. A consumable halfway through its life on a
   truck that has run this far is "none", not "minor". Read the expected-wear
   row above before you grade anything, and grade against THAT truck, not
   against a new one.

2. "not_visible" means no photograph in this set shows the component well
   enough to judge it. It is not a polite "none". If you cannot see it, say
   you cannot see it - the difference between "I looked and it is fine" and "I
   could not look" is the most useful thing in this whole answer.

3. Every severity you assign must cite the photo indices you saw it in. A
   finding with no photo behind it is not a finding. If the same wear shows in
   three photographs, cite all three on the one entry rather than writing it
   out three times.

4. Grade on the WORST DISTINCT defect and on how much of the truck was visible -
   never on how many findings there are. A well-photographed truck produces a
   long list because it was photographed well, not because it is in worse
   condition than one photographed badly. Cosmetic and minor entries must not
   pull the grade down; "poor" means something on this truck needs money spent
   on it before it works, and "fair" means real wear a buyer would negotiate
   over.

5. A truck with no visible defects and thin coverage is not "excellent" - it is
   ungraded coverage, so put what you could not cover in "not_assessable" and
   keep confidence low.

6. Do not estimate a price, and do not name one. You are reading condition.

7. Work only from the photographs in front of you. Do not infer mechanical
   condition from a clean exterior, and do not carry anything over from another
   truck you have looked at."""


def panelist_prompt(*, listing_id: str, make: str | None, model: str | None,
                    year: int | None, km: int | None, market: str,
                    photos: list[str]) -> str:
    """The full instrument handed to one panelist for one vehicle."""
    band, wear = wear_rows(km)
    km_line = f"{km:,} km (read off the seller's listing)" if km else "distance not stated"
    vehicle = " ".join(str(x) for x in (year, make, model) if x) or "tractor unit"
    photo_list = "\n".join(f"  [{i}] {p}" for i, p in enumerate(photos))

    anchors = "\n\n".join(family_blocks().values())
    if wear:
        wear_block = "\n".join(
            f"  {family:<14} {text}" for family, text in wear.items())
        wear_head = (
            f"This truck has run {km_line.split(' (')[0]}, which puts it in the "
            f"\"{band}\" distance band. At that distance the following is what "
            "ON SCHEDULE looks like. Anything matching these lines is \"none\", "
            "and belongs in your answer as a component you looked at and found "
            "unremarkable:")
    else:
        wear_block = "  (the seller stated no distance - grade against the registration year alone)"
        wear_head = ("No odometer reading was published for this truck, so there is no "
                     "expected-wear row to grade against:")

    return f"""{P.SYSTEM}

You are one of three independent readers grading the SAME vehicle. You will not
see the other two answers and they will not see yours. Answer as you actually
read it; do not hedge toward what you imagine an average reader would say.

=== THE VEHICLE ===

  listing      {listing_id}  ({market})
  vehicle      {vehicle}
  registered   {year or "not stated"}
  odometer     {km_line}
  photographs  {len(photos)}, all of them below, and they are all of this one truck

Open every one of these with the Read tool before you answer. The index in
square brackets is how you cite it:

{photo_list}

=== WHAT THE SEVERITY WORDS MEAN ===

{P.SEVERITY_RUBRIC}

=== WHAT EACH LEVEL LOOKS LIKE, PER COMPONENT ===

{anchors}

=== WHAT THIS TRUCK'S DISTANCE ALREADY PREDICTS ===

{wear_head}

{wear_block}

These rows are stated assumptions, domain-reasoned, not measured - the corpus
carries no maintenance records.

=== WORKED EXAMPLES ===

{P.WORKED_EXAMPLES}

=== TWO WORDS IN THE TEXT ABOVE THAT ARE NOT IN YOUR SCHEMA ===

The rubric and the worked examples are quoted verbatim from the production
prompt, which has fields yours does not. Where they say something "belongs in
strengths", your answer for that component is "none" - you looked at it and
found it unremarkable, which is the whole of rule 1 below. Ignore
"price_impact"; you are not asked for one.

=== THE RULES ===

{RULES}

=== WHAT TO RETURN ===

ONE JSON object and nothing else - no markdown fence, no commentary before or
after it. Every one of the {len(P.COMPONENTS)} component ids must appear as a key of
"components", exactly once, spelled exactly as listed:

{json.dumps(P.COMPONENTS, indent=2)}

{{
  "components": {{
    "<component_id>": {{
      "severity": one of {json.dumps(COMPONENT_STATES)},
      "note": string,      // short. why that level, or what you could not see.
      "photos": [integer]  // indices you judged it from. [] only for not_visible.
    }}
    // ... all {len(P.COMPONENTS)} ids
  }},
  "findings": [            // ONLY the components you gave a real severity to
    {{"component": string, "severity": one of {json.dumps(list(P.SEVERITIES))},
      "note": string, "photos": [integer]}}
  ],
  "grade": one of {json.dumps(GRADES)},
  "grade_reason": string,  // one or two sentences, naming the defect that set it
  "confidence": 0.0-1.0,
  "not_assessable": [string]   // what this photo set could not show you
}}"""
