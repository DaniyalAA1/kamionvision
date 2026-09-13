"""What this repo knows about a specific truck model, as opposed to trucks.

Until now the only model-specific thing in the pipeline was a string. The
identity pass read "F-MAX" off a grille, `passes.vehicle_line` turned it into
one sentence - "identified from the full set as a Ford Trucks F-MAX (high
sleeper, 4x2, 2018-2022)" - and that sentence was the entire extent of what
passes B, C and D knew about the vehicle in front of them. `EXPECTED_WEAR` is
keyed by view and distance band; a 2015 Cargo tractor and a 2023 F-MAX at the
same odometer received the same reference for what "on schedule" looks like.

`data/reference/models_tr.json` is the fix, and it is deliberately the same
artefact shape as `data/reference/new_prices_tr.json`: hand-curated, every row
carrying a source URL, a date and a `source_type`, and `app.cli doctor`
reporting its age. The reasoning is the same too - a spec card nobody can trace
is a hallucination with a filename, and this one is read into a prompt whose
output a buyer is asked to trust.

Three hard rules this module enforces rather than documents:

  * **No prices, ever.** `assert_priceless` walks the loaded card and raises on
    anything that looks like a currency figure. The separation that answers
    "is it more than a thin wrapper" is that the VLM never sees a price, and a
    spec card is a new and very plausible way to leak one. A test calls this.
  * **Absent file degrades to the old behaviour.** Every accessor returns None
    or an empty list when `models_tr.json` is missing, so the pipeline runs
    exactly as it did before this module existed.
  * **A card never overrides the photograph.** The lines this module emits are
    framed as what the model generally is, and the prompt that consumes them
    says to report what is actually visible. A spec card that contradicts the
    pixels is a spec card being read wrong.
"""
from __future__ import annotations

import json
import re
import threading

from .config import REPO

MODEL_SPECS = REPO / "data" / "reference" / "models_tr.json"

# Anything that smells like money. Deliberately broad: the cost of a false
# positive here is a maintainer rewording a spec note, and the cost of a false
# negative is a currency figure in front of the model that assigns severity.
_PRICE = re.compile(
    r"(₺|\$|€|£|\bTRY\b|\bUSD\b|\bEUR\b|\bTL\b|\blira\b|\bdolar\b|\beuro\b"
    r"|\bprice[sd]?\b|\bcost(s|ing)?\b|\bworth\b|\bfiyat\b)", re.IGNORECASE)

_REFERENCE: dict | None = None
_LOCK = threading.Lock()


class PriceLeak(ValueError):
    """A spec card carried something that reads as money."""


def assert_priceless(payload) -> None:
    """Raise if any string in the part that reaches a prompt looks like a price.

    The one invariant in this file worth a runtime check rather than a comment.
    `known_weak_points` is the realistic leak: the natural way to write one is
    "the AdBlue tank cracks and is expensive to replace", and "expensive" is a
    price signal reaching the pass that decides severity.

    Scoped to `models` deliberately. The file's header prose documents this very
    rule and so contains the word "price" several times over; none of it is ever
    read into a prompt, and a check that fired on its own documentation would be
    turned off within a day.
    """
    if isinstance(payload, dict) and "models" in payload:
        payload = payload["models"]

    # Maintainer-facing keys, never read into a prompt. `corpus` quotes dataset
    # counts, `sources` carries URLs and `basis` explains provenance - all three
    # legitimately discuss pricing. Note that `known_weak_points[].note` is NOT
    # skipped: it is prompt surface and it is the realistic leak.
    skip = {"corpus", "basis", "sources", "aliases", "maintainer_note"}
    def walk(node, path: str):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in skip:
                    continue
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, str):
            hit = _PRICE.search(node)
            if hit:
                raise PriceLeak(
                    f"{MODEL_SPECS.name}{path} reads as money ({hit.group(0)!r}): "
                    f"{node[:80]!r}. The VLM never sees a price - say what the "
                    f"component does, not what it costs.")
    walk(payload, "")


def available() -> bool:
    return MODEL_SPECS.exists()


def reference() -> dict:
    """The parsed card, price-checked once on load."""
    global _REFERENCE
    with _LOCK:
        if _REFERENCE is None:
            if not MODEL_SPECS.exists():
                _REFERENCE = {"models": [], "as_of": None, "missing": True}
            else:
                data = json.loads(MODEL_SPECS.read_text(encoding="utf-8"))
                assert_priceless(data)
                _REFERENCE = data
        return _REFERENCE


def _reset() -> None:
    """Drop the cache. Tests that write a temporary card call this."""
    global _REFERENCE
    with _LOCK:
        _REFERENCE = None


def as_of() -> str | None:
    return reference().get("as_of")


# --- normalising a model string -------------------------------------------
# The reason this exists: `anchor.lookup` matches `str(model).strip().upper()`
# against its rows exactly, so a vision model answering "F Max", "FMAX" or
# "F-MAX 500" misses the F-MAX row and falls through to the brand default.
# `normalise_brand` has existed in pricing/features.py since the first fit;
# nothing ever did the same job for the model half of the pair.

def _key(text: str | None) -> str:
    """Fold a free-text model to a comparison key: letters and digits only."""
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def normalise_model(make: str | None, model: str | None) -> str | None:
    """Free-text model -> the canonical id used by the card and the anchor.

    Falls back to the input uppercased when the card has nothing to say, so a
    brand the reference has never heard of still reaches the anchor exactly as
    it does today rather than becoming None.
    """
    from .pricing.features import normalise_brand

    if not model:
        return None
    brand = normalise_brand(make)
    wanted = _key(model)
    if not wanted:
        return None

    for row in reference().get("models", []):
        if normalise_brand(row.get("brand")) != brand:
            continue
        candidates = [row.get("model"), *(row.get("aliases") or [])]
        if any(_key(c) == wanted for c in candidates if c):
            return str(row["model"]).upper()

    # Not in the card. Longest alias that the answer *starts with* catches the
    # "F-MAX 500" shape, where the model is right and a trim has been appended.
    best = None
    for row in reference().get("models", []):
        if normalise_brand(row.get("brand")) != brand:
            continue
        for candidate in [row.get("model"), *(row.get("aliases") or [])]:
            ck = _key(candidate)
            if ck and wanted.startswith(ck) and (best is None or len(ck) > len(_key(best))):
                best = str(row["model"]).upper()
    return best or str(model).strip().upper()


def card(make: str | None, model: str | None) -> dict | None:
    """The spec row for a (brand, model), or None."""
    from .pricing.features import normalise_brand

    canonical = normalise_model(make, model)
    if not canonical:
        return None
    brand = normalise_brand(make)
    for row in reference().get("models", []):
        if (normalise_brand(row.get("brand")) == brand
                and str(row.get("model", "")).upper() == canonical):
            return row
    return None


def vocabulary(make: str | None = None) -> list[str]:
    """The closed model list for the identity schema's enum.

    Always ends in "other": a judge's truck may be a model this file has never
    heard of, and forcing a wrong pick from a closed list is worse than an
    honest "other". Empty when the card is missing, and the caller then leaves
    `model` as a free string exactly as it is today.
    """
    from .pricing.features import normalise_brand

    rows = reference().get("models", [])
    if make:
        brand = normalise_brand(make)
        rows = [r for r in rows if normalise_brand(r.get("brand")) == brand]
    names = sorted({str(r["model"]).upper() for r in rows if r.get("model")})
    return names + ["other"] if names else []


def brand_vocabulary() -> list[str]:
    rows = reference().get("models", [])
    return sorted({str(r["brand"]).upper() for r in rows if r.get("brand")})


# --- generations ----------------------------------------------------------

def generations(make: str | None, model: str | None) -> list[dict]:
    row = card(make, model)
    return list(row.get("generations") or []) if row else []


def generation_ids(make: str | None, model: str | None) -> list[str]:
    ids = [str(g["id"]) for g in generations(make, model) if g.get("id")]
    return ids + ["unknown"] if ids else []


def generation_for_year(make: str | None, model: str | None,
                        year: int | None) -> dict | None:
    """The generation whose stated year span contains `year`.

    Used only as a cross-check against what the photographs say - the pipeline
    asks the vision model for a generation from the visual markers and then
    compares. It is deliberately not used to fill the answer in: a card that
    answers the question makes the check circular.
    """
    if not year:
        return None
    for gen in generations(make, model):
        lo, hi = _year_span(gen.get("years"))
        if lo is not None and lo <= year <= (hi or 9999):
            return gen
    return None


def _year_span(text) -> tuple[int | None, int | None]:
    """"2018-2023" / "2018-present" / "2018" -> (2018, 2023) | (2018, None)."""
    years = [int(m.group(0)) for m in re.finditer(r"(?:19|20)\d{2}", str(text or ""))]
    if not years:
        return (None, None)
    if len(years) == 1:
        # An open span ("2018-present") has one year and a word after it.
        open_ended = re.search(r"\d{4}\s*[-–]\s*(present|now|current)",
                               str(text), re.IGNORECASE)
        return (years[0], None if open_ended else years[0])
    return (min(years), max(years))


# --- what the prompts actually consume ------------------------------------

def spec_lines(make: str | None, model: str | None, *,
               generation: str | None = None) -> list[str]:
    """The model's spec, as lines for the per-appraisal zone of a prompt.

    Deliberately short. This sits in every close-up call, so each line has to
    earn its tokens sixteen times over.
    """
    row = card(make, model)
    if not row:
        return []
    lines: list[str] = []
    if row.get("segment"):
        lines.append(f"Segment: {row['segment']}.")
    if row.get("cab"):
        lines.append(f"Cab: {row['cab']}.")
    if row.get("driveline"):
        lines.append(f"Driveline: {row['driveline']}.")
    if row.get("axle_configs"):
        lines.append("Axle configurations offered: "
                     + ", ".join(str(a) for a in row["axle_configs"]) + ".")
    gen = next((g for g in (row.get("generations") or [])
                if str(g.get("id")) == str(generation)), None) if generation else None
    if gen and gen.get("visual_markers"):
        lines.append(f"This has been read as the {gen.get('years', gen['id'])} "
                     f"generation, identified by: "
                     + "; ".join(gen["visual_markers"][:3]) + ".")
    return lines


def weak_points(make: str | None, model: str | None,
                components: list[str] | None = None) -> list[tuple[str, str]]:
    """(component, note) pairs, optionally filtered to the ones in this frame.

    The filter is what makes this affordable: a close-up of a tire has no
    business being told about this model's AdBlue tank, and `closeup_prompt`
    already knows which components belong to the view it is building.
    """
    row = card(make, model)
    if not row:
        return []
    wanted = set(components or [])
    out = []
    for entry in row.get("known_weak_points") or []:
        component = str(entry.get("component") or "")
        note = str(entry.get("note") or "").strip()
        if not component or not note:
            continue
        if wanted and component not in wanted:
            continue
        out.append((component, note))
    return out


def identity_tells(make: str | None, model: str | None) -> list[str]:
    row = card(make, model)
    return [str(t) for t in (row.get("identity_tells") or [])] if row else []


def sources(make: str | None, model: str | None) -> list[dict]:
    row = card(make, model)
    return list(row.get("sources") or []) if row else []


def coverage() -> dict:
    """What `app.cli doctor` reports: how much of this file is actually sourced."""
    rows = reference().get("models", [])
    cited = [r for r in rows if r.get("sources")]
    return {
        "present": available(),
        "as_of": as_of(),
        "models": len(rows),
        "models_with_sources": len(cited),
        "generations": sum(len(r.get("generations") or []) for r in rows),
        "weak_points": sum(len(r.get("known_weak_points") or []) for r in rows),
    }
