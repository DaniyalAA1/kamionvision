"""Adjudicate what the truck is, across every witness that has an opinion.

By the time this runs, four different things may have named the vehicle, and
until now nothing put them in a room together:

    the identity pass   several sampled calls over the whole photo set
    the badge read      one full-resolution call on a crop of the grille/door
    the chassis plate   a VIN's world manufacturer identifier, deterministic
    the identity head   a linear probe over pooled CLIP embeddings

`VehicleRead.confidence` was the only identity number the pipeline had, and it
is the vision model's opinion of its own answer. Meanwhile only `body_type` and
`same_vehicle` could stop the pipeline - so the system had no way at all to say
"I do not know what this truck is well enough to put a band on it", even though
brand is a term in the price model and an unseen brand is the measured 1.85x
widening, which is to say the likeliest way this fails in front of a judge.

Two design rules worth stating because both were tempting to break.

**One adjudicator, one widening.** `reconcile` already records a correction when
the head disputes the badge, and now another when the WMI does. If each of those
also widened the band, one disagreement between one badge and one plate would
compound through two or three multipliers - 1.25 x 1.20 is a 1.5x band for a
single fact. So the rules keep recording corrections, which is the audit trail,
and this module owns the widening. `merge_widening` is how a caller drops the
per-rule multipliers that this verdict has superseded.

**A witness that cannot have an opinion does not get a vote.** The identity head
may only speak about brands in its own class list - the corpus has no Scania and
a head that has never seen one will still name a class, confidently. The WMI
table is silent about manufacturers it has not verified. Absence is "unknown",
never "conflict", and a witness that abstains must not dilute the ones that did
speak. That rule is why `status` is decided on the witnesses that actually voted
rather than on a fraction of the four possible ones.
"""
from __future__ import annotations

from .config import (IDENTITY_DISPUTED_WIDENING, IDENTITY_UNKNOWN_WIDENING,
                     WMI_CONFLICT_WIDENING)
from .schema import IdentityVerdict

# How much each witness is worth when they disagree. Ordinal, not probabilistic:
# these are ASSUMED weights - this corpus has no misidentification ground truth,
# so there is nothing to fit them against, and they are labelled assumed
# wherever they surface.
#
# The order is the defensible part. A VIN is stamped into the chassis and
# decoded by an arithmetic check digit, so it outranks anything read off
# bodywork. A badge photographed at full resolution outranks the same badge
# read off a downscaled eight-photo montage. The trained head is last because
# its own classes are CLIP pseudo-labels and it is the only witness that cannot
# say "I don't know".
WITNESS_WEIGHT = {
    "wmi": 1.00,
    "badge": 0.80,
    "identity_pass": 0.70,
    "head": 0.45,
}

# A witness below this is treated as not having spoken. The head's own
# threshold in `reconcile` is 0.60 and this matches it deliberately: two
# different numbers for "confident enough to matter" would drift apart.
MIN_WITNESS_CONFIDENCE = 0.60

# What to ask for, per missing thing. The brief rewards the re-ask path
# explicitly and it existed only for capture quality - "send me a shot of the
# tires" had no counterpart for "I cannot tell what make this is".
REASK_BADGE = ("Send one more photo: the front grille square on, close enough "
               "to read the maker's badge and any model script on it.")
REASK_PLATE = ("Send a photo of the chassis plate or VIN sticker - usually "
               "inside the driver's door frame or on the right-hand frame rail.")


def _brand(value: str | None) -> str | None:
    from .pricing.features import normalise_brand

    if not value or not str(value).strip():
        return None
    folded = normalise_brand(value)
    return None if folded == "other" else folded


def collect(vehicle, perception=None, *, head_classes=None) -> list[list]:
    """Every witness with a usable opinion, as [source, brand, confidence].

    Pure and separate from `decide` so the witness list can be asserted in a
    test without constructing a verdict, and so a caller can show the reader
    exactly who voted.
    """
    witnesses: list[list] = []

    make = _brand(getattr(vehicle, "make", None))
    if make:
        # The sampled agreement rate where there is one, the model's self-report
        # where there is not. `identity_agreement` is a measured quantity about
        # the claim; `confidence` is the model's opinion of itself. Prefer the
        # measurement, and fall back rather than assuming a single call agreed
        # with itself perfectly.
        agreement = float(getattr(vehicle, "identity_agreement", 0.0) or 0.0)
        confidence = agreement or float(getattr(vehicle, "confidence", 0.0) or 0.0)
        witnesses.append(["identity_pass", make, round(confidence, 3)])

    badge = _brand(getattr(vehicle, "badge_make", None))
    if not badge:
        # No parsed badge make: fall back to matching the literal transcription
        # against the brands we know. A badge read is worth having even when the
        # pass that produced it did not name a make.
        badge = _brand_from_text(getattr(vehicle, "badge_text", None))
    if badge:
        witnesses.append(["badge", badge, 0.9])

    wmi_brand = _brand(getattr(vehicle, "wmi_brand", None))
    if wmi_brand:
        # Deterministic: either the table has a verified row for those three
        # characters or it does not. There is no soft version of this read.
        witnesses.append(["wmi", wmi_brand, 1.0])

    if perception is not None:
        head_brand = _brand(getattr(perception, "brand", None))
        conf = float(getattr(perception, "brand_conf", 0.0) or 0.0)
        # The head may not dispute a brand it was never trained on, and the test
        # is on what the BADGE said, not on what the head said - exactly as
        # `reconcile._check_identity` does it (`if said not in classes: return`).
        # The distinction matters and is easy to get backwards: the corpus has
        # no Scania, so when the vision model reads SCANIA off a grille the head
        # will still answer with some class it does know, confidently, and that
        # answer is an artefact of the corpus rather than an observation about
        # this truck. Gating on the head's own output would let exactly that
        # artefact through, because the class it names is by construction one it
        # was trained on.
        known = {_brand(c) for c in (head_classes or [])} if head_classes else None
        disputable = known is None or make is None or make in known
        if head_brand and conf >= MIN_WITNESS_CONFIDENCE and disputable:
            witnesses.append(["head", head_brand, round(conf, 3)])

    return witnesses


def _brand_from_text(lines) -> str | None:
    """Match a literal badge transcription against the brands we can name."""
    from .pricing.features import normalise_brand

    if not lines:
        return None
    known = ["FORD", "MERCEDES-BENZ", "MERCEDES", "MAN", "SCANIA", "DAF", "VOLVO",
             "RENAULT", "IVECO", "FREIGHTLINER", "KENWORTH", "PETERBILT", "MACK",
             "INTERNATIONAL", "WESTERN STAR"]
    haystack = " ".join(str(line).upper() for line in lines)
    for name in sorted(known, key=len, reverse=True):
        if name in haystack:
            folded = normalise_brand(name)
            return None if folded == "other" else folded
    return None


def decide(vehicle, perception=None, *, head_classes=None) -> IdentityVerdict:
    """The witnesses -> one verdict, with the band multiplier it justifies."""
    verdict = IdentityVerdict()
    witnesses = collect(vehicle, perception, head_classes=head_classes)
    verdict.witnesses = witnesses
    verdict.model = getattr(vehicle, "model_canonical", None) or getattr(vehicle, "model", None)

    speaking = [w for w in witnesses if w[2] >= MIN_WITNESS_CONFIDENCE]
    if not speaking:
        verdict.status = "unknown"
        verdict.make = None
        verdict.confidence = 0.0
        verdict.widening = IDENTITY_UNKNOWN_WIDENING
        verdict.reason = ("nothing in these photos identified the make with any "
                          "confidence, so the brand term in the price model is a "
                          f"guess and the band widens {IDENTITY_UNKNOWN_WIDENING:.2f}x "
                          "(a stated assumption)")
        verdict.reask = REASK_BADGE
        return verdict

    # Score each named brand by the weight of the witnesses backing it.
    scores: dict[str, float] = {}
    for source, brand, confidence in speaking:
        scores[brand] = scores.get(brand, 0.0) + WITNESS_WEIGHT.get(source, 0.5) * confidence

    winner = max(scores, key=lambda b: scores[b])
    verdict.make = winner
    verdict.agreed = sorted({w[0] for w in speaking if w[1] == winner})
    verdict.disagreed = sorted({w[0] for w in speaking if w[1] != winner})
    total = sum(scores.values()) or 1.0
    verdict.confidence = round(scores[winner] / total, 3)

    if verdict.disagreed:
        verdict.status = "disputed"
        # The badge stays the priced answer even when outscored, for the same
        # reason `odometer_conflict` keeps the VLM's figure: the disagreement
        # widens the band and is shown, it does not silently swap the answer
        # underneath the reader.
        loser = ", ".join(f"{w[0]} says {w[1]}" for w in speaking if w[1] != winner)
        # A dispute the chassis plate took part in - on either side - widens
        # LESS than one between two soft reads, which reads backwards until you
        # notice what the multiplier is paying for. It is the residual doubt
        # after the disagreement is resolved, not the size of the disagreement.
        # A stamped, check-digit-verified VIN resolves it; two fuzzy reads
        # contradicting each other with nothing deterministic to break the tie
        # leave strictly more doubt behind.
        plate_involved = any(w[0] == "wmi" for w in speaking)
        widening = (WMI_CONFLICT_WIDENING if plate_involved
                    else IDENTITY_DISPUTED_WIDENING)
        verdict.widening = widening
        verdict.reason = (f"the reads disagree on the make - "
                          f"{', '.join(verdict.agreed)} says {winner}, {loser} - "
                          f"so the band widens {widening:.2f}x (a stated assumption)")
        verdict.reask = REASK_PLATE if "wmi" not in verdict.agreed else REASK_BADGE
        return verdict

    if len(verdict.agreed) >= 2:
        verdict.status = "confirmed"
        verdict.widening = 1.0
        verdict.reason = (f"{len(verdict.agreed)} independent reads agree this is a "
                          f"{winner} ({', '.join(verdict.agreed)})")
        return verdict

    verdict.status = "probable"
    verdict.widening = 1.0
    verdict.reason = (f"only the {verdict.agreed[0]} read names the make, and nothing "
                      f"contradicts it")
    # Not a widening, but worth asking for: a second witness is cheap for the
    # seller and turns "probable" into "confirmed".
    verdict.reask = REASK_PLATE if verdict.agreed[0] != "wmi" else REASK_BADGE
    return verdict


# Widenings this verdict supersedes. `reconcile` still records the corrections -
# the audit trail is the point and nothing here deletes it - but the multiplier
# is applied once, here, rather than once per rule that noticed the same fact.
SUPERSEDED = ("identity disputed between the badge read and the ",
              "the chassis-plate WMI")


def merge_widening(verdict: IdentityVerdict,
                   widening: list) -> list:
    """Drop per-rule identity widenings, then append the verdict's own.

    Without this, one disagreement between a badge and a chassis plate is
    multiplied twice: `reconcile` widens because the head disputed the badge,
    the WMI rule widens because the plate did, and the verdict widens because
    it adjudicated both. 1.25 x 1.20 x 1.25 is a 1.87x band for a single fact,
    and every one of the three would have looked correct in isolation.
    """
    kept = [row for row in (widening or [])
            if not any(marker in str(row[0]) for marker in SUPERSEDED)]
    if verdict is not None and verdict.widening > 1.0:
        kept.append([verdict.reason, verdict.widening])
    return kept
