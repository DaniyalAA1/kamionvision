"""Reading one photograph k times - and one photo set k times - and combining.

One read of one photo was a single draw from a distribution nobody had
measured. The severity a finding came back with depended on that draw, and the
draw was multiplied straight into a price by `condition_adjustment`. Worse,
`Issue.confidence` - the number that multiplier is weighted by - was the
model's opinion of its own claim, which `reconcile`'s docstring has complained
about since it was written.

Reading each photo `config.CLOSEUP_SAMPLES` times fixes both at once. The
severity becomes an order statistic over the samples rather than a draw, and
`Issue.confidence` becomes the **agreement rate across samples** - a measured
quantity about the claim instead of a self-report about it. The model's own
number is kept on `Issue.self_confidence`, which makes the obvious calibration
study possible: does what it says about its confidence predict what it agrees
with itself about?

How the samples combine:

  k of k   severity = the median of the reported ordinals
  quorum   severity = the LOWER of the reported ordinals, and a
                      `sample_disagreement` correction
  below    kept, `corroborated=False`, capped below "major", and an
           `uncorroborated_finding` correction

"Lower of the two" is an assumption on day one, not a measurement: the measured
direction of error is upward, so the conservative order statistic is the right
estimator until a retest says otherwise.

All k samples of one photo go to the SAME backend. Mixing backends would
confound sample-to-sample disagreement with a difference between two models and
make the whole measurement meaningless - so a fallback only happens when a
sample actually fails, and it is recorded on the photo it happened to.

The same argument applies with more force to pass A, which is why
`identity_consensus` is here too. One read of the identity call was one draw
from an unmeasured distribution, and more rides on that draw than on any single
close-up: `make` picks the brand column in the price model, `model` picks the
anchor row, `body_type` can stop the pricing stage through
`pipeline.pricing_blocker`, and `same_vehicle` can stop the valuation outright.

How the identity samples combine:

  make, model, body_type, axle_config, cab_type, generation
           majority vote over the samples that named something. An abstention
           lowers the agreement rate; it never wins the vote, because deleting
           the only read that named the truck would cost the brand column
           entirely, and a low agreement already reaches `app/identity.py` as
           a weak witness and widens the band.
  same_vehicle
           NOT symmetric, and the asymmetry is the whole rule. False stops the
           valuation, and the prompt already says to set it true when not
           certain - so it takes `IDENTITY_QUORUM` samples to go false, and a
           split is recorded as a `Correction` either way.
  identity_agreement
           the measured share of surviving samples that named the winning
           make. `VehicleRead.confidence` stays the model's self-report, which
           is what makes the pair worth having: `identity.collect` prefers the
           measurement and falls back to the self-report.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import modelspec, vlm
from ..config import (CLOSEUP_QUORUM, CLOSEUP_SAMPLES, IDENTITY_QUORUM,
                      IDENTITY_SAMPLES)
from ..schema import Correction, Issue, PhotoFinding, VehicleRead
from . import passes, prompts

SEVERITY_RANK = {name: i for i, name in enumerate(prompts.SEVERITIES)}
IMPACT_RANK = {name: i for i, name in enumerate(prompts.IMPACTS)}


def _median_rank(ranks: list[int]) -> int:
    """The middle value, taking the LOWER of the two middles on an even count."""
    ordered = sorted(ranks)
    return ordered[(len(ordered) - 1) // 2]


def _dedupe(lines: list[str], limit: int) -> list[str]:
    out: list[str] = []
    for line in lines:
        text = (line or "").strip()
        if not text or any(passes._overlap(text, kept) >= passes.SAME_DEFECT
                           for kept in out):
            continue
        out.append(text)
    return out[:limit]


def _cluster(samples: list[PhotoFinding]) -> list[list[tuple[int, Issue]]]:
    """One cluster per distinct defect, carrying which sample each sighting came from.

    Reuses `passes._overlap` and `passes.SAME_DEFECT` rather than inventing a
    second matcher: two observations on the same component above that overlap
    are the same defect, whether they came from two photos or two reads of one.
    """
    clusters: list[list[tuple[int, Issue]]] = []
    for index, sample in enumerate(samples):
        for issue in sample.issues:
            for cluster in clusters:
                if cluster[0][1].component != issue.component:
                    continue
                if any(i == index for i, _ in cluster):
                    continue        # one sighting per sample per defect
                if passes._overlap(cluster[0][1].observation, issue.observation) \
                        >= passes.SAME_DEFECT:
                    cluster.append((index, issue))
                    break
            else:
                clusters.append([(index, issue)])
    return clusters


def combine_samples(samples: list[PhotoFinding], *, quorum: int | None = None
                    ) -> tuple[PhotoFinding, list[Correction]]:
    """k reads of one photograph into the one PhotoFinding the rest of the run sees."""
    base = samples[0]
    k = len(samples)
    quorum = CLOSEUP_QUORUM if quorum is None else quorum
    # With one surviving sample there is nothing to corroborate against, so the
    # quorum falls back to one rather than marking every finding uncorroborated
    # - that would be reporting a provider failure as evidence about the truck.
    quorum = max(1, min(quorum, k))

    merged = PhotoFinding(
        photo_id=base.photo_id, view=base.view, shows=base.shows,
        legible=sum(1 for s in samples if s.legible) * 2 >= k,
        cropped=base.cropped,
        # Keep one sample's geometry; averaging rectangles invents locations.
        component_regions=base.component_regions,
        strengths=_dedupe([g for s in samples for g in s.strengths], 8),
        cannot_tell=_dedupe([g for s in samples for g in s.cannot_tell], 6),
        confidence=round(sum(s.confidence for s in samples) / k, 3),
        backend=base.backend, samples=k,
        sample_errors=[e for s in samples for e in s.sample_errors],
        elapsed_s=round(sum(s.elapsed_s for s in samples), 2))

    # The odometer is digits, so the samples vote on it. A read two of three
    # agree on is worth more than one read, and `reconcile._check_odometer`
    # is about to test that number against what the seller typed.
    reads = [s.odometer_km for s in samples if s.odometer_km]
    if reads:
        best = max(set(reads), key=lambda km: (reads.count(km), -km))
        merged.odometer_km = best
        if len(set(reads)) > 1:
            merged.sample_errors.append(
                "the reads of this odometer disagreed: "
                + ", ".join(f"{km:,}" for km in sorted(set(reads)))
                + f"; kept {best:,}")

    corrections: list[Correction] = []
    for cluster in _cluster(samples):
        votes = [issue.severity for _, issue in cluster]
        ranks = [SEVERITY_RANK.get(v, 0) for v in votes]
        agreed = len(cluster)
        if agreed >= k:
            rank = _median_rank(ranks)
        else:
            rank = min(ranks)
        corroborated = agreed >= quorum
        severity = prompts.SEVERITIES[rank]

        # Pick the sighting that actually argued for the level we landed on, so
        # the prose and the ordinal are the same call's answer.
        chosen = next((issue for _, issue in cluster if issue.severity == severity),
                      cluster[0][1])
        impacts = [IMPACT_RANK.get(i.price_impact, 0) for _, i in cluster]
        impact_rank = _median_rank(impacts) if agreed >= k else min(impacts)

        issue = Issue(
            photo_id=base.photo_id, component=chosen.component,
            observation=chosen.observation, severity=severity,
            confidence=round(agreed / k, 3),
            price_impact=prompts.IMPACTS[impact_rank],
            magnitude=chosen.magnitude or next(
                (i.magnitude for _, i in cluster if i.magnitude), None),
            box=chosen.box or next((i.box for _, i in cluster if i.box), None),
            severity_votes=votes,
            self_confidence=round(sum(i.confidence for _, i in cluster) / agreed, 3),
            corroborated=corroborated,
            severity_provisional=severity)

        if not corroborated:
            if issue.severity == prompts.SEVERITIES[-1]:
                issue.severity = prompts.SEVERITIES[-2]
            corrections.append(Correction(
                kind="uncorroborated_finding", photo_id=base.photo_id,
                before=severity, after=issue.severity,
                detail=f"{issue.component}: {agreed} of {k} reads of this photo "
                       f"reported it; kept and cited, and it cannot be raised to "
                       f"\"{prompts.SEVERITIES[-1]}\" on one read"))
            issue.severity_provisional = issue.severity
        elif len(set(votes)) > 1:
            corrections.append(Correction(
                kind="sample_disagreement", photo_id=base.photo_id,
                before="/".join(votes), after=issue.severity,
                detail=f"{issue.component}: {k} reads of this photo did not agree "
                       f"on how bad it is; the lower reading stands"))
        merged.issues.append(issue)
    return merged, corrections


def closeup_consensus(chain, check, vehicle_line: str, *, tmpdir: Path,
                      max_tokens: int, expectation: str = "",
                      band: str | None = None, samples: int | None = None,
                      weak_points: list | tuple = (),
                      repair=None) -> tuple[PhotoFinding, list[Correction], list[str]]:
    """`samples` reads of one photo, in sequence, combined into one finding.

    Sequential rather than parallel per photo: the crop is cut once and the
    pool is already saturated by the other photos, so the wall clock is the
    same three waves either way and nothing races to write the same crop file.
    """
    t0 = time.time()
    samples = CLOSEUP_SAMPLES if samples is None else samples
    image, cropped = passes.closeup_image(check, tmpdir)
    good: list[PhotoFinding] = []
    errors: list[str] = []
    fallbacks: list[str] = []

    for _ in range(max(1, samples)):
        clients = list(chain)
        while clients:
            client = clients.pop(0)
            try:
                good.append(passes.closeup(
                    client, check, vehicle_line, max_tokens=max_tokens,
                    expectation=expectation, band=band, image=image,
                    cropped=cropped, weak_points=weak_points, repair=repair))
                break
            except Exception as exc:
                errors.append(f"{client.name}: {type(exc).__name__}: {exc}")
                if clients:
                    fallbacks.append(
                        f"photo {check.photo_id}: {client.name} failed "
                        f"({type(exc).__name__}), retried on {clients[0].name}")

    if not good:
        raise vlm.VLMError("; ".join(errors) or "no samples were taken")

    merged, corrections = combine_samples(good)
    merged.sample_errors.extend(errors)
    merged.elapsed_s = round(time.time() - t0, 2)
    return merged, corrections, fallbacks


# --- pass A, k times -------------------------------------------------------

@dataclass
class IdentityRead:
    """The combined identity read, and everything the stage has to record."""
    vehicle: VehicleRead
    same_vehicle: bool = True
    vehicle_mismatch: str = ""
    backend: str = ""
    model: str = ""
    # The backend that answered, handed on so passes B, C and D prefer it.
    client: Any = None
    corrections: list[Correction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)
    calls: list[list] = field(default_factory=list)


def _plain_key(value) -> str:
    return str(value or "").strip().upper()


def _brand_key(value) -> str:
    """Fold a make for comparison. "Ford" and "Ford Trucks" are one vote."""
    from ..pricing.features import normalise_brand

    return normalise_brand(value) if value else ""


def _vote(values: list, key=_plain_key) -> tuple[Any, int]:
    """The best-supported answer that names something, and how many named it.

    An abstention is not a vote. Two samples that could not read a make and one
    that could leaves the answer standing on one read - which is the honest
    outcome, because the agreement rate travels with it and
    `app/identity.py` treats a thin agreement as a weak witness and widens the
    band. Dropping the only read that named the truck would cost the brand
    column in the price model outright.

    Ties go to the value that appeared first, which is the first sample's - the
    answer the unsampled pass would have returned.
    """
    groups: dict[str, list] = {}
    for value in values:
        folded = key(value)
        if not folded:
            continue
        groups.setdefault(folded, []).append(value)
    if not groups:
        return None, 0
    winner = max(groups, key=lambda k: len(groups[k]))
    return groups[winner][0], len(groups[winner])


def combine_identity(reads: list[tuple], *, quorum: int | None = None
                     ) -> tuple[VehicleRead, bool, str, list[Correction]]:
    """k identity reads into the one VehicleRead the rest of the run sees."""
    vehicles = [r[0] for r in reads]
    sames = [bool(r[1]) for r in reads]
    mismatches = [str(r[2] or "").strip() for r in reads]
    k = len(reads)
    # With one surviving sample there is nothing to corroborate against, so the
    # quorum falls back to one rather than making a provider failure into
    # evidence about the truck. Same rule as `combine_samples`.
    quorum = max(1, min(IDENTITY_QUORUM if quorum is None else quorum, k))
    corrections: list[Correction] = []

    make, make_votes = _vote([v.make for v in vehicles], key=_brand_key)
    # The model is voted on its CANONICAL form, so "F Max 500" and "F-MAX" are
    # one vote for one truck rather than two answers that cancel each other.
    model, _ = _vote([v.model for v in vehicles],
                     key=lambda m: _plain_key(modelspec.normalise_model(make, m)))
    # The sample that argued for the make we landed on, so the free-text fields
    # that cannot be voted come from a read that agreed with the answer.
    chosen = next((v for v in vehicles if _brand_key(v.make) == _brand_key(make)),
                  vehicles[0])

    merged = VehicleRead(
        make=make,
        model=model,
        model_canonical=modelspec.normalise_model(make, model),
        body_type=_vote([v.body_type for v in vehicles])[0],
        cab_type=_vote([v.cab_type for v in vehicles])[0],
        axle_config=_vote([v.axle_config for v in vehicles])[0],
        approx_year_range=_vote([v.approx_year_range for v in vehicles])[0],
        generation=_vote([v.generation for v in vehicles])[0],
        generation_conf=chosen.generation_conf,
        year_evidence=chosen.year_evidence,
        # Averaged, and still a self-report: this is the model's opinion of its
        # own answer and it stays labelled as such. `identity_agreement` below
        # is the measured quantity and the two are deliberately not merged.
        confidence=round(sum(v.confidence for v in vehicles) / k, 3),
        identity_samples=k,
        identity_agreement=round(make_votes / k, 3),
    )

    # Badges are a witness list rather than a single answer, so they union
    # across the samples, ordered by how many reads saw each one.
    badges: dict[str, list[str]] = {}
    for vehicle in vehicles:
        for badge in vehicle.badges_seen:
            text = str(badge).strip()
            if text:
                badges.setdefault(text.upper(), []).append(text)
    merged.badges_seen = [group[0] for group in
                          sorted(badges.values(), key=lambda g: -len(g))][:8]

    # `same_vehicle` is the one field that is not symmetric. Setting it false
    # stops the valuation entirely, and the prompt already says to set it true
    # where the reader is not certain - so a single dissenting sample must not
    # be able to halt an appraisal, and it takes a quorum to get there. The
    # error this guards against is the expensive one: photos of one truck
    # routinely differ in light, mud, blur and angle, and one read in three
    # calling that a second vehicle would refuse a genuine listing.
    dissent = sum(1 for s in sames if not s)
    same = dissent < quorum
    if 0 < dissent < k:
        outcome = (f"it takes {quorum} of them to stop a valuation, so the set is "
                   f"priced and the doubt is printed beside it" if same else
                   "that is the quorum, so nothing is priced")
        corrections.append(Correction(
            kind="same_vehicle_disagreement",
            before="/".join("different" if not s else "same" for s in sames),
            after="one vehicle" if same else "more than one vehicle",
            detail=(f"{dissent} of {k} reads of this photo set called it more than "
                    f"one vehicle; {outcome}")))
    # Prefer a doubt written by a read that reached the same conclusion; keep
    # any doubt at all rather than none - the prompt asks for the wording even
    # when `same_vehicle` stays true.
    mismatch = next((m for s, m in zip(sames, mismatches) if m and s == same),
                    "") or next((m for m in mismatches if m), "")

    # A split on what the truck IS, recorded rather than averaged away. The
    # majority answer stands and `identity_agreement` carries how thin it was,
    # which is what `app/identity.py` widens the band on.
    makes = {_brand_key(v.make) for v in vehicles if v.make}
    models = {_plain_key(modelspec.normalise_model(v.make, v.model))
              for v in vehicles if v.model}
    if len(makes) > 1 or len(models) > 1:
        corrections.append(Correction(
            kind="identity_disagreement",
            before=" / ".join(sorted(makes | models)),
            after=" ".join(x for x in (make, model) if x) or "nothing",
            detail=(f"the {k} reads of this set did not name the same vehicle; "
                    f"{make_votes} of {k} named {make or 'nothing'}. The answer is "
                    f"the majority one and the agreement rate travels with it "
                    f"rather than the disagreement being dropped")))
    return merged, same, mismatch, corrections


def identity_consensus(chain, selected: list, declared: dict | None, *,
                       max_tokens: int, samples: int | None = None,
                       quorum: int | None = None, repair=None) -> IdentityRead:
    """`samples` reads of the identity call, on one backend, combined into one.

    Sequential, like `closeup_consensus`: the samples are three waves of one
    call rather than three calls racing, and pass B - sixteen photos times
    `CLOSEUP_SAMPLES` - is what the concurrency budget is for.

    Every sample goes to the backend that answered the first one. A fallback
    happens only when a sample actually fails, and it is recorded. A failed
    sample costs one sample; every sample failing raises, because there is
    nothing to describe and nothing to price.
    """
    samples = IDENTITY_SAMPLES if samples is None else samples
    photo_ids = [c.photo_id for c in selected]
    # The schema a repair is checked against has to be the schema the call was
    # asked with. Repairing an unparseable answer against a wider one lets the
    # model "fix" itself into a model name the narrowed call excluded.
    schema = passes.identity_schema(passes.declared_identity(declared)[0])
    good: list[tuple] = []
    errors: list[str] = []
    fallbacks: list[str] = []
    calls: list[list] = []
    pinned = None
    backend = model_id = ""

    for index in range(max(1, samples)):
        order = ([pinned] + [c for c in chain if c is not pinned]) if pinned else list(chain)
        for position, client in enumerate(order):
            try:
                response = passes.identity(client, selected, declared,
                                           max_tokens=max_tokens)
                try:
                    read = passes.parse_identity(response.text, photo_ids)
                except (ValueError, json.JSONDecodeError) as exc:
                    if repair is None:
                        raise
                    response = repair(client, response, exc, schema, max_tokens)
                    read = passes.parse_identity(response.text, photo_ids)
                    errors.append(f"identity sample {index + 1} was unparseable "
                                  f"({exc}); repaired")
            except Exception as exc:
                errors.append(f"identity sample {index + 1} on {client.name}: "
                              f"{type(exc).__name__}: {exc}")
                if position + 1 < len(order):
                    fallbacks.append(
                        f"identity sample {index + 1}: {client.name} failed "
                        f"({type(exc).__name__}), retried on {order[position + 1].name}")
                continue
            good.append(read)
            calls.append([f"identity {index + 1}/{samples}", response.elapsed_s])
            if pinned is None:
                pinned, backend, model_id = client, response.backend, response.model
            break

    if not good:
        raise vlm.VLMError("every identity call failed:\n  " + "\n  ".join(errors))

    vehicle, same, mismatch, corrections = combine_identity(good, quorum=quorum)
    return IdentityRead(vehicle=vehicle, same_vehicle=same, vehicle_mismatch=mismatch,
                        backend=backend, model=model_id, client=pinned,
                        corrections=corrections, warnings=errors,
                        fallbacks=fallbacks, calls=calls)
