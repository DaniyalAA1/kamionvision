"""Reading one photograph k times, and combining the reads.

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
"""
from __future__ import annotations

import time
from pathlib import Path

from .. import vlm
from ..config import CLOSEUP_QUORUM, CLOSEUP_SAMPLES
from ..schema import Correction, Issue, PhotoFinding
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
                    cropped=cropped, repair=repair))
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
