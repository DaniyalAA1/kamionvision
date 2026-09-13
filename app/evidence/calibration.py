"""Pass D - the severity the whole set decides, after the duplicates are folded.

Every severity that reaches this module was assigned by a call that saw one
photograph. A severity is a comparison ("worse than this truck should be"), and
a single frame has nothing to compare against: the close-up call was being
asked for a relative ordinal from an absolute-only observation, sixteen times,
and the answers were summed into a price. That is most of why a scuffed fuel
tank strap could grade a whole truck "fair".

So the close-up now reports two things - an ordinal it is told is provisional,
and a `Magnitude` that is answerable from one frame - and this pass, which sees
the finished list, decides the ordinal. It runs AFTER `merge_duplicates`
deliberately: `notes_block` numbers the pre-merge flat list, so a revision
riding in the synthesis response would revise a list in which three paraphrases
of one worn drive tire are still three rows.

The licence is asymmetric, because the measured direction of error is upward:

  lower       freely, as far as the evidence supports
  raise       by at most one level
  raise to    "major" only when the defect was seen in a second photograph
  create      never. Pass D may not add, delete or re-bind a finding, and being
              text-only it structurally cannot invent a photograph to cite.

Every change is a `Correction` in `reconcile`'s own format - before, after,
reason - and nothing is ever deleted. Same posture, same dataclass, same
rendering path: changing your mind is allowed, doing it quietly is not.
"""
from __future__ import annotations

from ..config import CALIBRATION_EFFORT
from ..schema import Correction, Issue, Magnitude, PhotoFinding
from . import prompts

SEVERITY_RANK = {name: i for i, name in enumerate(prompts.SEVERITIES)}
IMPACT_RANK = {name: i for i, name in enumerate(prompts.IMPACTS)}
MAJOR = prompts.SEVERITIES[-1]


# --- the absolute half of a finding ---------------------------------------

def parse_magnitude(raw) -> Magnitude | None:
    """The four absolute fields, or nothing.

    A partly-filled magnitude is still worth keeping - an extent with an
    unreadable state tells pass D more than silence does - but a value outside
    its enum is dropped rather than rounded to a neighbour, because rounding an
    unreadable answer is how inflation gets into a parser.
    """
    if not isinstance(raw, dict):
        return None
    extent = raw.get("extent") if raw.get("extent") in prompts.EXTENTS else ""
    state = raw.get("state") if raw.get("state") in prompts.STATES else ""
    blocks = raw.get("blocks_use") if raw.get("blocks_use") in prompts.BLOCKS_USE else ""
    consumable = bool(raw.get("consumable"))
    if not (extent or state or blocks or consumable):
        return None
    return Magnitude(extent=extent, state=state, consumable=consumable,
                     blocks_use=blocks)


def issue_suffix(issue: Issue, counts: dict[str, int] | None = None,
                 samples: int | None = None) -> str:
    """The parenthetical after an observation: level, magnitude, corroboration.

    Read by `notes_block` for pass C and by `findings_block` for pass D. The
    corroboration denominator is only added for pass D, which is the only
    reader that has been told what to do with it.
    """
    parts = [f"{issue.severity}, {issue.price_impact} price impact"]
    mag = issue.magnitude
    if mag:
        shape = ", ".join(x for x in (mag.extent, mag.state) if x)
        if mag.consumable:
            shape = (shape + ", a consumable").lstrip(", ")
        if mag.blocks_use in ("yes", "maybe"):
            shape = (shape + f"; blocks use: {mag.blocks_use}").lstrip("; ")
        if shape:
            parts.append(shape)
    if issue.severity_votes:
        agreed = len(issue.severity_votes)
        parts.append(f"reported by {agreed} of {max(samples or agreed, agreed)} reads "
                     f"of this photo" + ("" if issue.corroborated else ", uncorroborated"))
    seen = 1 + len(issue.also_seen_in or [])
    if counts is not None:
        total = max(seen, counts.get(issue.component, seen))
        parts.append(f"seen in {seen} of {total} frames that show this component")
    elif seen > 1:
        parts.append(f"seen in {seen} photos")
    return "; ".join(parts)


def findings_block(issues: list[Issue], counts: dict[str, int] | None = None,
                   samples: dict[int, int] | None = None) -> str:
    if not issues:
        return "  (none)"
    return "\n".join(f"  [{i}] {issue.component}: {issue.observation} "
                     f"({issue_suffix(issue, counts, (samples or {}).get(issue.photo_id))})"
                     for i, issue in enumerate(issues))


def confirmed_sound(findings: list[PhotoFinding], limit: int = 14) -> list[str]:
    """Every `strengths` line the fan-out produced, near-duplicates folded.

    Sixteen frames confirming the same intact mudflap is one fact, the same way
    sixteen frames of one worn tire is one finding. The list it produces is the
    first consumer `strengths` has ever had beyond being printed.
    """
    from .passes import SAME_DEFECT, _overlap

    out: list[str] = []
    for finding in findings:
        if finding.error:
            continue
        for good in finding.strengths:
            text = (good or "").strip()
            if not text or any(_overlap(text, kept) >= SAME_DEFECT for kept in out):
                continue
            out.append(text)
    return out[:limit]


# --- the call --------------------------------------------------------------

def calibrate(client, issues: list[Issue], *, vehicle_line: str, expectation: str,
              n_photos: int, coverage: str, strengths: list[str],
              counts: dict[str, int] | None, samples: dict[int, int] | None,
              max_tokens: int,
              effort: str | None = CALIBRATION_EFFORT):
    """Text-only, images=[]. It cannot see a photograph, so it cannot invent one."""
    prompt = prompts.calibration_prompt(
        vehicle=vehicle_line, expectation=expectation, n_photos=n_photos,
        coverage=coverage,
        strengths="\n".join(f"  - {s}" for s in strengths),
        findings=findings_block(issues, counts, samples), last=len(issues) - 1)
    return client.complete(
        prompt, [], system=prompts.SYSTEM, max_tokens=max_tokens, effort=effort,
        json_schema=prompts.CALIBRATION_SCHEMA if client.supports_structured_output else None)


def parse_calibration(text: str) -> dict:
    from .passes import extract_json

    data = extract_json(text)
    revisions = []
    for entry in data.get("revisions") or []:
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("finding"))
        except (TypeError, ValueError):
            continue
        revisions.append({
            "finding": index,
            "severity": entry.get("severity"),
            "price_impact": entry.get("price_impact"),
            "reason": str(entry.get("reason") or "").strip()[:400],
        })
    try:
        worst = int(data.get("worst_finding"))
    except (TypeError, ValueError):
        worst = None
    return {"revisions": revisions, "worst_finding": worst,
            "calibration_note": str(data.get("calibration_note") or "").strip()}


# --- the asymmetric licence -----------------------------------------------

def _granted(issue: Issue, want: int, rank: dict[str, int], *, severity: bool) -> tuple[int, str]:
    """What the licence allows, and why it is less than what was asked for."""
    before = rank[issue.severity if severity else issue.price_impact]
    if want <= before:
        return want, ""                      # lowering is free
    ceiling = before + 1
    if want > ceiling:
        return ceiling, "a raise of more than one level is not allowed"
    if severity and want == SEVERITY_RANK[MAJOR]:
        if not issue.corroborated:
            return min(ceiling, SEVERITY_RANK[MAJOR] - 1), (
                "only one of the reads of this photo reported it, so it cannot be "
                "raised to major")
        if not issue.also_seen_in:
            return min(ceiling, SEVERITY_RANK[MAJOR] - 1), (
                "it was seen in one photograph only, and a raise to major needs "
                "corroboration from a second")
    return want, ""


def apply_revisions(issues: list[Issue], data: dict) -> tuple[list[Correction], list[str]]:
    """Re-decide the severities in place, and write down every change.

    Never creates, deletes or re-binds: the list that goes in is the list that
    comes out, same length, same `photo_id` on every row. A finding the model
    skipped keeps its provisional level and says so in a parse warning - the
    revision is what needs a licence, not the absence of one.
    """
    corrections: list[Correction] = []
    warnings: list[str] = []
    seen: set[int] = set()

    for entry in data.get("revisions") or []:
        index = entry.get("finding")
        if not isinstance(index, int) or not 0 <= index < len(issues) or index in seen:
            continue
        seen.add(index)
        issue = issues[index]
        reason = entry.get("reason") or ""
        before_severity, before_impact = issue.severity, issue.price_impact
        clamped = ""

        # Severity first: how far it is ALLOWED to move decides how far the
        # price impact may move with it.
        severity_steps = None
        if entry.get("severity") in SEVERITY_RANK:
            want = SEVERITY_RANK[entry["severity"]]
            allowed, why = _granted(issue, want, SEVERITY_RANK, severity=True)
            if why:
                clamped = why
            severity_steps = allowed - SEVERITY_RANK[before_severity]
            issue.severity = prompts.SEVERITIES[allowed]

        impact_clamp = ""
        if entry.get("price_impact") in IMPACT_RANK:
            want = IMPACT_RANK[entry["price_impact"]]
            allowed, why_i = _granted(issue, want, IMPACT_RANK, severity=False)
            before_rank = IMPACT_RANK[before_impact]
            # The price impact may not outrun the severity behind it. The two
            # are one claim - "this is worse than it looked" - and the price
            # multiplies them (`SEVERITY_WEIGHT[s] * IMPACT_WEIGHT[i]`). Without
            # this, a raise to `major` refused for want of corroboration came
            # straight back on the other axis: moderate/medium (1.00 * 1.00)
            # became moderate/high (1.00 * 2.00), doubling the weight of a
            # finding the licence had just refused to promote. A rule that binds
            # one axis and not the other is decorative.
            if severity_steps is not None and allowed > before_rank:
                ceiling = before_rank + max(0, severity_steps)
                if allowed > ceiling:
                    impact_clamp = (
                        f"and its price impact stayed at \"{before_impact}\": impact "
                        f"cannot be raised further than the severity behind it")
                    allowed = ceiling
            elif why_i:
                impact_clamp = why_i
            issue.price_impact = prompts.IMPACTS[allowed]

        if clamped or impact_clamp:
            asked = "/".join(x for x in (entry.get("severity"), entry.get("price_impact")) if x)
            corrections.append(Correction(
                kind="severity_raise_clamped", photo_id=issue.photo_id,
                before=f"{before_severity}/{before_impact}",
                after=f"{issue.severity}/{issue.price_impact}",
                detail=f"the calibration pass asked for \"{asked}\" on "
                       f"{issue.component}: {'; '.join(x for x in (clamped, impact_clamp) if x)}"))

        if not issue.severity_provisional:
            issue.severity_provisional = before_severity
        if reason:
            issue.severity_reason = reason
        if (issue.severity, issue.price_impact) != (before_severity, before_impact):
            corrections.append(Correction(
                kind="severity_calibrated", photo_id=issue.photo_id,
                before=f"{before_severity}/{before_impact}",
                after=f"{issue.severity}/{issue.price_impact}",
                detail=f"{issue.component}: {reason}" if reason else issue.component))

    missing = [i for i in range(len(issues)) if i not in seen]
    if missing:
        warnings.append(
            f"the calibration pass returned no decision for finding(s) "
            f"{', '.join(str(i) for i in missing)}; their provisional severities stand")
    return corrections, warnings


def worst_note(issues: list[Issue], worst: int | None) -> str:
    """One sentence naming the worst thing on the truck, if the pass named it."""
    if worst is None or not 0 <= worst < len(issues):
        return ""
    issue = issues[worst]
    text = issue.observation.strip()
    if len(text) > 140:
        text = text[:139].rsplit(" ", 1)[0] + "..."
    return (f"The worst thing on this truck is the {issue.component.replace('_', ' ')}: "
            f"{text}")
