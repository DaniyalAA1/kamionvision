"""The card that travels with the artifact.

The pre-registration below is a CONSTANT in source, committed before the panel
was run, so the git history is the proof that the targets were written before
any result existed. The card renders it verbatim next to what actually
happened. Nothing here decides anything on its own - the decision is written
into the card by hand under "Verdict", because a threshold that auto-renders a
verdict is a threshold that stops being read.
"""
from __future__ import annotations

import json
from datetime import date

from app import config
from app.evidence import prompts as P
from panel import agreement as ag
from panel.protocol import (COMPONENT_STATES, RUBRIC_SHA, SEVERITY_SCALE,
                            panelist_prompt, rubric_text)

# --- written before the pilot ran ------------------------------------------
PREREGISTERED = """\
Written and committed before the first panelist was launched. The commit that
introduced `panel/card.py` predates the commit that introduced
`data/reference/condition_panel_v1.jsonl`; `git log --follow` on the two files
is the proof, and nothing below was edited after a number came back.

| # | Target | Threshold | Why this number |
|---|---|---|---|
| 1 | Ordinal Krippendorff α over per-component severity, `not_visible` treated as missing | **≥ 0.50** | Below this a silver standard cannot carry a severity MAE quoted to two decimals. Krippendorff's own guidance is 0.667 for tentative conclusions and 0.80 for firm ones; 0.50 is the floor at which the labels are worth keeping at all, not the bar for trusting them. |
| 2 | Exact grade agreement — all three panelists identical | **≥ 0.50 of vehicles** | Four ordered grades, three raters: chance unanimity is ~6%. Half is a low bar and deliberately so; the grade is the coarsest thing the panel produces and if it cannot be reproduced, nothing finer can. |
| 3 | Pairwise grade agreement | **≥ 0.67 of rater pairs** | Two of three agreeing on every vehicle. |
| 4 | Consensus grade distribution: share `good` or `excellent` | **≥ 0.60** | Both Turkish sources are OEM dealers selling RECONDITIONED stock photographed on prepped lots (`data/DATASET_CARD.md`, known limitation 2). A well-calibrated reader grades most of this corpus well. A pilot that comes back mostly `fair`/`poor` is either a rough sample or a rubric that is still inflated, and the card must say which it thinks it is. |
| 5 | `none` is the single most common component state, and ≥ 0.35 of all readings | **both** | Rule 1 of the prompt says so in as many words. If severities outnumber `none` among components the panel could see, the instrument is inflated and the reference would inherit it. |
| 6 | Schema-valid responses | **≥ 15 of 18 runs** | All 33 components present, every cited photo index in range, grade in the enum. Below this the protocol is the problem, not the rubric. |
| 7 | Vehicles triggering adjudication (all three grades differ) | **≤ 1 of 6** | The trigger exists to be rare. If it fires often the grade vocabulary is underspecified. |

Decision rule, also pre-registered:

* **Scale to 60** only if 6 holds AND 1 holds AND at least one of 2/3 holds.
* **Revise the protocol first** if 6 holds but 1 fails - the words are the
  problem, and re-running 180 heavy agent runs under words that do not
  reproduce is the expensive mistake this pilot exists to prevent.
* **Abandon the panel** if 6 fails, or if α is near zero with no identifiable
  cause in the answers themselves.
* Whatever the outcome, if α < 0.50 the card says so where the numbers are
  quoted, and no downstream metric derived from this file is quoted to two
  decimals."""

HONESTY = """\
## What this file is, and what it is not

This is a **silver standard, not ground truth**. Nobody has inspected these
trucks. `damaged` is null in 100% of 7,458 corpus rows, there is no Turkish
hasar / tramer / boyalı / değişen / ekspertiz field anywhere in the metadata,
and `data/DATASET_CARD.md` records the absence as known limitation 3. The
labels here are the readings of a panel of Claude Opus 5 agents, and they are
the best available reference precisely because no better one exists.

**The panel is better informed than the system it is used to measure, by
construction.** The panel sees every photograph of the vehicle at once, plus
the registration year and the odometer reading. The production close-up call
sees ONE photograph and is never told the distance. So the panel is not a
ceiling the pipeline could reach by being smarter at the same task - it is a
different, easier task. Part of any gap between panel and pipeline is an
information gap, not a capability gap, and no number derived from this file
should be presented as "how much worse the pipeline is at reading a photo".

**The panel and the pipeline share a rubric and a model family.** Both grade
under the same `SEVERITY_RUBRIC`, and both are Claude. A shared blind spot
would be invisible to this instrument: if the rubric is wrong about what
"moderate" means, panel and pipeline are wrong together and agreement goes up
rather than down. This is the load-bearing weakness of the whole design and it
is not fixable from inside the panel. What it CAN detect is inflation relative
to its own rubric, read with full information - which is exactly the failure it
was built for.

**Prices are asking prices and the panel never sees one.** No panelist is told
what the truck is listed for, and rule 6 of the prompt forbids naming a figure.
"""


def _fmt(alpha: ag.Alpha) -> str:
    a = "undefined" if alpha.alpha is None else f"{alpha.alpha:.3f}"
    note = f" ({alpha.note})" if alpha.note else ""
    return (f"{a} over {alpha.n_units} units / {alpha.n_observations} readings"
            f"{note}")


def render(*, rows: list[dict], stats: dict, run_id: str, model_id: str,
           example_prompt: str, verdict: str, exclusions: str,
           graded_at: str | None = None) -> str:
    graded_at = graded_at or date.today().isoformat()
    grade_rows = "\n".join(
        f"| {row['listing_id']} | {row['market']} | {row['stratum']} | "
        f"{row['year']} {row['make']} {row['model']} | {row['km']:,} | "
        f"{row['wear_band']} | {', '.join(row['agreement']['grades'])} | "
        f"**{row['consensus']['grade']}** |"
        for row in rows)

    return f"""\
# Condition panel v1 — the reference this repo measures condition against

| | |
|---|---|
| Artifact | `data/reference/condition_panel_v1.jsonl` — one row per vehicle, **all** panelist answers kept |
| Rows | {len(rows)} vehicles × {stats['n_panelists']} panelists = {stats['n_runs']} independent runs |
| Model | `{model_id}` |
| Graded | {graded_at} |
| Run | `{run_id}` |
| Rubric | `sha256:{RUBRIC_SHA[:16]}…` — `SEVERITY_RUBRIC` + `COMPONENT_ANCHORS` + `EXPECTED_WEAR` + `WORKED_EXAMPLES` from `app/evidence/prompts.py` |
| Status | **PILOT** — not the target-60 reference set. Built to find out whether the protocol works before 180 heavy agent runs are spent on it. |

{HONESTY}

## Pre-registered targets

{PREREGISTERED}

## What the pilot measured

| Target | Pre-registered | Measured | Met |
|---|---|---|---|
{stats['scorecard']}

### Agreement, in full

* Ordinal α, per-component severity, `not_visible` = missing: **{_fmt(stats['severity_alpha'])}**
* Nominal α, full six-word vocabulary including `not_visible`: {_fmt(stats['state_alpha'])}
* Nominal α, visible / not-visible only: {_fmt(stats['visibility_alpha'])}
* Ordinal α over the grade: {_fmt(stats['grade'].alpha)}
* Raw component-state agreement (share of rater pairs writing the same word): {stats['raw_state']}
* Exact grade agreement (all three): {stats['grade'].unanimous:.3f} · pairwise: {stats['grade'].pairwise:.3f}
* Vehicles triggering adjudication (all three grades differ): {len(stats['grade'].needs_adjudication)} — {stats['grade'].needs_adjudication or 'none'}

**Read α next to the raw agreement, not instead of it.** α corrects for chance,
and when nearly every answer is the same word the expected disagreement
collapses toward zero and α goes unstable even at high raw agreement. That is
the α paradox, and with `none` as the designed-for common answer this instrument
sits in exactly that regime. Where the two disagree, both are reported.

### The readings themselves

State distribution across all {stats['n_readings']} component readings:

{stats['state_table']}

Grade per vehicle:

| Vehicle | Market | Stratum | Truck | km | Band | Panel grades | Consensus |
|---|---|---|---|---|---|---|---|
{grade_rows}

Consensus grade distribution: {stats['grade_distribution']}

## Method

* **Selection** — `panel/select.py`, deterministic given seed `{stats['seed']}`.
  Reads `data/metadata/listings.csv` and `data/metadata/images.csv` only; never
  a photograph, so nothing about how a truck looks can influence whether it was
  chosen. The pilot's six are picked by stated rule, not by eye:
{stats['pilot_rules']}
* **Independence** — three separate agent invocations per vehicle, no shared
  context, no sight of each other's answers, all launched in one message so
  none could be influenced by an earlier one's result. A panel whose members
  can see each other is one rater with extra steps.
* **What each panelist got** — every original frame of the one vehicle, the
  registration year, the odometer reading, and the prompt built by
  `panel.protocol.panelist_prompt`. Degraded twins were not used.
* **Consensus** — per component, the ordinal MEDIAN of the visible readings, so
  consensus never invents a level nobody wrote; `not_visible` only when most of
  the panel could not see it. Grade: majority, else the ordinal median — never
  the worst of the three.
* **Severity scale** — `{' < '.join(SEVERITY_SCALE)}`. `not_visible` is off that
  scale and is treated as missing data, not as a fifth low level: a component
  nobody could see is an absent reading, and folding it into `none` would
  manufacture agreement out of two readers both failing to look.
* **`config.REPAIR_BANDS` is not in the prompt.** The rubric is anchored on
  repair effort rather than money so that teaching a model what a severity
  means never puts a currency figure in front of it. The lira reading of the
  same four words, for this card only and stamped
  `{config.REPAIR_BANDS_AS_OF}`: {json.dumps({k: v for k, v in config.REPAIR_BANDS.items()})} TRY.
  A test in `tests/test_panel.py` asserts none of it reaches the prompt.

## Exclusions

{exclusions}

## Verdict

{verdict}

## The panelist prompt, verbatim

This is the exact instrument, as handed to the panelists for
`{stats['example_vehicle']}`. Only the vehicle block and the expected-wear row
change between vehicles; every other word is identical for every run.

```text
{example_prompt}
```

## Re-scoring without re-running

Every panelist's full answer is in the artifact, so a rubric revision is a
re-score rather than a re-run:

```bash
.venv/bin/python -m panel.cli report      # recompute α and the card from the jsonl
```

A row whose `rubric_sha` no longer matches the current constants is not wrong —
it was graded under different words. `panel.store.check_rubric` reports the
mismatch rather than letting it pass.
"""
