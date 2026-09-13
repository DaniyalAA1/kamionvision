# Status — read this first

**This is the one file to read for "what's happening right now."** `CLAUDE.md` explains how the
system works and the invariants it encodes; this explains where it stands. Updated at every real
state change — a merge, a refit, a decision — **never on a timer**.

**Last updated:** 2026-09-12 — docs system bootstrapped; served-blend pricing measured and pushed
on a branch; improvement brainstorm opened (see `docs/BRAINSTORM.md`).

## If you are picking this up cold

- Hackathon entry for the Kamion challenge (`kamion-truck-appraisal-brief.md`): photos of a used
  truck in, a price range + condition report + honest refusals out. Judged live on unseen photos.
- The pipeline is built end to end: gate → evidence (one vision call per photo) → reconcile →
  price → report, as a CLI and a streamed web screen (`app.cli serve`, port 8000).
- Offline suite: **163 tests pass** on `worktree-pricing-served-blend` (158 on `main`).
- Pricing, measured on 84 Turkish listings / 24 spec groups: served estimate **R² 0.938, median
  error 3.1%, 80% band covers 80.9%** (branch) vs hedonic-only 0.842 / 4.2% / 80.3% (`main`).
- **Blocker for a live appraisal on this machine: no `.env`** — no vision-model key is set, so the
  gallery browses but a new appraisal cannot run. `app.cli doctor` reports it.
- New-price reference rows are **219 days old** (`doctor` warns). With the blend now weighting the
  anchor 1.00 for Ford/MAN, they carry the price — refresh them.

## ⚠️ Where the work actually lives

| Branch | Carries | State |
| --- | --- | --- |
| `main` | the full appraisal system | shipped |
| `worktree-pricing-served-blend` | measured served blend + its own band; drop_brand bug fix (decision 0001) | pushed, unmerged, needs teammate review |
| `worktree-devkit-graphify` | this docs system, graphify graph, brainstorm (stacked on the pricing branch) | pushed, unmerged |

## Right now

- Nothing long-running. The demo server may be running from the `main` checkout.
- Data: 200 vehicles, 3,729 photos + degraded twins; images are gitignored (`data/images/`).
- Price model `tr_only`; gate false-refusal 1 of 200; perception heads fitted 2026-09-12.

## What to do next, in order

1. Put a vision key in `.env` and run `app.cli doctor` until it is green — no live demo without it.
2. Teammate review + merge of `worktree-pricing-served-blend`; re-verify the new-price rows.
3. Pick from `docs/BRAINSTORM.md` — ranked against the four judging criteria.
4. Run `app.demo` (spends ~18 vision calls per case) before any commit that matters.

## Do NOT

- Pool EU/US listings into the price fit — measured worse (R² 0.70 / 0.65 vs 0.84). See `CLAUDE.md`.
- Fit an image→price head — `scripts/probe_residual_signal.py` found no signal above a permuted null.
- Scrape sahibinden, arabam, TruckPaper, Machinery Trader, Copart — ToS/hard blocks, ruled out.
- Run devkit's `init.sh` inside a git worktree: `.git` is a file there, so its `[[ ! -d .git ]]`
  check runs `git init` and nests a repo. Copy templates by hand (see PLAYBOOK).
