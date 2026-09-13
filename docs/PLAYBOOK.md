# Playbook — traps that have already bitten

Every entry cost real time to learn. **Symptom first**, because that is how the next person
arrives. Add entries when something bites, never speculatively. Data-pipeline and scraping traps
live in `CLAUDE.md` → "Gotchas already paid for".

## Project-specific traps

**`app.pricing.train` rewrites `models/price_model.json` every run.**
*Symptom:* a clean checkout shows the model file modified after "just looking at the numbers".
Even with no code change it restamps `fitted_at`. Run it in a branch, or `git checkout` the file.

**A git worktree has no `.venv`, no `.env`, and no `data/images/`.** All three are gitignored.
*Symptom:* `test_corpus_photos_resolve_on_disk` fails, or `.venv/bin/python` is missing.
Symlink them from the main checkout: `ln -s <main>/.venv .venv`, same for `data/images`.

**devkit's `init.sh` runs `git init` inside a worktree.** In a worktree `.git` is a file, so its
`[[ ! -d .git ]]` guard is true and it nests a fresh repo. Copy the templates by hand there.

**The served price was not the measured price.** *Symptom:* a model card quoting R² 0.84 next to
a band that covered 87%. `estimate()` blended two routes, but only one had been scored. Any change
to how `estimate()` combines things needs `train.py` to score *that* combination. Decision 0001.

**A diagnostic can slice the wrong columns and still print a plausible number.**
*Symptom:* "brand columns are worth −0.024 R²". `evaluate(drop_brand=True)` hard-coded 4 columns
and dropped `euro6` instead. Derive column counts from `features.MARKETS`, never literals.

## Seeded traps — project-agnostic, all real

**A graceful fallback that swallows an empty result hides total failure.** Log or count when a
fallback fires. (This repo's `fell_back_from` is the pattern done right.)

**Chained shell commands print success echoes after a mid-chain failure.** Check exit codes.

**Verify a commit landed by comparing SHAs** (`git rev-parse HEAD origin/<branch>`), never by
reading hook output.

**A new git worktree branches from the default branch** and is missing unmerged work — check the
branch table in `STATUS.md`.

**A correct number can quietly stop applying.** Record the conditions a measurement was taken
under (corpus size, fit date, FX date) and re-measure when they change.
