# Continuity — how to rehydrate full context from zero

Chat sessions are disposable; this repo is the memory. Any fresh session — any model, any tool,
after any compaction — reconstructs full context by reading, in order:

0. **Sync first: `git fetch origin && git pull`.** A local checkout can lag behind merged work.
   **This project keeps unmerged branches — `docs/STATUS.md`'s branch table is authoritative;
   pulling `main` alone will NOT show you everything.**
1. **`docs/STATUS.md`** — what is happening right now.
2. `CLAUDE.md` — how the system works, and the invariants it encodes. Binding.
3. `kamion-truck-appraisal-brief.md` — the sponsor's brief; ground truth for what gets judged.
4. `docs/decisions/` (newest first) — every strategic choice, its why, what it superseded.
5. `docs/PLAYBOOK.md` — traps that have already cost this project time.
6. `docs/BRAINSTORM.md` — open improvement ideas, ranked; not decisions.
7. `graphify-out/GRAPH_REPORT.md` — the codebase knowledge graph, for "what connects to what".

## Rules that keep this true

- **No state lives only in chat.** A decision made in conversation lands the same day as a
  decision-log entry. If it is not in git, it did not happen.
- **The docs change while you are not looking.** Teammates edit concurrently — that is why step 0
  is `git pull`, not a formality.
- **A doc can be confidently wrong.** Every measured number here was true under the corpus and fit
  it names. If you refit (`app.pricing.train`) or re-harvest, re-measure before quoting it.
- **Supersede, never rewrite, a decision.** The wrong turns are the most valuable content.
