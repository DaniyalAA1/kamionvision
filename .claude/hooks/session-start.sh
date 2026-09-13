#!/usr/bin/env bash
# Injected at every session start. Removes the "the agent didn't know where it was" failure class.
# Keep it FAST and quiet — it runs before every session.
echo "=== $(basename "$PWD") ==="
if git rev-parse --git-dir > /dev/null 2>&1; then
  branch=$(git branch --show-current 2>/dev/null)
  dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  echo "branch: ${branch:-<detached>}  |  uncommitted files: $dirty"
  if upstream=$(git rev-parse --abbrev-ref '@{u}' 2>/dev/null); then
    counts=$(git rev-list --left-right --count "@{u}...HEAD" 2>/dev/null)
    echo "vs $upstream: $(echo "$counts" | awk '{print $2" ahead, "$1" behind"}')"
  else
    echo "upstream: none set (branch not pushed)"
  fi
fi
if [ -f docs/STATUS.md ]; then
  echo "--- docs/STATUS.md ---"
  grep -m1 -A2 '^\*\*Last updated' docs/STATUS.md 2>/dev/null || head -5 docs/STATUS.md
  echo "--- read docs/STATUS.md before acting ---"
else
  echo "NOTE: no docs/STATUS.md — run /devkit to bootstrap the docs system."
fi
