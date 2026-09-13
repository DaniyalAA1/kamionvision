"""The artifact: JSON to diff against, markdown to read, and a noise floor.

The one rule this module exists to enforce: **every headline metric carries a
confidence interval, and `--diff` says `within noise` when a delta sits inside
the before-run's interval.** At n = 40-60 vehicles most deltas will be noise,
and a harness that reports noise as progress is worse than no harness at all -
it launders variance into a changelog entry. That is not polish, it is the
difference between this and `app/vlm/bench.py`.

`code_fingerprint` hashes four files separately rather than the tree, so a
diff between two runs can say WHICH workstream moved: a prompt edit, a
condition-rollup edit, a pricing edit or a pass-plumbing edit. `app/condition.py`
does not exist yet; an absent file fingerprints as `"absent"` rather than
raising, so this file is usable before the workstream that creates it lands.

Measured numbers and assumed ones are labelled differently here for the same
reason they are in the product's UI: an assumption that travels without its
label eventually gets quoted as a measurement.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import REPO

SCHEMA_VERSION = 1

# Hashed separately, not as a tree. The point is attribution: a moved number
# should name the workstream that moved it.
FINGERPRINT_FILES = {
    "prompts_sha": "app/evidence/prompts.py",
    "condition_sha": "app/condition.py",      # may not exist yet - see module docstring
    "pricing_sha": "app/pricing/model.py",
    "passes_sha": "app/evidence/passes.py",
}

MEASURED = "measured"
ASSUMED = "assumed"
DIAGNOSTIC = "diagnostic"   # reported, never a selection criterion


# --- intervals -------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """Score interval for a proportion. Correct at the small n this harness has.

    The normal approximation puts the interval for 0/8 at exactly [0, 0], which
    would let a suite report "zero false positives, confidently" off eight
    observations. Wilson does not.
    """
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def bootstrap_ci(units: list, statistic, *, draws: int = 2000, seed: int = 7,
                 alpha: float = 0.05) -> list[float] | None:
    """Percentile bootstrap, resampling UNITS.

    The unit is the vehicle, never the photo and never the finding. One tractor
    contributes 15-40 near-identical frames; resampling photos would treat
    those as independent observations and report an interval several times too
    tight - the same leak `splits.json` exists to prevent on the training side.
    """
    units = list(units)
    n = len(units)
    if n < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        try:
            value = statistic(sample)
        except (ZeroDivisionError, ValueError, TypeError):
            # A resample can contain no unit that supports the statistic - every
            # vehicle in the draw had zero matched findings, say. Dropping the
            # draw is right; substituting a zero would pull the interval toward
            # a value nothing measured.
            continue
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        values.append(float(value))
    if len(values) < 20:
        return None
    values.sort()
    lo = values[max(0, int(math.floor((alpha / 2) * len(values))))]
    hi = values[min(len(values) - 1, int(math.ceil((1 - alpha / 2) * len(values))) - 1)]
    return [round(lo, 4), round(hi, 4)]


# --- metrics ---------------------------------------------------------------

@dataclass
class Metric:
    name: str
    value: float | int | str | None
    ci: list[float] | None = None
    n: int = 0
    unit: str = ""
    basis: str = MEASURED        # measured | assumed | diagnostic
    note: str = ""
    headline: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Gate:
    """A pre-registered pass/fail, so a threshold cannot be chosen after the fact."""
    name: str
    value: float | None
    threshold: float
    direction: str = "below"     # below | above
    ci: list[float] | None = None
    basis: str = ASSUMED
    note: str = ""

    @property
    def passed(self) -> bool | None:
        if self.value is None:
            return None
        return self.value <= self.threshold if self.direction == "below" \
            else self.value >= self.threshold

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pass"] = self.passed
        return d


@dataclass
class SuiteResult:
    name: str
    status: str = "ok"           # ok | skipped | not_implemented | failed
    metrics: list[Metric] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    strata: dict = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "metrics": [m.to_dict() for m in self.metrics],
                "gates": [g.to_dict() for g in self.gates],
                "strata": self.strata, "caveats": self.caveats, "detail": self.detail}


# --- provenance ------------------------------------------------------------

def _sha(path: Path) -> str:
    if not path.exists():
        return "absent"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def code_fingerprint(root: Path = REPO) -> dict:
    return {label: _sha(root / rel) for label, rel in FINGERPRINT_FILES.items()}


def _git(*args: str, root: Path = REPO) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_info(root: Path = REPO) -> dict:
    sha = _git("rev-parse", "HEAD", root=root)
    return {"sha": sha[:7], "full_sha": sha,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", root=root),
            "dirty": bool(_git("status", "--porcelain", root=root))}


def run_id(model_id: str, *, sha: str | None = None, when: float | None = None) -> str:
    stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime(when))
    sha = sha or git_info()["sha"] or "nogit"
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in model_id)
    return f"{stamp}-{sha}-{safe}"


# --- the scorecard ---------------------------------------------------------

# Printed into every artifact. The replacement says out loud what it replaced,
# because the selection it replaced is still in the shipped model id.
BENCH_NOTE = (
    "The shipped vision model was selected by app/vlm/bench.py, which scores a "
    "backend on finding count, truck_vocab_share and mean observation length, "
    "on one fixture, against no ground truth. A model that fabricates twice as "
    "many long truck-vocabulary findings scores strictly better on every column "
    "it has. This harness is the replacement; truck_vocab_share survives here "
    "only as a diagnostic and is never a selection criterion.")

NO_GROUND_TRUTH_NOTE = (
    "This corpus has no condition ground truth: `damaged` is null in 100% of "
    "7,458 rows and there are no hasar/tramer/boyali fields (DATASET_CARD "
    "limitation 3). Every suite here manufactures its comparison from structure "
    "- a degraded twin is the same truck, a repeat is the same photo - or from "
    "the external panel set. None of them is a label.")


@dataclass
class Scorecard:
    run_id: str
    tier: str
    seed: int
    model: dict = field(default_factory=dict)
    git: dict = field(default_factory=dict)
    code_fingerprint: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)
    suites: dict = field(default_factory=dict)
    gates: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    sampling: dict = field(default_factory=dict)
    schema: int = SCHEMA_VERSION
    elapsed_s: float = 0.0

    @classmethod
    def start(cls, *, tier: str, seed: int, model: dict) -> "Scorecard":
        git = git_info()
        return cls(run_id=run_id(model.get("id", "unknown"), sha=git.get("sha")),
                   tier=tier, seed=seed, model=model, git=git,
                   code_fingerprint=code_fingerprint(),
                   notes=[BENCH_NOTE, NO_GROUND_TRUTH_NOTE])

    def add(self, result: SuiteResult) -> None:
        self.suites[result.name] = result.to_dict()
        for gate in result.gates:
            row = gate.to_dict()
            row["suite"] = result.name
            self.gates.append(row)

    def to_dict(self) -> dict:
        return {"schema": self.schema, "run_id": self.run_id, "git": self.git,
                "model": self.model, "code_fingerprint": self.code_fingerprint,
                "tier": self.tier, "seed": self.seed, "sampling": self.sampling,
                "calls": self.calls, "suites": self.suites, "gates": self.gates,
                "notes": self.notes, "elapsed_s": round(self.elapsed_s, 1)}

    def write(self, root: Path) -> Path:
        out = Path(root) / self.run_id
        (out / "raw").mkdir(parents=True, exist_ok=True)
        (out / "scorecard.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "scorecard.md").write_text(self.markdown(), encoding="utf-8")
        return out

    # --- rendering ---------------------------------------------------------

    def markdown(self) -> str:
        d = self.to_dict()
        L = [f"# eval {self.run_id}", ""]
        L.append(f"- tier **{self.tier}**, seed {self.seed}")
        L.append(f"- model `{self.model.get('backend','?')}/{self.model.get('id','?')}` "
                 f"effort `{self.model.get('effort','default')}`")
        L.append(f"- git `{self.git.get('sha','?')}` on `{self.git.get('branch','?')}`"
                 + (" (dirty tree)" if self.git.get("dirty") else ""))
        fp = ", ".join(f"{k.replace('_sha','')} `{v}`"
                       for k, v in self.code_fingerprint.items())
        L.append(f"- code fingerprint: {fp}")
        c = self.calls or {}
        L.append(f"- calls: **{c.get('new', 0)} new**, {c.get('cached', 0)} cached, "
                 f"{c.get('failed', 0)} failed - cache hit rate "
                 f"{100 * c.get('hit_rate', 0.0):.0f}%")
        if self.sampling:
            L.append(f"- sampling: {json.dumps(self.sampling, sort_keys=True)}")
        L.append("")

        if self.gates:
            L += ["## Gates", "", "| gate | value | threshold | basis | pass |",
                  "|---|---|---|---|---|"]
            for g in self.gates:
                mark = {True: "pass", False: "**FAIL**", None: "n/a"}[g.get("pass")]
                L.append(f"| `{g['suite']}.{g['name']}` | {_fmt(g['value'])}"
                         f"{_fmt_ci(g.get('ci'))} | {g['direction']} {g['threshold']} "
                         f"| {g['basis']} | {mark} |")
            L.append("")

        for name, suite in d["suites"].items():
            L.append(f"## {name}  — {suite['status']}")
            L.append("")
            if suite["metrics"]:
                L += ["| metric | value | 95% CI | n | basis |", "|---|---|---|---|---|"]
                for m in suite["metrics"]:
                    star = "**" if m.get("headline") else ""
                    L.append(f"| {star}`{m['name']}`{star} | {_fmt(m['value'])}"
                             f"{(' ' + m['unit']) if m['unit'] else ''} "
                             f"| {_ci_cell(m['ci'])} | {m['n']} | {m['basis']} |")
                L.append("")
            for note in suite.get("caveats") or []:
                L.append(f"> {note}")
                L.append("")
            for label, table in (suite.get("strata") or {}).items():
                if not table:
                    continue
                L.append(f"### by {label}")
                L.append("")
                cols = sorted({k for row in table.values() for k in row})
                L.append("| " + label + " | " + " | ".join(cols) + " |")
                L.append("|" + "---|" * (len(cols) + 1))
                for key in sorted(table):
                    L.append(f"| {key} | " + " | ".join(
                        _fmt(table[key].get(c)) for c in cols) + " |")
                L.append("")

        if self.notes:
            L.append("## Notes")
            L.append("")
            for note in self.notes:
                L.append(f"- {note}")
            L.append("")
        return "\n".join(L)


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _fmt_ci(ci) -> str:
    return f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else ""


def _ci_cell(ci) -> str:
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "*none*"


def load(run_dir: Path | str) -> dict:
    path = Path(run_dir)
    if path.is_dir():
        path = path / "scorecard.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --- the diff, and the noise floor -----------------------------------------

@dataclass
class DiffRow:
    suite: str
    metric: str
    before: float | None
    after: float | None
    delta: float | None
    before_ci: list | None
    after_ci: list | None
    within_noise: bool
    headline: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _overlaps(a, b) -> bool:
    return bool(a and b and a[0] <= b[1] and b[0] <= a[1])


def is_noise(before: dict, after: dict) -> tuple[bool, str]:
    """Deliberately conservative: it is harder to claim a real move than a null one.

    Two ways a delta is called noise. The design's rule is the first: the new
    value sits inside the interval the old run already spanned, so the old run
    could not have told them apart. The second catches the mirror case, where
    the new run is the imprecise one - overlapping intervals are not a
    difference either, and letting that through would reward shrinking n.
    """
    bv, av = before.get("value"), after.get("value")
    if not isinstance(bv, (int, float)) or not isinstance(av, (int, float)):
        return True, "non-numeric"
    bci, aci = before.get("ci"), after.get("ci")
    if bci and bci[0] <= av <= bci[1]:
        return True, "after value inside the before-run interval"
    if _overlaps(bci, aci):
        return True, "intervals overlap"
    if not bci and not aci:
        return True, "no interval on either run - cannot distinguish from noise"
    return False, ""


def diff(before: dict, after: dict) -> list[DiffRow]:
    rows: list[DiffRow] = []
    for suite in sorted(set(before.get("suites", {})) | set(after.get("suites", {}))):
        b_metrics = {m["name"]: m for m in
                     before.get("suites", {}).get(suite, {}).get("metrics", [])}
        a_metrics = {m["name"]: m for m in
                     after.get("suites", {}).get(suite, {}).get("metrics", [])}
        for name in sorted(set(b_metrics) | set(a_metrics)):
            b, a = b_metrics.get(name, {}), a_metrics.get(name, {})
            bv, av = b.get("value"), a.get("value")
            delta = (av - bv) if isinstance(bv, (int, float)) and isinstance(av, (int, float)) \
                else None
            noise, reason = is_noise(b, a) if (b and a) else (True, "metric only in one run")
            rows.append(DiffRow(suite=suite, metric=name, before=bv, after=av,
                                delta=round(delta, 4) if delta is not None else None,
                                before_ci=b.get("ci"), after_ci=a.get("ci"),
                                within_noise=noise, reason=reason,
                                headline=bool(a.get("headline") or b.get("headline"))))
    return rows


def render_diff(before: dict, after: dict, rows: list[DiffRow]) -> str:
    L = [f"# diff  {before.get('run_id')}  ->  {after.get('run_id')}", ""]
    moved = {k: (before["code_fingerprint"].get(k), after["code_fingerprint"].get(k))
             for k in set(before.get("code_fingerprint", {}))
             | set(after.get("code_fingerprint", {}))}
    changed = [k.replace("_sha", "") for k, (b, a) in sorted(moved.items()) if b != a]
    L.append("- code that moved: " + (", ".join(f"`{c}`" for c in changed)
                                      if changed else "*nothing - same four files*"))
    for label, card in (("before", before), ("after", after)):
        c = card.get("calls", {})
        L.append(f"- {label} calls: {c.get('new', 0)} new / {c.get('cached', 0)} cached "
                 f"({100 * c.get('hit_rate', 0.0):.0f}% hits), tier {card.get('tier')}")
    if before.get("tier") != after.get("tier"):
        L.append(f"- **tiers differ** ({before.get('tier')} vs {after.get('tier')}): "
                 f"different n, and at smoke tier a different sampling depth. "
                 f"These two runs are not comparable; nothing below is a result.")
    if before.get("seed") != after.get("seed"):
        L.append(f"- **seeds differ** ({before.get('seed')} vs {after.get('seed')}): the "
                 f"two runs are not on the same vehicles, so every delta below "
                 f"carries sampling variation on top of its own interval.")
    L += ["", "| suite | metric | before | after | delta | verdict |",
          "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (not r.headline, r.suite, r.metric)):
        star = "**" if r.headline else ""
        verdict = "within noise" if r.within_noise else "moved"
        L.append(f"| {r.suite} | {star}`{r.metric}`{star} | {_fmt(r.before)}"
                 f"{_fmt_ci(r.before_ci)} | {_fmt(r.after)}{_fmt_ci(r.after_ci)} "
                 f"| {_fmt(r.delta)} | {verdict} |")
    n_moved = sum(1 for r in rows if not r.within_noise)
    L += ["", f"{n_moved} of {len(rows)} metrics moved beyond the before-run's "
              f"interval. The rest are reported as noise on purpose: at this n "
              f"most deltas are, and a harness that calls them progress is worse "
              f"than none."]
    return "\n".join(L)
