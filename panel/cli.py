"""prompts | ingest | report — the three steps of a panel run.

    .venv/bin/python -m panel.cli prompts --pilot --out <dir>
    .venv/bin/python -m panel.cli ingest  --dir <dir> --run-id pilot-YYYY-MM-DD
    .venv/bin/python -m panel.cli report

`prompts` writes one instrument per (vehicle, panelist) and a manifest. The
panelists themselves are agent invocations launched by hand, in one message, so
that none of them can see an earlier one's answer - that independence cannot be
enforced from inside this file, so it is written down in the card instead.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from panel import agreement as ag, card, store
from panel.protocol import (COMPONENT_STATES, NOT_VISIBLE, RUBRIC_SHA,
                            SEVERITY_SCALE, panelist_prompt, validate_response)
from panel.select import (PILOT_RULES, SEED, Vehicle, load_vehicles, manifest,
                          select_pilot, select_target)

PANELISTS = ("p1", "p2", "p3")


def _vehicles(args) -> list[Vehicle]:
    target = select_target(seed=args.seed)
    return select_pilot(target, seed=args.seed) if args.pilot else target


def cmd_prompts(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _vehicles(args)
    index = []
    for v in rows:
        prompt = panelist_prompt(
            listing_id=v.listing_id, make=v.make, model=v.model, year=v.year,
            km=v.km, market=v.market, photos=v.photos)
        path = out / f"{v.listing_id}.prompt.txt"
        path.write_text(prompt, encoding="utf-8")
        index.append({"listing_id": v.listing_id, "source_key": v.source_key,
                      "stratum": v.stratum, "prompt": str(path),
                      "n_photos": v.n_photos, "km": v.km, "year": v.year,
                      "make": v.make, "model": v.model,
                      "responses": {p: str(out / f"{v.listing_id}__{p}.json")
                                    for p in PANELISTS}})
    (out / "manifest.json").write_text(
        json.dumps({**manifest(rows, name="pilot" if args.pilot else "target60"),
                    "panelists": list(PANELISTS), "index": index},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} vehicles x {len(PANELISTS)} panelists -> {out}")
    for entry in index:
        print(f"  {entry['listing_id']:<10} {entry['n_photos']:>3} photos  {entry['prompt']}")


def _load_response(path: Path, n_photos: int) -> tuple[dict | None, list[str]]:
    if not path.exists():
        return None, ["no response file"]
    text = path.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None, [f"not JSON: {exc}"]
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc2:
            return None, [f"not JSON: {exc2}"]
    return obj, validate_response(obj, n_photos=n_photos)


def cmd_ingest(args) -> None:
    src = Path(args.dir)
    index = json.loads((src / "manifest.json").read_text(encoding="utf-8"))["index"]
    by_id = {v.listing_id: v for v in load_vehicles()}
    timings = {}
    if (src / "timings.json").exists():
        timings = json.loads((src / "timings.json").read_text(encoding="utf-8"))

    rows, bad = [], 0
    for entry in index:
        vehicle = by_id[entry["listing_id"]]
        vehicle.stratum = entry["stratum"]
        panelists = []
        for name, path in entry["responses"].items():
            obj, errors = _load_response(Path(path), entry["n_photos"])
            bad += bool(errors)
            panelists.append({
                "panelist": name,
                "ok": not errors,
                "errors": errors,
                **timings.get(f"{entry['listing_id']}__{name}", {}),
                "response": obj,
            })
        rows.append(store.build_row(vehicle, panelists, run_id=args.run_id,
                                    model_id=args.model))
    path = store.write_rows(rows)
    print(f"wrote {len(rows)} rows -> {path}  ({bad} panelist answers with errors)")


def compute_stats(rows: list[dict], *, seed: int = SEED) -> dict:
    vectors, grades = store.panel_view(rows)
    sev = ag.severity_alpha(vectors)
    state = ag.state_alpha(vectors)
    vis = ag.visibility_alpha(vectors)
    grade = ag.grade_agreement(grades)
    units = list(ag.component_units(vectors).values())
    raw_state, _ = ag.percent_agreement(units)

    readings = Counter(s for unit in units for s in unit)
    n_readings = sum(readings.values())
    visible = n_readings - readings[NOT_VISIBLE]
    state_table = "| state | readings | share of all | share of visible |\n|---|---|---|---|\n"
    state_table += "\n".join(
        f"| `{s}` | {readings[s]} | {readings[s] / n_readings:.1%} | "
        + (f"{readings[s] / visible:.1%} |" if s != NOT_VISIBLE and visible else "— |")
        for s in COMPONENT_STATES)

    n_runs = sum(len(r["panelists"]) for r in rows)
    n_ok = sum(1 for r in rows for p in r["panelists"] if p["ok"])
    consensus_grades = Counter(r["consensus"]["grade"] for r in rows)
    good = sum(consensus_grades[g] for g in ("good", "excellent"))
    none_share = readings["none"] / n_readings if n_readings else 0.0
    top_state = readings.most_common(1)[0][0] if readings else None

    checks = [
        ("1 ordinal α, per-component severity", "≥ 0.50",
         "undefined" if sev.alpha is None else f"{sev.alpha:.3f}",
         sev.alpha is not None and sev.alpha >= 0.50),
        ("2 exact grade agreement", "≥ 0.50",
         f"{grade.unanimous:.3f}", (grade.unanimous or 0) >= 0.50),
        ("3 pairwise grade agreement", "≥ 0.67",
         f"{grade.pairwise:.3f}", (grade.pairwise or 0) >= 0.67),
        ("4 consensus `good` or `excellent`", "≥ 0.60",
         f"{good}/{len(rows)} = {good / len(rows):.2f}", good / len(rows) >= 0.60),
        ("5 `none` most common state, ≥ 0.35 of readings", "both",
         f"most common `{top_state}`, none = {none_share:.1%}",
         top_state == "none" and none_share >= 0.35),
        ("6 schema-valid responses", "≥ 15 of 18",
         f"{n_ok}/{n_runs}", n_ok >= 15),
        ("7 vehicles needing adjudication", "≤ 1 of 6",
         f"{len(grade.needs_adjudication)}/{len(rows)}",
         len(grade.needs_adjudication) <= 1),
    ]
    scorecard = "\n".join(
        f"| {name} | {target} | **{value}** | {'yes' if ok else '**no**'} |"
        for name, target, value, ok in checks)

    return {
        "severity_alpha": sev, "state_alpha": state, "visibility_alpha": vis,
        "grade": grade, "raw_state": f"{raw_state:.3f}" if raw_state else "n/a",
        "n_panelists": max(len(r["panelists"]) for r in rows),
        "n_runs": n_runs, "n_ok": n_ok, "n_readings": n_readings,
        "state_table": state_table, "scorecard": scorecard, "checks": checks,
        "grade_distribution": ", ".join(f"{g} {n}" for g, n in consensus_grades.most_common()),
        "seed": seed,
        "pilot_rules": "\n".join(f"  {i + 1}. `{k}` — {why}"
                                 for i, (k, why) in enumerate(PILOT_RULES)),
        "example_vehicle": rows[0]["listing_id"],
    }


def cmd_report(args) -> None:
    rows = store.read_rows()
    if not rows:
        raise SystemExit("no artifact yet - run ingest first")
    stale = store.check_rubric(rows)
    if stale:
        print(f"WARNING: {len(stale)} row(s) graded under another rubric: {stale}")
    stats = compute_stats(rows)
    for name, target, value, ok in stats["checks"]:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<46} {target:<10} {value}")
    if args.card:
        by_id = {v.listing_id: v for v in load_vehicles()}
        first = rows[0]
        v = by_id[first["listing_id"]]
        example = panelist_prompt(listing_id=v.listing_id, make=v.make, model=v.model,
                                  year=v.year, km=v.km, market=v.market, photos=v.photos)
        text = card.render(rows=rows, stats=stats, run_id=first["run_id"],
                           model_id=first["model_id"], example_prompt=example,
                           verdict=Path(args.verdict).read_text(encoding="utf-8")
                           if args.verdict else "_not yet written_",
                           exclusions=Path(args.exclusions).read_text(encoding="utf-8")
                           if args.exclusions else "_none_",
                           graded_at=first["graded_at"])
        store.CARD.write_text(text, encoding="utf-8")
        print(f"wrote {store.CARD}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="panel.cli")
    ap.add_argument("--seed", type=int, default=SEED)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompts"); p.set_defaults(fn=cmd_prompts)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--out", required=True)

    p = sub.add_parser("ingest"); p.set_defaults(fn=cmd_ingest)
    p.add_argument("--dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", default=store.PANEL_MODEL)

    p = sub.add_parser("report"); p.set_defaults(fn=cmd_report)
    p.add_argument("--card", action="store_true")
    p.add_argument("--verdict")
    p.add_argument("--exclusions")

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
