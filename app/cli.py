"""Command line entry point.

    .venv/bin/python -m app.cli doctor
    .venv/bin/python -m app.cli appraise data/images/tr_truckmarket/14790 --year 2020 --km 420000
    .venv/bin/python -m app.cli serve
    .venv/bin/python -m app.cli demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report as report_mod
from .config import GATE_THRESHOLDS, PRICE_MODEL, REPO


def cmd_doctor(args) -> int:
    from . import vlm
    from .config import (ANTHROPIC_MODEL, BACKEND_CHAIN, BACKEND_OVERRIDE,
                         CURSOR_MODEL, MAX_EVIDENCE_PHOTOS)

    ok = True
    print("kamionvision doctor\n" + "=" * 60)

    print("\nvision backends (resolution order: "
          f"{BACKEND_OVERRIDE or ' -> '.join(BACKEND_CHAIN)})")
    ready_any = False
    for status in vlm.probe_all():
        if status.ready:
            mark = "  ok  "
            if not BACKEND_OVERRIDE or status.name == BACKEND_OVERRIDE:
                ready_any = True
        elif status.account_blocked:
            mark = " BILL "
        else:
            mark = "  --  "
        print(f"  [{mark}] {status.name:<10} {status.detail}")
    if not ready_any:
        ok = False
        print("\n  The selected vision backend is not usable. Set CURSOR_API_KEY "
              "in .env\n  (Cursor needs a user API key from cursor.com/dashboard -> "
              "API & SSH Keys;\n  the cursor-agent CLI login is a session token "
              "and the SDK rejects it.)")

    print("\nlocal models")
    for label, path, howto in (
        ("gate thresholds", GATE_THRESHOLDS, "python -m app.calibrate_gate"),
        ("price model", PRICE_MODEL, "python -m app.pricing.train"),
        ("perception heads", REPO / "models" / "perception.json",
         "python scripts/cache_embeddings.py && python -m app.perception.train"),
        ("yolov8n weights", REPO / "models" / "yolov8n.pt", "downloaded on first use"),
    ):
        if path.exists():
            print(f"  [  ok  ] {label:<16} {path.relative_to(REPO)} "
                  f"({path.stat().st_size / 1024:.0f} KB)")
        else:
            ok = False
            print(f"  [ MISS ] {label:<16} {path.relative_to(REPO)} - run: {howto}")

    if PRICE_MODEL.exists():
        from .pricing import load_model
        m = load_model()
        cal, meta = m.calibration, m.meta
        print(f"\nprice model  ({meta.get('training_set')}, fitted {meta.get('fitted_at')})")
        print(f"  {meta['n_listings']} listings -> {meta['n_groups']} distinct specs")
        print(f"  R2 {cal['r2_oof']:.3f}   median error {cal['median_ape_oof']:.1f}%")
        for level in ("0.5", "0.8", "0.9"):
            cov = cal.get(f"coverage_{level}")
            if cov is not None:
                print(f"  {float(level) * 100:.0f}% band covers {cov * 100:.1f}% "
                      f"of held-out listings")
        print(f"  unseen-brand widening {m.widening['unknown_brand']}x")
        a = m.anchor or {}
        if a.get("ok"):
            print(f"\nnew-price anchor  (second pricing route, needs no comparable)")
            print(f"  retention curve R2 {a['r2_oof']:+.3f}, median error "
                  f"{a['median_ape_oof']}% on {a['n']} TR listings")
            test = (a.get("unseen_brand_test") or {})
            for key, r in test.items():
                if key.startswith("_"):
                    continue
                print(f"  {key} held out (n={r['n_held_out']}): band "
                      f"{r['band_factor_widened_only']:.2f}x -> {r['band_factor_with_anchor']:.2f}x, "
                      f"coverage {r['coverage_widened_only']:.2f} -> {r['coverage_with_anchor']:.2f}, "
                      f"error {r['median_ape_widened_only']:.1f}% -> {r['median_ape_with_anchor']:.1f}%")
        else:
            print(f"  new-price anchor: not fitted ({a.get('reason', 'no retention curve')})")

    perception_path = REPO / "models" / "perception.json"
    if perception_path.exists():
        import json
        meta = json.loads(perception_path.read_text(encoding="utf-8"))["meta"]
        print(f"\nperception heads  (fitted {meta.get('fitted_at')})")
        print(f"  {meta['n_images']} images over {meta['n_vehicles']} vehicles, "
              f"folds grouped by vehicle")
        print(f"  degradation AUC {meta['degradation_auc']} (real labels), "
              f"severity R2 {meta['severity_r2']}")
        print(f"  view stability under degradation {meta['view_agreement_head']:.3f} "
              f"vs {meta['view_agreement_zeroshot']:.3f} zero-shot")
        print(f"  brand accuracy {meta['identity_accuracy']:.3f} vs "
              f"{meta['identity_majority']:.3f} majority")

    ref = REPO / "data" / "reference" / "new_prices_tr.json"
    if ref.exists():
        import datetime
        import json
        rows = [r for r in json.loads(ref.read_text(encoding="utf-8"))["rows"] if r.get("list_price")]
        oldest = min(r["as_of"] for r in rows) if rows else "-"
        age = (datetime.date.today() - datetime.date.fromisoformat(oldest)).days if rows else 0
        print(f"\nnew-price reference: {len(rows)} priced rows, oldest stamped {oldest} "
              f"({age} days ago)")
        if age > 120:
            print("  [ WARN ] a list price this old should be re-checked against its source URL")

    print("\ndevice")
    try:
        from . import vision
        print(f"  torch device: {vision.device()}")
    except Exception as exc:
        ok = False
        print(f"  [ FAIL ] {type(exc).__name__}: {exc}")

    print(f"\nevidence photos per call: {MAX_EVIDENCE_PHOTOS}")
    print(f"models: cursor={CURSOR_MODEL}  anthropic={ANTHROPIC_MODEL}")
    print("\n" + ("all good" if ok else "some checks failed"))
    return 0 if ok else 1


def cmd_appraise(args) -> int:
    from . import pipeline

    photos = pipeline.collect_photos(args.folder)
    if not photos:
        print(f"no images found in {args.folder}", file=sys.stderr)
        return 2
    if args.limit:
        photos = photos[:args.limit]

    declared = {k: v for k, v in
                (("year", args.year), ("km", args.km), ("make", args.make),
                 ("asking_price", args.asking))
                if v is not None}

    def note(step, detail):
        if not args.json:
            print(f"  ... {step}: {detail}", file=sys.stderr, flush=True)

    result = pipeline.appraise(photos, declared, market=args.market,
                               backend=args.backend, on_step=note)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report_mod.render_text(result))

    if args.html:
        from .export import write_html
        out = write_html(result, args.html)
        print(f"\nwrote {out} ({out.stat().st_size / 1024:.0f} KB, opens offline)",
              file=sys.stderr)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\nsaved {out}", file=sys.stderr)
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    print(f"kamionvision UI -> http://{args.host}:{args.port}")
    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo
    return run_demo(only=args.case, backend=args.backend)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kamionvision",
                                 description="Appraise a used semi-tractor from photos.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check backends, models and calibration")
    d.set_defaults(func=cmd_doctor)

    a = sub.add_parser("appraise", help="appraise a folder of photos")
    a.add_argument("folder")
    a.add_argument("--year", type=int)
    a.add_argument("--km", type=float)
    a.add_argument("--make")
    a.add_argument("--asking", type=float,
                   help="the seller's asking price, to be judged against the comparables")
    a.add_argument("--market", default="TR", choices=["TR", "US"])
    a.add_argument("--backend", help="pin a vision backend (cursor | anthropic)")
    a.add_argument("--limit", type=int, help="use only the first N photos")
    a.add_argument("--json", action="store_true", help="emit the raw Appraisal JSON")
    a.add_argument("--save", help="also write the JSON here")
    a.add_argument("--html", help="write a standalone offline HTML report here")
    a.set_defaults(func=cmd_appraise)

    s = sub.add_parser("serve", help="run the demo web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("demo", help="run the rehearsed demo cases")
    m.add_argument("--case", help="run only this case")
    m.add_argument("--backend")
    m.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
