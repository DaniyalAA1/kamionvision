"""Freeze an appraisal into a single self-contained HTML file.

Demo insurance. `hackathon-plan.md`'s checklist asks for a backup in case the
live run fails, and a screen recording cannot be re-opened at a judge's
question. This writes the same screen the server renders - gauge, photo grid,
findings, the lot - as one file with the stylesheet, the script and every photo
inlined as data URIs. It opens with no server, no network and no API key, which
is exactly the situation a failed live demo puts you in.

    .venv/bin/python -m app.cli appraise demo/tr_clean --html out.html
    .venv/bin/python -m app.demo --export demo_reports/
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from .config import WEB
from .schema import Appraisal
from .vlm.base import encode_jpeg

# Photos are re-encoded at this width before being inlined. A 20-photo set of
# 4000px dealer originals would produce a 40 MB HTML file; at 640 the whole
# report lands around 1-2 MB and the thumbnails and lightbox still read.
EXPORT_LONG_EDGE = 640


def _data_uri(path: Path) -> str:
    raw, _ = encode_jpeg(path, long_edge=EXPORT_LONG_EDGE)
    return "data:image/jpeg;base64," + base64.standard_b64encode(raw).decode("ascii")


def build_html(appraisal: Appraisal, *, title: str | None = None) -> str:
    payload = appraisal.to_dict()

    # Replace served URLs with inlined images, keyed by the gate's photo_id.
    photos = {}
    for check in appraisal.gate.photos:
        source = Path(check.path)
        if source.exists():
            try:
                photos[str(check.photo_id)] = _data_uri(source)
            except Exception:
                continue
    payload["photo_urls"] = photos

    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8")

    html = html.replace('<link rel="stylesheet" href="/static/styles.css">',
                        f"<style>\n{css}\n</style>")
    html = html.replace('<script src="/static/app.js"></script>',
                        "<script>\nwindow.KAMION_APPRAISAL = "
                        + json.dumps(payload, ensure_ascii=False)
                        + ";\n</script>\n<script>\n" + js + "\n</script>")
    if title:
        html = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html, count=1)
    return html


def write_html(appraisal: Appraisal, out: str | Path, *, title: str | None = None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(appraisal, title=title), encoding="utf-8")
    return out


INDEX_CSS = """
body { margin:0; background:#0d1318; color:#e9f0f4;
       font:400 16px/1.55 "Barlow", ui-sans-serif, system-ui, sans-serif; }
main { max-width:820px; margin:0 auto; padding:2rem 1.5rem 4rem; }
h1 { font-family:"Barlow Condensed", sans-serif; font-size:2rem; margin:0 0 .3rem; }
p.sub { color:#93a7b5; margin:0 0 2rem; }
a.case { display:grid; grid-template-columns:1fr auto; gap:.2rem 1rem;
         padding:.85rem 1rem; margin-bottom:.5rem; text-decoration:none; color:inherit;
         background:#131c23; border:1px solid #1c2831; border-left:2px solid #3b5162;
         border-radius:3px; }
a.case:hover { background:#17222a; border-left-color:#f0a23c; }
a.case b { font-family:"Barlow Condensed", sans-serif; font-size:1.1rem; font-weight:600; }
a.case span { color:#6d8291; font-size:.88rem; grid-column:1; }
a.case em { font-style:normal; color:#f0a23c; font-size:.85rem; align-self:center; }
a.case.refused { border-left-color:#e05b4f; }
a.case.refused em { color:#e05b4f; }
"""


def write_index(entries: list[dict], out_dir: str | Path) -> Path:
    """An index over exported reports, so the backup is browsable on stage."""
    out_dir = Path(out_dir)
    rows = []
    for e in entries:
        cls = "case refused" if e["status"] in ("refused", "need_more_photos") else "case"
        rows.append(
            f'<a class="{cls}" href="{e["file"]}"><b>{e["title"]}</b>'
            f'<em>{e["status"]}</em><span>{e["headline"]}</span></a>')
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>KamionVision — saved appraisals</title>"
        '<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600'
        '&family=Barlow+Condensed:wght@500;600&display=swap" rel="stylesheet">'
        f"<style>{INDEX_CSS}</style></head><body><main>"
        "<h1>KamionVision — saved appraisals</h1>"
        "<p class=\"sub\">Offline copies of the rehearsed cases. Each opens with no "
        "server and no network.</p>" + "\n".join(rows) + "</main></body></html>")
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
