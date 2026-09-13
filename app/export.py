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

# Scripts that live under app/web/js/ but are not part of the appraisal
# screen's ES module graph.
NON_MODULE_SCRIPTS = {"landing"}


def _data_uri(path: Path) -> str:
    raw, _ = encode_jpeg(path, long_edge=EXPORT_LONG_EDGE)
    return "data:image/jpeg;base64," + base64.standard_b64encode(raw).decode("ascii")


def _inline_fonts(css: str) -> str:
    """Rewrite url(/static/assets/fonts/x.woff2) to a data URI.

    Without this the frozen file falls back to system fonts, which is the one
    thing an offline copy of a drawing should not do - the whole sheet is set
    in a condensed technical face and a proportional substitute reflows it.
    """
    def sub(match: re.Match) -> str:
        name = match.group(1)
        path = WEB / "assets" / "fonts" / name
        if not path.exists():
            return match.group(0)
        b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        return f'url("data:font/woff2;base64,{b64}")'

    return re.sub(r'url\("/static/assets/fonts/([^"]+)"\)', sub, css)


def _module_url(source: str) -> str:
    """One ES module as a data: URL, with its imports made bare.

    A data: URL has no base, so `./dom.js` inside an inlined module cannot
    resolve. Rewriting every intra-app import to a bare `kamion:` specifier
    lets the import map resolve it regardless of where the importing module
    came from, which keeps the real module graph rather than concatenating the
    files and hoping the names do not collide.
    """
    source = re.sub(r"from '\./(?:js/)?([a-z]+)\.js'", r"from 'kamion:\1'", source)
    b64 = base64.standard_b64encode(source.encode("utf-8")).decode("ascii")
    return f"data:text/javascript;base64,{b64}"


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

    # --- stylesheets, in the order the page links them ---
    sheets = re.findall(r'<link rel="stylesheet" href="/static/(styles/[^"]+)">', html)
    css = "\n".join((WEB / name).read_text(encoding="utf-8") for name in sheets)
    css = _inline_fonts(css)
    first = f'<link rel="stylesheet" href="/static/{sheets[0]}">'
    html = html.replace(first, f"<style>\n{css}\n</style>", 1)
    for name in sheets[1:]:
        html = html.replace(f'<link rel="stylesheet" href="/static/{name}">', "", 1)

    # --- the vendored animation library ---
    motion = (WEB / "vendor" / "motion.min.js").read_text(encoding="utf-8")
    html = html.replace('<script src="/static/vendor/motion.min.js"></script>',
                        f"<script>\n{motion}\n</script>", 1)

    # --- the module graph, plus the payload and the drawing ---
    # The landing page's script is a classic IIFE that drives a pixel mascot on
    # `/`; it is not part of the appraisal screen's module graph and nothing
    # imports it. Inlining it added a dead data: URL to every frozen report.
    imports = {f"kamion:{m.stem}": _module_url(m.read_text(encoding="utf-8"))
               for m in sorted((WEB / "js").glob("*.js"))
               if m.stem not in NON_MODULE_SCRIPTS}
    entry = re.sub(r"from '\./js/([a-z]+)\.js'", r"from 'kamion:\1'",
                   (WEB / "app.js").read_text(encoding="utf-8"))
    elevation = (WEB / "assets" / "tractor-elevation.svg").read_text(encoding="utf-8")

    bundle = (
        '<script type="importmap">\n'
        + json.dumps({"imports": imports}) + "\n</script>\n"
        + "<script>\nwindow.KAMION_APPRAISAL = "
        + json.dumps(payload, ensure_ascii=False) + ";\n"
        + "window.KAMION_ELEVATION = " + json.dumps(elevation) + ";\n</script>\n"
        + '<script type="module">\n' + entry + "\n</script>")
    html = html.replace('<script type="module" src="/static/app.js"></script>', bundle, 1)

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


def _index_faces() -> str:
    """The two display faces, inlined, so the index is set like the reports.

    The index is a sibling file in the output directory with no assets beside
    it, so it cannot link the woff2 the way the sheet does.
    """
    wanted = [("Barlow", 400, "barlow-normal-400-latin.woff2"),
              ("Barlow", 600, "barlow-normal-600-latin.woff2"),
              ("Barlow Condensed", 600, "barlow-condensed-normal-600-latin.woff2")]
    out = []
    for family, weight, name in wanted:
        path = WEB / "assets" / "fonts" / name
        if not path.exists():
            continue
        b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        out.append(f'@font-face{{font-family:"{family}";font-weight:{weight};'
                   f'font-display:swap;'
                   f'src:url("data:font/woff2;base64,{b64}") format("woff2")}}')
    return "".join(out)


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
        f"<style>{_index_faces()}{INDEX_CSS}</style></head><body><main>"
        "<h1>KamionVision — saved appraisals</h1>"
        "<p class=\"sub\">Offline copies of the rehearsed cases. Each opens with no "
        "server and no network.</p>" + "\n".join(rows) + "</main></body></html>")
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
