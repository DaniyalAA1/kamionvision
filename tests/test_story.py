"""The landing page's scroll story is a frozen run, and has to stay one.

The page argues that this system is not a thin wrapper. That argument is
worthless if the page's own evidence is copywriting, so these tests exist to
make authored prose about the truck fail the suite rather than ship.

Three things are pinned. Every claim rendered into `landing.html` appears
verbatim in `story.json`; the region is exactly what `scripts/freeze_story.py`
would regenerate, so it cannot be hand-edited; and the measured 80.3% coverage
figure stays welded to the band it was measured on.
"""
import json
import re
import subprocess
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "app" / "web"
STORY = WEB / "assets" / "story"
LANDING = WEB / "landing.html"
START, END = "<!-- story:start -->", "<!-- story:end -->"


def region() -> str:
    page = LANDING.read_text(encoding="utf-8")
    return page.split(START, 1)[1].split(END, 1)[0]


class Text(HTMLParser):
    """The visible text of the region, and the src/href it references."""

    def __init__(self, source):
        super().__init__()
        self.chunks, self.refs, self.attrs = [], [], []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        self.attrs.append((tag, d))
        for key in ("src", "href"):
            if d.get(key, "").startswith("/static/"):
                self.refs.append(d[key])
        if d.get("alt"):
            self.chunks.append(d["alt"])

    def handle_data(self, data):
        if data.strip():
            self.chunks.append(data.strip())


class StoryFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.story = json.loads((STORY / "story.json").read_text(encoding="utf-8"))
        cls.region = region()
        cls.text = Text(cls.region)


class TheRegionIsGenerated(StoryFixture):
    """`freeze_story.py --check` regenerates the region and diffs it.

    Without this the markup is just HTML someone can edit, and the moment it is
    edited the page stops being a record of a run and starts being a brochure
    that happens to contain some real numbers.
    """

    def test_the_markers_are_present(self):
        page = LANDING.read_text(encoding="utf-8")
        self.assertEqual(page.count(START), 1)
        self.assertEqual(page.count(END), 1)
        self.assertLess(page.index(START), page.index(END))

    def test_the_story_json_and_its_photos_are_committed(self):
        self.assertTrue((STORY / "story.json").is_file())
        for photo in self.story["photos"]:
            self.assertTrue((STORY / photo["file"]).is_file(), photo["file"])
        if self.story["gate"]["lot"]:
            self.assertTrue((STORY / "lot.jpg").is_file())

    def test_every_static_reference_resolves(self):
        self.assertTrue(self.text.refs, "the story renders no assets at all")
        for ref in self.text.refs:
            self.assertTrue((WEB / ref.removeprefix("/static/")).is_file(), ref)


class EveryClaimComesFromTheRun(StoryFixture):
    """No sentence about this truck may be authored in the markup."""

    def claims(self):
        """Every sentence in the JSON that describes the vehicle."""
        out = []
        for photo in self.story["photos"]:
            out.append(photo["shows"])
            out += [i["observation"] for i in photo["issues"]]
            out += photo["strengths"]
        out += self.story["condition"]["cannot_tell"]
        out.append(self.story["condition"]["grade_reason"])
        out.append(self.story["gate"]["subject_evidence"])
        if self.story["merge"]["example"]:
            out.append(self.story["merge"]["example"]["observation"])
        ask = self.story["price"].get("asking")
        if ask:
            out += [ask["summary"], ask["label"]]
        return [c for c in out if c]

    def test_the_claims_on_the_page_are_in_the_json(self):
        blob = " ".join(self.text.chunks)
        missing = [c for c in self.claims() if c not in blob]
        # Two are allowed to be absent: `subject_evidence` is capitalised into
        # the sentence it joins, and `grade_reason` likewise. Both are checked
        # by their own distinctive tail below.
        self.assertLessEqual(len(missing), 2, missing)

    def test_the_two_reshaped_claims_still_appear(self):
        """Two claims are reshaped rather than quoted, and both are lossless.

        `subject_evidence` opens a sentence, so its first letter is capitalised.
        `grade_reason` opens by restating the grade - "good: 6 finding(s), ..." -
        next to a sentence that has already said "Condition good", so the
        restatement is dropped. Neither edit may remove a word of the substance,
        which is what these two assertions are for.
        """
        blob = " ".join(self.text.chunks)
        evidence = self.story["gate"]["subject_evidence"]
        self.assertIn(evidence[1:], blob, evidence)

        reason = self.story["condition"]["grade_reason"]
        grade = self.story["condition"]["grade"]
        self.assertTrue(reason.startswith(f"{grade}: "),
                        f"grade_reason no longer opens with the grade: {reason!r}")
        self.assertIn(reason.split(": ", 1)[1], blob, reason)

    def test_the_identity_witnesses_are_listed_rather_than_summed(self):
        """Act 1 names each independent read, or it names none of them.

        "Two independent reads agree this is a Ford" is checkable; a single
        confidence decimal is not, and collapsing the witnesses into one would
        throw away the only part a reader can verify. The verdict line is also
        the third claim `render()` reshapes: `IdentityVerdict.reason` ends with
        the witness ids it used, which the list directly above has just named
        properly, so the trailing parenthetical is dropped.
        """
        ident = self.story.get("identity") or {}
        if not ident.get("witnesses"):
            self.skipTest("this run produced no identity verdict")
        blob = " ".join(self.text.chunks)
        for w in ident["witnesses"]:
            self.assertIn(w["said"], blob, w["name"])
        reason = re.sub(r"\s*\([^)]*\)\s*$", "", ident["reason"])
        self.assertIn(reason, blob)
        # Dropping the parenthetical may not drop a word of the claim itself.
        self.assertEqual(reason, ident["reason"].split(" (")[0])
        if ident.get("year_evidence"):
            self.assertIn(ident["year_evidence"], blob)

    def test_a_disputed_identity_would_not_be_drawn_as_agreement(self):
        ident = self.story.get("identity") or {}
        if not ident.get("witnesses"):
            self.skipTest("this run produced no identity verdict")
        expected = "agree" if not ident["disagreed"] else "differ"
        self.assertIn(f'class="witness-verdict {expected}"', self.region)

    def test_no_photo_verdict_cites_a_photo_that_is_not_shipped(self):
        shipped = {p["photo_id"] for p in self.story["photos"]}
        for photo in self.story["photos"]:
            self.assertIn(photo["photo_id"], shipped)
            self.assertTrue((STORY / photo["file"]).is_file())

    def test_the_severity_words_are_the_closed_set(self):
        allowed = {"cosmetic", "minor", "moderate", "major"}
        for photo in self.story["photos"]:
            for issue in photo["issues"]:
                self.assertIn(issue["severity"], allowed)
                self.assertIn(f'class="sev-{issue["severity"]}"', self.region)


class TheMeasuredNumberStaysWithItsBand(StoryFixture):
    """The measured 80.3% coverage belongs to the comparable-asking band.

    `tokens.css` and CLAUDE.md are unambiguous about this and the repo has
    already shipped it wrong twice, in opposite directions. On a landing page
    the failure is worse: it would put a measured guarantee on a number that
    carries none, in the largest type on the page.
    """

    def test_the_coverage_figure_belongs_to_the_baseline(self):
        self.assertEqual(self.story["price"]["measured"]["belongs_to"], "baseline")

    def test_the_coverage_figure_is_printed_only_in_the_baseline_row(self):
        coverage = f'{self.story["price"]["measured"]["coverage"]}%'
        rows = re.findall(r'<div class="band-row"[^>]*data-band-row="(\w+)"(.*?)</div>',
                          self.region, re.S)
        self.assertTrue(rows, "the price bands did not render")
        for name, body in rows:
            if name == "baseline":
                self.assertIn(coverage, body)
            else:
                self.assertNotIn(coverage, body, f"measured coverage leaked into {name}")

    def test_the_adjusted_row_disclaims_a_guarantee(self):
        adjusted = re.search(r'data-band-row="adjusted"(.*?)</div>', self.region, re.S)
        self.assertIsNotNone(adjusted)
        self.assertIn("Coverage on this row is unmeasured", adjusted.group(1))

    def test_asking_prices_are_named_as_asking_prices(self):
        self.assertIn("asking prices rather than confirmed sale prices", self.region)

    def test_a_clean_frame_has_no_empty_subtitle(self):
        self.assertNotIn("beat-clean", self.region)
        self.assertNotIn("Nothing wrong found", self.region)
        self.assertNotIn("Nothing flagged", self.region)


class TheFrozenRunHasNotGoneStale(StoryFixture):
    """The band on the front page has to be one the system would still produce.

    A frozen run is a record, and a record is allowed to be dated - the page
    says which run and when. What it is not allowed to be is a band the current
    price model would no longer give, printed in the largest type on the site
    next to the word "measured". Those two look identical from inside the JSON,
    so the fit is stamped and compared.

    Fixing a failure here means re-running the appraisal and re-freezing, which
    costs vision calls. That is the correct price: the alternative is a landing
    page that quietly disagrees with `models/price_model.json`.
    """

    @classmethod
    def shipped(cls):
        import json as _json
        path = REPO / "models" / "price_model.json"
        if not path.is_file():
            return None
        return _json.loads(path.read_text(encoding="utf-8"))

    def test_the_band_came_from_the_price_model_on_disk(self):
        model = self.shipped()
        if not model:
            self.skipTest("no price model is committed")
        fitted = (model.get("meta") or {}).get("fitted_at")
        self.assertEqual(
            self.story["price"]["measured"]["fitted_at"], fitted,
            "the price model has been refit since the story was frozen. Re-run\n"
            "  .venv/bin/python -m app.cli appraise demo/tr_clean --year 2021 "
            "--km 164374 --make Ford --asking 2550000 --market TR "
            "--save data/reference/story_appraisal.json\n"
            "and then scripts/freeze_story.py, or the page shows a band the "
            "current model would not produce.")

    def test_the_coverage_on_the_page_is_the_models_own(self):
        model = self.shipped()
        if not model:
            self.skipTest("no price model is committed")
        level = self.story["price"]["interval_level"]
        measured = (model.get("calibration") or {}).get(f"coverage_{level}")
        if measured is None:
            self.skipTest(f"the model reports no coverage_{level}")
        self.assertAlmostEqual(self.story["price"]["measured"]["coverage"],
                               round(measured * 100, 1), places=1)

    def test_the_page_stamps_the_fit_next_to_the_figure(self):
        # Undated, "80.3% - measured" reads as a present-tense claim about the
        # system rather than about one fit of one model.
        stamp = (self.story["price"]["measured"]["fitted_at"] or "")[:10]
        self.assertTrue(stamp, "the story records no price-model fit date")
        baseline = re.search(r'data-band-row="baseline"(.*?)</div>', self.region, re.S)
        self.assertIsNotNone(baseline)
        self.assertIn(stamp, baseline.group(1))


class TheStaticDocumentIsTheFallback(StoryFixture):
    """The choreography is an enhancement, never the content.

    Every scroll-driven rule in `story.css` is scoped to `.is-driven`, which
    `js/story/index.js` adds only after it has measured the acts. If that scope
    ever leaks, a reader with reduced motion, a narrow window or no JavaScript
    gets a stack of absolutely-positioned photographs on top of each other.
    """

    CSS = (WEB / "styles" / "story.css").read_text(encoding="utf-8")

    def test_the_pinning_rules_are_scoped_to_is_driven(self):
        for rule in ("position: sticky", "height: 100svh"):
            for line in [l for l in self.CSS.splitlines() if rule in l]:
                block = self.CSS[:self.CSS.index(line)].rsplit("}", 1)[-1]
                self.assertIn("is-driven", block, f"{rule} escaped .is-driven")

    def media_block(self, query):
        """The body of one `@media` rule, brace-matched rather than sliced."""
        at = f"@media ({query})"
        start = self.CSS.index(at) + len(at)
        open_brace = self.CSS.index("{", start)
        depth, i = 1, open_brace + 1
        while depth and i < len(self.CSS):
            depth += {"{": 1, "}": -1}.get(self.CSS[i], 0)
            i += 1
        return self.CSS[open_brace + 1:i - 1]

    def test_both_fallbacks_unpin_the_stage(self):
        for query in ("prefers-reduced-motion: reduce", "max-width: 819px"):
            block = self.media_block(query)
            self.assertIn("position: static", block, query)
            self.assertIn("height: auto", block, query)
            # The absolutely-positioned layers have to come back into flow too,
            # or the six photographs sit on top of each other.
            self.assertIn(".beat", block, query)

    def test_the_js_modules_it_imports_exist(self):
        js = WEB / "js" / "story"
        for module in js.glob("*.js"):
            for spec in re.findall(r"from\s+'([^']+)'", module.read_text(encoding="utf-8")):
                target = (module.parent / spec).resolve()
                self.assertTrue(target.is_file(), f"{module.name} imports missing {spec}")

    def test_the_story_never_reaches_the_network(self):
        for module in (WEB / "js" / "story").glob("*.js"):
            src = module.read_text(encoding="utf-8")
            for banned in ("fetch(", "XMLHttpRequest", "//cdn", "https://"):
                self.assertNotIn(banned, src, f"{module.name} contains {banned}")


class TheOdometerCrossCheckIsReproducible(StoryFixture):
    """Act 4 claims two independent readings of the dashboard agree.

    That claim is only worth printing if it can be re-taken, so this re-runs
    the OCR against the shipped photograph. It skips rather than fails when
    RapidOCR is absent, which is the same soft dependency `reconcile` has.
    """

    def test_the_ocr_still_reads_what_the_page_claims(self):
        odo = self.story["cross_checks"]["odometer"]
        if not odo["ocr"].get("available"):
            self.skipTest("no OCR reading was frozen")
        try:
            from app import odometer
        except ImportError:
            self.skipTest("RapidOCR is not installed")
        photo = next((p for p in self.story["photos"]
                      if p["odometer_km"] is not None), None)
        self.assertIsNotNone(photo, "act 4 claims an odometer but ships no dashboard")
        again = odometer.read(str(STORY / photo["file"]))
        self.assertEqual(again.km, odo["ocr"]["km"])

    def test_three_readings_agree_or_the_page_says_they_do_not(self):
        odo = self.story["cross_checks"]["odometer"]
        if odo["agree"]:
            self.assertEqual(odo["declared"], odo["vlm"])
            self.assertEqual(odo["vlm"], odo["ocr"]["km"])
            self.assertIn("All three agree", self.region)
        else:
            self.assertIn("readings differ", self.region)


class RegeneratingTheRegionChangesNothing(StoryFixture):
    """The end-to-end check: `--check` exits 0 against the committed page."""

    def test_freeze_story_check_passes(self):
        source = REPO / "data" / "reference" / "story_appraisal.json"
        if not source.is_file():
            self.skipTest("the source appraisal is not committed")
        # freeze_story.py defaults to demo/tr_clean, which `--build` produces and
        # .gitignore keeps out of the repo; on a checkout without it, --check
        # exits on "no such photo folder" before it can compare anything.
        if not (REPO / "demo" / "tr_clean").is_dir():
            self.skipTest("demo fixtures not built (python -m app.demo --build)")
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "freeze_story.py"),
             str(source), "--check"],
            capture_output=True, text=True, cwd=REPO)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
