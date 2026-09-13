"""Offline regression tests. No API calls, no network, ~2 s.

`python -m app.demo` is the real end-to-end check, but it spends a vision call
per case, so it is not something to run on every edit. These cover the pure
logic underneath it - the parts that have actually broken during this build:

  * JSON extraction from a model response (fenced, trailing prose, truncated)
  * dropping evidence that cites a photo which was never sent
  * the price interval's ordering and currency conversion
  * the condition adjustment staying inside its measured cap
  * capture metrics agreeing with the thresholds they were calibrated against

Run: .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import base64
import json
import re
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from app import evidence, gate, report, subject, vision
from app.config import USD_TRY
from app.schema import (Appraisal, Detection, EvidenceReport, GateDecision,
                        GateReport, Issue, PhotoCheck)


def _check(photo_id: int, name: str = "x.jpg") -> PhotoCheck:
    return PhotoCheck(photo_id=photo_id, path=f"/tmp/{name}", filename=name, usable=True)


class ExtractJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(evidence.extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(evidence.extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_trailing_prose(self):
        # Models sometimes append a sentence after the object.
        self.assertEqual(evidence.extract_json('{"a": 1}\nHope this helps!'), {"a": 1})

    def test_brace_inside_string(self):
        # The brace counter must not be fooled by a brace in a string value,
        # which is common in observations quoting part numbers.
        self.assertEqual(evidence.extract_json('{"a": "} not the end"}'),
                         {"a": "} not the end"})

    def test_truncated_raises(self):
        with self.assertRaises(ValueError):
            evidence.extract_json('{"a": 1, "b": ')


class ParseIdentity(unittest.TestCase):
    """Pass A answers what the truck is and whether it is one truck."""

    def payload(self, **over):
        base = {"make": "Ford", "model": "F-MAX", "body_type": "tractor_unit",
                "cab_type": "high sleeper", "axle_config": "4x2",
                "approx_year_range": "2019-2023", "badges_seen": ["F-MAX"],
                "confidence": 0.9, "same_vehicle": True, "vehicle_mismatch": ""}
        base.update(over)
        return json.dumps(base)

    def test_reads_the_vehicle(self):
        v, same, mismatch = evidence.parse_identity(self.payload())
        self.assertEqual((v.make, v.model), ("Ford", "F-MAX"))
        self.assertTrue(same)
        self.assertEqual(mismatch, "")

    def test_a_mismatch_is_carried_not_swallowed(self):
        _, same, mismatch = evidence.parse_identity(
            self.payload(same_vehicle=False, vehicle_mismatch="two different plates"))
        self.assertFalse(same)
        self.assertIn("plates", mismatch)

    def test_blank_strings_become_none(self):
        v, _, _ = evidence.parse_identity(self.payload(make="", model=None))
        self.assertIsNone(v.make)
        self.assertIsNone(v.model)


class ParseCloseup(unittest.TestCase):
    """Pass B is one call per photo, which is what makes the citation structural."""

    def setUp(self):
        self.check = _check(7, "a.jpg")
        self.check.view = "tire_wheel"

    def payload(self, **over):
        base = {"shows": "the near-side steer tire", "legible": True,
                "odometer_km": None,
                "observations": [{"component": "steer_tires",
                                  "observation": "worn to the bars on the outer shoulder",
                                  "severity": "moderate", "confidence": 0.8,
                                  "price_impact": "medium"}],
                "strengths": ["rim is straight and free of kerbing"],
                "cannot_tell": ["tread depth in millimetres"], "confidence": 0.7}
        base.update(over)
        return json.dumps(base)

    def test_photo_id_comes_from_the_caller_not_the_model(self):
        # The whole point of the fan-out: the call was given exactly one photo,
        # so a claim cannot cite a frame that was never sent. There is no index
        # for the model to get wrong.
        f = evidence.parse_closeup(self.payload(photo_id=999), self.check, cropped=False)
        self.assertEqual(f.photo_id, 7)
        self.assertEqual(f.issues[0].photo_id, 7)

    def test_unknown_component_is_kept_but_not_laundered(self):
        f = evidence.parse_closeup(self.payload(observations=[
            {"component": "flux_capacitor", "observation": "hmm", "severity": "minor",
             "confidence": 0.5, "price_impact": "low"}]), self.check, cropped=False)
        self.assertEqual(f.issues[0].component, "flux_capacitor")
        self.assertNotIn("flux_capacitor", evidence.COMPONENTS)

    def test_bad_enum_values_round_down_and_say_so(self):
        """An unparseable severity is not a minor defect.

        Every default here used to round UP - severity to `minor`, impact to
        `low`, confidence to 0.5 - so a field nobody could read was worth real
        money. The honest state is "the model did not grade this finding": it
        is kept, shown with its photograph, and weighted at zero.
        """
        f = evidence.parse_closeup(self.payload(observations=[
            {"component": "steer_tires", "observation": "x", "severity": "catastrophic",
             "confidence": "nope", "price_impact": "ruinous"}]), self.check, cropped=False)
        issue = f.issues[0]
        self.assertIn(issue.severity, evidence.SEVERITIES)
        self.assertIn(issue.price_impact, evidence.IMPACTS)
        self.assertEqual(issue.severity, "cosmetic")
        self.assertEqual(issue.price_impact, "none")
        self.assertTrue(issue.ungraded)
        self.assertEqual(issue.confidence, 0.3)
        from app import condition
        self.assertEqual(condition.demerit(issue), 0.0)

    def test_an_observation_with_no_text_is_not_a_finding(self):
        f = evidence.parse_closeup(self.payload(observations=[
            {"component": "steer_tires", "observation": "   ", "severity": "minor",
             "confidence": 0.5, "price_impact": "low"}]), self.check, cropped=False)
        self.assertEqual(f.issues, [])

    def test_strengths_and_gaps_survive(self):
        f = evidence.parse_closeup(self.payload(), self.check, cropped=False)
        self.assertEqual(len(f.strengths), 1)
        self.assertEqual(len(f.cannot_tell), 1)

    def test_an_implausible_odometer_is_dropped(self):
        # A guessed mileage is worse than none: it is cross-checked against what
        # the seller typed, and a bad read fabricates a contradiction.
        for bad in (0, -5, 99_000_000, "lots", None):
            f = evidence.parse_closeup(self.payload(odometer_km=bad), self.check,
                                       cropped=False)
            self.assertIsNone(f.odometer_km, bad)

    def test_a_real_odometer_is_kept(self):
        f = evidence.parse_closeup(self.payload(odometer_km=164374), self.check,
                                   cropped=False)
        self.assertEqual(f.odometer_km, 164374)

    def parsed_box(self, box, *, cropped=False):
        f = evidence.parse_closeup(self.payload(observations=[{
            "component": "steer_tires", "observation": "worn",
            "severity": "minor", "confidence": 0.5, "price_impact": "low",
            "box": box,
        }]), self.check, cropped=cropped)
        return f.issues[0].box

    def test_a_valid_box_is_kept(self):
        # The screen draws this on the photo the finding cites. A claim that
        # cannot point at pixels is still a finding; it just opens unmarked.
        self.assertEqual(self.parsed_box([0.2, 0.3, 0.4, 0.25]),
                         [0.2, 0.3, 0.4, 0.25])

    def test_a_missing_or_junk_box_is_dropped(self):
        for junk in (None, [], [0.1, 0.2], [0, 0, 0, 0],
                     [1.5, 0.1, 0.2, 0.2], [0.1, 0.1, -0.2, 0.2],
                     "nope", [0.1, 0.1, 0.2, "wide"]):
            self.assertIsNone(self.parsed_box(junk), junk)

    def test_a_cropped_box_is_mapped_onto_the_original(self):
        # The model saw the padded subject crop; the lightbox shows the listing
        # photo. The box has to land on the same pixels, not on the crop.
        self.check.width, self.check.height = 1000, 800
        self.check.subject_box = [200, 100, 700, 500]
        # 8% pad on a 500×400 box → crop (160, 68, 740, 532) in a 1000×800 frame.
        self.assertEqual([round(v, 6) for v in self.parsed_box([0.0, 0.0, 1.0, 1.0],
                                                              cropped=True)],
                         [0.16, 0.085, 0.58, 0.58])

    def test_an_uncropped_box_is_not_remapped(self):
        self.check.width, self.check.height = 1000, 800
        self.check.subject_box = [200, 100, 700, 500]
        self.assertEqual(self.parsed_box([0.1, 0.2, 0.3, 0.4]),
                         [0.1, 0.2, 0.3, 0.4])


class MergeDuplicates(unittest.TestCase):
    """One worn tire seen in three frames is one finding with three photos."""

    def _issues(self, n):
        return [Issue(photo_id=i, component="steer_tires", observation=f"o{i}",
                      severity="minor", confidence=0.5) for i in range(n)]

    def test_merged_findings_become_corroboration(self):
        flat = self._issues(3)
        out = evidence.merge_duplicates(flat, [{"keep": 0, "merge": [1, 2]}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].photo_id, 0)
        self.assertEqual(sorted(out[0].also_seen_in), [1, 2])

    def test_nothing_is_deleted_without_being_recorded(self):
        # Same posture as fell_back_from and Correction: changing your mind is
        # allowed, doing it where nobody can see it is not.
        flat = self._issues(3)
        out = evidence.merge_duplicates(flat, [{"keep": 0, "merge": [1, 2]}])
        cited = {out[0].photo_id, *out[0].also_seen_in}
        self.assertEqual(cited, {0, 1, 2})

    def test_out_of_range_indices_are_ignored(self):
        flat = self._issues(2)
        out = evidence.merge_duplicates(flat, [{"keep": 9, "merge": [0]},
                                               {"keep": 0, "merge": [99]}])
        self.assertEqual(len(out), 2)

    def test_garbage_does_not_lose_findings(self):
        flat = self._issues(2)
        out = evidence.merge_duplicates(flat, ["nonsense", {"merge": [1]}, None])
        self.assertEqual(len(out), 2)

    def test_no_duplicates_leaves_every_finding_alone(self):
        flat = self._issues(4)
        out = evidence.merge_duplicates(flat, [])
        self.assertEqual(len(out), 4)
        self.assertTrue(all(not i.also_seen_in for i in out))


class ParseSynthesis(unittest.TestCase):
    def test_missing_summary_keys_are_filled_not_omitted(self):
        data = evidence.parse_synthesis(json.dumps({"condition_summary": {"tires": "ok"}}))
        self.assertEqual(set(data["condition_summary"]), set(evidence.SUMMARY_KEYS))
        self.assertEqual(data["condition_summary"]["engine_driveline"],
                         "not visible in these photos")

    def test_an_invented_grade_falls_back_to_unknown(self):
        data = evidence.parse_synthesis(json.dumps({"condition_grade": "immaculate"}))
        self.assertEqual(data["condition_grade"], "unknown")


class JsonSchema(unittest.TestCase):
    def test_strict_shape(self):
        """Strict mode requires every property listed in `required`."""
        closeup = evidence.closeup_schema()
        nodes = [closeup, closeup["properties"]["observations"]["items"],
                 evidence.synthesis_schema(),
                 evidence.synthesis_schema()["properties"]["condition_summary"]]
        from app.evidence.prompts import IDENTITY_SCHEMA
        nodes.append(IDENTITY_SCHEMA)
        for node in nodes:
            self.assertFalse(node["additionalProperties"])
            self.assertEqual(set(node["required"]), set(node["properties"]))

    def test_enums_match_the_parser(self):
        item = evidence.closeup_schema()["properties"]["observations"]["items"]
        obs = item["properties"]
        self.assertEqual(obs["component"]["enum"], evidence.COMPONENTS)
        self.assertEqual(obs["severity"]["enum"], evidence.SEVERITIES)
        self.assertEqual(obs["price_impact"]["enum"], evidence.IMPACTS)
        # Strict structured output requires every property. Null is how a
        # close-up that cannot point at pixels still returns a legal object.
        self.assertEqual(obs["box"]["type"], ["array", "null"])
        self.assertIn("box", item["required"])

    def test_every_component_rolls_up_into_a_summary_key(self):
        # The deterministic fallback used when the synthesis call fails groups
        # findings through this map. A component missing from it would silently
        # vanish from the summary rather than error.
        for component in evidence.COMPONENTS:
            self.assertIn(component, evidence.COMPONENT_SUMMARY)
            self.assertIn(evidence.COMPONENT_SUMMARY[component], evidence.SUMMARY_KEYS)

    def test_every_canonical_view_has_its_own_checklist(self):
        # A view with no checklist falls back to four generic questions, which
        # is exactly the shallow prompt the fan-out exists to replace.
        from app import vision
        for view in vision.VIEW_LABELS:
            self.assertIn(view, evidence.VIEW_QUESTIONS, view)
            self.assertGreaterEqual(len(evidence.questions_for(view)), 4)


def _module_strings(module):
    """Every string constant a module holds, however deeply nested."""
    def walk(name, node):
        if isinstance(node, str):
            yield name, node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield from walk(f"{name}[{key!r}]", value)
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                yield from walk(f"{name}[{i}]", value)

    for attr, value in vars(module).items():
        if not attr.startswith("__"):
            yield from walk(attr, value)


class SeverityRubric(unittest.TestCase):
    """What the four words in `SEVERITIES` are allowed to mean.

    They feed a weight table that runs 147x from end to end, and until the
    rubric landed nothing in the repo said what any of them was - so the same
    worn tire could come back "minor" or "major" depending on the light. These
    are the checks that keep the rubric honest, and the last one is the price
    invariant: the levels are anchored on repair effort so that no currency
    figure ever has to reach a model.
    """

    @classmethod
    def setUpClass(cls):
        from app import vision
        from app.evidence import prompts
        cls.P = prompts
        cls.views = set(vision.VIEW_LABELS)

    def test_every_canonical_view_has_its_own_anchors(self):
        # Same posture as the checklists: a view with no anchors is a view
        # graded against nothing, and an anchor for a view that does not exist
        # is a typo that would never be injected.
        self.assertEqual(set(self.P.COMPONENT_ANCHORS), self.views)
        for view, block in self.P.COMPONENT_ANCHORS.items():
            for level in self.P.SEVERITIES:
                self.assertIn(level, block, f"{view} anchors skip {level}")

    def test_every_canonical_view_says_what_to_confirm(self):
        self.assertEqual(set(self.P.VIEW_CONFIRMATIONS), self.views)
        for view, items in self.P.VIEW_CONFIRMATIONS.items():
            self.assertGreaterEqual(len(items), 3, view)

    def test_every_canonical_view_has_an_expected_wear_row_per_band(self):
        self.assertEqual(set(self.P.EXPECTED_WEAR), self.views)
        for view, bands in self.P.EXPECTED_WEAR.items():
            self.assertEqual(set(bands), set(self.P.WEAR_BAND_IDS), view)
            for band, rows in bands.items():
                self.assertTrue(rows and all(rows), f"{view}/{band}")

    def test_the_wear_band_is_chosen_by_distance(self):
        self.assertIsNone(self.P.wear_band(None))
        self.assertIsNone(self.P.wear_band(0))
        self.assertEqual(self.P.wear_band(80_000), "low")
        self.assertEqual(self.P.wear_band(149_999), "low")
        self.assertEqual(self.P.wear_band(150_000), "mid")
        self.assertEqual(self.P.wear_band(799_999), "high")
        self.assertEqual(self.P.wear_band(1_200_000), "very_high")

    def test_every_component_named_in_an_anchor_exists(self):
        # The only machine-readable ids in an anchor are component ids - schema
        # field names are quoted, as they are everywhere else in prompts.py. A
        # misspelt id would send the close-up call after an enum value that the
        # parser then drops.
        for view, block in self.P.COMPONENT_ANCHORS.items():
            prose = re.sub(r'"[^"]*"', " ", block)
            named = set(re.findall(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", prose))
            self.assertTrue(named, f"{view} anchors name no component")
            self.assertLessEqual(named, set(self.P.COMPONENTS), view)

    def test_the_anchors_cover_every_component(self):
        # A component with no anchor can still be reported, and would then be
        # graded against nothing at all. The families partition COMPONENTS.
        covered = [c for ids in self.P.ANCHOR_COMPONENTS.values() for c in ids]
        self.assertEqual(sorted(covered), sorted(self.P.COMPONENTS))

    def test_the_rubric_defines_the_four_levels_and_no_others(self):
        levels = re.findall(r"^([a-z_]+) {2,}", self.P.SEVERITY_RUBRIC, re.M)
        self.assertEqual(set(levels), set(self.P.SEVERITIES))

    def test_two_of_the_worked_examples_are_not_findings(self):
        # The failure this rubric exists to fix is not an invented defect. It
        # is a real observation promoted a level to make it worth writing down,
        # so the examples have to show the model declining to do that.
        verdicts = re.findall(r"->\s*(not an observation|cosmetic|minor|moderate|major)",
                              self.P.WORKED_EXAMPLES)
        self.assertEqual(len(verdicts), 6)
        null = [v for v in verdicts if v in ("not an observation", "cosmetic")]
        self.assertGreaterEqual(len(null), 2, verdicts)

    def test_no_prompt_anywhere_carries_a_repair_price(self):
        # `config.REPAIR_BANDS` is what the four levels cost in lira, and it is
        # for the README and the panel card only. The rubric anchors severity
        # on repair effort - a workshop morning, a component replacement - so
        # that teaching the model what a severity means never puts a currency
        # figure in front of it, and "the VLM never sees or emits a price"
        # stays whole.
        from app.config import REPAIR_BANDS
        figures = {n for band in REPAIR_BANDS.values() for n in band if n}
        priced = {f"{n:,}" for n in figures} | {str(n) for n in figures}
        for name, text in _module_strings(self.P):
            # Reported by name rather than by assertNotIn, which would dump a
            # whole prompt into the failure.
            for figure in sorted(priced):
                if figure in text:
                    self.fail(f"{name} quotes the repair band figure {figure}")
            # Case-sensitive on the currency codes on purpose: \bTRY\b under
            # re.I matches the verb "try", which a prompt is allowed to say.
            money = (re.search(r"[₺$€£]|\bTRY\b|\bUSD\b|\bTL\b", text)
                     or re.search(r"\blira\b", text, re.I))
            if money:
                self.fail(f"{name} carries a currency: {money.group(0)!r}")


class Pricing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.pricing import load_model
        cls.model = load_model()

    def estimate(self, **kw):
        from app.pricing import model as M
        params = dict(year=2021, km=300000, make="FORD", market="TR", evidence=None)
        params.update(kw)
        return M.estimate(self.model, **params)

    def test_band_is_ordered_and_contains_the_point(self):
        e = self.estimate()
        self.assertTrue(e.ok)
        self.assertLess(e.low, e.point)
        self.assertLess(e.point, e.high)

    def test_more_km_is_worth_less(self):
        self.assertLess(self.estimate(km=900000).point, self.estimate(km=150000).point)

    def test_older_is_worth_less(self):
        self.assertLess(self.estimate(year=2013).point, self.estimate(year=2023).point)

    def test_currency_conversion_is_the_stamped_rate(self):
        e = self.estimate()
        self.assertAlmostEqual(e.point_usd, e.point / USD_TRY, delta=e.point_usd * 0.01)

    def test_prediction_is_anchored_to_the_real_market(self):
        """Order-of-magnitude guard, in absolute terms.

        The ratio test above is internally consistent even when both numbers
        are wrong by the same factor, and that is exactly what happened:
        switching the model target to native currency left `estimate` still
        multiplying by the TRY rate, and a 2021 F-MAX came out at 118,000,000
        TRY instead of 2,435,000 - a 48x error that every relative assertion
        in this file sailed past. Real 2021 F-MAX listings in the corpus ask
        2.35-2.55M TRY.
        """
        e = self.estimate(year=2021, km=164374)
        self.assertGreater(e.point, 1_000_000)
        self.assertLess(e.point, 6_000_000)
        # And the USD view has to land in a plausible band for a used tractor.
        self.assertGreater(e.point_usd, 10_000)
        self.assertLess(e.point_usd, 200_000)

    def test_band_brackets_the_real_asking_price_of_a_known_spec(self):
        """The comparable band for a listing IN the corpus should contain it."""
        e = self.estimate(year=2021, km=164374)
        self.assertLessEqual(e.baseline_low, 2_550_000)
        self.assertGreaterEqual(e.baseline_high, 2_550_000)

    def test_missing_inputs_decline_rather_than_guess(self):
        for kw in ({"year": None}, {"km": None}):
            with self.subTest(**kw):
                e = self.estimate(**kw)
                self.assertFalse(e.ok)
                self.assertEqual(e.point, 0)

    def test_unknown_brand_widens_the_band(self):
        known = self.estimate(make="FORD")
        unseen = self.estimate(make="SCANIA")
        self.assertGreater(unseen.high - unseen.low, known.high - known.low)
        self.assertTrue(unseen.widened)

    def test_condition_adjustment_respects_its_cap(self):
        """Even an implausible pile of major defects cannot exceed the cap."""
        from app.pricing import model as M
        ev = EvidenceReport(issues=[
            Issue(photo_id=0, component="chassis_frame", observation="x",
                  severity="major", confidence=1.0, price_impact="high")] * 50)
        adj = M.condition_adjustment(ev, self.model.residual_std)
        # `multiplier` is rounded to 4dp for display, so the comparison has to
        # allow half a unit in that last place.
        floor = np.exp(-self.model.residual_std) - 5e-5
        self.assertGreaterEqual(adj.multiplier, floor)
        self.assertLessEqual(adj.pct, 0.0)
        self.assertGreater(adj.cap_pct, 0.0)

    def test_no_issues_means_no_adjustment(self):
        from app.pricing import model as M
        adj = M.condition_adjustment(EvidenceReport(), self.model.residual_std)
        self.assertEqual(adj.multiplier, 1.0)

    def test_baseline_band_is_reported_separately(self):
        """The measured coverage belongs to the asking band, so it must survive."""
        from app.pricing import model as M
        ev = EvidenceReport(issues=[
            Issue(photo_id=0, component="drive_tires", observation="x",
                  severity="moderate", confidence=0.9, price_impact="medium")] * 4)
        e = M.estimate(self.model, year=2021, km=300000, make="FORD",
                       market="TR", evidence=ev)
        self.assertNotEqual(e.baseline_point, e.point)
        self.assertLess(e.baseline_low, e.baseline_high)


class AskingPriceVerdict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.pricing import load_model
        cls.model = load_model()

    def estimate(self, asking):
        from app.pricing import model as M
        return M.estimate(self.model, year=2021, km=300000, make="FORD",
                          market="TR", evidence=None, asking_price=asking)

    def test_inside_the_comparable_band_reads_as_market_rate(self):
        e = self.estimate(1)          # get the band first
        mid = (e.baseline_low + e.baseline_high) / 2
        v = self.estimate(mid).asking
        self.assertTrue(v.inside_comparable_band)
        self.assertEqual(v.label, "in line with the market")

    def test_above_and_below_are_distinguished(self):
        e = self.estimate(1)
        self.assertEqual(self.estimate(e.baseline_high * 1.4).asking.label, "above the market")
        self.assertEqual(self.estimate(e.baseline_low * 0.6).asking.label, "below the market")

    def test_percentages_have_the_right_sign(self):
        e = self.estimate(1)
        high = self.estimate(e.baseline_high * 1.4).asking
        self.assertGreater(high.vs_comparables_pct, 0)
        low = self.estimate(e.baseline_low * 0.6).asking
        self.assertLess(low.vs_comparables_pct, 0)

    def test_absent_when_not_supplied(self):
        from app.pricing import model as M
        e = M.estimate(self.model, year=2021, km=300000, make="FORD",
                       market="TR", evidence=None)
        self.assertIsNone(e.asking)


class ImageFormats(unittest.TestCase):
    def test_heic_is_accepted_when_the_decoder_is_installed(self):
        from app import config
        if not config.HEIF_SUPPORT:
            self.skipTest("pillow-heif not installed")
        self.assertIn(".heic", config.IMAGE_SUFFIXES)

    def test_cli_and_web_accept_the_same_set(self):
        """A file the folder walk collects must not be rejected by the upload."""
        from app import config, server
        self.assertEqual(set(server.ALLOWED), set(config.IMAGE_SUFFIXES))


class CaptureMetrics(unittest.TestCase):
    def test_blur_lowers_the_variance_below_the_floor(self):
        import cv2
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(sharp, (61, 61), 0)
        floor = gate.thresholds()["capture"]["blur_laplacian_var_floor"]
        self.assertGreater(gate.capture_metrics(sharp)["blur_laplacian_var"], floor)
        self.assertLess(gate.capture_metrics(blurred)["blur_laplacian_var"], floor)

    def test_black_frame_is_below_the_exposure_floor(self):
        black = np.zeros((400, 600, 3), dtype=np.uint8)
        m = gate.capture_metrics(black)
        self.assertLess(m["brightness"], gate.thresholds()["capture"]["brightness_floor"])

    def test_quality_score_is_bounded(self):
        rng = np.random.default_rng(1)
        for img in (np.zeros((400, 600, 3), np.uint8),
                    np.full((400, 600, 3), 255, np.uint8),
                    rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)):
            s = gate.quality_score(gate.capture_metrics(img), 600, 400)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class PricingBlockers(unittest.TestCase):
    """Conditions where the comparables cannot honestly price what is shown."""

    def make(self, **kw):
        from app.schema import VehicleRead
        ev = EvidenceReport()
        ev.vehicle = VehicleRead(body_type=kw.pop("body_type", "tractor_unit"),
                                 confidence=kw.pop("confidence", 0.9))
        for k, v in kw.items():
            setattr(ev, k, v)
        return ev

    def test_tractor_unit_is_priceable(self):
        from app.pipeline import pricing_blocker
        self.assertIsNone(pricing_blocker(self.make()))

    def test_unstated_body_type_is_not_blocked(self):
        """A missing read is not evidence of the wrong vehicle."""
        from app.pipeline import pricing_blocker
        self.assertIsNone(pricing_blocker(self.make(body_type=None)))
        self.assertIsNone(pricing_blocker(self.make(body_type="")))

    def test_rigid_blocks_pricing(self):
        from app.pipeline import pricing_blocker
        blocked = pricing_blocker(self.make(body_type="rigid"))
        self.assertIsNotNone(blocked)
        self.assertIn("rigid", blocked[0])

    def test_low_confidence_body_read_does_not_block(self):
        """'I don't know what this is' is different from 'I know it's a rigid'."""
        from app.pipeline import pricing_blocker
        self.assertIsNone(pricing_blocker(self.make(body_type="rigid", confidence=0.05)))

    def test_mixed_vehicles_block_and_quote_the_mismatch(self):
        from app.pipeline import pricing_blocker
        blocked = pricing_blocker(self.make(same_vehicle=False,
                                            vehicle_mismatch="plates differ"))
        self.assertIsNotNone(blocked)
        self.assertIn("plates differ", blocked[0])

    def test_mixed_vehicles_outrank_body_type(self):
        """If it is not even one vehicle, body type is the lesser problem."""
        from app.pipeline import pricing_blocker
        blocked = pricing_blocker(self.make(body_type="rigid", same_vehicle=False))
        self.assertIn("not all of the same truck", blocked[0])

    def test_none_evidence_is_safe(self):
        from app.pipeline import pricing_blocker
        self.assertIsNone(pricing_blocker(None))

    def test_gate_report_records_subject_clusters(self):
        g = GateReport(subject_clusters=2)
        self.assertEqual(g.to_dict()["subject_clusters"], 2)

    def test_clip_clusters_block_pricing_when_calibrated(self):
        from app.pipeline import pricing_blocker
        gate_rep = GateReport(subject_clusters=3)
        blocked = pricing_blocker(self.make(), gate=gate_rep, min_clusters=2)
        self.assertIsNotNone(blocked)
        self.assertIn("more than one truck", blocked[0].lower())

    def test_clip_clusters_do_not_block_below_threshold(self):
        from app.pipeline import pricing_blocker
        gate_rep = GateReport(subject_clusters=1)
        self.assertIsNone(pricing_blocker(self.make(), gate=gate_rep, min_clusters=2))

    def test_missing_threshold_does_not_block(self):
        from app.pipeline import pricing_blocker
        gate_rep = GateReport(subject_clusters=9)
        self.assertIsNone(pricing_blocker(self.make(), gate=gate_rep, min_clusters=None))

    def test_measured_artifact_blocks_at_three_clusters(self):
        from app.pipeline import mixed_cluster_threshold, pricing_blocker
        self.assertEqual(mixed_cluster_threshold(), 3)
        blocked = pricing_blocker(self.make(), gate=GateReport(subject_clusters=3))
        self.assertIsNotNone(blocked)
        self.assertIsNone(pricing_blocker(
            self.make(), gate=GateReport(subject_clusters=2)))

    def test_confident_rigid_tag_blocks_pricing(self):
        from app.pipeline import BODY_TYPE_GATE_CONF, pricing_blocker
        gate_rep = GateReport(body_tag="rigid", body_tag_conf=0.80)
        blocked = pricing_blocker(self.make(body_type=None, confidence=0.0),
                                  gate=gate_rep, min_rigid_conf=BODY_TYPE_GATE_CONF)
        self.assertIsNotNone(blocked)
        self.assertIn("rigid", blocked[0].lower())

    def test_unmeasured_rigid_tag_does_not_block(self):
        from app.pipeline import pricing_blocker
        gate_rep = GateReport(body_tag="rigid", body_tag_conf=0.99)
        self.assertIsNone(pricing_blocker(
            self.make(body_type=None, confidence=0.0),
            gate=gate_rep, min_rigid_conf=None))

    def test_low_confidence_rigid_tag_does_not_block(self):
        from app.pipeline import BODY_TYPE_GATE_CONF, pricing_blocker
        gate_rep = GateReport(body_tag="rigid", body_tag_conf=0.20)
        self.assertIsNone(pricing_blocker(
            self.make(body_type=None, confidence=0.0),
            gate=gate_rep, min_rigid_conf=BODY_TYPE_GATE_CONF))

    def test_tractor_tag_does_not_block(self):
        from app.pipeline import pricing_blocker
        gate_rep = GateReport(body_tag="tractor_unit", body_tag_conf=0.99)
        self.assertIsNone(pricing_blocker(self.make(), gate=gate_rep))


class SeedClusters(unittest.TestCase):
    """Mixed-listing CLIP count uses whole-vehicle seeds, not every winner."""

    def _cand(self, photo_id, emb):
        from app.subject import Candidate
        return Candidate(
            photo_id=photo_id,
            det=Detection(label="truck", confidence=0.9, box=[0, 0, 10, 10],
                          area_frac=0.3),
            label="truck", confidence=0.9, box=[0, 0, 10, 10],
            area_frac=0.3, centre_hit=True, far=0.1, clipped_sides=0,
            min_side_px=100, emb=emb)

    def _check(self, photo_id, view):
        c = _check(photo_id)
        c.view = view
        return c

    def test_two_dissimilar_seed_exteriors_are_two_clusters(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        n = subject.cluster_seed_winners(
            [self._cand(1, a), self._cand(2, b)],
            [self._check(1, "exterior_front"), self._check(2, "exterior_side")],
            seeded_from=[1, 2])
        self.assertEqual(n, 2)

    def test_similar_seed_exteriors_are_one_cluster(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.99, 0.01], dtype=np.float32)
        b = b / np.linalg.norm(b)
        n = subject.cluster_seed_winners(
            [self._cand(1, a), self._cand(2, b)],
            [self._check(1, "exterior_front"), self._check(2, "exterior_side")],
            seeded_from=[1, 2])
        self.assertEqual(n, 1)

    def test_tire_winners_do_not_count_as_a_second_vehicle(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        n = subject.cluster_seed_winners(
            [self._cand(1, a), self._cand(2, b)],
            [self._check(1, "exterior_front"), self._check(2, "tire_wheel")],
            seeded_from=[1, 2])
        self.assertEqual(n, 1)


class MixedVehicleCalibration(unittest.TestCase):
    """Threshold arithmetic for the mixed-listing / rigid-tag artifacts."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_mixed_vehicle.py"
        spec = importlib.util.spec_from_file_location("calibrate_mixed_vehicle", path)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_smallest_k_inside_half_percent_budget(self):
        # One known-single at 3 clusters is 0.5% of 200, so k=2 is admissible.
        flag, fp = self.mod.choose_threshold([1] * 199 + [3])
        self.assertEqual(flag, 2)
        self.assertEqual(fp, 0.005)

    def test_threshold_sits_above_the_highest_known_single_when_needed(self):
        # Ten listings all split once: k=2 fails the budget, k=3 never fires.
        flag, fp = self.mod.choose_threshold([2] * 10)
        self.assertEqual(flag, 3)
        self.assertEqual(fp, 0.0)

    def test_body_tag_stays_display_only_when_corpus_overlaps_demo(self):
        body = self.mod.summarise_body_tags(
            [{"body_tag": "rigid", "body_tag_conf": 0.90}],
            {"rigid_truck": {"body_tag": "rigid", "body_tag_conf": 0.80}})
        self.assertFalse(body["blocks"])
        self.assertIsNone(body["block_conf"])

    def test_body_tag_blocks_only_above_every_corpus_rigid(self):
        body = self.mod.summarise_body_tags(
            [{"body_tag": "tractor_unit", "body_tag_conf": 0.70},
             {"body_tag": "rigid", "body_tag_conf": 0.60}],
            {"rigid_truck": {"body_tag": "rigid", "body_tag_conf": 0.85}})
        self.assertTrue(body["blocks"])
        self.assertGreater(body["block_conf"], 0.60)
        self.assertLessEqual(body["block_conf"], 0.85)


class BackendChain(unittest.TestCase):
    """Ordering logic only, against stub backends.

    Deliberately does not probe the real providers: `Cursor.me()` is a network
    call, which made this flaky and made an "offline" suite take three times as
    long for no extra coverage.
    """

    def setUp(self):
        from app import vlm
        from app.vlm.base import BackendStatus, VLMBackend

        class Stub(VLMBackend):
            ready = True

            def probe(self):
                return BackendStatus(self.name, self.ready, "stub")

        self.vlm = vlm
        self.saved = dict(vlm._REGISTRY)
        vlm._REGISTRY.clear()
        for name in ("alpha", "beta", "gamma"):
            vlm._REGISTRY[name] = type(f"{name}Stub", (Stub,), {"name": name})
        self.chain_patch = unittest.mock.patch(
            "app.config.BACKEND_CHAIN", ("alpha", "beta", "gamma"))
        self.override_patch = unittest.mock.patch("app.config.BACKEND_OVERRIDE", None)
        self.chain_patch.start()
        self.override_patch.start()

    def tearDown(self):
        self.chain_patch.stop()
        self.override_patch.stop()
        self.vlm._REGISTRY.clear()
        self.vlm._REGISTRY.update(self.saved)

    def test_default_order_follows_the_configured_chain(self):
        self.assertEqual([b.name for b in self.vlm.resolve_chain()],
                         ["alpha", "beta", "gamma"])

    def test_pin_moves_to_front_without_truncating(self):
        """A pin must not disable failover - the demo has to survive an outage."""
        self.assertEqual([b.name for b in self.vlm.resolve_chain("gamma")],
                         ["gamma", "alpha", "beta"])

    def test_unready_backends_are_skipped(self):
        self.vlm._REGISTRY["beta"].ready = False
        self.assertEqual([b.name for b in self.vlm.resolve_chain()], ["alpha", "gamma"])

    def test_all_unready_raises_with_every_reason(self):
        for name in ("alpha", "beta", "gamma"):
            self.vlm._REGISTRY[name].ready = False
        with self.assertRaises(self.vlm.VLMError) as ctx:
            self.vlm.resolve_chain()
        for name in ("alpha", "beta", "gamma"):
            self.assertIn(name, str(ctx.exception))

    def test_unknown_pin_raises(self):
        with self.assertRaises(self.vlm.VLMError):
            self.vlm.resolve_chain("not-a-backend")


class Rendering(unittest.TestCase):
    def test_refusal_renders(self):
        a = Appraisal(status="refused", headline="Not a truck.",
                      gate=GateReport(decision=GateDecision.REFUSE_NOT_A_TRUCK,
                                      headline="Not a truck."))
        self.assertIn("Not a truck", report.render_text(a))

    def test_findings_cite_a_filename_not_an_index(self):
        g = GateReport(photos=[_check(3, "steer.jpg")], usable_photo_ids=[3])
        ev = EvidenceReport(issues=[Issue(photo_id=3, component="steer_tires",
                                          observation="worn", severity="moderate",
                                          confidence=0.8, price_impact="medium")])
        out = report.render_text(Appraisal(gate=g, evidence=ev, headline="x"))
        self.assertIn("steer.jpg", out)


class PhotoSelection(unittest.TestCase):
    def test_prefers_view_diversity_over_count(self):
        """Thirty frames of one tire must not crowd out the other views."""
        photos = []
        for i in range(30):
            c = _check(i, f"{i}.jpg")
            c.view, c.capture_quality = "tire_wheel", 0.9
            photos.append(c)
        for i, view in enumerate(["exterior_front_34", "dashboard_odometer", "interior_cab"]):
            c = _check(100 + i, f"v{i}.jpg")
            c.view, c.capture_quality = view, 0.5
            photos.append(c)
        picked = evidence.select_photos(GateReport(photos=photos), limit=8)
        self.assertEqual(len(picked), 8)
        self.assertGreaterEqual(len({c.view for c in picked}), 4)

    def test_drops_unusable_photos(self):
        good, bad = _check(1), _check(2)
        bad.usable = False
        picked = evidence.select_photos(GateReport(photos=[good, bad]), limit=10)
        self.assertEqual([c.photo_id for c in picked], [1])


class GateEventContract(unittest.TestCase):
    """The web screen paints the gate report while the vision call is still
    running, so the pipeline has to hand it over mid-run rather than only in
    the finished Appraisal."""

    def _gate(self, decision=GateDecision.PASS, **over):
        g = GateReport(decision=decision, headline="h",
                       photos=[_check(0), _check(1)], usable_photo_ids=[0, 1],
                       views_present=["tire_wheel"], **over)
        return g

    def _run(self, gate_report):
        from app import pipeline
        seen = []
        ev = EvidenceReport()

        def record_gate(g):
            # Order matters: the callback is useless if it only fires once the
            # vision call it is meant to cover has already returned.
            seen.append(("gate", g, len(seen)))

        def fake_evidence(gate, declared=None, *, backend=None, on_photo=None):
            seen.append(("evidence", None, len(seen)))
            return ev

        with unittest.mock.patch.object(pipeline.gate_stage, "run",
                                        return_value=gate_report), \
             unittest.mock.patch.object(pipeline.evidence_stage, "run",
                                        side_effect=fake_evidence), \
             unittest.mock.patch.object(pipeline.evidence_stage, "select_photos",
                                        return_value=[_check(0)]):
            result = pipeline.appraise([Path("a.jpg")], on_gate=record_gate)
        return result, seen

    def test_fires_once_with_the_report_before_evidence(self):
        gate_report = self._gate()
        _, seen = self._run(gate_report)
        kinds = [s[0] for s in seen]
        self.assertEqual(kinds.count("gate"), 1)
        self.assertIs(seen[0][1], gate_report)
        self.assertLess(kinds.index("gate"), kinds.index("evidence"))

    def test_fires_on_a_refusal_too(self):
        # The refused frame is the whole explanation, so the screen needs the
        # report on the path that never reaches the vision call at all.
        _, seen = self._run(self._gate(GateDecision.REFUSE_NOT_A_TRUCK))
        self.assertEqual([s[0] for s in seen], ["gate"])

    def test_absent_callback_changes_nothing(self):
        from app import pipeline
        with unittest.mock.patch.object(pipeline.gate_stage, "run",
                                        return_value=self._gate(
                                            GateDecision.REFUSE_NO_PHOTOS)):
            result = pipeline.appraise([Path("a.jpg")])
        self.assertEqual(result.status, "refused")

    def test_the_vision_call_gets_fewer_frames_than_the_gate_passed(self):
        # The screen names the frames one by one while it waits, and it reads
        # them off the gate event rather than the usable list, because
        # select_photos caps at MAX_EVIDENCE_PHOTOS. Counting usable frames
        # would name frames that were never sent.
        from app.config import MAX_EVIDENCE_PHOTOS
        photos = []
        for i in range(MAX_EVIDENCE_PHOTOS + 6):
            c = _check(i, f"{i}.jpg")
            c.view, c.capture_quality = "tire_wheel", 0.9
            photos.append(c)
        gate_report = GateReport(decision=GateDecision.PASS, photos=photos,
                                 usable_photo_ids=[c.photo_id for c in photos])
        sent = evidence.select_photos(gate_report)
        self.assertEqual(len(sent), MAX_EVIDENCE_PHOTOS)
        self.assertLess(len(sent), len(gate_report.usable_photo_ids))

    def test_on_step_still_takes_exactly_two_arguments(self):
        # app/cli.py and app/demo.py both pass a two-parameter callback. Adding
        # a third positional argument to the pipeline's `note` call broke every
        # CLI appraise, and a `lambda *a` test callback hid it.
        from app import pipeline
        calls = []

        def two_arg_note(step, detail):
            calls.append(step)

        photos = [_check(0), _check(1)]
        gate_report = GateReport(decision=GateDecision.PASS, headline="h",
                                 photos=photos, usable_photo_ids=[0, 1])
        with unittest.mock.patch.object(pipeline.gate_stage, "run",
                                        return_value=gate_report), \
             unittest.mock.patch.object(pipeline.evidence_stage, "run",
                                        return_value=EvidenceReport()):
            pipeline.appraise([Path("a.jpg")], on_step=two_arg_note)
        self.assertIn("gate", calls)
        self.assertIn("evidence", calls)


class StaticAssets(unittest.TestCase):
    """The screen serves fonts, modules and the elevation out of
    subdirectories, which the old flat handler rejected."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from app.server import app as server_app
        self.client = TestClient(server_app)

    def test_serves_a_nested_file(self):
        self.assertEqual(self.client.get("/static/js/dom.js").status_code, 200)
        self.assertEqual(
            self.client.get("/static/assets/tractor-elevation.svg").status_code, 200)

    def test_still_refuses_to_climb_out_of_the_web_root(self):
        for path in ("/static/../config.py", "/static/../../README.md",
                     "/static/js/../../config.py"):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_gate_event_names_the_frames_the_vision_call_will_get(self):
        # The screen walks these frame by frame while it waits, so the list has
        # to come from the same function the evidence stage uses.
        import inspect

        from app import server
        src = inspect.getsource(server.appraise)
        self.assertIn("evidence.select_photos(gate)", src)
        self.assertIn("evidence_photo_ids", src)

    def test_every_stylesheet_and_module_the_page_asks_for_resolves(self):
        html = (Path("app/web/index.html")).read_text(encoding="utf-8")
        refs = re.findall(r'(?:href|src)="/static/([^"]+)"', html)
        self.assertTrue(refs)
        for ref in refs:
            self.assertEqual(self.client.get(f"/static/{ref}").status_code, 200, ref)


class ElevationDrawing(unittest.TestCase):
    """The drawing is only useful if its zones are the real component and view
    vocabularies. A typo here would silently stop a finding from ever lighting
    anything up."""

    SVG = Path("app/web/assets/tractor-elevation.svg")
    JS = Path("app/web/js/elevation.js")

    # Conditions rather than parts: observed on a panel that is already drawn.
    ALIASED = {"paint_finish", "corrosion", "fluid_leaks"}

    def zones(self):
        import xml.etree.ElementTree as ET
        return [g.get("data-component")
                for g in ET.parse(self.SVG).iter() if g.get("data-component")]

    def test_no_zone_invents_a_component(self):
        self.assertEqual([z for z in self.zones() if z not in evidence.COMPONENTS], [])

    def test_every_component_has_a_zone_or_a_documented_alias(self):
        drawn = set(self.zones())
        missing = [c for c in evidence.COMPONENTS
                   if c not in drawn and c not in self.ALIASED]
        self.assertEqual(missing, [])

    def test_each_zone_is_drawn_once(self):
        z = self.zones()
        self.assertEqual([x for x in set(z) if z.count(x) > 1], [])

    def test_aliases_point_at_zones_that_exist(self):
        src = self.JS.read_text(encoding="utf-8")
        alias_block = src[src.index("const ALIAS = {"):]
        alias_block = alias_block[:alias_block.index("};")]
        targets = re.findall(r":\s*'([a-z_]+)'", alias_block)
        self.assertEqual(set(re.findall(r"^\s*([a-z_]+):", alias_block, re.M)),
                         self.ALIASED)
        drawn = set(self.zones())
        for t in targets:
            self.assertIn(t, drawn)

    def test_view_map_covers_the_whole_view_vocabulary(self):
        from app.vision import VIEW_LABELS
        src = self.JS.read_text(encoding="utf-8")
        block = src[src.index("const VIEW_ZONES = {"):]
        block = block[:block.index("\n};")]
        mapped = re.findall(r"^\s{2}([a-z_0-9]+):", block, re.M)
        self.assertEqual(sorted(mapped), sorted(VIEW_LABELS))

    def test_view_map_only_names_drawn_zones(self):
        src = self.JS.read_text(encoding="utf-8")
        block = src[src.index("const VIEW_ZONES = {"):]
        block = block[:block.index("\n};")]
        named = set(re.findall(r"'([a-z_]+)'", block))
        drawn = set(self.zones())
        self.assertEqual(sorted(n for n in named if n not in drawn), [])


class FrozenExport(unittest.TestCase):
    """`app/export.py` inlines the screen into one offline file. Splitting the
    front end into modules and a styles/ directory broke it once already, so
    the contract is a test: nothing in the frozen file may point at a path the
    server would have had to serve."""

    def _appraisal(self):
        gate_report = GateReport(decision=GateDecision.REFUSE_NOT_A_TRUCK,
                                 headline="not a truck", photos=[],
                                 usable_photo_ids=[], views_present=[])
        return Appraisal(status="refused", headline="not a truck",
                         gate=gate_report, version="test")

    def test_frozen_file_references_nothing_the_server_would_serve(self):
        from app import export
        html = export.build_html(self._appraisal())
        self.assertNotIn('href="/static/', html)
        self.assertNotIn('src="/static/', html)
        self.assertNotIn("fonts.googleapis.com", html)

    def test_every_module_and_stylesheet_is_inlined(self):
        """A module the export forgets is a screen that half-renders offline.

        `landing.js` lives under js/ but drives the landing page at `/` and is
        not in the appraisal screen's module graph, so it is excluded by name
        rather than by hoping nobody notices it in the bundle.
        """
        from app import export
        from app.config import WEB
        from app.export import NON_MODULE_SCRIPTS
        html = export.build_html(self._appraisal())
        for mod in (WEB / "js").glob("*.js"):
            if mod.stem in NON_MODULE_SCRIPTS:
                self.assertNotIn(f'"kamion:{mod.stem}"', html, mod.name)
                continue
            self.assertIn(f'"kamion:{mod.stem}"', html, mod.name)

    def test_inlined_modules_have_no_unresolvable_relative_imports(self):
        # A data: URL has no base, so a surviving './x.js' would fail to load.
        from app import export
        from app.config import WEB
        for mod in sorted((WEB / "js").glob("*.js")):
            rewritten = export._module_url(mod.read_text(encoding="utf-8"))
            decoded = base64.standard_b64decode(
                rewritten.split(",", 1)[1]).decode("utf-8")
            self.assertEqual(re.findall(r"from '\./", decoded), [], mod.name)

    def test_the_drawing_travels_with_the_file(self):
        from app import export
        html = export.build_html(self._appraisal())
        self.assertIn("window.KAMION_ELEVATION", html)
        self.assertIn("data-component", html)


class PartIcons(unittest.TestCase):
    """Every closed component has a shop-manual glyph. A missing key would
    silently draw the generic truck, which is the old wall-of-prose failure
    wearing a different hat."""

    SRC = Path("app/web/js/icons.js").read_text(encoding="utf-8")

    def _object(self, name):
        match = re.search(rf"export const {name} = \{{(.*?)\n\}}", self.SRC, re.S)
        self.assertIsNotNone(match, name)
        return dict(re.findall(r"([a-z0-9_]+):\s*'([^']*)'", match.group(1)))

    def test_every_component_has_a_glyph(self):
        from app.evidence.prompts import COMPONENTS
        self.assertEqual(sorted(self._object("COMPONENT_ICONS")), sorted(COMPONENTS))

    def test_every_summary_key_has_a_glyph(self):
        from app.evidence.prompts import SUMMARY_KEYS
        self.assertEqual(sorted(self._object("SUMMARY_ICONS")), sorted(SUMMARY_KEYS))

    def test_every_mapped_kind_has_a_path(self):
        paths = self._object("PATHS")
        kinds = set(self._object("COMPONENT_ICONS").values())
        kinds |= set(self._object("SUMMARY_ICONS").values())
        self.assertEqual(sorted(k for k in kinds if k not in paths), [])

    def test_the_screen_draws_them(self):
        for name in ("reasoning.js", "panels.js", "gallery.js"):
            src = Path("app/web/js", name).read_text(encoding="utf-8")
            self.assertIn("from './icons.js'", src, name)
            self.assertIn("partIcon", src, name)

    def test_every_summary_key_has_a_workshop_title(self):
        from app.evidence.prompts import SUMMARY_KEYS
        src = Path("app/web/js/panels.js").read_text(encoding="utf-8")
        match = re.search(r"const SYSTEM_TITLE = \{(.*?)\n\}", src, re.S)
        self.assertIsNotNone(match)
        titles = dict(re.findall(r"([a-z0-9_]+):\s*'([^']*)'", match.group(1)))
        self.assertEqual(sorted(titles), sorted(SUMMARY_KEYS))


class ScreenChrome(unittest.TestCase):
    """The live screen's motion and corners have a habit of drifting back
    into a second look. These pin the current ones: no defocus on the rail,
    a lightweight inspection sweep, and one corner radius."""

    STYLES = Path("app/web/styles")

    def test_inspection_cards_do_not_blur(self):
        css = (self.STYLES / "reasoning.css").read_text(encoding="utf-8")
        self.assertNotIn("filter: blur", css)
        self.assertNotIn("blur(", css)

    def test_scan_is_lightweight_and_respects_reduced_motion(self):
        css = (self.STYLES / "run.css").read_text(encoding="utf-8")
        js = Path("app/web/js/run.js").read_text(encoding="utf-8")
        self.assertIn("@keyframes optical-sweep", css)
        self.assertIn("prefers-reduced-motion: reduce", css)
        self.assertIn(".scan.on::before { animation: none; }", css)
        self.assertNotIn("fillScanGrid", js)
        html = Path("app/web/index.html").read_text(encoding="utf-8")
        self.assertIn('id="scan"', html)
        self.assertIn('id="scan-status" role="status"', html)

    def test_price_ranges_are_separate_and_directly_labelled(self):
        js = Path("app/web/js/band.js").read_text(encoding="utf-8")
        self.assertIn("Photo-adjusted estimate", js)
        self.assertIn("Comparable-market baseline", js)
        self.assertIn("Both rows use the same price scale", js)
        self.assertIn("not confirmed sale prices", js)
        self.assertIn("above chart scale", js)
        self.assertIn("below chart scale", js)

    def test_appraisal_surfaces_share_one_corner_radius(self):
        for name in ("base.css", "gallery.css", "run.css",
                     "reasoning.css", "result.css", "tokens.css"):
            css = (self.STYLES / name).read_text(encoding="utf-8")
            self.assertNotIn("999px", css, name)
            self.assertNotRegex(css, r"border-radius:\s*\d+px", name)


if __name__ == "__main__":
    unittest.main()


class SubjectBox(unittest.TestCase):
    """Which truck in the frame is the one being sold.

    Decided in the gate rather than the browser. The old screen picked the
    largest vehicle box client-side while the vision model was handed the whole
    frame, so on a dealer-lot photo the box a viewer saw and the pixels the
    model read were different trucks.
    """

    def _check(self, *boxes, width=1000, height=600):
        c = _check(0)
        c.width, c.height = width, height
        c.detections = [Detection(label=lbl, confidence=conf, box=list(box),
                                  area_frac=abs((box[2] - box[0]) * (box[3] - box[1]))
                                  / float(width * height))
                        for lbl, conf, box in boxes]
        return c

    def test_no_vehicle_box_means_no_subject(self):
        # A tire close-up contains no truck-shaped object and is still a photo
        # of the truck; it simply has no box to draw.
        self.assertIsNone(gate.pick_subject(self._check()))

    def test_the_only_truck_is_the_subject(self):
        c = self._check(("truck", 0.9, (100, 100, 600, 500)))
        self.assertEqual(gate.pick_subject(c), [100, 100, 600, 500])

    def test_the_bigger_truck_wins(self):
        c = self._check(("truck", 0.9, (10, 10, 120, 120)),
                        ("truck", 0.9, (300, 100, 800, 500)))
        self.assertEqual(gate.pick_subject(c), [300, 100, 800, 500])

    def test_centring_breaks_a_tie_between_similar_trucks(self):
        # Two trucks of the same size in a lot shot: the one the photographer
        # framed is the one being sold.
        c = self._check(("truck", 0.9, (0, 0, 300, 300)),
                        ("truck", 0.9, (350, 150, 650, 450)))
        self.assertEqual(gate.pick_subject(c), [350, 150, 650, 450])

    def test_a_car_is_never_the_subject(self):
        c = self._check(("car", 0.95, (0, 0, 900, 580)),
                        ("truck", 0.6, (400, 200, 700, 450)))
        self.assertEqual(gate.pick_subject(c), [400, 200, 700, 450])

    def test_a_clipped_centre_subject_beats_a_whole_truck_at_the_edge(self):
        """The case that sent the box to the wrong truck on a real frame.

        On `demo/tr_clean/000.jpg` the subject ran off the top of the frame, so
        YOLO measured it at 15% of the area against 23% for a whole white
        tractor parked to the left, and area picked the white one. A box
        clipped by the frame edge is always under-measured; the frame's centre
        is not.
        """
        # 1000x600, so the centre of the frame is (500, 300).
        c = self._check(("truck", 0.81, (30, 120, 470, 560)),   # whole, left of centre
                        ("truck", 0.45, (400, 0, 700, 420)))    # clipped at the top
        self.assertEqual(gate.pick_subject(c), [400, 0, 700, 420])

    def test_when_two_boxes_hold_the_centre_the_bigger_one_wins(self):
        c = self._check(("truck", 0.7, (420, 240, 560, 360)),
                        ("truck", 0.7, (150, 60, 880, 560)))
        self.assertEqual(gate.pick_subject(c), [150, 60, 880, 560])

    def test_nothing_on_the_centre_falls_back_to_area_and_position(self):
        c = self._check(("truck", 0.8, (10, 10, 210, 210)),
                        ("truck", 0.8, (700, 350, 990, 590)))
        self.assertIsNotNone(gate.pick_subject(c))

    def test_competing_vehicles_are_counted_excluding_the_subject(self):
        c = self._check(("truck", 0.9, (300, 100, 800, 500)),
                        ("truck", 0.8, (10, 100, 200, 400)),
                        ("car", 0.7, (820, 300, 980, 420)))
        c.subject_box = gate.pick_subject(c)
        self.assertEqual(gate.competing_vehicles(c), 2)


class SubjectCrop(unittest.TestCase):
    """When the close-up call gets the crop instead of the whole frame."""

    def _check(self, subject, others=(), width=1000, height=600):
        c = _check(0)
        c.width, c.height = width, height
        boxes = [("truck", 0.9, subject)] + list(others)
        c.detections = [Detection(label=lbl, confidence=conf, box=list(box),
                                  area_frac=abs((box[2] - box[0]) * (box[3] - box[1]))
                                  / float(width * height))
                        for lbl, conf, box in boxes]
        c.subject_box = list(subject)
        return c

    def test_a_lone_truck_is_sent_whole(self):
        # Nothing to be confused by, and cropping would throw away the ground
        # line and the background a buyer reads for context.
        self.assertFalse(evidence.wants_crop(self._check((200, 100, 700, 500))))

    def test_a_lot_shot_is_cropped_to_the_subject(self):
        c = self._check((350, 150, 650, 450),
                        [("truck", 0.8, (0, 150, 300, 450))])
        self.assertTrue(evidence.wants_crop(c))

    def test_a_truck_already_filling_the_frame_is_not_cropped(self):
        c = self._check((5, 5, 995, 595), [("truck", 0.8, (0, 0, 60, 60))])
        self.assertFalse(evidence.wants_crop(c))

    def test_no_subject_box_is_never_cropped(self):
        c = _check(0)
        c.width, c.height = 1000, 600
        self.assertFalse(evidence.wants_crop(c))


class PhotoStream(unittest.TestCase):
    """`on_photo` is a separate callback because `on_step` is pinned at two
    arguments - widening it broke every CLI appraise once already."""

    def _gate(self, n=3):
        photos = []
        for i in range(n):
            c = _check(i, f"{i}.jpg")
            c.view, c.capture_quality = "tire_wheel", 0.9
            photos.append(c)
        return GateReport(decision=GateDecision.PASS, headline="h", photos=photos,
                          usable_photo_ids=[c.photo_id for c in photos])

    def test_fires_once_per_photo_and_not_through_on_step(self):
        from app import pipeline
        from app.schema import PhotoFinding
        gate_report = self._gate(3)
        seen, steps = [], []

        def fake_evidence(g, declared=None, *, backend=None, on_photo=None):
            for c in g.photos:
                on_photo(PhotoFinding(photo_id=c.photo_id, view=c.view))
            return EvidenceReport()

        with unittest.mock.patch.object(pipeline.gate_stage, "run",
                                        return_value=gate_report), \
             unittest.mock.patch.object(pipeline.evidence_stage, "run",
                                        side_effect=fake_evidence):
            pipeline.appraise([Path("a.jpg")],
                              on_step=lambda step, detail: steps.append(step),
                              on_photo=lambda f: seen.append(f.photo_id))
        self.assertEqual(sorted(seen), [0, 1, 2])
        self.assertIn("evidence", steps)

    def test_absent_callback_changes_nothing(self):
        from app import pipeline
        with unittest.mock.patch.object(pipeline.gate_stage, "run",
                                        return_value=self._gate(2)), \
             unittest.mock.patch.object(pipeline.evidence_stage, "run",
                                        return_value=EvidenceReport()):
            result = pipeline.appraise([Path("a.jpg")])
        self.assertIn(result.status, ("ok", "need_more_photos", "ok_with_requests"))


class FanOutFailure(unittest.TestCase):
    """One lost frame is not a lost appraisal; every lost frame is."""

    class _Client:
        supports_structured_output = False
        name = "fake"

        def __init__(self, fail_on=()):
            self.fail_on = set(fail_on)
            self.calls = 0

        def complete(self, prompt, images, **kw):
            from app.vlm.base import VLMError, VLMResponse
            self.calls += 1
            # By filename rather than by call number: each photo is now read
            # CLOSEUP_SAMPLES times by a pool, so "the second call" is whatever
            # thread got there first and would make this test a coin toss.
            if images and images[0].name in self.fail_on:
                raise VLMError("provider said no")
            return VLMResponse(text=json.dumps({
                "shows": "a tire", "legible": True, "odometer_km": None,
                "observations": [], "strengths": [], "cannot_tell": [],
                "confidence": 0.5}), backend="fake", model="fake")

    def _checks(self, n):
        out = []
        for i in range(n):
            c = _check(i, f"{i}.jpg")
            c.view = "tire_wheel"
            out.append(c)
        return out

    def test_a_failed_photo_is_recorded_not_swallowed(self):
        from app.evidence import stage as run_module
        import tempfile
        client = self._Client(fail_on={"2.jpg"})
        report = EvidenceReport()
        with tempfile.TemporaryDirectory() as tmp:
            findings = run_module._fan_out([client], self._checks(3), "a truck",
                                           Path(tmp), None, report)
        failed = [f for f in findings if f.error]
        self.assertEqual(len(failed), 1)
        self.assertTrue(any("could not be read" in w for w in report.parse_warnings))

    def test_the_surviving_photos_still_produce_findings(self):
        from app.evidence import stage as run_module
        import tempfile
        client = self._Client(fail_on={"1.jpg"})
        report = EvidenceReport()
        with tempfile.TemporaryDirectory() as tmp:
            findings = run_module._fan_out([client], self._checks(3), "a truck",
                                           Path(tmp), None, report)
        self.assertEqual(sum(1 for f in findings if not f.error), 2)


class FallbackSynthesis(unittest.TestCase):
    """Losing the synthesis call should cost the prose, not the findings."""

    def test_findings_are_grouped_without_a_model(self):
        from app.evidence.stage import _fallback_synthesis
        from app.schema import PhotoFinding
        flat = [Issue(photo_id=0, component="steer_tires", observation="worn",
                      severity="major", confidence=0.8),
                Issue(photo_id=1, component="engine_bay", observation="oil film",
                      severity="minor", confidence=0.6)]
        findings = [PhotoFinding(photo_id=0, view="tire_wheel",
                                 cannot_tell=["tread in mm"])]
        data = _fallback_synthesis(findings, flat)
        self.assertEqual(set(data["condition_summary"]), set(evidence.SUMMARY_KEYS))
        self.assertIn("worn", data["condition_summary"]["tires"])
        self.assertIn("oil film", data["condition_summary"]["engine_driveline"])
        self.assertIn("tread in mm", data["coverage_gaps"])

    def test_the_fallback_does_not_grade_at_all(self):
        """The regression this replaces, pinned.

        The old fallback graded on `max(severity)` with `default=-1` and no
        `-1` key in the dict, so sixteen spotless photos returned `unknown`
        while ONE cosmetic scuff upgraded the same truck to `good`. Grading
        now belongs to `app.condition.rollup` on both paths, so the two differ
        only in prose.
        """
        from app.evidence.stage import _fallback_synthesis
        self.assertEqual(_fallback_synthesis([], [])["condition_grade"], "")
        flat = [Issue(photo_id=0, component="paint_finish", observation="light scuff",
                      severity="cosmetic", confidence=0.6, price_impact="none")]
        self.assertEqual(_fallback_synthesis([], flat)["condition_grade"], "")

    def test_an_unseen_system_says_so_rather_than_claiming_it_is_fine(self):
        from app.evidence.stage import _fallback_synthesis
        data = _fallback_synthesis([], [])
        self.assertEqual(data["condition_summary"]["fifth_wheel_coupling"],
                         "not visible in these photos")


class Gallery(unittest.TestCase):
    """One grid, every vehicle, filterable by lot on the client."""

    @classmethod
    def setUpClass(cls):
        from app import gallery as g
        cls.g = g
        cls.cards = g.cards()

    def test_the_rehearsed_cases_are_pinned_to_the_front(self):
        # A gallery that could only offer real tractors could not demonstrate a
        # refusal, and refusing well is its own line in the brief's rubric.
        demo = [c for c in self.cards if c["demo"]]
        self.assertTrue(all(c["demo"] for c in self.cards[:len(demo)]))
        self.assertGreaterEqual(len(demo), 6)

    def test_every_card_can_be_rendered(self):
        for card in self.cards:
            self.assertTrue(card["id"])
            self.assertIn("make", card)
            self.assertIsInstance(card["n_photos"], int)

    def test_a_vehicle_backing_a_rehearsed_case_appears_once(self):
        ids = [c["id"] for c in self.cards]
        self.assertEqual(len(ids), len(set(ids)))

    def test_makes_are_normalised_so_a_filter_chip_is_one_manufacturer(self):
        # The corpus spells one make several ways because three harvesters
        # wrote it; FORD and Ford must not be two chips.
        names = [m["name"] for m in self.g.facets()["makes"]]
        self.assertEqual(len(names), len({n.lower() for n in names}))
        self.assertIn("Ford", names)

    def test_corpus_photos_resolve_on_disk(self):
        card = next(c for c in self.cards if not c["demo"])
        paths = self.g.photo_paths(card["source_key"], card["listing_id"])
        self.assertEqual(len(paths), card["n_photos"])
        self.assertTrue(paths[0].exists())

    def test_the_cover_index_is_inside_the_photo_list(self):
        for card in self.cards:
            if card["demo"] or not card["cover"]:
                continue
            index = int(card["cover"].rsplit("/", 1)[1])
            self.assertLess(index, card["n_photos"])

    def test_rehearsed_cases_inherit_the_backing_lot(self):
        demos = {c["case_id"]: c for c in self.cards if c["demo"]}
        self.assertEqual(demos["tr_clean"]["market"], "TR")
        self.assertEqual(demos["tr_phone"]["market"], "TR")
        self.assertEqual(demos["unseen_brand"]["market"], "US")
        self.assertIsNone(demos["not_a_truck"]["market"])
        real = [c for c in self.cards if not c["demo"]]
        self.assertTrue(real)
        self.assertTrue(all(c.get("market") in ("TR", "US") for c in real))
        self.assertGreater(sum(1 for c in real if c["market"] == "TR"), 0)
        self.assertGreater(sum(1 for c in real if c["market"] == "US"), 0)
        lots = self.g.facets()["markets"]
        self.assertEqual(lots["TR"] + lots["US"], len(real))

    def test_the_lot_filter_does_not_change_how_a_truck_is_priced(self):
        # A US card on the wall is still sent as market=TR. The old dropdown
        # asked a Turkish fit for a number in dollars; this one must not.
        # The lot control is a Country facet in js/filters.js now, which is
        # where the browse vocabulary lives; app.js is still the only place a
        # market reaches the pipeline.
        app = Path("app/web/app.js").read_text(encoding="utf-8")
        filters = Path("app/web/js/filters.js").read_text(encoding="utf-8")
        self.assertIn("p.set('market', 'TR')", app)
        self.assertIn("Türkiye", filters)
        self.assertIn("gallery-market", filters)

    def test_no_card_on_the_wall_is_exempt_from_the_filters(self):
        # The rehearsed cases used to be pinned into the grid AND excused from
        # every chip by `if (card.demo) return true`, so narrowing to Ford left
        # a motorcycle sitting among the Fords. They have their own shelf now,
        # and the grid is fed from a pool with no demo cards in it at all.
        src = Path("app/web/js/gallery.js").read_text(encoding="utf-8")
        self.assertNotIn("card.demo) return true", src)
        self.assertIn("demos = cards.filter((c) => c.demo)", src)
        self.assertIn("listings = cards.filter((c) => !c.demo)", src)
        # The shelf draws the demos; everything paged draws from `listings`.
        self.assertIn("shelf.replaceChildren(...demos.map(nodeFor))", src)
        self.assertIn("listings.filter((c) => inSearch(c) && facets.matches(c))", src)

    def test_the_wall_is_paged_rather_than_grown(self):
        # "Show 24 more" only went one way and hid how much there was.
        src = Path("app/web/js/gallery.js").read_text(encoding="utf-8")
        html = Path("app/web/index.html").read_text(encoding="utf-8")
        self.assertNotIn("more trucks", src)
        self.assertNotIn("gallery-more", html)
        self.assertIn('id="gallery-pager"', html)
        self.assertIn("pager.render", src)
        # Any narrowing goes back to page one: staying on page 4 of a two-page
        # result set is how "the filter did nothing" happens.
        self.assertRegex(src, r"function narrow\(\) \{\s*page = 1;")

    def test_every_filter_group_reads_a_key_the_cards_carry(self):
        # A typo in a group id is a filter that silently matches nothing.
        src = Path("app/web/js/filters.js").read_text(encoding="utf-8")
        block = re.search(r"export const GROUPS = \[(.*?)\n\];", src, re.S).group(1)
        ids = re.findall(r"\{ id: '([a-z_]+)'", block)
        self.assertEqual(ids, ["make", "year", "km", "market", "capture"])
        card = next(c for c in self.cards if not c["demo"])
        for key in ids:
            self.assertIn(key, card, key)

    def test_the_photo_filter_can_reach_every_capture_label(self):
        # `mixed` was in the data from the first harvest and had no chip, so
        # 103 of 197 trucks could not be filtered to at all.
        src = Path("app/web/js/filters.js").read_text(encoding="utf-8")
        labels = {c["capture"] for c in self.cards if not c["demo"]}
        for label in labels:
            self.assertIn(f"c.capture === '{label}'", src, label)

    def test_the_upload_path_survives_the_gallery_overhaul(self):
        # The brief is judged on photos the team has never seen, and the
        # dropzone is the only way those reach the system. It moved behind a
        # button; it did not go away.
        html = Path("app/web/index.html").read_text(encoding="utf-8")
        app = Path("app/web/app.js").read_text(encoding="utf-8")
        self.assertIn('id="dropzone"', html)
        self.assertIn('id="own-toggle"', html)
        self.assertIn('aria-controls="own-panel"', html)
        self.assertIn("uploadFiles", app)
        # The seller fields feed declaredParams() whether or not the panel is
        # open, so they have to stay in the document rather than be built on
        # demand.
        for field in ("f-year", "f-km", "f-make", "f-asking"):
            self.assertIn(f'id="{field}"', html)


class NearDuplicateMerge(unittest.TestCase):
    """Sixteen calls describe one worn drive tire sixteen ways.

    The first real fan-out produced three separate `major` findings for one
    shoulder-worn drive tire, which read as three problems and dragged the
    condition grade to `poor`. The model's own duplicate list missed the
    paraphrases, so a deterministic pass runs behind it.
    """

    def issue(self, photo_id, component, text, severity="major"):
        return Issue(photo_id=photo_id, component=component, observation=text,
                     severity=severity, confidence=0.8)

    def test_paraphrases_of_one_defect_collapse(self):
        flat = [
            self.issue(0, "drive_tires",
                       "The outer drive tire has severe irregular shoulder wear with "
                       "extensive tread-block tearing and chunking"),
            self.issue(1, "drive_tires",
                       "The outer drive tire shows uneven scalloped tread-block wear "
                       "with irregular shoulder tearing across the visible ribs"),
        ]
        out = evidence.merge_duplicates(flat, [])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].also_seen_in, [1])

    def test_two_genuinely_different_defects_stay_apart(self):
        flat = [
            self.issue(0, "drive_tires", "outer drive tire worn to the wear bars"),
            self.issue(1, "drive_tires", "inner drive tire sidewall has a deep cut "
                                         "exposing cord near the bead"),
        ]
        self.assertEqual(len(evidence.merge_duplicates(flat, [])), 2)

    def test_different_components_never_merge(self):
        # Identical wording on two parts is two findings, not one.
        text = "heavy corrosion with flaking paint and pitting across the surface"
        flat = [self.issue(0, "chassis_frame", text),
                self.issue(1, "fuel_tank", text)]
        self.assertEqual(len(evidence.merge_duplicates(flat, [])), 2)

    def test_the_models_own_pairing_still_wins_first(self):
        flat = [self.issue(0, "steer_tires", "worn to the bars"),
                self.issue(1, "steer_tires", "completely unrelated wording here"),
                self.issue(2, "steer_tires", "another unrelated description")]
        out = evidence.merge_duplicates(flat, [{"keep": 0, "merge": [1, 2]}])
        self.assertEqual(len(out), 1)
        self.assertEqual(sorted(out[0].also_seen_in), [1, 2])

    def test_corroboration_from_both_passes_is_kept(self):
        flat = [self.issue(0, "drive_tires", "outer drive tire shoulder wear tearing"),
                self.issue(1, "drive_tires", "unrelated rim damage wording entirely"),
                self.issue(2, "drive_tires", "outer drive tire shoulder wear and tearing")]
        out = evidence.merge_duplicates(flat, [{"keep": 1, "merge": []}])
        by_photo = {i.photo_id: i for i in out}
        self.assertIn(0, by_photo)
        self.assertEqual(by_photo[0].also_seen_in, [2])

    def test_noise_words_alone_do_not_merge(self):
        # Two short observations sharing only filler must not be folded.
        flat = [self.issue(0, "mirrors_visor", "the mirror is there and visible"),
                self.issue(1, "mirrors_visor", "there is a crack across the glass")]
        self.assertEqual(len(evidence.merge_duplicates(flat, [])), 2)


class BrandPalette(unittest.TestCase):
    """The landing page and the appraisal screen are one product.

    `styles/landing.css` is a standalone page that loads nothing else, so it
    carries its own `:root`. `styles/tokens.css` mirrors its brand colours
    rather than importing them. Two copies drift, so this asserts they agree:
    change a brand colour in one place and this fails until it changes in both.
    """

    #  tokens.css name  ->  landing.css name
    SHARED = {"--lot": "--paper", "--ink": "--ink",
              "--signal": "--orange", "--edge": "--line"}

    @classmethod
    def setUpClass(cls):
        from app.config import WEB

        def palette(path):
            text = (WEB / "styles" / path).read_text()
            return dict(re.findall(r"(--[a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{3,8})", text))
        cls.app = palette("tokens.css")
        cls.landing = palette("landing.css")

    def test_the_brand_colours_agree(self):
        for ours, theirs in self.SHARED.items():
            self.assertIn(theirs, self.landing, theirs)
            self.assertEqual(self.app.get(ours), self.landing[theirs],
                             f"{ours} must equal landing's {theirs}")

    def test_the_landing_still_defines_what_we_mirror(self):
        # Renaming a token on the landing side would make the check above pass
        # vacuously if it were written with .get on both sides.
        for theirs in self.SHARED.values():
            self.assertRegex(self.landing[theirs], r"^#[0-9a-fA-F]{6}$")


class PagesAreConnected(unittest.TestCase):
    """`/` is the landing page and `/app` is the appraisal screen. A visitor
    has to be able to get from either to the other, and the two have to look
    like the same product when they do."""

    @classmethod
    def setUpClass(cls):
        from app.config import WEB
        cls.web = WEB
        cls.landing = (WEB / "landing.html").read_text()
        cls.app = (WEB / "index.html").read_text()

    def test_the_landing_offers_the_app(self):
        self.assertGreaterEqual(self.landing.count('href="/app"'), 4)

    def test_the_app_offers_the_way_back(self):
        self.assertIn('href="/"', self.app)

    def test_both_routes_are_served(self):
        from fastapi.testclient import TestClient
        from app.server import app as server
        client = TestClient(server)
        self.assertIn("Zero guesswork", client.get("/").text)
        self.assertIn('id="dropzone"', client.get("/app").text)

    def test_one_wordmark_across_the_click(self):
        # Two different logos either side of one link is two products.
        for markup in ('class="brand"', 'class="brand-light"'):
            self.assertIn(markup, self.landing, markup)
            self.assertIn(markup, self.app, markup)

    def test_the_landing_assets_the_app_page_borrows_exist(self):
        for name in ("assets/kip.svg", "assets/demo-truck.jpg"):
            self.assertTrue((self.web / name).is_file(), name)


class SubjectDrawContract(unittest.TestCase):
    """The browser must not have to recompute which box is the subject.

    It used to find it by exact float equality on all four coordinates, which
    held only because `subject_box` was literally an element of `detections`.
    The moment the subject is decided anywhere else that match fails, `shown`
    is empty and the screen draws no box at all - silently.
    """

    JS = Path("app/web/js/frames.js")

    def test_the_subject_is_read_off_is_subject_first(self):
        src = self.JS.read_text(encoding="utf-8")
        self.assertIn("d.is_subject", src)
        lookup = src[src.index("const subjectOf"):src.index("function drawBoxes")]
        # is_subject before sameBox, not instead of it: an older frozen export
        # carries the subject as coordinates and nothing else.
        self.assertLess(lookup.index("is_subject"), lookup.index("sameBox"))
        self.assertIn("sameBox", lookup)

    def test_the_schema_emits_every_field_the_screen_reads(self):
        det = Detection(label="truck", confidence=0.9, box=[0, 0, 1, 1],
                        area_frac=0.5).to_dict()
        self.assertIn("is_subject", det)
        self.assertFalse(det["is_subject"])
        self.assertIn("subject_basis", _check(0).to_dict())

    def test_the_refusal_colour_is_a_separate_channel(self):
        # Truck detection is set-level: a box earns the refusal colour only
        # when the whole SET was refused for not being a truck. `is_subject`
        # must never reach `data-disqualifying`.
        src = self.JS.read_text(encoding="utf-8")
        for line in src.splitlines():
            if "dataset.disqualifying" in line:
                self.assertIn("blockedLabel", line, line)
        self.assertIn("refusedAsNotATruck = decision === 'refuse_not_a_truck'", src)


def _boxes(*boxes, width=1000, height=600, view="unknown", view_conf=0.0):
    """A PhotoCheck carrying synthetic detections. Same shape the SubjectBox
    helper builds, hoisted so the newer classes can share it."""
    c = _check(0)
    c.width, c.height = width, height
    c.view, c.view_conf = view, view_conf
    c.detections = [Detection(label=lbl, confidence=conf, box=list(box),
                              area_frac=abs((box[2] - box[0]) * (box[3] - box[1]))
                              / float(width * height))
                    for lbl, conf, box in boxes]
    return c


class VehicleDedup(unittest.TestCase):
    """One physical vehicle, one box.

    COCO runs NMS per class, so a tractor comes back as truck 0.71, bus 0.44
    and car 0.31 at the same pixels. The duplicates were counted as competing
    vehicles - which triggers a crop - and one of them could be picked as the
    subject in its own right.
    """

    def test_one_vehicle_three_labels_collapses_to_the_truck(self):
        c = _boxes(("truck", 0.71, (300, 100, 800, 500)),
                   ("bus", 0.44, (305, 104, 795, 498)),
                   ("car", 0.31, (298, 98, 802, 502)))
        kept = subject.dedupe_vehicles(c.detections)
        self.assertEqual([(d.label, d.confidence) for d in kept], [("truck", 0.71)])

    def test_two_trucks_side_by_side_both_survive(self):
        c = _boxes(("truck", 0.9, (0, 100, 300, 500)),
                   ("truck", 0.8, (320, 100, 620, 500)))
        self.assertEqual(len(subject.dedupe_vehicles(c.detections)), 2)

    def test_a_cab_inside_a_whole_rig_survives_as_two_candidates(self):
        # Nested, but not the same object: IoU well under the merge threshold.
        c = _boxes(("truck", 0.9, (100, 100, 900, 500)),
                   ("truck", 0.6, (100, 150, 400, 480)))
        self.assertEqual(len(subject.dedupe_vehicles(c.detections)), 2)

    def test_a_motorcycle_over_a_truck_is_never_merged_away(self):
        # COCO_DISQUALIFYING is fed by exactly these boxes, and the gate's
        # worst possible error is refusing a real listing.
        c = _boxes(("truck", 0.87, (100, 100, 900, 500)),
                   ("motorcycle", 0.62, (105, 105, 895, 495)))
        kept = subject.dedupe_vehicles(c.detections)
        self.assertEqual(sorted(d.label for d in kept), ["motorcycle", "truck"])

    def test_the_duplicates_stop_counting_as_competition(self):
        c = _boxes(("truck", 0.71, (300, 100, 800, 500)),
                   ("bus", 0.44, (305, 104, 795, 498)),
                   ("car", 0.60, (298, 98, 802, 502)),
                   ("truck", 0.8, (10, 100, 250, 450)))
        c.detections = subject.dedupe_vehicles(c.detections)
        c.subject_box = gate.pick_subject(c)
        self.assertEqual(gate.competing_vehicles(c), 1)


class SubjectScore(unittest.TestCase):
    """What the frame score buys over "a centred box wins outright".

    An override has no crossover point. Any 0.26-confidence box straddling the
    centre pixel eliminated a 0.95-confidence box filling a third of the frame,
    which is how an engine-bay close-up came to be cropped to a background
    lorry. The nine cases in SubjectBox are the regression floor; these are the
    ones the old rule got wrong.
    """

    def test_the_reported_bug_a_big_truck_beats_a_speck_on_the_centre(self):
        c = _boxes(("truck", 0.95, (0, 80, 480, 518)),     # 35% of frame, off-centre
                   ("truck", 0.26, (460, 265, 540, 332)))  # 0.9%, holds the centre
        self.assertEqual(gate.pick_subject(c), [0, 80, 480, 518])

    def test_confidence_counts(self):
        # Mirror-image boxes, identical area and identical distance from the
        # centre. Under area x centrality they tie and the answer is whichever
        # one YOLO happened to emit first.
        c = _boxes(("truck", 0.3, (600, 150, 900, 450)),
                   ("truck", 0.9, (100, 150, 400, 450)))
        self.assertEqual(gate.pick_subject(c), [100, 150, 400, 450])

    def test_a_centred_box_beats_a_rival_twice_its_size(self):
        c = _boxes(("truck", 0.8, (420, 220, 620, 380)),
                   ("truck", 0.8, (20, 20, 340, 220)))
        self.assertEqual(gate.pick_subject(c), [420, 220, 620, 380])

    def test_and_loses_to_one_four_times_its_size(self):
        c = _boxes(("truck", 0.8, (420, 220, 620, 380)),
                   ("truck", 0.8, (20, 20, 660, 220)))
        self.assertEqual(gate.pick_subject(c), [20, 20, 660, 220])

    def test_a_car_wins_when_it_is_the_only_vehicle_in_frame(self):
        # COCO labels a tight cab shot `car`. Excluding the class outright
        # loses real subjects; it is only barred when something better is
        # on offer, which is what test_a_car_is_never_the_subject pins.
        c = _boxes(("car", 0.9, (100, 100, 600, 500)))
        self.assertEqual(gate.pick_subject(c), [100, 100, 600, 500])

    def test_exactly_one_detection_is_flagged_as_the_subject(self):
        c = _boxes(("truck", 0.9, (300, 100, 800, 500)),
                   ("truck", 0.8, (10, 100, 200, 400)),
                   ("car", 0.7, (820, 300, 980, 420)))
        box = gate.pick_subject(c)
        flagged = [d for d in c.detections if d.is_subject]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(list(flagged[0].box), box)

    def test_a_sub_threshold_winner_is_appended_to_the_detections(self):
        # The screen is asked to draw the subject box. A candidate below the
        # 0.25 the screen shows would otherwise be a box it cannot find.
        c = _boxes(("car", 0.9, (0, 0, 200, 200)))
        faint = Detection(label="truck", confidence=0.19, box=[200, 80, 800, 520],
                          area_frac=(600 * 440) / 600000)
        c._candidates = subject.build_candidates(c, [faint])
        self.assertEqual(gate.pick_subject(c), [200, 80, 800, 520])
        self.assertIn(faint, c.detections)
        self.assertTrue(faint.is_subject)

    def test_the_real_tr_clean_000_frame_not_just_the_stylised_one(self):
        """The detections YOLOv8n actually returns on `demo/tr_clean/000.jpg`.

        `SubjectBox.test_a_clipped_centre_subject_beats_a_whole_truck_at_the_edge`
        stylises this frame, and it stylises it with the clipping the wrong way
        round: on the real photograph it is the BACKGROUND tractor that is
        flush against x=0, and the subject the photographer framed sits clear
        of every edge. Edge relief for any box touching an edge scored the
        wrong truck 0.378 against the right one's 0.349 while the synthetic
        test stayed green.
        """
        c = _boxes(("car", 0.249, (309.3, 506.7, 1145.1, 992.2)),
                   ("truck", 0.805, (0.0, 213.1, 526.4, 898.6)),
                   ("truck", 0.454, (549.3, 76.4, 998.6, 588.4)),
                   ("bus", 0.677, (1240.3, 302.4, 1439.1, 634.9)),
                   ("truck", 0.549, (1107.0, 403.6, 1246.9, 579.1)),
                   ("car", 0.519, (985.2, 478.8, 1151.3, 597.3)),
                   width=1440, height=1080)
        self.assertEqual(gate.pick_subject(c), [549.3, 76.4, 998.6, 588.4])


class PartViewSubject(unittest.TestCase):
    """A close-up of a component has no subject, not a distant one.

    The bug this pins, in the user's words: "the app appraises trucks in the
    background while something like a steering wheel or even ENGINE is in the
    foreground". YOLO finds something truck-shaped in the yard, `pick_subject`
    returns it, `wants_crop` fires, and the close-up call is handed a crop of a
    lorry forty metres away under the line "this image has been cropped to the
    one vehicle being sold; other vehicles in the original frame were
    deliberately excluded". It is then asked the engine-bay checklist about it.
    """

    CASES = Path("tests/subject_cases.json")

    def _check(self, view, view_conf, *boxes, width=1000, height=600):
        return _boxes(*boxes, width=width, height=height,
                      view=view, view_conf=view_conf)

    def test_a_truck_in_the_yard_behind_an_engine_bay_is_not_the_subject(self):
        c = self._check("engine_bay", 0.7, ("truck", 0.6, (700, 40, 950, 200)))
        self.assertIsNone(gate.pick_subject(c))
        self.assertFalse(evidence.wants_crop(c))
        self.assertIn("yard behind it", c.subject_basis)

    def test_an_unconfident_view_tag_does_not_suppress_anything(self):
        # A part view has to be a CONFIDENT part view. Below the threshold the
        # tag is not evidence of anything and the geometry decides alone.
        c = self._check("engine_bay", 0.3, ("truck", 0.6, (700, 40, 950, 200)))
        self.assertIsNotNone(gate.pick_subject(c))

    def test_the_close_up_frame_itself_still_keeps_its_box(self):
        # YOLO calls a dashboard filling the frame a truck. That box is the
        # photograph, not the yard, and nulling it would be over-correction.
        c = self._check("dashboard_odometer", 0.9, ("truck", 0.5, (2, 2, 998, 598)))
        self.assertEqual(gate.pick_subject(c), [2, 2, 998, 598])
        # ... and it is still never cropped: the crop can only remove the
        # thing the per-view checklist is about.
        self.assertFalse(evidence.wants_crop(c))

    def test_a_tiny_box_fails_on_pixels_even_at_a_friendly_area(self):
        c = self._check("tire_wheel", 0.8, ("truck", 0.9, (100, 100, 140, 140)),
                        width=200, height=140)
        self.assertIsNone(gate.pick_subject(c))

    def test_damage_detail_suppresses_a_crop_but_does_not_vouch_for_a_truck(self):
        """One name was doing two jobs whose costs run in opposite directions.

        For the GATE, `damage_detail` may not vouch that a set is genuine truck
        close-ups: it is the taxonomy's catch-all and it matched a parked
        motorcycle at 0.42, so admitting it would let a motorcycle listing past
        the refusal ladder. For CROPPING, it is unambiguously a close-up of one
        part, and excluding it meant a damage shot could be cropped to a lorry
        behind it. Deciding wrongly that a frame is a close-up costs one
        uncropped frame; deciding wrongly that a close-up is an exterior costs a
        confident description of the wrong truck.

        Forced by measurement: the prompt ensemble moved chassis frames into
        `damage_detail` and background_truck_rejected_rate fell 0.90 -> 0.73.
        """
        self.assertIn("damage_detail", subject.CLOSE_UP_VIEWS)
        self.assertNotIn("damage_detail", subject.TRUCK_PART_VIEWS)
        c = self._check("damage_detail", 0.9, ("truck", 0.6, (700, 40, 950, 200)))
        self.assertIsNone(gate.pick_subject(c))      # not cropped to the background
        self.assertFalse(evidence.wants_crop(c))

    def test_a_whole_vehicle_frame_is_untouched_by_the_rule(self):
        c = self._check("exterior_front_34", 0.9, ("truck", 0.6, (700, 40, 950, 200)))
        self.assertEqual(gate.pick_subject(c), [700, 40, 950, 200])

    def test_the_prototype_can_rescue_a_mistagged_three_quarter(self):
        c = self._check("fifth_wheel", 0.6, ("truck", 0.8, (200, 100, 800, 500)))
        pool = subject.build_candidates(c)
        pool[0].sim_abs = 0.99
        identity = subject.SubjectIdentity(prototype=np.ones(4, dtype=np.float32),
                                           method="recurring_vehicle")
        with unittest.mock.patch.object(subject, "PART_VIEW_RESCUE_SIM", 0.80):
            self.assertFalse(subject.is_scenery(c, pool[0], identity))
            self.assertTrue(subject.is_scenery(c, pool[0], subject.NO_IDENTITY))

    # --- the real frames, end to end -------------------------------------

    def _corpus(self, listing_id, view):
        cases = json.loads(self.CASES.read_text(encoding="utf-8"))["cases"]
        row = next(c for c in cases if c["listing_id"] == listing_id and c["view"] == view)
        c = _check(0)
        c.width, c.height = row["width"], row["height"]
        c.view, c.view_conf = row["view"], row["view_conf"]
        c.detections = [Detection(label=l, confidence=cf, box=b, area_frac=a)
                        for l, cf, b, a in row["detections"]]
        c._candidates = subject.build_candidates(
            c, [d for d in c.detections if d.confidence < subject.DETECTION_CONF])
        c.detections = [d for d in c.detections if d.confidence >= subject.DETECTION_CONF]
        return c

    def test_the_tire_close_up_with_a_row_of_lorries_across_the_top(self):
        # us_selectrucks/256409/007.jpg: eight truck boxes crammed into the top
        # 15% of a 2000x1500 tire shot. The gate cropped to one of them.
        c = self._corpus("256409", "tire_wheel")
        self.assertIsNone(gate.pick_subject(c))
        self.assertFalse(evidence.wants_crop(c))

    def test_the_dashboard_with_a_truck_seen_through_the_windscreen(self):
        # us_selectrucks/256409/014.jpg: truck 0.649 over 3.3% of the frame,
        # through the glass. The odometer checklist was asked about it.
        c = self._corpus("256409", "dashboard_odometer")
        self.assertIsNone(gate.pick_subject(c))
        self.assertFalse(evidence.wants_crop(c))

    def test_the_engine_bay_with_the_yard_behind_it(self):
        # us_selectrucks/232615/010.jpg: truck 0.212 over 3.3% of a 5712x4284
        # engine bay. The candidate pool reaches below 0.25, so without this
        # rule the wider pool would make it worse, not better.
        c = self._corpus("232615", "engine_bay")
        self.assertIsNone(gate.pick_subject(c))
        self.assertFalse(evidence.wants_crop(c))


class SubjectCropGeometry(unittest.TestCase):
    """What `wants_crop` measures, and how small a crop it will write.

    The four SubjectCrop cases are the regression floor; these are the ones
    the shipped rule got wrong.
    """

    def _check(self, subject_box, others=(), width=1000, height=600):
        c = _boxes(("truck", 0.9, subject_box), *others, width=width, height=height)
        c.subject_box = list(subject_box)
        c.detections[0].is_subject = True
        return c

    def test_a_subject_at_sixty_percent_is_not_cropped_to_eighty_one(self):
        # CROP_PAD is 0.08 a side: 1.16x on each axis, 1.35x on area. The
        # unpadded box passed the < 0.70 test and the crop it produced covered
        # 0.81 of the frame - while prompts.py told the model the other
        # vehicles had been deliberately excluded.
        c = self._check((110, 40, 890, 560), [("truck", 0.8, (0, 0, 100, 100))])
        x1, y1, x2, y2 = c.subject_box
        self.assertLess((x2 - x1) * (y2 - y1) / 600000.0,
                        evidence.passes.CROP_MAX_SUBJECT_FRAC)
        self.assertFalse(evidence.wants_crop(c))

    def test_a_crop_under_a_hundred_and_sixty_pixels_is_refused(self):
        self.assertIsNone(
            evidence.passes.subject_crop_rect([100, 100, 220, 220], 2000, 1500))

    def test_a_speck_of_a_competitor_does_not_trigger_a_crop(self):
        # A parked hatchback 80 m behind a lone truck used to be worth
        # throwing away the ground line for.
        c = self._check((200, 100, 700, 500), [("car", 0.8, (960, 560, 995, 595))])
        self.assertFalse(evidence.wants_crop(c))
        c = self._check((200, 100, 700, 500), [("car", 0.8, (0, 300, 300, 590))])
        self.assertTrue(evidence.wants_crop(c))

    def test_competition_is_counted_when_the_box_is_not_byte_identical(self):
        # The subject no longer has to be an element of `detections` by
        # identity - `is_subject` is the key, and the float match is a
        # fallback for an older frozen export.
        c = self._check((200, 100, 700, 500), [("truck", 0.8, (0, 150, 180, 450))])
        c.subject_box = [200.0000001, 100, 700, 500]
        self.assertEqual(gate.competing_vehicles(c), 1)


class FocusScenery(unittest.TestCase):
    """A box less in focus than its frame is the yard, not the subject.

    The one subject rule that does not consult the CLIP view tag. 36.8% of the
    corpus carries an exterior tag and the four exterior classes have median
    confidences of 0.34 to 0.55 - a Ford dashboard in the corpus is tagged
    `exterior_front` at 0.38 and a stripped engine bay `exterior_rear` at 0.39.
    Every view-gated rule misses both frames entirely.
    """

    def _gray(self, sharp_box=True):
        """A 600x400 frame: noise everywhere, and a 120x80 patch at (60,40)
        that is either noisy (sharp) or flat (blurred)."""
        rng = np.random.default_rng(4)
        g = rng.integers(0, 255, (400, 600), dtype=np.uint8)
        if not sharp_box:
            g[40:120, 60:180] = 128          # flat patch -> near-zero Laplacian
        return g

    def test_a_blurred_box_scores_below_one(self):
        r = subject.focus_ratio(self._gray(sharp_box=False), [60, 40, 180, 120])
        self.assertLess(r, subject.FOCUS_SCENERY_RATIO)

    def test_a_sharp_box_scores_around_one(self):
        r = subject.focus_ratio(self._gray(sharp_box=True), [60, 40, 180, 120])
        self.assertGreater(r, subject.FOCUS_SCENERY_RATIO)

    def test_no_image_is_neutral_never_suppressing(self):
        self.assertEqual(subject.focus_ratio(None, [0, 0, 10, 10]), 1.0)

    def test_a_box_filling_the_frame_is_neutral(self):
        g = self._gray()
        self.assertEqual(subject.focus_ratio(g, [0, 0, 600, 400]), 1.0)

    def test_a_tiny_box_is_neutral(self):
        self.assertEqual(subject.focus_ratio(self._gray(), [0, 0, 5, 5]), 1.0)

    def test_focus_suppresses_without_any_part_view_tag(self):
        """The reason this rule exists: it fires when the view tag is wrong."""
        check = PhotoCheck(photo_id=0, path="x.jpg", filename="x.jpg")
        check.width, check.height = 600, 400
        check.view, check.view_conf = "exterior_front", 0.38   # the real mislabel
        det = Detection(label="truck", confidence=0.6, box=[60, 40, 180, 120],
                        area_frac=(120 * 80) / (600 * 400))
        cand = subject._candidate(check, det)
        cand.focus_ratio = 0.2
        self.assertTrue(subject.is_scenery(check, cand, subject.NO_IDENTITY))

    def test_a_sharp_small_box_on_a_mistagged_frame_survives(self):
        """Focus and area catch different failures; neither subsumes the other."""
        check = PhotoCheck(photo_id=0, path="x.jpg", filename="x.jpg")
        check.width, check.height = 600, 400
        check.view, check.view_conf = "exterior_front", 0.38
        det = Detection(label="truck", confidence=0.6, box=[60, 40, 180, 120],
                        area_frac=(120 * 80) / (600 * 400))
        cand = subject._candidate(check, det)
        cand.focus_ratio = 2.3          # small, but genuinely in focus
        self.assertFalse(subject.is_scenery(check, cand, subject.NO_IDENTITY))


class ViewPromptEnsemble(unittest.TestCase):
    """Several templates per view class, max-pooled.

    One template per class was measured at 83.3%/86.7% (dev/held-out) on
    whole-vehicle vs component, with 18.9%/15.8% of genuine component close-ups
    coming back as an exterior view. That is the error that matters: an
    `exterior_front` tag hands the frame the exterior question bank, counts it
    toward whole-vehicle coverage, and exempts it from every part-view rule in
    app/subject.py. The ensemble takes it to 91.7%/96.7% and 10.8%/0.0%.
    """

    def test_every_label_has_several_templates(self):
        for label in vision.VIEW_LABELS:
            self.assertGreaterEqual(
                len(vision.VIEW_PROMPTS[label]), 3,
                f"{label} lost its ensemble; one template per class measured 83%")

    def test_labels_and_prompt_keys_are_the_same_vocabulary(self):
        self.assertEqual(list(vision.VIEW_PROMPTS), vision.VIEW_LABELS)

    def test_the_eleven_view_ids_are_unchanged(self):
        """VIEW_QUESTIONS, VIEW_ZONES in elevation.js and the perception head's
        classes are all keyed on these. Adding a template is free; renaming a
        class is not."""
        self.assertEqual(vision.VIEW_LABELS, [
            "exterior_front", "exterior_front_34", "exterior_side", "exterior_rear",
            "interior_cab", "dashboard_odometer", "tire_wheel", "engine_bay",
            "chassis_undercarriage", "fifth_wheel", "damage_detail"])

    def test_no_template_is_shared_between_two_classes(self):
        seen = {}
        for label, prompts in vision.VIEW_PROMPTS.items():
            for p in prompts:
                self.assertNotIn(p, seen, f"{label} and {seen.get(p)} share a template")
                seen[p] = label

    def test_nothing_unpacks_the_prompts_as_pairs_any_more(self):
        """`for k, _ in VIEW_PROMPTS` silently became "unpack the label string"
        when this turned into a dict, and cost two call sites."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        checked = 0
        for path in root.rglob("*.py"):
            if ".venv" in path.parts:
                continue
            src = path.read_text()
            if "VIEW_PROMPTS" not in src or "VIEW_PROMPTS = [" in src:
                continue        # scripts/clean_dataset.py keeps its own list of pairs
            checked += 1
            rel = path.relative_to(root)
            self.assertNotIn("for k, _ in VIEW_PROMPTS", src, str(rel))
            self.assertNotRegex(src, r"VIEW_PROMPTS\[[^\]]+\]\[0\]", str(rel))
        self.assertGreater(checked, 0, "the scan found nothing; it has stopped working")


class SubjectAreaFloor(unittest.TestCase):
    """No box this small is the vehicle being sold, in any view.

    The part-view rule needs the classifier to have called the frame a close-up
    first. This one does not, which is the point: it is what still holds when
    the view tag is wrong, and the view tag is wrong on roughly 7% of component
    close-ups even after the prompt ensemble.
    """

    def _check(self, view, conf, box, width=1200, height=900):
        c = PhotoCheck(photo_id=0, path="x.jpg", filename="x.jpg")
        c.width, c.height = width, height
        c.view, c.view_conf = view, conf
        x1, y1, x2, y2 = box
        c.detections = [Detection(label="truck", confidence=0.8, box=list(box),
                                  area_frac=abs((x2 - x1) * (y2 - y1)) / (width * height))]
        return c

    def test_a_truck_in_the_distance_is_not_the_subject_on_an_exterior_frame(self):
        """The composite this was built for: a whole truck pasted into the top
        fifth of a close-up occupies 2.56% of the frame. The frame then reads as
        an exterior view precisely BECAUSE a truck is visible in it, so every
        view-gated rule stands down."""
        c = self._check("exterior_front_34", 0.9, (100, 40, 292, 184))   # 2.56%
        self.assertIsNone(gate.pick_subject(c))

    def test_a_normal_subject_is_far_above_the_floor(self):
        c = self._check("exterior_front_34", 0.9, (120, 90, 1080, 810))  # 64%
        self.assertIsNotNone(gate.pick_subject(c))

    def test_the_floor_sits_below_the_measured_whole_vehicle_distribution(self):
        """Measured over 201 confident whole-vehicle frames: median 0.554,
        q10 0.311, q05 0.137. The floor must stay well under q05 or it starts
        eating real subjects."""
        self.assertLess(subject.SUBJECT_MIN_AREA_FRAC, 0.137)
        self.assertGreater(subject.SUBJECT_MIN_AREA_FRAC, 0.0256)
