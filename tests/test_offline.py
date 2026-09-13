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

from app import evidence, gate, report
from app.config import USD_TRY
from app.schema import (Appraisal, EvidenceReport, GateDecision, GateReport,
                        Issue, PhotoCheck)


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


class ParseEvidence(unittest.TestCase):
    def setUp(self):
        self.selected = [_check(7, "a.jpg"), _check(11, "b.jpg")]

    def payload(self, **over):
        base = {
            "vehicle": {"make": "Ford", "model": "F-MAX", "odometer_km": 164374,
                        "odometer_photo_id": 1, "badges_seen": ["F-MAX"],
                        "confidence": 0.9},
            "per_photo": [],
            "issues": [{"photo_id": 0, "component": "steer_tires",
                        "observation": "worn to the bars", "severity": "moderate",
                        "confidence": 0.8, "price_impact": "medium"}],
            "condition_summary": {}, "condition_grade": "good",
            "coverage_gaps": [], "confidence": 0.7,
        }
        base.update(over)
        return json.dumps(base)

    def test_indices_map_back_to_gate_ids(self):
        """The model sees 0..n-1; the report must cite the gate's photo ids."""
        r = evidence.parse(self.payload(), self.selected)
        self.assertEqual(r.issues[0].photo_id, 7)
        self.assertEqual(r.vehicle.odometer_photo_id, 11)

    def test_uncited_issue_is_dropped(self):
        """An issue naming a photo that was never sent is not evidence."""
        r = evidence.parse(self.payload(issues=[
            {"photo_id": 99, "component": "drive_tires", "observation": "invented",
             "severity": "major", "confidence": 1.0, "price_impact": "high"}]),
            self.selected)
        self.assertEqual(r.issues, [])
        self.assertTrue(any("uncited" in w for w in r.parse_warnings))

    def test_unknown_component_is_normalised_not_trusted(self):
        r = evidence.parse(self.payload(issues=[
            {"photo_id": 0, "component": "flux_capacitor", "observation": "hmm",
             "severity": "minor", "confidence": 0.5, "price_impact": "low"}]),
            self.selected)
        self.assertEqual(r.issues[0].component, "flux_capacitor")
        self.assertTrue(any("unknown component" in w for w in r.parse_warnings))

    def test_bad_enum_values_fall_back(self):
        r = evidence.parse(self.payload(issues=[
            {"photo_id": 0, "component": "steer_tires", "observation": "x",
             "severity": "catastrophic", "confidence": "nope", "price_impact": "ruinous"}]),
            self.selected)
        self.assertIn(r.issues[0].severity, evidence.SEVERITIES)
        self.assertIn(r.issues[0].price_impact, evidence.IMPACTS)
        self.assertEqual(r.issues[0].confidence, 0.5)

    def test_summary_keys_always_present(self):
        r = evidence.parse(self.payload(), self.selected)
        self.assertEqual(set(r.condition_summary), set(evidence.SUMMARY_KEYS))


class JsonSchema(unittest.TestCase):
    def test_strict_shape(self):
        """Strict mode requires every property listed in `required`."""
        s = evidence.json_schema()
        for node in (s, s["properties"]["vehicle"], s["properties"]["issues"]["items"]):
            self.assertFalse(node["additionalProperties"])
            self.assertEqual(set(node["required"]), set(node["properties"]))

    def test_enums_match_the_parser(self):
        issue = evidence.json_schema()["properties"]["issues"]["items"]["properties"]
        self.assertEqual(issue["component"]["enum"], evidence.COMPONENTS)
        self.assertEqual(issue["severity"]["enum"], evidence.SEVERITIES)


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

        def fake_evidence(gate, declared=None, *, backend=None):
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
        from app import export
        from app.config import WEB
        html = export.build_html(self._appraisal())
        for mod in sorted((WEB / "js").glob("*.js")):
            self.assertIn(f'"kamion:{mod.stem}"', html, mod.name)
        # the import map has to cover what the entry point actually imports
        entry = (WEB / "app.js").read_text(encoding="utf-8")
        for name in re.findall(r"from '\./js/([a-z]+)\.js'", entry):
            self.assertIn(f'"kamion:{name}"', html, name)

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


if __name__ == "__main__":
    unittest.main()
