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

    def test_bad_enum_values_fall_back(self):
        f = evidence.parse_closeup(self.payload(observations=[
            {"component": "steer_tires", "observation": "x", "severity": "catastrophic",
             "confidence": "nope", "price_impact": "ruinous"}]), self.check, cropped=False)
        self.assertIn(f.issues[0].severity, evidence.SEVERITIES)
        self.assertIn(f.issues[0].price_impact, evidence.IMPACTS)
        self.assertEqual(f.issues[0].confidence, 0.5)

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
        obs = evidence.closeup_schema()["properties"]["observations"]["items"]["properties"]
        self.assertEqual(obs["component"]["enum"], evidence.COMPONENTS)
        self.assertEqual(obs["severity"]["enum"], evidence.SEVERITIES)
        self.assertEqual(obs["price_impact"]["enum"], evidence.IMPACTS)

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
            if self.calls in self.fail_on:
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
        client = self._Client(fail_on={2})
        report = EvidenceReport()
        with tempfile.TemporaryDirectory() as tmp:
            findings = run_module._fan_out(client, self._checks(3), "a truck",
                                           Path(tmp), None, report)
        failed = [f for f in findings if f.error]
        self.assertEqual(len(failed), 1)
        self.assertTrue(any("could not be read" in w for w in report.parse_warnings))

    def test_the_surviving_photos_still_produce_findings(self):
        from app.evidence import stage as run_module
        import tempfile
        client = self._Client(fail_on={1})
        report = EvidenceReport()
        with tempfile.TemporaryDirectory() as tmp:
            findings = run_module._fan_out(client, self._checks(3), "a truck",
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
        self.assertEqual(data["condition_grade"], "poor")
        self.assertIn("tread in mm", data["coverage_gaps"])

    def test_an_unseen_system_says_so_rather_than_claiming_it_is_fine(self):
        from app.evidence.stage import _fallback_synthesis
        data = _fallback_synthesis([], [])
        self.assertEqual(data["condition_summary"]["fifth_wheel_coupling"],
                         "not visible in these photos")


class Gallery(unittest.TestCase):
    """One grid, every vehicle, no market split."""

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
