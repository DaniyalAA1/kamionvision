"""Offline tests for the trained perception heads, reconciliation and the anchor.

No API calls, no network, no model download - every test here either builds a
tiny synthetic artifact or asserts a pure function. The things covered are the
ones that would fail silently:

  * the scalar feature order agreeing between training and inference, which is
    invisible if it breaks and quietly wrong
  * reconciliation downgrading a claim rather than deleting it
  * the identity head being forbidden from contradicting a badge for a brand it
    was never trained on - the corpus has no Scania, and a head that has never
    seen one will still name a class, confidently
  * the anchor never narrowing the band below the interval whose coverage was
    actually measured
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from PIL import Image

from app import reconcile
from app.perception import heads
from app.pricing import anchor
from app.schema import (Correction, EvidenceReport, GateReport, Issue,
                        PerceptionReport, PhotoCheck, PhotoPerception, VehicleRead)


def _photo(pid, **kw):
    c = PhotoCheck(photo_id=pid, path=f"/tmp/{pid}.jpg", filename=f"{pid}.jpg")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _perception(*photos, brand=None, conf=0.0):
    return PerceptionReport(photos=list(photos), brand=brand, brand_conf=conf)


class Vin(unittest.TestCase):

    def test_check_digit_accepts_a_vin_from_the_us_corpus(self):
        import pandas as pd
        from app.config import LISTINGS_CSV
        from app.vin import check_digit_ok
        vin = (pd.read_csv(LISTINGS_CSV)["vin"].dropna().astype(str)
                 .loc[lambda s: s.str.len() == 17].iloc[0])
        self.assertTrue(check_digit_ok(vin), vin)

    def test_ioq_substituted_before_length_check(self):
        from app.vin import normalise
        self.assertEqual(normalise("IOQ"), "100")

    def test_model_year_uses_the_modern_cycle_when_position_seven_is_a_letter(self):
        from app.vin import model_year
        self.assertEqual(model_year("3AKJHHDR3LSLJ7839"), 2020)

    def test_read_abstains_without_an_exact_length_candidate(self):
        from app import vin
        with unittest.mock.patch.object(vin, "_tokens", return_value=[("short", 0.9)]):
            reading = vin.read("/tmp/plate.jpg")
        self.assertIsNone(reading.vin)


class ScalarContract(unittest.TestCase):
    """The training matrix and the inference vector must agree, in order."""

    def test_scalar_row_length_matches_the_declared_names(self):
        row = heads.scalar_row(blur_laplacian_var=100.0, brightness=0.5, contrast_rms=0.2,
                               dark_clipped_frac=0.0, bright_clipped_frac=0.0,
                               colourfulness=30.0, capture_quality=0.8,
                               width=1000, height=1000)
        self.assertEqual(len(row), len(heads.SCALARS))

    def test_trainer_builds_its_matrix_through_the_same_function(self):
        import inspect

        from app.perception import train
        src = inspect.getsource(train.scalars_matrix)
        self.assertIn("scalar_row", src,
                      "the trainer must build features through heads.scalar_row, or the "
                      "artifact's column order can silently diverge from inference")

    def test_photo_matrix_width_is_embedding_plus_scalars(self):
        c = _photo(0, width=800, height=600)
        c._embedding = np.zeros(512, dtype=np.float32)
        self.assertEqual(heads.photo_matrix([c]).shape, (1, 512 + len(heads.SCALARS)))

    def test_a_photo_with_no_embedding_is_skipped_not_guessed(self):
        report = heads.run([_photo(0)])
        self.assertEqual(report.photos, [])

    def test_trainer_max_pools_view_templates_by_class(self):
        from types import SimpleNamespace

        import torch

        from app import vision
        from app.perception import train

        tagger = SimpleNamespace(
            view_bank=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            view_owner=torch.tensor([0, 0, 1]),
        )
        with (unittest.mock.patch.object(vision, "clip", return_value=tagger),
              unittest.mock.patch.object(vision, "VIEW_LABELS", ["whole", "part"])):
            pred = train.zero_shot_views(np.array([[-1.0, 0.0]]))

        self.assertEqual(pred.tolist(), ["part"])


class Reconciliation(unittest.TestCase):

    def _evidence(self, observation, photo_id=0, confidence=0.9, make=None):
        ev = EvidenceReport(issues=[Issue(photo_id=photo_id, component="steer_tires",
                                          observation=observation, severity="moderate",
                                          confidence=confidence, price_impact="medium")])
        ev.vehicle = VehicleRead(make=make)
        return ev

    def test_fine_detail_claim_on_a_degraded_photo_is_downgraded(self):
        ev = self._evidence("tread depth looks close to the 3mm limit")
        p = _perception(PhotoPerception(photo_id=0, severity=0.8, fine_detail_ok=False,
                                        degradations=["motion_blur"]))
        rep = reconcile.apply(GateReport(), p, ev)
        self.assertEqual(len(rep.of_kind("unsupported_detail")), 1)
        self.assertLess(ev.issues[0].confidence, 0.9)

    def test_a_downgraded_claim_is_never_deleted(self):
        ev = self._evidence("tread depth is marginal")
        p = _perception(PhotoPerception(photo_id=0, severity=0.99, fine_detail_ok=False))
        reconcile.apply(GateReport(), p, ev)
        self.assertEqual(len(ev.issues), 1)
        self.assertGreaterEqual(ev.issues[0].confidence, reconcile.CONFIDENCE_FLOOR)

    def test_a_coarse_claim_on_a_degraded_photo_is_left_alone(self):
        ev = self._evidence("the front bumper is missing entirely")
        p = _perception(PhotoPerception(photo_id=0, severity=0.9, fine_detail_ok=False))
        rep = reconcile.apply(GateReport(), p, ev)
        self.assertEqual(rep.of_kind("unsupported_detail"), [])
        self.assertEqual(ev.issues[0].confidence, 0.9)

    def test_a_clean_photo_supports_a_fine_detail_claim(self):
        ev = self._evidence("tread depth around 5mm")
        p = _perception(PhotoPerception(photo_id=0, severity=0.05, fine_detail_ok=True))
        reconcile.apply(GateReport(), p, ev)
        self.assertEqual(ev.issues[0].confidence, 0.9)

    def test_identity_conflict_is_raised_for_a_brand_the_head_knows(self):
        ev = self._evidence("scuffed", make="FORD")
        p = _perception(PhotoPerception(photo_id=0), brand="MAN", conf=0.9)
        with unittest.mock.patch.object(reconcile, "_identity_classes",
                                        return_value=["FORD", "MAN"]):
            rep = reconcile.apply(GateReport(), p, ev)
        self.assertEqual(len(rep.of_kind("identity_conflict")), 1)
        self.assertTrue(rep.widening)

    def test_the_head_may_not_dispute_a_brand_it_was_never_trained_on(self):
        # The corpus has no Scania. A head that has never seen one still names
        # a class, confidently - and must not be allowed to contradict a badge.
        ev = self._evidence("scuffed", make="SCANIA")
        p = _perception(PhotoPerception(photo_id=0), brand="FORD", conf=0.99)
        with unittest.mock.patch.object(reconcile, "_identity_classes",
                                        return_value=["FORD", "MAN"]):
            rep = reconcile.apply(GateReport(), p, ev)
        self.assertEqual(rep.of_kind("identity_conflict"), [])
        self.assertEqual(rep.widening, [])

    def test_an_unconfident_head_stays_quiet(self):
        ev = self._evidence("scuffed", make="FORD")
        p = _perception(PhotoPerception(photo_id=0), brand="MAN", conf=0.4)
        with unittest.mock.patch.object(reconcile, "_identity_classes",
                                        return_value=["FORD", "MAN"]):
            rep = reconcile.apply(GateReport(), p, ev)
        self.assertEqual(rep.of_kind("identity_conflict"), [])

    def test_a_view_the_head_can_see_withdraws_the_re_ask(self):
        gate_report = GateReport(missing_views=["tire_wheel"],
                                 requests=["a close-up of one steer tire and one drive tire, "
                                           "square to the tread so the grooves are readable"])
        p = _perception(PhotoPerception(photo_id=0, view="tire_wheel", view_conf=0.95))
        rep = reconcile.apply(gate_report, p, EvidenceReport())
        self.assertEqual(len(rep.of_kind("coverage_restored")), 1)
        self.assertEqual(gate_report.missing_views, [])
        self.assertEqual(gate_report.requests, [])
        self.assertIn("tire_wheel", gate_report.views_present)

    def test_an_unconfident_view_does_not_withdraw_a_re_ask(self):
        gate_report = GateReport(missing_views=["tire_wheel"], requests=["send tires"])
        p = _perception(PhotoPerception(photo_id=0, view="tire_wheel", view_conf=0.3))
        reconcile.apply(gate_report, p, EvidenceReport())
        self.assertEqual(gate_report.missing_views, ["tire_wheel"])

    def test_a_part_tagged_frame_cannot_create_whole_vehicle_coverage(self):
        gate_report = GateReport(
            photos=[_photo(0, usable=True, view="exterior_front")],
            missing_views=["exterior_front"],
            requests=["send a whole-vehicle front three-quarter photo"],
            blocks_pricing=True,
        )
        p = _perception(PhotoPerception(
            photo_id=0, view="exterior_front", view_conf=0.95,
            framing="part", framing_conf=0.95))

        rep = reconcile.apply(gate_report, p, EvidenceReport())

        self.assertEqual(rep.of_kind("coverage_restored"), [])
        self.assertEqual(len(rep.of_kind("framing_override")), 1)
        self.assertEqual(gate_report.missing_views, ["exterior_front"])
        self.assertNotIn("exterior_front", gate_report.views_present)

    def test_a_part_tagged_frame_cannot_clear_an_existing_pricing_block(self):
        gate_report = GateReport(
            photos=[_photo(0, usable=True, view="exterior_side")],
            views_present=["exterior_side"],
            blocks_pricing=True,
        )
        p = _perception(PhotoPerception(
            photo_id=0, view="exterior_side", view_conf=0.95,
            framing="part", framing_conf=0.95))

        reconcile.apply(gate_report, p, EvidenceReport())

        self.assertTrue(gate_report.blocks_pricing)

    def test_no_perception_is_a_no_op(self):
        ev = self._evidence("tread depth 2mm")
        rep = reconcile.apply(GateReport(), None, ev)
        self.assertEqual(rep.n, 0)
        self.assertEqual(ev.issues[0].confidence, 0.9)


class Odometer(unittest.TestCase):
    """The fourth rule: an OCR read reconciled against the VLM's odometer.

    `odometer.read` is mocked so these need neither RapidOCR nor a real image -
    what is under test is the reconciliation logic, not the OCR engine.
    """

    def _gate(self, view="dashboard_odometer", usable=True):
        return GateReport(photos=[_photo(3, view=view, usable=usable,
                                         capture_quality=0.9)])

    def _ev(self, odometer_km=None):
        return EvidenceReport(vehicle=VehicleRead(odometer_km=odometer_km,
                                                  odometer_photo_id=None))

    def _read(self, km, confidence=0.85):
        from app.odometer import OdometerRead
        return OdometerRead(km=km, confidence=confidence, text=f"{km}km")

    def test_joined_miles_converts_to_km(self):
        from app import odometer as O
        reading = O._select([("242780mi", 0.9, [0, 0, 10, 10])])
        self.assertEqual(reading.km, round(242780 * 1.60934))
        self.assertEqual(reading.rule, "joined_miles")

    def test_adjacent_miles_converts_to_km(self):
        from app import odometer as O
        reading = O._select([
            ("242780", 0.9, [0, 0, 60, 20]),
            ("miles", 0.8, [65, 0, 95, 20]),
        ])
        self.assertEqual(reading.km, round(242780 * 1.60934))
        self.assertEqual(reading.rule, "adjacent_miles")

    def test_kmh_still_rejected(self):
        from app import odometer as O
        reading = O._select([("90km/h", 0.99, [0, 0, 10, 10])])
        self.assertIsNone(reading.km)

    def test_trip_decimal_still_rejected(self):
        from app import odometer as O
        reading = O._select([("976.6km", 0.99, [0, 0, 10, 10])])
        self.assertIsNone(reading.km)

    def test_subject_crop_is_tried_before_full_frame(self):
        from app import odometer as O
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dashboard.jpg"
            Image.new("RGB", (400, 300)).save(path)
            with unittest.mock.patch.object(
                    O, "_tokens",
                    side_effect=[
                        [("242780mi", 0.9, [0, 0, 10, 10])],
                    ]) as tokens:
                reading = O.read(path, subject_box=[100, 100, 250, 220])
        self.assertEqual(reading.km, round(242780 * 1.60934))
        self.assertEqual(tokens.call_count, 1)
        self.assertIsInstance(tokens.call_args.args[0], np.ndarray)

    def test_full_frame_is_retried_after_crop_abstains(self):
        from app import odometer as O
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dashboard.jpg"
            Image.new("RGB", (400, 300)).save(path)
            with unittest.mock.patch.object(
                    O, "_tokens",
                    side_effect=[[], [("305273km", 0.9, [0, 0, 10, 10])]]) as tokens:
                reading = O.read(path, subject_box=[100, 100, 250, 220])
        self.assertEqual(reading.km, 305273)
        self.assertEqual(tokens.call_count, 2)
        self.assertEqual(tokens.call_args.args[0], path)

    def test_crop_smaller_than_minimum_side_is_not_used(self):
        from app import odometer as O
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dashboard.jpg"
            Image.new("RGB", (400, 300)).save(path)
            with unittest.mock.patch.object(
                    O, "_tokens",
                    return_value=[("305273km", 0.9, [0, 0, 10, 10])]) as tokens:
                reading = O.read(path, subject_box=[100, 100, 130, 130])
        self.assertEqual(reading.km, 305273)
        tokens.assert_called_once_with(path)

    def test_a_missing_odometer_is_recovered_and_fed_to_pricing(self):
        ev = self._ev(odometer_km=None)
        with unittest.mock.patch("app.odometer.read", return_value=self._read(305273)):
            rep = reconcile.apply(self._gate(), None, ev)
        self.assertEqual(len(rep.of_kind("odometer_recovered")), 1)
        self.assertEqual(ev.vehicle.odometer_km, 305273)      # now available to price
        self.assertEqual(ev.vehicle.odometer_photo_id, 3)

    def test_an_agreeing_read_is_silent(self):
        ev = self._ev(odometer_km=305000)
        with unittest.mock.patch("app.odometer.read", return_value=self._read(305273)):
            rep = reconcile.apply(self._gate(), None, ev)
        self.assertEqual(rep.n, 0)
        self.assertEqual(rep.widening, [])

    def test_a_conflicting_read_widens_but_keeps_the_priced_figure(self):
        ev = self._ev(odometer_km=500000)
        with unittest.mock.patch("app.odometer.read", return_value=self._read(305273)):
            rep = reconcile.apply(self._gate(), None, ev)
        self.assertEqual(len(rep.of_kind("odometer_conflict")), 1)
        self.assertEqual([w[1] for w in rep.widening],
                         [reconcile.ODOMETER_CONFLICT_WIDENING])
        self.assertEqual(ev.vehicle.odometer_km, 500000)      # vision figure unchanged

    def test_an_abstaining_read_changes_nothing(self):
        ev = self._ev(odometer_km=None)
        with unittest.mock.patch("app.odometer.read", return_value=self._read(None)):
            rep = reconcile.apply(self._gate(), None, ev)
        self.assertEqual(rep.n, 0)
        self.assertIsNone(ev.vehicle.odometer_km)

    def test_a_low_confidence_read_does_not_dispute_the_vision_figure(self):
        # A smeared dashboard OCR is unsure of must not override a legible
        # reading. Below the conflict-confidence bar the disagreement is dropped.
        ev = self._ev(odometer_km=184113)
        low = self._read(204113, confidence=reconcile.ODOMETER_CONFLICT_CONFIDENCE - 0.05)
        with unittest.mock.patch("app.odometer.read", return_value=low):
            rep = reconcile.apply(self._gate(), None, ev)
        self.assertEqual(rep.n, 0)
        self.assertEqual(rep.widening, [])
        self.assertEqual(ev.vehicle.odometer_km, 184113)

    def test_a_frame_backing_the_vision_figure_blocks_a_lone_misread(self):
        # Two dashboard frames: one confidently agrees with the vision figure,
        # the other confidently reads a number that never appears on any dash.
        # The corroborated reading wins - no conflict is raised.
        gate = GateReport(photos=[_photo(1, view="dashboard_odometer",
                                         usable=True, capture_quality=0.9),
                                  _photo(2, view="dashboard_odometer",
                                         usable=True, capture_quality=0.9)])
        ev = self._ev(odometer_km=184113)
        reads = {"/tmp/1.jpg": self._read(204113, confidence=0.95),   # misread, highest conf
                 "/tmp/2.jpg": self._read(184000, confidence=0.90)}   # backs the vision figure
        with unittest.mock.patch("app.odometer.read",
                                 side_effect=lambda path, **kw: reads[str(path)]):
            rep = reconcile.apply(gate, None, ev)
        self.assertEqual(len(rep.of_kind("odometer_conflict")), 0)
        self.assertEqual(ev.vehicle.odometer_km, 184113)

    def test_corroborated_ocr_overrides_an_unsupported_vision_misread(self):
        # The reported bug: the vision model read 204,113 km, which appears on no
        # dashboard, while two frames read the true 184,113. The priced figure
        # must become the corroborated OCR reading, not the vision misread.
        gate = GateReport(photos=[_photo(1, view="dashboard_odometer",
                                         usable=True, capture_quality=0.9),
                                  _photo(2, view="dashboard_odometer",
                                         usable=True, capture_quality=0.9)])
        ev = self._ev(odometer_km=204113)               # the vision misread
        reads = {"/tmp/1.jpg": self._read(184113, confidence=0.90),
                 "/tmp/2.jpg": self._read(184000, confidence=0.85)}
        with unittest.mock.patch("app.odometer.read",
                                 side_effect=lambda path, **kw: reads[str(path)]):
            rep = reconcile.apply(gate, None, ev)
        self.assertEqual(len(rep.of_kind("odometer_overridden")), 1)
        self.assertEqual(len(rep.of_kind("odometer_conflict")), 0)
        self.assertEqual(ev.vehicle.odometer_km, 184113)   # priced figure corrected
        self.assertEqual(rep.widening, [])                 # corroborated, so no widening

    def test_a_lone_confident_read_still_only_flags_and_keeps_the_vision_figure(self):
        # One dashboard frame, no corroboration: the read could itself be the
        # misread, so the conflict is surfaced but the vision figure stays priced.
        ev = self._ev(odometer_km=500000)
        with unittest.mock.patch("app.odometer.read",
                                 return_value=self._read(305273, confidence=0.9)):
            rep = reconcile.apply(self._gate(), None, ev)
        self.assertEqual(len(rep.of_kind("odometer_conflict")), 1)
        self.assertEqual(len(rep.of_kind("odometer_overridden")), 0)
        self.assertEqual(ev.vehicle.odometer_km, 500000)   # unchanged

    def test_the_ocr_abstains_below_the_legibility_floor(self):
        # The engine itself: a token read under the floor yields no km, and the
        # reason names the confidence rather than emitting the digits.
        from app import odometer as O
        reading = O._select([("204113km", O.MIN_CONFIDENCE - 0.1, [0, 0, 10, 10])])
        self.assertIsNone(reading.km)
        self.assertEqual(reading.text, "204113km")       # what it saw is still reported
        self.assertIn("legibility floor", reading.reason)

    def test_a_legible_read_above_the_floor_is_kept(self):
        from app import odometer as O
        reading = O._select([("184113km", O.MIN_CONFIDENCE + 0.3, [0, 0, 10, 10])])
        self.assertEqual(reading.km, 184113)

    def test_the_rule_does_not_run_without_a_dashboard_frame(self):
        ev = self._ev(odometer_km=None)
        with unittest.mock.patch("app.odometer.read",
                                 side_effect=AssertionError("must not OCR")) as m:
            reconcile.apply(self._gate(view="exterior_side"), None, ev)
        m.assert_not_called()

    def test_reconcile_passes_the_gate_subject_box_to_ocr(self):
        box = [10, 20, 300, 250]
        gate = self._gate()
        gate.photos[0].subject_box = box
        with unittest.mock.patch("app.odometer.read",
                                 return_value=self._read(None)) as read:
            reconcile.apply(gate, None, self._ev())
        read.assert_called_once_with("/tmp/3.jpg", subject_box=box)


class VinReconciliation(unittest.TestCase):

    def _gate(self):
        return GateReport(photos=[_photo(7, view="chassis_undercarriage", usable=True,
                                                capture_quality=0.9)])

    @staticmethod
    def _read(year=2020):
        from app.vin import VinRead
        return VinRead(vin="3AKJHHDR3LSLJ7839", year=year, wmi="3AK",
                       check_ok=True, confidence=0.91)

    def test_a_missing_year_is_recovered_from_the_vin(self):
        ev = EvidenceReport(vehicle=VehicleRead())
        with unittest.mock.patch("app.vin.read", return_value=self._read()):
            rep = reconcile.apply(self._gate(), None, ev, {})
        self.assertEqual(len(rep.of_kind("vin_recovered")), 1)
        self.assertEqual(ev.vehicle.vin, "3AKJHHDR3LSLJ7839")
        self.assertEqual(ev.vehicle.vin_year, 2020)

    def test_a_conflicting_year_widens_but_keeps_the_declared_year_for_pricing(self):
        ev = EvidenceReport(vehicle=VehicleRead())
        with unittest.mock.patch("app.vin.read", return_value=self._read(2020)):
            rep = reconcile.apply(self._gate(), None, ev, {"year": 2024})
        self.assertEqual(len(rep.of_kind("vin_conflict")), 1)
        self.assertEqual([w[1] for w in rep.widening], [reconcile.VIN_CONFLICT_WIDENING])


class Anchor(unittest.TestCase):

    def test_the_reference_table_is_well_formed_and_cited(self):
        for row in anchor.reference()["rows"]:
            for key in ("brand", "model", "as_of", "source", "source_type", "source_url"):
                self.assertTrue(row.get(key), f"{row.get('model')} is missing {key}")
            self.assertIn(row["source_type"], anchor.SOURCE_WIDENING)

    def test_a_priced_row_is_found_by_brand_and_model(self):
        self.assertIsNotNone(anchor.lookup("Ford Trucks", "F-MAX"))

    def test_an_unknown_brand_has_no_anchor(self):
        self.assertIsNone(anchor.lookup("Kenworth", "T680"))

    def test_the_model_alias_beats_the_brand_default(self):
        # The corpus files pre-F-MAX Ford tractors under the model "TRUCKS";
        # their equivalent is the 1845T, not the flagship the default picks.
        self.assertEqual(anchor.lookup("FORD", "TRUCKS")["model"], "1845T")

    def test_a_row_with_no_price_is_not_offered_as_an_anchor(self):
        self.assertIsNone(anchor.lookup("SCANIA", "R-SERIES"))

    def test_the_anchor_is_turkish_market_only(self):
        est = anchor.estimate({"ok": True}, year=2021, km=100000, make="FORD",
                              model="F-MAX", market="US")
        self.assertFalse(est.ok)

    def test_blending_favours_the_more_precise_route(self):
        mu, sd, w = anchor.blend(math.log(100.0), 0.10, math.log(200.0), 0.10)
        self.assertAlmostEqual(w, 0.5, places=6)
        self.assertAlmostEqual(mu, (math.log(100.0) + math.log(200.0)) / 2, places=6)
        _, _, w_tight = anchor.blend(math.log(100.0), 0.40, math.log(200.0), 0.10)
        self.assertGreater(w_tight, 0.9)

    def test_blending_never_increases_the_error(self):
        _, sd, _ = anchor.blend(0.0, 0.10, 0.1, 0.10)
        self.assertLess(sd, 0.10)


class AnchorInEstimate(unittest.TestCase):
    """The fitted artifact has to actually reach the price."""

    @classmethod
    def setUpClass(cls):
        from app.pricing.model import load_model
        cls.model = load_model()
        if not cls.model.anchor.get("ok"):
            raise unittest.SkipTest("no retention curve - run app.pricing.train")

    def _estimate(self, **kw):
        from app.pricing.model import estimate
        return estimate(self.model, **kw)

    def test_an_unseen_brand_with_a_reference_price_beats_bare_widening(self):
        withref = self._estimate(year=2021, km=500000, make="MERCEDES-BENZ",
                                 vehicle_model="Actros")
        noref = self._estimate(year=2021, km=500000, make="SCANIA", vehicle_model="R450")
        self.assertTrue(withref.anchor.ok)
        self.assertFalse(noref.anchor.ok)
        self.assertLess(withref.high - withref.low, noref.high - noref.low)

    def test_the_band_never_narrows_below_the_measured_interval(self):
        # The 80.3% coverage belongs to the unwidened hedonic band. The anchor
        # may claw back a widening; it may not claim a tighter band than the
        # one that was actually measured.
        lo_off, hi_off = self.model.offsets["0.8"]
        est = self._estimate(year=2021, km=164374, make="FORD", vehicle_model="F-MAX")
        floor = math.exp(hi_off) / math.exp(lo_off)
        # Prices are rounded to the nearest 1000 for display, so the ratio of
        # two rounded ends can sit a hair under the exact one. Tolerate exactly
        # that much and no more: half a step on each end, relative to the band.
        tol = 1000.0 / est.baseline_low
        self.assertGreaterEqual(est.baseline_high / est.baseline_low, floor - tol)

    def test_a_missing_reference_degrades_to_the_shipped_behaviour(self):
        est = self._estimate(year=2021, km=500000, make="SCANIA", vehicle_model="R450")
        self.assertTrue(est.ok)
        self.assertTrue(any("1.85" in w for w in est.widened))

    def test_the_anchor_records_where_its_number_came_from(self):
        est = self._estimate(year=2021, km=164374, make="FORD", vehicle_model="F-MAX")
        self.assertTrue(est.anchor.source_url.startswith("http"))
        self.assertTrue(est.anchor.as_of)
        self.assertIn("retained", est.anchor.basis)


if __name__ == "__main__":
    unittest.main()
