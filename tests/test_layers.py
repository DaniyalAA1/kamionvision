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
import unittest
import unittest.mock

import numpy as np

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

    def test_the_rule_does_not_run_without_a_dashboard_frame(self):
        ev = self._ev(odometer_km=None)
        with unittest.mock.patch("app.odometer.read",
                                 side_effect=AssertionError("must not OCR")) as m:
            reconcile.apply(self._gate(view="exterior_side"), None, ev)
        m.assert_not_called()


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


class StackWeight(unittest.TestCase):
    """The learned blend share, which replaced inverse variance where measured."""

    def test_a_perfect_route_takes_all_the_weight(self):
        from app.pricing.train import _stack_weight
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(_stack_weight(y, y + [0.3, -0.2, 0.1, -0.4], y), 1.0)
        self.assertAlmostEqual(_stack_weight(y, y, y + [0.3, -0.2, 0.1, -0.4]), 0.0)

    def test_the_weight_is_a_share(self):
        from app.pricing.train import _stack_weight
        rng = np.random.default_rng(0)
        y = rng.normal(size=50)
        w = _stack_weight(y, y + rng.normal(size=50), y - 3 + rng.normal(size=50))
        self.assertTrue(0.0 <= w <= 1.0)


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

    def _band_ratio_at_least(self, est, offsets):
        lo_off, hi_off = offsets
        floor = math.exp(hi_off) / math.exp(lo_off)
        # Prices are rounded to the nearest 1000 for display, so the ratio of
        # two rounded ends can sit a hair under the exact one. Tolerate exactly
        # that much and no more: half a step on each end, relative to the band.
        tol = 1000.0 / est.baseline_low
        self.assertGreaterEqual(est.baseline_high / est.baseline_low, floor - tol)

    def test_the_band_never_narrows_below_the_measured_interval(self):
        # An unseen make keeps the inverse-variance blend, which was never
        # measured, so it may claw back a widening but never claim a band
        # tighter than the hedonic one whose coverage was.
        est = self._estimate(year=2021, km=500000, make="MERCEDES-BENZ",
                             vehicle_model="Actros")
        self.assertEqual(est.model_card["estimator"], "hedonic")
        self._band_ratio_at_least(est, self.model.offsets["0.8"])

    def test_a_measured_blend_serves_its_own_band_and_its_own_numbers(self):
        # Ford with an official new price is the case the blend was scored on:
        # the band and the coverage printed beside it must both be the blend's.
        blend = self.model.anchor.get("blend") or {}
        if not blend.get("ok"):
            self.skipTest("no measured blend - run app.pricing.train")
        est = self._estimate(year=2021, km=164374, make="FORD", vehicle_model="F-MAX")
        self.assertEqual(est.model_card["estimator"], "blend")
        self.assertEqual(est.model_card["coverage"], blend["calibration"]["coverage_0.8"])
        self.assertAlmostEqual(est.anchor.weight, blend["weight_anchor"], places=3)
        self._band_ratio_at_least(est, blend["offsets"]["0.8"])

    def test_the_blend_band_is_labelled_honestly(self):
        # Narrowing is only allowed because it was measured; an "80%" band that
        # covers far off 80% on held-out listings would be a claim, not a fact.
        blend = self.model.anchor.get("blend") or {}
        if not blend.get("ok"):
            self.skipTest("no measured blend - run app.pricing.train")
        self.assertTrue(0.0 <= blend["weight_anchor"] <= 1.0)
        self.assertAlmostEqual(blend["calibration"]["coverage_0.8"], 0.8, delta=0.05)

    def test_a_trade_press_price_does_not_borrow_the_measured_blend(self):
        from app.pricing import anchor as A
        row = A.lookup("MERCEDES-BENZ", "Actros")
        self.assertEqual(row["source_type"], "trade_press")
        est = self._estimate(year=2021, km=500000, make="MERCEDES-BENZ",
                             vehicle_model="Actros")
        self.assertNotEqual(est.model_card["estimator"], "blend")

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
