"""The chassis plate as a brand witness, and the model card as a year witness.

Two rules live here, and both exist for the same reason: the trained identity
head may only dispute brands in its own class list, and the corpus has no
Scania, no DAF and no Volvo. A judge arriving with an unseen brand is the
measured 1.85x failure mode of this system, and until the WMI was consumed
nothing at all could cross-check the badge on one.

Everything here is offline. `vin.read` is monkeypatched, so no test needs
RapidOCR, an image, or a network call - what is under test is the
reconciliation, not the OCR engine.

The load-bearing assertions are the negative ones. A WMI absent from the table
means *unknown*, never *conflict*: it must never dispute a badge the vision
model can plainly read. That is the same discipline as "the identity head may
not dispute a brand it was never trained on", and it is the assertion that will
catch a future maintainer who decides an unknown three-character prefix is
evidence of a lie.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from app import reconcile, vin
from app.config import WMI_CONFLICT_WIDENING
from app.schema import EvidenceReport, GateReport, PhotoCheck, VehicleRead


def _photo(pid=7, **kw):
    check = PhotoCheck(photo_id=pid, path=f"/tmp/{pid}.jpg", filename=f"{pid}.jpg")
    check.usable = True
    check.view = "chassis_undercarriage"
    check.capture_quality = 0.9
    for key, value in kw.items():
        setattr(check, key, value)
    return check


def _gate(**kw):
    return GateReport(photos=[_photo(**kw)])


def _ev(make=None, **kw):
    return EvidenceReport(vehicle=VehicleRead(make=make, **kw))


def _read(wmi="NM0", year=None, check_ok=True, confidence=0.91):
    """A VinRead with a chosen WMI, built rather than OCR'd.

    `year=None` by default so a WMI test never also trips the year rules - the
    two halves of `_check_vin` are asserted separately on purpose.
    """
    body = "X" * (17 - len(wmi))
    return vin.VinRead(vin=f"{wmi}{body}", year=year, wmi=wmi,
                       check_ok=check_ok, confidence=confidence)


class WmiTable(unittest.TestCase):
    """data/reference/wmi.json: hand-curated, stamped and cited, like the anchor."""

    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads(vin.WMI_TABLE.read_text(encoding="utf-8"))["rows"]

    def test_every_row_names_a_three_character_wmi_and_a_brand(self):
        for row in self.rows:
            self.assertRegex(str(row.get("wmi", "")), r"^[A-Z0-9]{3}$")
            self.assertTrue(str(row.get("brand", "")).strip(), row.get("wmi"))

    def test_no_wmi_appears_twice(self):
        keys = [row["wmi"] for row in self.rows]
        self.assertEqual(len(keys), len(set(keys)), sorted(keys))

    def test_a_wmi_never_contains_the_three_forbidden_characters(self):
        # I, O and Q are excluded from a VIN by ISO 3779 precisely because they
        # are confusable with 1 and 0. `vin.normalise` substitutes them on the
        # way in, so a table key containing one could never be matched.
        for row in self.rows:
            self.assertNotRegex(row["wmi"], r"[IOQ]", row["wmi"])

    def test_every_verified_row_carries_the_source_url_it_was_checked_against(self):
        for row in self.rows:
            if row.get("verified"):
                self.assertTrue(str(row.get("source_url") or "").startswith("http"),
                                f"{row['wmi']} is verified with no source URL")
                self.assertTrue(row.get("source_type"), row["wmi"])

    def test_every_brand_survives_the_price_model_s_brand_normaliser(self):
        # `identity.collect` drops a witness whose brand folds to "other", so a
        # row that normalises to "other" would be silently inert.
        from app.pricing.features import normalise_brand
        for row in self.rows:
            self.assertNotEqual(normalise_brand(row["brand"]), "other", row["wmi"])

    def test_no_row_lists_a_spelling_the_brand_normaliser_already_folds(self):
        # A redundant alternate is not harmful but it is misleading: it implies
        # normalise_brand needs help where it does not, and the next person
        # adding a row copies it.
        from app.pricing.features import normalise_brand
        for row in self.rows:
            accepted = [row["brand"]] + (row.get("also_badged") or [])
            folded = [normalise_brand(b) for b in accepted]
            self.assertEqual(len(folded), len(set(folded)), row["wmi"])

    def test_every_european_brand_tolerates_the_trucks_suffix_on_the_badge(self):
        # What a vision model actually types. normalise_brand folds FORD TRUCKS
        # into FORD and nothing else, so every other marque needs the spelling
        # carried in the table or the rule disputes a badge that is correct.
        from app.pricing.features import normalise_brand
        by_brand = {}
        for row in self.rows:
            accepted = [row["brand"]] + (row.get("also_badged") or [])
            by_brand.setdefault(normalise_brand(row["brand"]), set()).update(
                normalise_brand(b) for b in accepted)
        for brand in ("SCANIA", "DAF", "VOLVO", "MAN", "MERCEDES-BENZ", "MACK"):
            spelling = normalise_brand(f"{brand} TRUCKS")
            self.assertIn(spelling, by_brand[brand],
                          f"a badge reading '{brand} Trucks' would be disputed")

    def test_the_table_covers_the_brands_the_price_fit_has_never_seen(self):
        # The entire point of this table. The hedonic fit is 78/84 Ford; these
        # are the brands a judge is most likely to arrive with.
        from app.pricing.features import normalise_brand
        covered = {normalise_brand(r["brand"]) for r in self.rows if r.get("verified")}
        for brand in ("SCANIA", "DAF", "VOLVO", "MAN", "MERCEDES-BENZ",
                      "IVECO", "RENAULT", "FORD"):
            self.assertIn(brand, covered)


class WmiLookup(unittest.TestCase):
    """`vin.wmi_brand`: deterministic, cached, and silent when it does not know."""

    def tearDown(self):
        vin._reset_wmi()

    def _table(self, rows):
        """Point the lookup at a temporary table for the duration of a with-block."""
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"rows": rows}, tmp)
        tmp.close()
        vin._reset_wmi()
        return unittest.mock.patch.object(vin, "WMI_TABLE", Path(tmp.name))

    def test_a_verified_row_returns_its_brand(self):
        self.assertEqual(vin.wmi_brand("NM0"), "FORD")

    def test_a_wmi_that_is_not_in_the_table_returns_none(self):
        # Absence is "unknown", never "conflict". ZZZ is not an assigned WMI.
        self.assertIsNone(vin.wmi_brand("ZZZ"))

    def test_an_unverified_row_is_ignored(self):
        with self._table([{"wmi": "AAA", "brand": "ACME", "verified": False}]):
            self.assertIsNone(vin.wmi_brand("AAA"))

    def test_an_unverified_row_is_readable_only_when_explicitly_asked_for(self):
        with self._table([{"wmi": "AAA", "brand": "ACME", "verified": False}]):
            self.assertEqual(vin.wmi_brand("AAA", allow_unverified=True), "ACME")

    def test_a_missing_table_returns_none_rather_than_raising(self):
        # Soft dependency, and loud about it: silence here would mean a repo
        # missing its reference file looked identical to one whose plates were
        # all unreadable.
        vin._reset_wmi()
        with unittest.mock.patch.object(vin, "WMI_TABLE", Path("/nonexistent/wmi.json")), \
             self.assertLogs("app.vin", level="WARNING"):
            self.assertIsNone(vin.wmi_brand("NM0"))

    def test_a_corrupt_table_returns_none_rather_than_raising(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        tmp.write("{not json")
        tmp.close()
        vin._reset_wmi()
        with unittest.mock.patch.object(vin, "WMI_TABLE", Path(tmp.name)), \
             self.assertLogs("app.vin", level="WARNING"):
            self.assertIsNone(vin.wmi_brand("NM0"))

    def test_case_and_whitespace_are_folded(self):
        self.assertEqual(vin.wmi_brand(" nm0 "), "FORD")

    def test_the_letters_a_vin_cannot_contain_are_substituted_on_the_way_in(self):
        # RapidOCR reads the letter O for the digit 0 constantly. `read` already
        # normalises before slicing, but a caller passing a raw string must land
        # in the same place.
        self.assertEqual(vin.wmi_brand("NMO"), "FORD")

    def test_no_wmi_at_all_returns_none(self):
        self.assertIsNone(vin.wmi_brand(None))
        self.assertIsNone(vin.wmi_brand(""))
        self.assertIsNone(vin.wmi_brand("NM"))

    def test_the_alternate_badges_include_the_primary_brand_first(self):
        self.assertEqual(vin.wmi_brands("NM0")[0], "FORD")

    def test_a_manufacturer_holding_several_wmis_maps_them_all_to_one_brand(self):
        # PACCAR builds Kenworths under more than one WMI; the table records
        # every one it verified rather than the first one someone found.
        self.assertEqual(vin.wmi_brand("1XK"), vin.wmi_brand("1NK"))


class WmiReconciliation(unittest.TestCase):
    """The rule: the plate's manufacturer against the badge the VLM read."""

    def _apply(self, make, reading, declared=None):
        with unittest.mock.patch.object(vin, "read", return_value=reading):
            ev = _ev(make=make)
            report = reconcile.apply(_gate(), None, ev, declared or {})
        return ev, report

    def test_a_wmi_that_is_not_in_the_table_never_disputes_the_badge(self):
        # The assertion this whole file exists for. ZZZ is unknown, the badge
        # plainly reads SCANIA, and the system must say nothing.
        ev, report = self._apply("Scania", _read(wmi="ZZZ"))
        self.assertEqual(report.of_kind("wmi_conflict"), [])
        self.assertEqual(report.widening, [])
        self.assertIsNone(ev.vehicle.wmi_brand)
        self.assertEqual(ev.vehicle.wmi, "ZZZ")     # still recorded, just mute

    def test_an_unverified_row_cannot_dispute_the_badge_either(self):
        with unittest.mock.patch.object(vin, "wmi_brands", return_value=[]):
            ev, report = self._apply("Scania", _read(wmi="AAA"))
        self.assertEqual(report.corrections, [])
        self.assertEqual(report.widening, [])

    def test_an_agreeing_wmi_is_silent_but_records_the_confirmation(self):
        # Silence matches every other rule in reconcile.py: they speak only when
        # they have something to say. The confirmation is not lost - it lands on
        # the vehicle, which is where `identity.collect` reads it from.
        ev, report = self._apply("Ford Trucks", _read(wmi="NM0"))
        self.assertEqual(report.corrections, [])
        self.assertEqual(report.widening, [])
        self.assertEqual(ev.vehicle.wmi, "NM0")
        self.assertEqual(ev.vehicle.wmi_brand, "FORD")

    def test_a_disagreeing_wmi_records_a_correction_and_widens(self):
        ev, report = self._apply("Scania", _read(wmi="NM0"))
        conflicts = report.of_kind("wmi_conflict")
        self.assertEqual(len(conflicts), 1)
        self.assertEqual([w[1] for w in report.widening], [WMI_CONFLICT_WIDENING])
        self.assertIn("SCANIA", conflicts[0].before.upper())
        self.assertIn("FORD", conflicts[0].after.upper())

    def test_a_disagreeing_wmi_keeps_the_badge_as_the_priced_make(self):
        # Same posture as odometer_conflict, which keeps the vision figure as
        # the priced one and widens. Nothing here overwrites a read.
        ev, _ = self._apply("Scania", _read(wmi="NM0"))
        self.assertEqual(ev.vehicle.make, "Scania")

    def test_the_widening_is_labelled_a_stated_assumption(self):
        _, report = self._apply("Scania", _read(wmi="NM0"))
        self.assertIn("assumption", report.widening[0][0].lower())

    def test_the_widening_carries_the_marker_the_identity_verdict_supersedes(self):
        # `identity.merge_widening` drops this multiplier once the verdict has
        # adjudicated the same disagreement. It matches on a literal substring,
        # so the label and that marker have to stay welded together.
        from app.identity import SUPERSEDED
        _, report = self._apply("Scania", _read(wmi="NM0"))
        self.assertTrue(any(marker in report.widening[0][0] for marker in SUPERSEDED),
                        report.widening[0][0])

    def test_a_badge_spelling_the_price_normaliser_does_not_fold_is_not_a_conflict(self):
        # normalise_brand folds FORD TRUCKS -> FORD but has no rule for RENAULT
        # TRUCKS -> RENAULT, so the table carries the badge spellings itself.
        ev, report = self._apply("Renault Trucks", _read(wmi="VF6"))
        self.assertEqual(report.corrections, [])
        self.assertEqual(report.widening, [])
        # The spelling written onto the vehicle is the one that agreed, so the
        # identity verdict downstream sees agreement rather than re-deriving it.
        self.assertEqual(ev.vehicle.wmi_brand, "RENAULT TRUCKS")

    def test_no_badge_read_at_all_is_not_a_conflict(self):
        ev, report = self._apply(None, _read(wmi="NM0"))
        self.assertEqual(report.corrections, [])
        self.assertEqual(report.widening, [])
        self.assertEqual(ev.vehicle.wmi_brand, "FORD")   # still a witness

    def test_no_vin_read_leaves_the_vehicle_untouched(self):
        ev, report = self._apply("Scania", vin.VinRead(reason="no candidate"))
        self.assertIsNone(ev.vehicle.wmi)
        self.assertIsNone(ev.vehicle.wmi_brand)
        self.assertEqual(report.corrections, [])

    def test_an_unavailable_ocr_engine_no_ops_rather_than_raising(self):
        # Soft dependency, exactly like the odometer rule: absent RapidOCR the
        # rule does nothing and the appraisal continues.
        with unittest.mock.patch.object(vin, "read",
                                        side_effect=ImportError("no rapidocr")):
            ev = _ev(make="Scania")
            report = reconcile.apply(_gate(), None, ev, {})
        self.assertEqual(report.corrections, [])
        self.assertIsNone(ev.vehicle.wmi)

    def test_the_rule_reuses_the_single_ocr_pass_the_year_rule_already_made(self):
        # One read per frame. Doubling the OCR cost to ask the same plate a
        # second question would be the obvious wrong way to build this.
        with unittest.mock.patch.object(vin, "read",
                                        return_value=_read(wmi="NM0")) as read:
            reconcile.apply(_gate(), None, _ev(make="Ford"), {})
        self.assertEqual(read.call_count, 1)


class VinYearGuard(unittest.TestCase):
    """The year half of `_check_vin`, and why it may not speak for every VIN."""

    def test_a_european_vin_is_not_given_a_north_american_model_year(self):
        # Position 10 is a model-year code under FMVSS 565, a North American
        # rule; ISO 3779 does not mandate it and European heavy trucks do not
        # follow it. This VIN is real - it is in the VIN list published with
        # Australian Government recall REC-006624, covering DAF CF tractors
        # built between 2019 and 2025 - and the North American decode makes it
        # a 1994 truck.
        from app.vin import check_digit_ok, model_year, north_american
        european = "XLRATM430RG524824"
        self.assertFalse(north_american(european))
        self.assertFalse(check_digit_ok(european))
        self.assertIsNone(model_year(european))

    def test_the_region_test_and_not_only_the_check_digit_decides(self):
        # Roughly one VIN in eleven satisfies the check digit by chance. Left
        # to the check digit alone, ~9% of European trucks would be handed a
        # year off a character that was never a year, and `pricing/model.py`
        # falls back to `vin_year` when nothing else supplies one.
        from app.vin import check_digit_ok, model_year
        # The DAF VIN above with character 9 set to the digit the North
        # American algorithm wants - which is what one European VIN in eleven
        # looks like anyway.
        lucky = "XLRATM438RG524824"
        self.assertTrue(check_digit_ok(lucky))
        self.assertIsNone(model_year(lucky))             # region still says no

    def test_a_vin_whose_check_digit_fails_does_not_price_or_dispute_a_year(self):
        # The reconcile half of the same guard: even handed a decoded year, the
        # rule declines to price or dispute on a VIN that failed its check
        # digit. Acting on it would hand pricing/model.py a 1994 year for a
        # 2023 truck and raise a confident, wrong `vin_conflict` at the seller.
        european = "XLRATM430RG524824"
        reading = vin.VinRead(vin=european, year=1994, wmi="XLR",
                              check_ok=False, confidence=0.9)
        with unittest.mock.patch.object(vin, "read", return_value=reading):
            ev = _ev(make="DAF")
            report = reconcile.apply(_gate(), None, ev, {"year": 2023})
        self.assertIsNone(ev.vehicle.vin_year)
        self.assertEqual(report.of_kind("vin_conflict"), [])
        self.assertEqual(report.of_kind("vin_recovered"), [])
        self.assertEqual(ev.vehicle.vin, european)      # the read is still shown

    def test_the_brand_cross_check_still_runs_on_that_same_vin(self):
        # The check digit gates the YEAR, not the manufacturer. Every candidate
        # in the table is a distinct three-character prefix, so an OCR slip that
        # lands on a different brand's registered WMI is a different and much
        # less likely failure than a year code that was never meant to be read.
        reading = vin.VinRead(vin="XLRATM430RG524824", year=1994, wmi="XLR",
                              check_ok=False, confidence=0.9)
        with unittest.mock.patch.object(vin, "read", return_value=reading):
            ev = _ev(make="DAF")
            reconcile.apply(_gate(), None, ev, {"year": 2023})
        self.assertEqual(ev.vehicle.wmi_brand, "DAF")

    def test_a_valid_check_digit_still_recovers_a_missing_year(self):
        # The pre-existing behaviour, pinned so the guard above cannot quietly
        # turn the whole year rule off.
        reading = vin.VinRead(vin="3AKJHHDR3LSLJ7839", year=2020, wmi="3AK",
                              check_ok=True, confidence=0.91)
        with unittest.mock.patch.object(vin, "read", return_value=reading):
            ev = _ev(make="Freightliner")
            report = reconcile.apply(_gate(), None, ev, {})
        self.assertEqual(len(report.of_kind("vin_recovered")), 1)
        self.assertEqual(ev.vehicle.vin_year, 2020)


class GenerationReconciliation(unittest.TestCase):
    """The model card's year spans against the generation read off the pixels."""

    def _apply(self, vehicle_kw, declared=None, generations=None):
        ev = EvidenceReport(vehicle=VehicleRead(**vehicle_kw))
        gate = GateReport(photos=[])             # no frames: the VIN rule no-ops
        patches = []
        if generations is not None:
            from app import modelspec
            patches.append(unittest.mock.patch.object(
                modelspec, "generations", return_value=generations))
            patches.append(unittest.mock.patch.object(
                modelspec, "generation_for_year",
                side_effect=lambda make, model, year: next(
                    (g for g in generations if _spans(g, year)), None)))
        for patch in patches:
            patch.start()
        try:
            return ev, reconcile.apply(gate, None, ev, declared or {})
        finally:
            for patch in patches:
                patch.stop()

    def test_it_no_ops_when_the_card_has_no_generations_for_that_model(self):
        # A rule that fired on an empty card would fire on every appraisal in
        # the repo. `models_tr.json` shipped with every `generations` list
        # empty; the research has since landed, so the empty card is asserted
        # by patching rather than by relying on the file staying bare.
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX", "model_canonical": "F-MAX",
             "generation": "mk2", "generation_conf": 0.9},
            declared={"year": 2021}, generations=[])
        self.assertEqual(report.of_kind("generation_conflict"), [])
        self.assertEqual(report.widening, [])

    def test_it_no_ops_for_a_model_the_real_card_has_never_heard_of(self):
        # Against the shipped card, unpatched. A judge's Kenworth has no row,
        # so there is no span to check a year against and nothing to say.
        _, report = self._apply(
            {"make": "Kenworth", "model": "T680",
             "generation": "t680-2012", "generation_conf": 0.95},
            declared={"year": 2019})
        self.assertEqual(report.corrections, [])

    def test_the_shipped_card_agrees_with_itself_on_a_real_f_max(self):
        # The path that will actually run in the room: the corpus is 78/84
        # Ford and the F-MAX row now carries two generations.
        from app import modelspec
        gens = modelspec.generations("Ford Trucks", "F-MAX")
        self.assertTrue(gens, "the F-MAX row lost its generations")
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX",
             "generation": gens[0]["id"], "generation_conf": 0.9},
            declared={"year": 2021})
        self.assertEqual(report.corrections, [])

    def test_a_generation_id_the_card_does_not_list_is_not_a_disagreement(self):
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX",
             "generation": "mk7", "generation_conf": 0.95},
            declared={"year": 2021},
            generations=[{"id": "mk1", "years": "2018-2022"}])
        self.assertEqual(report.corrections, [])

    def test_overlapping_generations_do_not_manufacture_a_conflict(self):
        # Manufacturers sell an old and a new range side by side through a
        # changeover and the card records that: Scania's R-series runs
        # 2004-2017 and the next-generation cab starts in 2016, so a 2017
        # truck is legitimately either. `generation_for_year` returns the
        # first match, so testing agreement with it would call a correctly
        # read next-gen 2017 Scania a conflict.
        _, report = self._apply(
            {"make": "Scania", "model": "R-SERIES",
             "generation": "scania-next-gen-2016", "generation_conf": 0.9},
            declared={"year": 2017},
            generations=[{"id": "scania-r-2004", "years": "2004-2017"},
                         {"id": "scania-next-gen-2016", "years": "2016-present"}])
        self.assertEqual(report.corrections, [])

    def test_it_no_ops_when_the_vision_pass_read_no_generation(self):
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX", "generation": None},
            declared={"year": 2021},
            generations=[{"id": "mk1", "years": "2018-2022"}])
        self.assertEqual(report.corrections, [])

    def test_it_no_ops_when_no_year_is_available_from_any_source(self):
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX",
             "generation": "mk1", "generation_conf": 0.9},
            declared={},
            generations=[{"id": "mk1", "years": "2018-2022"}])
        self.assertEqual(report.corrections, [])

    def test_an_agreeing_generation_is_silent(self):
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX",
             "generation": "mk1", "generation_conf": 0.9},
            declared={"year": 2021},
            generations=[{"id": "mk1", "years": "2018-2022"},
                         {"id": "mk2", "years": "2023-present"}])
        self.assertEqual(report.corrections, [])

    def test_a_disagreeing_generation_records_a_correction(self):
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX", "generation": "mk2",
             "generation_conf": 0.9, "year_evidence": "one-piece grille surround"},
            declared={"year": 2021},
            generations=[{"id": "mk1", "years": "2018-2022"},
                         {"id": "mk2", "years": "2023-present"}])
        conflicts = report.of_kind("generation_conflict")
        self.assertEqual(len(conflicts), 1)
        self.assertIn("mk2", conflicts[0].before)
        self.assertIn("mk1", conflicts[0].after)
        self.assertIn("2021", conflicts[0].detail)

    def test_a_disagreeing_generation_does_not_widen_the_band(self):
        # It records a disagreement about a trim-level fact. The band belongs
        # to the identity verdict and the year, neither of which this changes.
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX",
             "generation": "mk2", "generation_conf": 0.9},
            declared={"year": 2021},
            generations=[{"id": "mk1", "years": "2018-2022"},
                         {"id": "mk2", "years": "2023-present"}])
        self.assertEqual(report.widening, [])

    def test_a_year_outside_every_span_on_the_card_is_silent(self):
        # An incomplete card cannot adjudicate. Same discipline as an unknown
        # WMI: the card not covering a year means unknown, never conflict.
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX",
             "generation": "mk1", "generation_conf": 0.9},
            declared={"year": 2009},
            generations=[{"id": "mk1", "years": "2018-2022"}])
        self.assertEqual(report.corrections, [])

    def test_an_unknown_generation_is_not_a_disagreement(self):
        # "unknown" is the honest answer the closed enum offers; disputing it
        # would punish the model for declining to guess.
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX",
             "generation": "unknown", "generation_conf": 0.9},
            declared={"year": 2021},
            generations=[{"id": "mk1", "years": "2018-2022"}])
        self.assertEqual(report.corrections, [])

    def test_a_low_confidence_generation_read_is_not_promoted_to_a_dispute(self):
        _, report = self._apply(
            {"make": "Ford Trucks", "model": "F-MAX",
             "generation": "mk2", "generation_conf": 0.2},
            declared={"year": 2021},
            generations=[{"id": "mk1", "years": "2018-2022"},
                         {"id": "mk2", "years": "2023-present"}])
        self.assertEqual(report.corrections, [])

    def test_the_vin_year_is_used_when_the_seller_declared_nothing(self):
        ev = EvidenceReport(vehicle=VehicleRead(
            make="Ford Trucks", model="F-MAX", generation="mk2",
            generation_conf=0.9, vin_year=2021))
        from app import modelspec
        gens = [{"id": "mk1", "years": "2018-2022"},
                {"id": "mk2", "years": "2023-present"}]
        with unittest.mock.patch.object(modelspec, "generations", return_value=gens), \
             unittest.mock.patch.object(
                 modelspec, "generation_for_year",
                 side_effect=lambda make, model, year: next(
                     (g for g in gens if _spans(g, year)), None)):
            report = reconcile.apply(GateReport(photos=[]), None, ev, {})
        self.assertEqual(len(report.of_kind("generation_conflict")), 1)


def _spans(generation: dict, year) -> bool:
    """The half of `modelspec.generation_for_year` the fake needs."""
    from app.modelspec import _year_span
    if not year:
        return False
    low, high = _year_span(generation.get("years"))
    return low is not None and low <= int(year) <= (high or 9999)


if __name__ == "__main__":
    unittest.main()
