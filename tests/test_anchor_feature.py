"""The model name reaches the price, twice over.

Two changes are covered here and they are halves of one thing - until both
landed, the model of the truck was a string that the pricing stage carried
around and never used.

  * `anchor.lookup` folds free text through `modelspec` before matching, so a
    vision model answering "F Max" or "Cargo 1848T" reaches the row it means
    instead of falling through to the brand default without saying so.
  * `features` carries log of that published price as a COLUMN in the hedonic
    design, which is what lets an F-MAX and a Cargo-derived tractor of the same
    age and mileage be different trucks to the regression.

The second one changed the shipped numbers, so a good share of what follows
pins the things that make those numbers honest rather than the numbers: that
the interval is still built out-of-fold, that no currency conversion got inside
the fit, that the blend can never claim a band tighter than the one whose
coverage was measured, and that a reference file which is simply not there
degrades to the old behaviour rather than raising.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import modelspec
from app.config import LISTINGS_CSV, USD_TRY
from app.pricing import anchor, load_model
from app.pricing import features as F
from app.pricing import model as M
from app.pricing import train as T


class WithoutTheReferenceFiles:
    """Context manager: run with one or both reference tables off disk.

    Both modules cache their parsed file in a module global, so the caches have
    to be dropped on the way in and on the way out or the next test inherits an
    empty table.
    """

    def __init__(self, *, specs: bool = False, prices: bool = False):
        self.specs, self.prices = specs, prices

    def __enter__(self):
        self._specs_path = modelspec.MODEL_SPECS
        self._prices_path = anchor.NEW_PRICES
        missing = Path("/nonexistent/reference/not-here.json")
        if self.specs:
            modelspec.MODEL_SPECS = missing
        if self.prices:
            anchor.NEW_PRICES = missing
        self._reset()
        return self

    def __exit__(self, *exc):
        modelspec.MODEL_SPECS = self._specs_path
        anchor.NEW_PRICES = self._prices_path
        self._reset()
        return False

    @staticmethod
    def _reset():
        modelspec._reset()
        anchor._REFERENCE = None
        F._REFERENCE_LOG_MEAN = None


def old_lookup(make, model):
    """`anchor.lookup` exactly as it read before the fold was added.

    Kept here rather than described, because the soft-dependency claim is
    "identical to the old behaviour" and the only way to assert that is to have
    the old behaviour to hand.
    """
    try:
        rows = [r for r in anchor.reference()["rows"] if r.get("list_price")]
    except FileNotFoundError:
        return None
    brand = F.normalise_brand(make)
    if brand == "other":
        return None
    wanted = str(model or "").strip().upper()
    wanted = anchor.MODEL_ALIASES.get((brand, wanted), wanted)
    for row in rows:
        if row["brand"] == brand and str(row["model"]).upper() == wanted:
            return row
    for row in rows:
        if row["brand"] == brand and row.get("default"):
            return row
    return None


def matched(make, model):
    row = anchor.lookup(make, model)
    return f"{row['brand']} {row['model']}" if row else None


class AnchorLookupFoldsTheModel(unittest.TestCase):
    """Task A: the row a free-text model string actually reaches."""

    def test_every_spelling_of_the_f_max_reaches_the_f_max_row(self):
        # The identity pass reads a badge. It does not read the corpus's
        # spelling of a badge.
        for spelling in ("F-MAX", "F Max", "FMAX", "f-max", "F-MAX 500",
                         "Ford Trucks F-MAX"):
            with self.subTest(spelling):
                self.assertEqual(matched("FORD", spelling), "FORD F-MAX")

    def test_the_cargo_family_reaches_the_row_that_replaced_it(self):
        # Every one of these is a pre-F-MAX Ford Otosan tractor. None of them
        # is buyable new, so the anchor borrows the 1845T's price.
        for spelling in ("TRUCKS", "CARGO", "Cargo", "1848T", "1845T",
                         "Ford Trucks", "Cargo 1848T"):
            with self.subTest(spelling):
                self.assertEqual(matched("FORD", spelling), "FORD 1845T")

    def test_the_fold_runs_before_the_reference_row_alias_and_not_after(self):
        """The ordering is the whole of this change and it is invisible if wrong.

        `MODEL_ALIASES` is keyed on canonical ids. Apply it first and "CARGO"
        is not yet "TRUCKS", so it never matches and the truck is priced off an
        F-MAX - an 8.6% error in the anchor, silently.
        """
        self.assertEqual(modelspec.normalise_model("FORD", "CARGO"), "TRUCKS")
        self.assertIn(("FORD", "TRUCKS"), anchor.MODEL_ALIASES)
        self.assertEqual(matched("FORD", "CARGO"), "FORD 1845T")
        self.assertNotEqual(matched("FORD", "CARGO"), matched("FORD", "F-MAX"))

    def test_the_two_ford_rows_are_a_real_price_difference(self):
        """If they were the same figure none of the above would matter."""
        fmax = anchor.lookup("FORD", "F Max")["list_price"]
        cargo = anchor.lookup("FORD", "Cargo 1848T")["list_price"]
        self.assertGreater(abs(math.log(cargo / fmax)), 0.05)

    def test_a_model_the_table_has_never_heard_of_still_gets_the_brand_default(self):
        # MAN made a TGA. Nobody sells one new, and there is no TGA row.
        row = anchor.lookup("MAN", "TGA")
        self.assertIsNotNone(row)
        self.assertTrue(row.get("default"))
        self.assertEqual(row["model"], "TGX")

    def test_a_brand_with_no_rows_at_all_is_still_none(self):
        for make in ("FREIGHTLINER", "KENWORTH", "PETERBILT", None, ""):
            with self.subTest(make):
                self.assertIsNone(anchor.lookup(make, "ANYTHING"))

    def test_a_deliberate_null_list_price_is_never_returned(self):
        """Scania's Turkish list exists only as a JPG and the row says so.

        A row without a figure is a documented gap, not a price of zero.
        """
        self.assertIsNone(anchor.lookup("SCANIA", "R500"))
        self.assertIsNone(anchor.lookup("VOLVO", "FH 500"))
        self.assertIsNone(anchor.lookup("DAF", "XF 106"))

    def test_the_fold_is_a_soft_dependency_on_the_spec_card(self):
        """With models_tr.json off disk, lookup is byte-for-byte what it was."""
        pairs = [("FORD", "F-MAX"), ("FORD", "TRUCKS"), ("FORD", "F Max"),
                 ("FORD", "CARGO"), ("MAN", "TGS"), ("MAN", "TGA"),
                 ("MERCEDES-BENZ", "ACTROS"), ("SCANIA", "R500"),
                 ("FREIGHTLINER", "CASCADIA 126"), ("FORD", None)]
        with WithoutTheReferenceFiles(specs=True):
            self.assertFalse(modelspec.available())
            for make, model in pairs:
                with self.subTest(make=make, model=model):
                    self.assertEqual(anchor.lookup(make, model),
                                     old_lookup(make, model))

    def test_the_corpus_strings_were_already_right_and_still_are(self):
        """The honest half of the measurement: listings.csv does not move.

        Every (make, model) in the packaged corpus already matched exactly,
        because the reference table was hand-built against those very strings.
        The fold is for what a vision model says, not for what the CSV says -
        and a change that quietly re-pointed a corpus row at a different price
        would be a regression, not a win.
        """
        listings = pd.read_csv(LISTINGS_CSV)
        for make, model in {(r.make, r.model) for r in listings.itertuples()}:
            with self.subTest(make=make, model=model):
                self.assertEqual(anchor.lookup(make, model), old_lookup(make, model))


class TheNewPriceColumn(unittest.TestCase):
    """Task B: what the truck cost new, as a term in the hedonic fit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_the_shipped_design_carries_it(self):
        self.assertTrue(self.model.uses_new_price)
        self.assertEqual(self.model.columns[-2:], F.NEW_PRICE_COLUMNS)
        self.assertTrue(self.model.new_price.get("in_design"))

    def test_the_column_order_agrees_across_the_three_places_that_build_it(self):
        """`design_columns`, `to_vector` and `matrix` are one design or none.

        A disagreement here does not raise: it silently multiplies a
        coefficient by the wrong number.
        """
        listings = pd.read_csv(LISTINGS_CSV)
        df = F.build_frame(listings)
        brands = F.brand_vocabulary(df)
        for flag in (True, False):
            with self.subTest(new_price=flag):
                cols = F.design_columns(brands, new_price=flag)
                mat = F.matrix(df, brands, new_price=flag)
                feats = F.row_features(2021, 300000, "FORD", "TR",
                                       vehicle_model="F-MAX")
                vec = F.to_vector(feats, brands, new_price=flag)
                self.assertEqual(len(cols), mat.shape[1])
                self.assertEqual(len(cols), len(vec))
        self.assertEqual(len(self.model.columns), len(self.model.coef))
        self.assertEqual(len(self.model.columns), len(self.model.mean))

    def test_turning_it_off_leaves_exactly_the_old_design(self):
        listings = pd.read_csv(LISTINGS_CSV)
        df = F.build_frame(listings)
        brands = F.brand_vocabulary(df)
        old = F.design_columns(brands, new_price=False)
        self.assertEqual(old, F.BASE_COLUMNS + [f"brand_{b}" for b in brands])
        self.assertEqual(F.matrix(df, brands, new_price=False).shape[1], len(old))

    def test_the_model_name_now_changes_the_price(self):
        """The point of the whole exercise, stated as an assertion.

        Same year, same kilometres, same brand, different model. Before this
        column these two were the same truck to the regression.
        """
        fmax = M.estimate(self.model, year=2019, km=500000, make="FORD",
                          market="TR", vehicle_model="F-MAX")
        cargo = M.estimate(self.model, year=2019, km=500000, make="FORD",
                           market="TR", vehicle_model="Cargo 1848T")
        self.assertGreater(fmax.baseline_point, cargo.baseline_point)
        self.assertGreater(abs(math.log(fmax.baseline_point / cargo.baseline_point)), 0.02)

    def test_free_text_and_the_corpus_spelling_price_identically(self):
        """Task A reaching Task B: the fold has to survive into the estimate."""
        def point(model_string):
            return M.estimate(self.model, year=2016, km=600000, make="FORD",
                              market="TR", vehicle_model=model_string).point
        self.assertEqual(point("TRUCKS"), point("Cargo 1848T"))
        self.assertEqual(point("TRUCKS"), point("1845T"))
        self.assertEqual(M.estimate(self.model, year=2021, km=300000, make="FORD",
                                    market="TR", vehicle_model="F Max").point,
                         M.estimate(self.model, year=2021, km=300000, make="FORD",
                                    market="TR", vehicle_model="F-MAX").point)

    def test_a_truck_with_no_published_new_price_gets_nothing_from_the_column(self):
        """Missing is a defined state, and its contribution is exactly zero.

        Not "the truck is priced as it was before the column existed" - adding
        a column refits the others and it is not. The claim being pinned is
        narrower and is the one that matters: the column itself says nothing
        about a truck nobody publishes a price for.
        """
        feats = F.row_features(2019, 500000, "SCANIA", "TR", vehicle_model="R500")
        self.assertIsNone(feats["log_new_price"])
        vector = self.model.vector(feats)
        z = (vector - np.array(self.model.mean)) / np.array(self.model.scale)
        parts = z * np.array(self.model.coef)
        for col in F.NEW_PRICE_COLUMNS:
            i = self.model.columns.index(col)
            self.assertLess(abs(float(parts[i])), 1e-4,
                            f"{col} moved the price of a truck it knows nothing about")

    def test_the_column_is_clipped_to_the_support_it_was_fitted_on(self):
        """A four-point support and a +2.4 slope is a short lever.

        Without the clip, a reference row for a 20M-lira truck would be
        extrapolated into a price uplift nothing in this corpus supports.
        """
        lo, hi = self.model.new_price["log_range"]
        i = self.model.columns.index("log_new_price")
        for value, expected in ((math.log(20_000_000), hi),
                                (math.log(1_000_000), lo),
                                ((lo + hi) / 2, (lo + hi) / 2)):
            with self.subTest(value=value):
                feats = F.row_features(2019, 500000, "FORD", "TR")
                feats["log_new_price"] = value
                vec = self.model.vector(feats)
                self.assertAlmostEqual(float(vec[i]), expected, places=6)

    def test_the_reference_table_is_turkish_and_the_column_says_so(self):
        """A lira figure beside a dollar listing is not a constant shift.

        The market dummy absorbs an FX constant exactly. It cannot absorb a
        per-row TRY price attached to some US listings and not others.
        """
        self.assertEqual(tuple(F.NEW_PRICE_MARKETS), ("TR",))
        self.assertIsNone(F.new_price_for("FORD", "F-MAX", "US"))
        self.assertIsNone(F.new_price_for("FORD", "F-MAX", "EU"))
        self.assertIsNotNone(F.new_price_for("FORD", "F-MAX", "TR"))
        df = F.build_frame(pd.read_csv(LISTINGS_CSV))
        non_tr = df[df.market.str.upper() != "TR"]
        self.assertTrue(non_tr.log_new_price.isna().all())

    def test_the_whole_column_is_a_soft_dependency(self):
        """Neither reference file on disk: a defined vector, not an exception."""
        with WithoutTheReferenceFiles(specs=True, prices=True):
            feats = F.row_features(2021, 300000, "FORD", "TR", vehicle_model="F-MAX")
            self.assertIsNone(feats["log_new_price"])
            vec = F.to_vector(feats, self.model.brands, new_price=True)
            self.assertEqual(len(vec), len(self.model.columns))
            self.assertTrue(np.isfinite(vec).all())
            df = F.build_frame(pd.read_csv(LISTINGS_CSV))
            mat = F.matrix(df, F.brand_vocabulary(df), new_price=True)
            self.assertTrue(np.isfinite(mat).all())


class TheGateItHadToPass(unittest.TestCase):
    """The measurement, and the rule that decided on it.

    CLAUDE.md's instruction about this fit is that a change has to beat the
    numbers out-of-fold or not ship. These pin the shape of that argument - the
    rule, the basis, the folds - rather than the digits, which move on a refit.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.gate = cls.model.new_price.get("gate", {})

    def test_both_sides_of_the_comparison_are_recorded(self):
        for key in ("r2_without", "r2_with", "median_ape_without", "median_ape_with",
                    "sigma_without", "sigma_with",
                    "coverage_0.8_without", "coverage_0.8_with"):
            self.assertIn(key, self.gate)

    def test_the_shipped_decision_follows_the_rule_it_states(self):
        gain = self.gate["r2_with"] - self.gate["r2_without"]
        loss = self.gate["coverage_0.8_without"] - self.gate["coverage_0.8_with"]
        expected = (gain >= T.NEW_PRICE_MIN_GAIN
                    and loss <= T.NEW_PRICE_MAX_COVERAGE_LOSS)
        self.assertEqual(bool(self.model.new_price["in_design"]), expected)
        self.assertTrue(self.model.uses_new_price, "the column shipped, so it won")

    def test_it_did_not_buy_accuracy_with_calibration(self):
        """The band is the product. The point estimate is not.

        An R2 that improves while the 80% interval stops covering 80% is a
        worse model wearing a better number, and the gate refuses that trade.
        """
        self.assertGreaterEqual(
            self.gate["coverage_0.8_with"],
            self.gate["coverage_0.8_without"] - T.NEW_PRICE_MAX_COVERAGE_LOSS)
        self.assertGreater(self.gate["coverage_0.8_with"], 0.75)
        self.assertLess(self.gate["coverage_0.8_with"], 0.88)

    def test_the_measurement_is_out_of_fold_and_grouped_by_spec(self):
        """In-sample residuals on 24 spec groups once produced an 80% band that
        covered 70%. The basis string is where that lesson is written down."""
        basis = self.gate.get("basis", "").lower()
        self.assertIn("out-of-fold", basis)
        self.assertIn("(market, brand, model, year, price)",
                      self.model.meta.get("grouping", ""))

    def test_the_interval_still_comes_from_out_of_fold_residuals(self):
        """The offsets on the artifact must be wider than the in-sample spread.

        A ridge fit on this many groups has in-sample residuals far tighter
        than its real prediction error, so an interval built from them would be
        visibly narrower than `residual_std` implies. This is the cheap check
        that the offsets did not quietly come from the wrong residuals.
        """
        lo, hi = self.model.offsets["0.8"]
        listings = pd.read_csv(LISTINGS_CSV)
        df = F.build_frame(listings)
        fit_df = df[df.market.str.upper() == "TR"]
        brands = F.brand_vocabulary(fit_df, min_n=3)
        X, y = F.matrix(fit_df, brands), fit_df.y.to_numpy()
        ridge, mean, scale = T._fit(X, y)
        in_sample = y - T._predict(ridge, mean, scale, X)
        self.assertGreater(hi - lo, float(np.std(in_sample)) * 2)
        self.assertGreater(self.model.residual_std, float(np.std(in_sample)))

    def test_the_folds_are_grouped_by_spec_not_by_row(self):
        df = F.build_frame(pd.read_csv(LISTINGS_CSV))
        tr = df[df.market.str.upper() == "TR"]
        self.assertLess(tr.group.nunique(), len(tr))
        sample = tr.group.iloc[0]
        self.assertEqual(len(str(sample).split("|")), 5)

    def test_the_reported_sigma_is_the_one_the_cap_uses(self):
        self.assertAlmostEqual(self.model.residual_std,
                               self.gate["sigma_with"], places=4)
        self.assertLess(self.gate["sigma_with"], self.gate["sigma_without"])

    def test_the_column_is_labelled_measured_and_assumed_separately(self):
        basis = self.model.new_price["basis"]
        self.assertIn("MEASURED", basis)
        self.assertIn("ASSUMED", basis)
        # The clip is the assumed half, and it has to say what it rests on.
        self.assertIn("clipped", basis.lower())
        lo, hi = self.model.new_price["log_range"]
        self.assertLess(lo, hi)
        self.assertGreaterEqual(self.model.new_price["distinct_prices"], 2)

    def test_the_column_is_not_the_fold_key_wearing_a_disguise(self):
        """The objection that has to be answered, not waved at.

        `log_new_price` is a deterministic function of (brand, model) and the
        fold key contains both, so every held-out group shares its column value
        with training rows of the same model. Two nulls in the artifact answer
        different halves of that, and this pins both.
        """
        p = self.model.new_price.get("permutation_test", {})
        self.assertTrue(p.get("ran"), p)
        # Across ROWS: the crude story, and it dies loudly.
        self.assertLess(p["across_rows"]["p"], 0.01)
        self.assertLess(p["across_rows"]["max"], p["observed_r2"])
        # Across MODELS: every permutation is the same partition, so a per-model
        # dummy scores identically on all of them. The spread is what the VALUES
        # are worth, and it is large.
        am = p["across_models"]
        self.assertGreater(am["max"] - am["min"], 0.2)
        self.assertLessEqual(am["rank_of_truth"], 3)
        # And the honest half: with this few models the test cannot return a
        # significant p at all, so it must report its own floor rather than
        # letting a reader mistake 0.083 for a null result.
        self.assertIn("p_floor", am)
        self.assertGreaterEqual(am["p_exact"], am["p_floor"])


class InvariantsTheColumnCouldHaveBroken(unittest.TestCase):
    """The four rules in CLAUDE.md that a change to the price fit endangers."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def estimate(self, **kw):
        params = dict(year=2021, km=164374, make="FORD", market="TR",
                      evidence=None, vehicle_model="F-MAX")
        params.update(kw)
        return M.estimate(self.model, **params)

    def test_no_fx_got_inside_the_fit(self):
        """A 2021 F-MAX asks 2.35-2.55M TRY. It does not ask 118,000,000.

        The same absolute-magnitude guard `test_offline` carries, repeated here
        because this change rewrote the design matrix and this is the failure
        every relative assertion in the suite sailed past last time.
        """
        e = self.estimate()
        self.assertEqual(e.currency, "TRY")
        self.assertGreater(e.point, 1_500_000)
        self.assertLess(e.point, 4_000_000)
        self.assertAlmostEqual(e.point_usd, e.point / USD_TRY, delta=e.point_usd * 0.01)
        self.assertEqual(self.model.meta["target"].count("own currency"), 1)

    def test_the_comparable_band_still_brackets_a_known_listing(self):
        """The band is half as wide now, so this is a real constraint again."""
        e = self.estimate()
        self.assertLessEqual(e.baseline_low, 2_550_000)
        self.assertGreaterEqual(e.baseline_high, 2_350_000)

    def test_the_blend_may_claw_a_widening_back_and_never_cut_below_one(self):
        """The floor, now that both routes read the same reference figure.

        Inverse variance assumes independence. The column and the anchor share
        their input, so the blended sd is optimistic by construction, and the
        floor at 1.0 is the only thing between that and a band narrower than
        the one whose 80.3% was measured.
        """
        lo_off, hi_off = self.model.offsets["0.8"]
        for make, model_name in (("FORD", "F-MAX"), ("FORD", "Cargo"),
                                 ("MAN", "TGS"), ("MERCEDES-BENZ", "Actros 1845"),
                                 ("SCANIA", "R500")):
            with self.subTest(make=make):
                e = self.estimate(make=make, vehicle_model=model_name)
                self.assertTrue(e.ok)
                width = math.log(e.baseline_high / e.baseline_low)
                # Both ends are rounded to the nearest 1000 TRY for display, so
                # up to 500 can come off each. On a ~2.4M band that is ~4e-4 in
                # log space - larger than the margin being tested, which is why
                # the slack is computed rather than guessed at.
                slack = 1000.0 / e.baseline_low
                self.assertGreaterEqual(width, (hi_off - lo_off) - slack,
                                        "the band went below the measured interval")

    def test_a_weaker_source_now_widens_the_band_and_not_only_the_anchor(self):
        """A trade-press figure moves the hedonic estimate by ~2.4x its error.

        Before the column it could only mis-weight the anchor. `anchor.sigma`
        is no longer sufficient on its own, so the same stated assumption has
        to reach the band.
        """
        mercedes = self.estimate(make="MERCEDES-BENZ", vehicle_model="Actros 1845")
        self.assertEqual(anchor.lookup("MERCEDES-BENZ", "Actros 1845")["source_type"],
                         "trade_press")
        self.assertTrue(any("trade-press" in w for w in mercedes.widened),
                        mercedes.widened)
        ford = self.estimate()
        self.assertFalse(any("trade-press" in w for w in ford.widened))

    def test_an_unseen_brand_still_widens_and_still_says_why(self):
        known = self.estimate()
        unseen = self.estimate(make="SCANIA", vehicle_model="R500")
        self.assertGreater(math.log(unseen.baseline_high / unseen.baseline_low),
                           math.log(known.baseline_high / known.baseline_low))
        self.assertTrue(any("not in the fitted comparables" in w
                            for w in unseen.widened))

    def test_no_image_embedding_reached_the_fit(self):
        """`probe_residual_signal.py` measured that and it said no.

        The design matrix is five numbers, a brand and a published price. If a
        column ever appears here that a camera produced, that probe has to be
        re-run and beaten first.
        """
        allowed = set(F.BASE_COLUMNS) | set(F.NEW_PRICE_COLUMNS)
        for col in self.model.columns:
            self.assertTrue(col in allowed or col.startswith("brand_"), col)

    def test_the_condition_cap_names_the_columns_it_is_residual_to(self):
        """The sigma halved. The sentence beside it has to have moved too."""
        e = self.estimate()
        self.assertIn("what the model costs new", e.adjustment.cap_basis)
        self.assertIn("measured", e.adjustment.cap_basis)
        plain = M.condition_adjustment(None, 0.0988)
        self.assertNotIn("what the model costs new", plain.cap_basis)

    def test_the_condition_cap_is_one_residual_sigma_of_the_shipped_fit(self):
        adj = M.condition_adjustment(None, self.model.residual_std)
        self.assertAlmostEqual(adj.cap_pct,
                               round((1 - math.exp(-self.model.residual_std)) * 100, 1),
                               places=1)

    def test_a_stale_artifact_without_the_column_still_loads_and_predicts(self):
        """`columns` is the contract, not a module constant.

        Someone refitting on an older checkout, or a models/ directory that
        predates this change, must not get a shape error - they must get the
        design their file was actually fitted with.
        """
        d = self.model.to_dict()
        keep = [i for i, c in enumerate(d["columns"])
                if c not in F.NEW_PRICE_COLUMNS]
        stale = M.PriceModel(**{**d,
                                "columns": [d["columns"][i] for i in keep],
                                "coef": [d["coef"][i] for i in keep],
                                "mean": [d["mean"][i] for i in keep],
                                "scale": [d["scale"][i] for i in keep],
                                "new_price": {}})
        self.assertFalse(stale.uses_new_price)
        e = M.estimate(stale, year=2021, km=164374, make="FORD", market="TR",
                       vehicle_model="F-MAX", evidence=None)
        self.assertTrue(e.ok)
        self.assertGreater(e.point, 1_000_000)
        self.assertLess(e.point, 6_000_000)


class TheModelReachesThePriceThroughTheEvidence(unittest.TestCase):
    """End of the wire: what the identity pass reads has to arrive at the fit."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def evidence(self, model_name):
        from app.schema import EvidenceReport, VehicleRead
        ev = EvidenceReport()
        ev.vehicle = VehicleRead(make="Ford Trucks", model=model_name,
                                 body_type="tractor_unit", confidence=0.9)
        return ev

    def test_the_model_the_photos_named_changes_the_number(self):
        declared = {"year": 2019, "km": 500000}
        fmax = M.price_from_evidence(self.model, self.evidence("F Max"), declared,
                                     market="TR")
        cargo = M.price_from_evidence(self.model, self.evidence("Cargo 1848T"),
                                      declared, market="TR")
        self.assertTrue(fmax.ok and cargo.ok)
        self.assertNotEqual(fmax.baseline_point, cargo.baseline_point)
        self.assertEqual(fmax.inputs["new_price_try"],
                         int(anchor.lookup("FORD", "F-MAX")["list_price"]))
        self.assertEqual(cargo.inputs["new_price_try"],
                         int(anchor.lookup("FORD", "1845T")["list_price"]))

    def test_the_sellers_word_on_the_model_beats_the_badge(self):
        declared = {"year": 2019, "km": 500000, "model": "F-MAX"}
        e = M.price_from_evidence(self.model, self.evidence("Cargo"), declared,
                                  market="TR")
        self.assertEqual(e.inputs["model"], "F-MAX")
        self.assertEqual(e.inputs["new_price_try"],
                         int(anchor.lookup("FORD", "F-MAX")["list_price"]))

    def test_no_model_at_all_falls_back_to_the_brand_default(self):
        """A vision model that cannot read the badge is not a failure state."""
        declared = {"year": 2019, "km": 500000}
        e = M.price_from_evidence(self.model, self.evidence(None), declared,
                                  market="TR")
        self.assertTrue(e.ok)
        self.assertEqual(e.inputs["new_price_try"],
                         int(anchor.lookup("FORD", None)["list_price"]))


if __name__ == "__main__":
    unittest.main()
