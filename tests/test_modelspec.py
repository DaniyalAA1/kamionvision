"""The spec card's accessors, and the price guard that decides what may load.

Most of this file is the guard. `assert_priceless` runs on every load of
data/reference/models_tr.json and it is the only thing standing between a
hand-edited reference file and a currency figure in front of the pass that
assigns severity - so both of its failure directions are pinned here.

The false-positive cases are not padding. The first researched card tripped the
guard six times on "worth noting", "worth a close-up" and "worth knowing when
judging corrosion", and once on "Euro 6", which is the emission standard every
tractor unit in this market is described by. A guard that fires on the most
common phrases in its own subject matter gets switched off, and then the real
leak goes through - so the precision is the feature, not a convenience.
"""
import json
import tempfile
import unittest
from pathlib import Path

from app import modelspec


class PriceGuard(unittest.TestCase):
    LEGAL = [
        "Euro 6 diesel", "Euro-6", "Euro VI", "Euro 5", "EURO 6 emission standard",
        "worth noting", "worth a close-up", "worth knowing when judging corrosion",
        "ZF TraXon 12-speed automated manual", "600 L aluminium fuel tank",
        "3,600 mm wheelbase", "13 L / 510 PS / 2600 Nm",
    ]
    LEAKS = [
        "worth 500,000", "worth ₺2M", "worth about 300000",
        "costs $85,000", "expensive to replace", "cheaper than a new one",
        "the price of a replacement", "EUR 40000", "list price 7,433,143 TRY",
        "fiyat listesi", "pricing varies",
    ]

    def test_spec_language_is_not_mistaken_for_money(self):
        for text in self.LEGAL:
            with self.subTest(text=text):
                modelspec.assert_priceless([{"note": text}])

    def test_a_currency_figure_is_caught(self):
        for text in self.LEAKS:
            with self.subTest(text=text):
                with self.assertRaises(modelspec.PriceLeak):
                    modelspec.assert_priceless([{"note": text}])

    def test_it_walks_nested_structures(self):
        payload = {"models": [{"known_weak_points": [{"note": "costs a fortune"}]}]}
        with self.assertRaises(modelspec.PriceLeak):
            modelspec.assert_priceless(payload)

    def test_maintainer_keys_are_skipped(self):
        # The file's own documentation discusses pricing at length and none of
        # it reaches a prompt. A check that fired on its own docs would be
        # deleted within a day.
        modelspec.assert_priceless({"models": [{
            "basis": "derived from the new-price reference table",
            "sources": [{"url": "https://example.com/price-list"}],
            "maintainer_note": "anchor.py aliases this to the 1845T new-price row",
        }]})

    def test_a_weak_point_note_is_never_skipped(self):
        # The realistic leak, and the reason `note` stays in scope while
        # `maintainer_note` does not.
        with self.assertRaises(modelspec.PriceLeak):
            modelspec.assert_priceless({"models": [{
                "known_weak_points": [{"component": "adblue_tank",
                                       "note": "cracks, and a replacement is expensive"}]}]})

    def test_the_shipped_card_loads(self):
        modelspec._reset()
        self.assertTrue(modelspec.available())
        self.assertTrue(modelspec.reference()["models"])


class Normalising(unittest.TestCase):
    def setUp(self):
        modelspec._reset()

    def test_spacing_and_punctuation_fold(self):
        for raw in ("F Max", "FMAX", "f-max", "F-MAX"):
            self.assertEqual(modelspec.normalise_model("FORD", raw), "F-MAX")

    def test_a_trailing_trim_still_finds_the_model(self):
        self.assertEqual(modelspec.normalise_model("FORD", "F-MAX 500"), "F-MAX")

    def test_a_corpus_model_name_folds_through_its_alias(self):
        self.assertEqual(modelspec.normalise_model("FORD", "1848T"), "TRUCKS")

    def test_the_brand_is_folded_too(self):
        self.assertEqual(modelspec.normalise_model("Ford Trucks", "F-MAX"), "F-MAX")

    def test_an_unknown_model_passes_through_rather_than_vanishing(self):
        # Returning None here would lose a model the anchor could still match.
        self.assertEqual(modelspec.normalise_model("KRONE", "SDP27"), "SDP27")

    def test_no_model_is_none(self):
        self.assertIsNone(modelspec.normalise_model("FORD", None))


class Degradation(unittest.TestCase):
    """Absent card: every accessor empty, nothing raises."""

    def setUp(self):
        self._real = modelspec.MODEL_SPECS
        modelspec.MODEL_SPECS = Path(tempfile.gettempdir()) / "kamion-no-such-card.json"
        modelspec._reset()

    def tearDown(self):
        modelspec.MODEL_SPECS = self._real
        modelspec._reset()

    def test_accessors_are_empty_and_quiet(self):
        self.assertFalse(modelspec.available())
        self.assertEqual(modelspec.vocabulary(), [])
        self.assertEqual(modelspec.generations("FORD", "F-MAX"), [])
        self.assertEqual(modelspec.spec_lines("FORD", "F-MAX"), [])
        self.assertEqual(modelspec.weak_points("FORD", "F-MAX"), [])
        self.assertEqual(modelspec.identity_tells("FORD", "F-MAX"), [])
        self.assertIsNone(modelspec.card("FORD", "F-MAX"))

    def test_normalise_still_returns_the_raw_model(self):
        self.assertEqual(modelspec.normalise_model("FORD", "F-MAX"), "F-MAX")


class Generations(unittest.TestCase):
    def setUp(self):
        modelspec._reset()

    def test_year_spans_parse(self):
        self.assertEqual(modelspec._year_span("2018-2023"), (2018, 2023))
        self.assertEqual(modelspec._year_span("2018-present"), (2018, None))
        self.assertEqual(modelspec._year_span("2018"), (2018, 2018))
        self.assertEqual(modelspec._year_span(None), (None, None))

    def test_a_year_selects_its_generation(self):
        got = modelspec.generation_for_year("FORD", "F-MAX", 2021)
        self.assertIsNotNone(got)
        self.assertIn("2018", str(got["years"]))

    def test_a_year_outside_every_span_selects_nothing(self):
        self.assertIsNone(modelspec.generation_for_year("FORD", "F-MAX", 1995))

    def test_no_year_selects_nothing(self):
        self.assertIsNone(modelspec.generation_for_year("FORD", "F-MAX", None))


class WeakPointFiltering(unittest.TestCase):
    def setUp(self):
        modelspec._reset()

    def test_filtering_to_a_view_drops_the_rest(self):
        got = modelspec.weak_points("FORD", "F-MAX", components=["adblue_tank"])
        for component, _note in got:
            self.assertEqual(component, "adblue_tank")

    def test_an_unfiltered_call_returns_everything(self):
        self.assertGreaterEqual(len(modelspec.weak_points("FORD", "F-MAX")), 1)

    def test_every_component_is_in_the_closed_enum(self):
        from app.evidence.prompts import COMPONENTS
        for row in modelspec.reference()["models"]:
            for entry in row.get("known_weak_points") or []:
                with self.subTest(model=row["model"], component=entry["component"]):
                    self.assertIn(entry["component"], COMPONENTS)


class Provenance(unittest.TestCase):
    """An uncited spec card is a hallucination with a filename."""

    def setUp(self):
        modelspec._reset()

    def test_every_model_carries_at_least_one_source(self):
        for row in modelspec.reference()["models"]:
            with self.subTest(model=row["model"]):
                self.assertTrue(row.get("sources"),
                                f"{row['brand']} {row['model']} has no source URL")

    def test_every_source_has_a_url(self):
        for row in modelspec.reference()["models"]:
            for src in row.get("sources") or []:
                with self.subTest(model=row["model"]):
                    self.assertTrue(str(src.get("url", "")).startswith("http"))

    def test_visual_markers_exist_for_every_generation(self):
        for row in modelspec.reference()["models"]:
            for gen in row.get("generations") or []:
                with self.subTest(model=row["model"], gen=gen.get("id")):
                    self.assertTrue(gen.get("visual_markers"),
                                    "a generation with no visual marker cannot be "
                                    "read off a photograph, which is its only job")


if __name__ == "__main__":
    unittest.main()
