"""Offline tests for the condition panel. No API calls, no agent runs.

The one that matters most is `KrippendorffAlpha`. An α you cannot verify is
worse than no α, so the implementation in `panel/agreement.py` is pinned three
ways here: against the figures published with the reference dataset, against a
second and structurally different computation of the same quantity written
inside this file, and against the degenerate cases where the answer is known by
inspection.

Provenance of the pinned numbers: the 3-coder x 15-unit dataset below is the
one distributed with the `krippendorff` PyPI package, whose documented results
are nominal 0.691 and interval 0.811. Ordinal is not published with it, so
0.8067 was cross-checked against that package directly - 40 random 3-rater
designs with missing data, max absolute difference 0.0 across nominal, ordinal
and interval. The package is deliberately NOT a dependency of this repo; the
cross-check ran in a throwaway venv and the number it produced is pinned here.
"""
from __future__ import annotations

import json
import random
import re
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

from app import config
from app.evidence import prompts as P
from panel import agreement as ag, protocol, select, store

# 3 coders x 15 units, missing values as None.
REFERENCE = [
    [None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3, 3, None, 3],
    [1, None, 2, 1, 3, 3, 4, 3, None, None, None, None, None, None, None],
    [None, None, 2, 1, 3, 4, 4, None, 2, 1, 1, 3, 3, None, 4],
]
REFERENCE_UNITS = [[row[i] for row in REFERENCE] for i in range(15)]


def _alpha_by_pairs(units, metric, values):
    """A second implementation, organised differently on purpose.

    Instead of building a coincidence matrix it walks the rater pairs inside
    each unit for D_o and every ordered pair of observations in the whole
    sample for D_e. Same quantity, different route.
    """
    usable = [[v for v in u if v is not None] for u in units]
    usable = [u for u in usable if len(u) >= 2]
    counts = {v: 0.0 for v in values}
    for unit in usable:
        for v in unit:
            counts[v] += 1.0
    n = sum(counts.values())

    def delta(a, b):
        i, j = values.index(a), values.index(b)
        if metric == "nominal":
            return 0.0 if i == j else 1.0
        if metric == "interval":
            return float((i - j) ** 2)
        lo, hi = min(i, j), max(i, j)
        return float((sum(counts[v] for v in values[lo:hi + 1])
                      - (counts[values[lo]] + counts[values[hi]]) / 2) ** 2)

    d_o = sum(2 * delta(a, b) / (len(u) - 1)
              for u in usable for a, b in combinations(u, 2)) / n
    d_e = sum(counts[a] * counts[b] * delta(a, b)
              for a in values for b in values) / (n * (n - 1))
    return 1.0 - d_o / d_e


class KrippendorffAlpha(unittest.TestCase):
    def test_nominal_matches_published(self):
        r = ag.krippendorff_alpha(REFERENCE_UNITS, metric="nominal", values=[1, 2, 3, 4])
        self.assertAlmostEqual(r.alpha, 0.691, places=3)
        self.assertEqual((r.n_units, r.n_observations), (12, 26))

    def test_interval_matches_published(self):
        r = ag.krippendorff_alpha(REFERENCE_UNITS, metric="interval", values=[1, 2, 3, 4])
        self.assertAlmostEqual(r.alpha, 0.811, places=3)

    def test_ordinal_matches_cross_checked_value(self):
        r = ag.krippendorff_alpha(REFERENCE_UNITS, metric="ordinal", values=[1, 2, 3, 4])
        self.assertAlmostEqual(r.alpha, 0.8067, places=4)

    def test_second_implementation_agrees(self):
        rng = random.Random(7)
        for _ in range(30):
            k = rng.choice([3, 4, 5])
            values = list(range(k))
            units = [[rng.choice(values) if rng.random() > 0.25 else None
                      for _ in range(3)] for _ in range(rng.randint(6, 40))]
            for metric in ("nominal", "ordinal", "interval"):
                mine = ag.krippendorff_alpha(units, metric=metric, values=values).alpha
                if mine is None:
                    continue
                self.assertAlmostEqual(mine, _alpha_by_pairs(units, metric, values),
                                       places=10, msg=f"{metric} {units}")

    def test_perfect_agreement_is_one(self):
        units = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1]]
        self.assertEqual(ag.krippendorff_alpha(units, metric="ordinal",
                                               values=[1, 2, 3]).alpha, 1.0)

    def test_no_variance_is_undefined_not_one(self):
        r = ag.krippendorff_alpha([[1, 1, 1], [1, 1, 1]], metric="ordinal", values=[1, 2])
        self.assertIsNone(r.alpha)
        self.assertIn("undefined", r.note)

    def test_units_with_one_reading_drop_out(self):
        r = ag.krippendorff_alpha([[1, 1], [2, None], [None, None]],
                                  metric="nominal", values=[1, 2])
        self.assertEqual(r.n_units, 1)

    def test_systematic_disagreement_is_negative(self):
        units = [[1, 2], [2, 1]] * 4
        self.assertLess(ag.krippendorff_alpha(units, metric="nominal",
                                              values=[1, 2]).alpha, 0)


class RubricHash(unittest.TestCase):
    def test_stable(self):
        self.assertEqual(protocol.rubric_sha(), protocol.RUBRIC_SHA)
        self.assertEqual(len(protocol.RUBRIC_SHA), 64)

    def test_covers_all_four_constants(self):
        text = protocol.rubric_text()
        self.assertIn(P.SEVERITY_RUBRIC, text)
        self.assertIn(P.WORKED_EXAMPLES, text)
        for view in P.COMPONENT_ANCHORS:
            self.assertIn(json.dumps(P.COMPONENT_ANCHORS[view])[1:-1], text)
        self.assertIn(P.EXPECTED_WEAR["tire_wheel"]["high"][0], text)

    def test_moves_when_the_rubric_moves(self):
        original = P.SEVERITY_RUBRIC
        try:
            P.SEVERITY_RUBRIC = original + " "
            self.assertNotEqual(protocol.rubric_sha(), protocol.RUBRIC_SHA)
        finally:
            P.SEVERITY_RUBRIC = original
        self.assertEqual(protocol.rubric_sha(), protocol.RUBRIC_SHA)


class AnchorBlocks(unittest.TestCase):
    def test_every_family_recovered(self):
        blocks = protocol.family_blocks()
        self.assertEqual(set(blocks), set(P.ANCHOR_COMPONENTS))

    def test_families_partition_components(self):
        covered = [c for ids in P.ANCHOR_COMPONENTS.values() for c in ids]
        self.assertEqual(sorted(covered), sorted(P.COMPONENTS))

    def test_wear_rows_cover_every_family(self):
        band, rows = protocol.wear_rows(500_000)
        self.assertEqual(band, "high")
        self.assertEqual(set(rows), set(P.ANCHOR_COMPONENTS))

    def test_no_wear_rows_without_a_distance(self):
        self.assertEqual(protocol.wear_rows(None), (None, {}))


class PanelistPrompt(unittest.TestCase):
    def prompt(self, km=480_000, photos=3):
        return protocol.panelist_prompt(
            listing_id="14190", make="MAN", model="TGS", year=2016, km=km,
            market="TR", photos=[f"/x/{i:03d}.jpg" for i in range(photos)])

    def test_embeds_the_rubric_verbatim(self):
        text = self.prompt()
        self.assertIn(P.SEVERITY_RUBRIC, text)
        self.assertIn(P.WORKED_EXAMPLES, text)
        self.assertIn(P.SYSTEM, text)

    def test_embeds_every_anchor_block_verbatim(self):
        text = self.prompt()
        for view, value in P.COMPONENT_ANCHORS.items():
            for block in value.split("\n\n"):
                self.assertIn(block, text, f"{view} anchor block missing")

    def test_embeds_only_this_distance_band(self):
        text = self.prompt(km=45_000)
        self.assertIn(P.EXPECTED_WEAR["tire_wheel"]["low"][0], text)
        self.assertNotIn(P.EXPECTED_WEAR["tire_wheel"]["very_high"][0], text)
        self.assertIn('"low" distance band', text)

    def test_rule_one_makes_none_the_expected_answer(self):
        text = self.prompt()
        self.assertIn('"none" means the component is visible and in the state its age '
                      'and distance predict.', " ".join(text.split()))
        self.assertIn("more often than any severity", text)

    def test_grade_wording_is_the_production_wording(self):
        flat = " ".join(self.prompt().split())
        production = " ".join(P.SYNTHESIS_PROMPT.split())
        for phrase in ('Grade on the WORST DISTINCT defect and on how much of the truck '
                       'was visible - never on how many findings there are.',
                       '"poor" means something on this truck needs money spent on it '
                       'before it works, and "fair" means real wear a buyer would '
                       'negotiate over.'):
            self.assertIn(phrase, production)
            self.assertIn(phrase, flat)

    def test_repair_bands_never_reach_the_prompt(self):
        text = self.prompt()
        figures = {str(b) for pair in config.REPAIR_BANDS.values()
                   for b in pair if b}
        for figure in figures:
            self.assertNotIn(figure, text, f"a lira figure ({figure}) reached the panel")
        for word in ("TRY", "lira", "₺", "REPAIR_BANDS"):
            self.assertNotIn(word, text)
        self.assertIn("Do not estimate a price, and do not name one.", text)

    def test_lists_every_component_and_every_photo(self):
        text = self.prompt(photos=5)
        for c in P.COMPONENTS:
            self.assertIn(c, text)
        for i in range(5):
            self.assertIn(f"[{i}] /x/{i:03d}.jpg", text)

    def test_no_distance_says_so(self):
        self.assertIn("No odometer reading was published", self.prompt(km=None))


class ResponseValidation(unittest.TestCase):
    def good(self, **over):
        body = {
            "components": {c: {"severity": "none", "note": "", "photos": [0]}
                           for c in P.COMPONENTS},
            "findings": [], "grade": "good", "grade_reason": "clean",
            "confidence": 0.7, "not_assessable": [],
        }
        body.update(over)
        return body

    def test_clean(self):
        self.assertEqual(protocol.validate_response(self.good(), n_photos=3), [])

    def test_missing_component(self):
        body = self.good()
        body["components"].pop("steer_tires")
        self.assertTrue(any("missing" in e for e in
                            protocol.validate_response(body, n_photos=3)))

    def test_unknown_component(self):
        body = self.good()
        body["components"]["wipers"] = {"severity": "none", "note": "", "photos": []}
        self.assertTrue(any("unknown" in e for e in
                            protocol.validate_response(body, n_photos=3)))

    def test_photo_index_out_of_range(self):
        body = self.good()
        body["components"]["steer_tires"]["photos"] = [9]
        self.assertTrue(any("outside" in e for e in
                            protocol.validate_response(body, n_photos=3)))

    def test_severity_must_cite_a_photo(self):
        body = self.good()
        body["components"]["steer_tires"] = {"severity": "moderate", "note": "worn",
                                             "photos": []}
        self.assertTrue(any("cites no photo" in e for e in
                            protocol.validate_response(body, n_photos=3)))

    def test_bad_grade_and_confidence(self):
        errors = protocol.validate_response(self.good(grade="ok", confidence=4),
                                            n_photos=3)
        self.assertEqual(len(errors), 2)

    def test_finding_severity_cannot_be_none(self):
        body = self.good(findings=[{"component": "steer_tires", "severity": "none",
                                    "note": "x", "photos": [0]}])
        self.assertTrue(any("not a level" in e for e in
                            protocol.validate_response(body, n_photos=3)))

    def test_schema_requires_every_component(self):
        schema = protocol.response_schema()
        self.assertEqual(schema["properties"]["components"]["required"],
                         list(P.COMPONENTS))

    def test_not_visible_is_off_the_severity_scale(self):
        self.assertNotIn(protocol.NOT_VISIBLE, protocol.SEVERITY_SCALE)
        self.assertEqual(protocol.SEVERITY_SCALE[0], "none")
        self.assertEqual(protocol.SEVERITY_SCALE[1:], list(P.SEVERITIES))

    def test_severity_vector_fills_gaps_with_not_visible(self):
        vec = protocol.severity_vector({"components": {"steer_tires":
                                                       {"severity": "minor"}}})
        self.assertEqual(vec["steer_tires"], "minor")
        self.assertEqual(vec["drive_tires"], protocol.NOT_VISIBLE)
        self.assertEqual(len(vec), len(P.COMPONENTS))


class PanelAggregation(unittest.TestCase):
    def test_adjudication_only_when_all_three_differ(self):
        self.assertTrue(ag.adjudication_required(["good", "fair", "poor"]))
        self.assertFalse(ag.adjudication_required(["good", "good", "poor"]))
        self.assertFalse(ag.adjudication_required(["good", "fair"]))

    def test_consensus_takes_the_median_not_the_worst(self):
        vectors = [{"steer_tires": s} for s in ("none", "minor", "major")]
        self.assertEqual(ag.consensus(vectors)["steer_tires"], "minor")

    def test_consensus_not_visible_needs_a_majority(self):
        two_blind = [{"steer_tires": "not_visible"}, {"steer_tires": "not_visible"},
                     {"steer_tires": "minor"}]
        self.assertEqual(ag.consensus(two_blind)["steer_tires"], "not_visible")
        one_blind = [{"steer_tires": "not_visible"}, {"steer_tires": "minor"},
                     {"steer_tires": "minor"}]
        self.assertEqual(ag.consensus(one_blind)["steer_tires"], "minor")

    def test_consensus_grade_prefers_the_majority(self):
        self.assertEqual(ag.consensus_grade(["good", "good", "poor"]), "good")

    def test_consensus_grade_with_no_majority_is_the_median(self):
        self.assertEqual(ag.consensus_grade(["excellent", "good", "fair"]), "good")

    def test_not_visible_is_missing_for_the_ordinal_alpha(self):
        vectors = {"v1": [{c: "not_visible" for c in P.COMPONENTS}] * 3}
        self.assertEqual(ag.severity_alpha(vectors).n_units, 0)
        self.assertIsNone(ag.severity_alpha(vectors).alpha)

    def test_grade_agreement_counts_pairs_and_unanimity(self):
        result = ag.grade_agreement({"a": ["good"] * 3, "b": ["good", "good", "fair"]})
        self.assertAlmostEqual(result.unanimous, 0.5)
        self.assertAlmostEqual(result.pairwise, 4 / 6)
        self.assertEqual(result.confusion[("good", "fair")], 2)


class Selection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all = select.load_vehicles(check_disk=False)
        cls.target = select.select_target(cls.all)
        cls.pilot = select.select_pilot(cls.target)

    def test_deterministic(self):
        again = select.select_target(select.load_vehicles(check_disk=False))
        self.assertEqual([v.listing_id for v in self.target],
                         [v.listing_id for v in again])

    def test_target_size_and_split(self):
        self.assertEqual(len(self.target), select.TARGET_N)
        self.assertEqual(sum(v.market == "TR" for v in self.target), select.TARGET_TR)
        self.assertEqual(sum(v.market == "US" for v in self.target), select.TARGET_US)

    def test_every_tr_man_is_in(self):
        man = {v.listing_id for v in self.all if v.market == "TR" and v.make == "MAN"}
        self.assertEqual(len(man), 6)
        self.assertTrue(man <= {v.listing_id for v in self.target})

    def test_us_spreads_across_makes(self):
        makes = {v.make for v in self.target if v.market == "US"}
        self.assertGreaterEqual(len(makes), 6)

    def test_tr_spans_the_wear_bands(self):
        bands = {v.wear_band for v in self.target if v.market == "TR"}
        self.assertTrue({"low", "mid", "high", "very_high"} <= bands)

    def test_pilot_is_six_inside_the_target(self):
        self.assertEqual(len(self.pilot), 6)
        self.assertTrue({v.listing_id for v in self.pilot}
                        <= {v.listing_id for v in self.target})

    def test_pilot_spans_and_names_its_strata(self):
        self.assertEqual([v.stratum for v in self.pilot],
                         [name for name, _ in select.PILOT_RULES])
        self.assertIn("US", {v.market for v in self.pilot})
        self.assertGreaterEqual(len({v.make for v in self.pilot}), 3)

    def test_pilot_photos_are_on_disk(self):
        for v in select.select_pilot(select.select_target(
                select.load_vehicles(check_disk=True))):
            self.assertGreaterEqual(v.n_photos, select.MIN_PHOTOS)
            for path in v.photos:
                self.assertTrue(Path(path).exists(), path)

    def test_selection_never_reads_a_photo(self):
        """Nothing about how a truck looks may influence whether it is chosen."""
        source = Path(select.__file__).read_text(encoding="utf-8")
        imports = re.findall(r"^(?:import|from)\s+(\S+)", source, re.M)
        for banned in ("PIL", "cv2", "torch", "app.vision", "imagehash"):
            self.assertNotIn(banned, imports)
        self.assertEqual(re.findall(r"(\w+)\.open\(", source), ["images", "listings"])

    def test_prompt_maps_the_two_foreign_field_names(self):
        text = protocol.panelist_prompt(listing_id="x", make="FORD", model="F-MAX",
                                        year=2020, km=300_000, market="TR",
                                        photos=["/x/0.jpg"])
        self.assertIn("strengths", P.SEVERITY_RUBRIC)
        self.assertIn("price_impact", P.WORKED_EXAMPLES)
        self.assertIn('your answer for that component is "none"', text)


class Artifact(unittest.TestCase):
    def row(self):
        vehicle = select.Vehicle(
            listing_id="14190", source_key="tr_truckmarket", market="TR",
            make="MAN", model="TGS", year=2016, km=870_022, wear_band="very_high",
            n_photos=2, photos=[str(select.DATA / "images/x/000.jpg")],
            stratum="tr_man")
        answers = []
        for i, sev in enumerate(("minor", "moderate", "minor")):
            body = {"components": {c: {"severity": "none", "note": "", "photos": [0]}
                                   for c in P.COMPONENTS},
                    "findings": [], "grade": ["good", "fair", "good"][i],
                    "grade_reason": "x", "confidence": 0.6, "not_assessable": []}
            body["components"]["steer_tires"]["severity"] = sev
            answers.append({"panelist": f"p{i + 1}", "ok": True, "errors": [],
                            "response": body})
        return store.build_row(vehicle, answers, run_id="test")

    def test_keeps_every_panelist_answer(self):
        row = self.row()
        self.assertEqual(len(row["panelists"]), 3)
        self.assertEqual(row["agreement"]["grades"], ["good", "fair", "good"])
        self.assertEqual(row["consensus"]["grade"], "good")
        self.assertEqual(row["consensus"]["components"]["steer_tires"], "minor")
        self.assertFalse(row["agreement"]["adjudication_required"])

    def test_stamped_with_the_rubric_it_was_graded_under(self):
        self.assertEqual(self.row()["rubric_sha"], protocol.RUBRIC_SHA)

    def test_round_trip_and_stale_rubric_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.jsonl"
            store.write_rows([self.row()], path)
            rows = store.read_rows(path)
            self.assertEqual(store.check_rubric(rows), [])
            rows[0]["rubric_sha"] = "0" * 64
            self.assertEqual(store.check_rubric(rows), ["0" * 64])

    def test_photos_are_stored_repo_relative(self):
        self.assertEqual(self.row()["photos"], ["images/x/000.jpg"])

    def test_panel_view_feeds_agreement(self):
        vectors, grades = store.panel_view([self.row()])
        self.assertEqual(len(vectors["14190"]), 3)
        self.assertEqual(grades["14190"], ["good", "fair", "good"])
        self.assertIsNotNone(ag.severity_alpha(vectors))


if __name__ == "__main__":
    unittest.main()
