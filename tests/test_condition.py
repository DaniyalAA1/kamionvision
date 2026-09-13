"""The condition rollup, the grade ladder, and the properties of the multiplier.

Kept out of `test_offline.py` on purpose - that file is being edited by two
other workstreams and this one is all new surface.

The property tests at the bottom are the point of the file. Nothing like them
existed, and every bug this workstream was opened to fix was a property nobody
had written down: sixteen restatements of one worn tire cost 3.9 pp more than
one statement of it; a spotless truck priced identically to an unphotographed
one; sixteen clean photos graded `unknown` while one cosmetic scuff upgraded
them to `good`; and the grade and the price disagreed about which of two trucks
was worse. Each of those is now one assertion.
"""
from __future__ import annotations

import math
import random
import unittest

from app import condition
from app.evidence import passes, prompts
from app.pricing import model as M
from app.schema import EvidenceReport, Issue, PhotoFinding

# The measured cap. `test_the_shipped_model_still_carries_the_measured_numbers`
# pins this against the artifact rather than letting the two drift.
CAP_LOG = 0.0988

ALL_VIEWS = ["exterior_front_34", "exterior_side", "exterior_front", "exterior_rear",
             "interior_cab", "dashboard_odometer", "tire_wheel", "engine_bay",
             "chassis_undercarriage", "fifth_wheel", "damage_detail",
             "tire_wheel", "exterior_side", "interior_cab", "exterior_front_34",
             "engine_bay"]


def photos(n: int = 16, *, strengths: bool = True, legible: bool = True):
    """A well-shot sixteen-photo set: every view, every frame readable."""
    return [PhotoFinding(photo_id=i, view=v, legible=legible,
                         strengths=[f"the {v.replace('_', ' ')} is in good order"]
                         if strengths else [])
            for i, v in enumerate(ALL_VIEWS[:n])]


def issue(photo_id=0, component="drive_tires", severity="minor", impact="low",
          confidence=0.8, observation="outer shoulder worn against the centre ribs",
          **kw):
    return Issue(photo_id=photo_id, component=component, observation=observation,
                 severity=severity, confidence=confidence, price_impact=impact, **kw)


def adjust(issues, findings=None):
    """Run a scenario the way the pipeline does: rollup, grade, then price."""
    findings = photos() if findings is None else findings
    roll = condition.rollup(issues, findings)
    ev = EvidenceReport(issues=list(issues), photo_findings=list(findings),
                        condition=roll, condition_grade=roll.grade)
    return roll, M.condition_adjustment(ev, CAP_LOG)


# --- the vocabularies ------------------------------------------------------

class Families(unittest.TestCase):
    """The 14 families partition the 33-id component enum, exactly.

    A component that falls out of the partition silently stops contributing to
    coverage and can never merge with the id beside it, which is the failure
    `corrosion` / `chassis_frame` / `undercarriage` actually had.
    """

    def test_the_partition_covers_every_component_exactly_once(self):
        mapped = [c for _, (_, members) in condition.FAMILIES.items() for c in members]
        self.assertEqual(len(mapped), len(set(mapped)), "a component is in two families")
        self.assertEqual(sorted(mapped), sorted(prompts.COMPONENTS))

    def test_the_family_weights_sum_to_one(self):
        total = sum(w for w, _ in condition.FAMILIES.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_the_three_floating_ids_land_where_they_are_observed(self):
        # The cross-id merge the old exact-id gate could never do.
        self.assertEqual(condition.family_of("corrosion"),
                         condition.family_of("chassis_frame"))
        self.assertEqual(condition.family_of("fluid_leaks"),
                         condition.family_of("engine_bay"))
        self.assertEqual(condition.family_of("paint_finish"),
                         condition.family_of("cab_exterior_panels"))

    def test_a_lit_fault_lamp_is_never_diluted_into_cab_interior(self):
        self.assertNotEqual(condition.family_of("warning_lights"),
                            condition.family_of("cab_interior_seats"))

    def test_an_invented_component_still_costs_money(self):
        roll, adj = adjust([issue(component="flux_capacitor", severity="major",
                                  impact="high", confidence=0.9)])
        self.assertGreater(roll.demerit, 0.0)
        self.assertLess(adj.multiplier, 1.0)

    def test_every_family_declares_the_views_that_show_it(self):
        self.assertEqual(sorted(condition.VIEW_COVERAGE), sorted(condition.FAMILIES))

    def test_the_view_map_only_names_real_views(self):
        from app.vision import VIEW_LABELS
        named = {v for p, s in condition.VIEW_COVERAGE.values() for v in (*p, *s)}
        self.assertTrue(named <= set(VIEW_LABELS), named - set(VIEW_LABELS))

    def test_a_damage_close_up_grants_coverage_to_nothing(self):
        # It can show any part of the truck, so crediting it would credit
        # everything - which is the "absence of evidence" failure in miniature.
        named = {v for p, s in condition.VIEW_COVERAGE.values() for v in (*p, *s)}
        self.assertNotIn("damage_detail", named)
        roll = condition.rollup([], [PhotoFinding(photo_id=0, view="damage_detail",
                                                  strengths=["looks fine"])])
        self.assertEqual(roll.coverage, 0.0)


# --- layer 1 and 2 ---------------------------------------------------------

class Demerit(unittest.TestCase):
    def test_an_ungraded_finding_weighs_nothing(self):
        self.assertEqual(condition.demerit(
            issue(severity="cosmetic", impact="none", ungraded=True)), 0.0)

    def test_an_enum_value_outside_the_vocabulary_weighs_nothing(self):
        # It used to round up to the minor/low weight, which turned a garbage
        # observation into real money.
        self.assertEqual(condition.demerit(issue(severity="catastrophic")), 0.0)
        self.assertEqual(condition.demerit(issue(impact="ruinous")), 0.0)

    def test_a_nonsense_confidence_does_not_crash_or_inflate(self):
        for bad in (float("nan"), float("inf"), None, "lots"):
            d = condition.demerit(issue(confidence=bad))
            self.assertTrue(math.isfinite(d), bad)
            self.assertLessEqual(d, condition.SEVERITY_WEIGHT["minor"]
                                 * condition.IMPACT_WEIGHT["low"])

    def test_confidence_has_a_floor_not_a_zero(self):
        # A downgrade must never become a silent deletion - reconcile's posture.
        self.assertGreater(condition.demerit(issue(confidence=0.0)), 0.0)


class Saturation(unittest.TestCase):
    """The worst defect on a subsystem counts in full, the rest decay."""

    def test_the_second_finding_counts_forty_percent_of_the_first(self):
        self.assertAlmostEqual(condition.family_demerit([1.0, 1.0]), 1.4, places=6)
        self.assertAlmostEqual(condition.family_demerit([1.0, 1.0, 1.0]), 1.56, places=6)

    def test_order_does_not_matter(self):
        self.assertAlmostEqual(condition.family_demerit([0.2, 2.0, 0.5]),
                               condition.family_demerit([2.0, 0.5, 0.2]), places=9)

    def test_a_subsystem_can_never_cost_more_than_its_worst_possible_defect(self):
        self.assertEqual(condition.family_demerit([4.4] * 200), condition.FAMILY_CAP)
        self.assertEqual(condition.FAMILY_CAP,
                         max(condition.SEVERITY_WEIGHT.values())
                         * max(condition.IMPACT_WEIGHT.values()))

    def test_two_subsystems_add_where_one_saturates(self):
        """The pair that was indistinguishable before, and must not be now.

        Ten minor findings on ONE subsystem and ten across TEN both priced at
        exactly -1.8%, because the sum was completely component-blind. A
        genuinely tireder truck has to cost more than one over-described tire.
        """
        one = [issue(photo_id=i, component="drive_tires", observation=t)
               for i, t in enumerate(DISTINCT_TIRE)]
        spread = [issue(photo_id=i, component=c, observation=DISTINCT_TIRE[i])
                  for i, c in enumerate(TEN_SUBSYSTEMS)]
        _, one_adj = adjust(one)
        _, spread_adj = adjust(spread)
        self.assertLess(spread_adj.pct, one_adj.pct - 1.0)


DISTINCT_TIRE = [
    "outer shoulder worn low against the centre ribs",
    "sidewall weather checking along the bead area",
    "tread block tearing near the kerb face",
    "valve stem perished at the grommet",
    "kerbing damage around the rim flange",
    "one wheel nut indicator rotated out of line",
    "brand mismatch against the opposite drive axle",
    "a retread buff line visible under the cap edge",
    "stone holding between two adjacent ribs",
    "light hub staining behind the wheel face",
]
TEN_SUBSYSTEMS = ["steer_tires", "drive_tires", "wheels_rims", "fifth_wheel",
                  "chassis_frame", "air_suspension", "cab_exterior_panels",
                  "cab_steps", "windscreen_glass", "fuel_tank"]


# --- the merge -------------------------------------------------------------

class Overlap(unittest.TestCase):
    SHORT = "outer shoulder worn to the wear bars"
    LONG = ("the outer shoulder of the near-side drive tire is worn down to the "
            "wear bars while the centre ribs still carry good depth, which reads "
            "as an alignment fault rather than mileage")

    def test_a_short_and_a_long_description_of_one_defect_now_merge(self):
        # Jaccard scores this pair 0.23 and refused; containment scores 1.00.
        self.assertLess(len(passes._tokens(self.SHORT) | passes._tokens(self.LONG))
                        and len(passes._tokens(self.SHORT) & passes._tokens(self.LONG))
                        / len(passes._tokens(self.SHORT) | passes._tokens(self.LONG)),
                        passes.SAME_DEFECT)
        self.assertGreaterEqual(passes._overlap(self.SHORT, self.LONG),
                                passes.SAME_DEFECT)

    def test_an_identical_string_is_always_a_duplicate(self):
        self.assertEqual(passes._overlap("x", "x"), 1.0)

    def test_a_very_short_note_is_judged_on_jaccard_alone(self):
        # Every token of a three-word note lands inside almost any long one,
        # so containment there would merge unrelated findings.
        self.assertEqual(passes._overlap("rust", self.LONG), 0.0)

    def test_two_different_defects_on_one_tire_stay_apart(self):
        other = "inner drive tire sidewall has a deep cut exposing cord near the bead"
        self.assertLess(passes._overlap(self.SHORT, other), passes.SAME_DEFECT)


class CrossIdMerge(unittest.TestCase):
    TEXT = ("heavy scaling corrosion lifting the paint along the top flange of "
            "the frame rail behind the cab")

    def test_one_defect_reported_under_two_ids_folds(self):
        flat = [issue(0, "chassis_frame", observation=self.TEXT),
                issue(1, "corrosion", observation=self.TEXT)]
        out = passes.merge_duplicates(flat, [])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].also_seen_in, [1])

    def test_two_families_never_fold_however_identical_the_wording(self):
        flat = [issue(0, "chassis_frame", observation=self.TEXT),
                issue(1, "fuel_tank", observation=self.TEXT)]
        self.assertEqual(len(passes.merge_duplicates(flat, [])), 2)

    def test_across_ids_the_wording_has_to_carry_more(self):
        self.assertGreater(passes.SAME_DEFECT_CROSS_ID, passes.SAME_DEFECT)


class Survivor(unittest.TestCase):
    """The highest severity two frames independently support."""

    def test_the_worked_examples(self):
        R = condition.SEVERITY_RANK
        for ranks, want in (
                ([R["minor"], R["major"]], "minor"),
                ([R["moderate"], R["major"]], "moderate"),
                ([R["major"], R["major"]], "major"),
                ([R["minor"], R["major"], R["major"]], "major"),
                ([R["minor"], R["minor"], R["moderate"], R["major"]], "moderate")):
            with self.subTest(ranks=ranks):
                got = passes.corroborated_rank(ranks)
                self.assertEqual(got, R[want])

    def test_a_single_witness_keeps_its_own_claim(self):
        self.assertEqual(passes.corroborated_rank([3]), 3)

    def test_it_does_not_depend_on_photo_order(self):
        text = Overlap.LONG
        a = passes.merge_duplicates([issue(0, observation=text, severity="major"),
                                     issue(1, observation=text, severity="minor")], [])
        b = passes.merge_duplicates([issue(0, observation=text, severity="minor"),
                                     issue(1, observation=text, severity="major")], [])
        self.assertEqual(a[0].severity, b[0].severity)
        self.assertEqual(a[0].severity, "minor")

    def test_the_levels_that_lost_are_printed_not_dropped(self):
        text = Overlap.LONG
        out = passes.merge_duplicates([issue(0, observation=text, severity="major"),
                                       issue(1, observation=text, severity="minor")], [])
        self.assertEqual(out[0].severity_span, ["major", "minor"])

    def test_a_unanimous_group_has_no_span_to_show(self):
        text = Overlap.LONG
        out = passes.merge_duplicates([issue(0, observation=text, severity="minor"),
                                       issue(1, observation=text, severity="minor")], [])
        self.assertEqual(out[0].severity_span, [])

    def test_the_survivor_keeps_the_photo_the_report_cites(self):
        text = Overlap.LONG
        out = passes.merge_duplicates([issue(3, observation=text),
                                       issue(9, observation=text)], [])
        self.assertEqual(out[0].photo_id, 3)


class MergedConfidence(unittest.TestCase):
    def test_corroboration_raises_it_but_never_to_certainty(self):
        self.assertEqual(passes.merged_confidence([0.6]), 0.6)
        self.assertEqual(passes.merged_confidence([0.6, 0.6]), 0.684)
        self.assertEqual(passes.merged_confidence([0.6, 0.6, 0.6]), 0.75)
        self.assertLessEqual(passes.merged_confidence([1.0] * 20),
                             passes.CONFIDENCE_CEILING)

    def test_it_is_monotone_in_every_witness(self):
        base = [0.4, 0.5, 0.6]
        self.assertGreaterEqual(passes.merged_confidence(base + [0.7]),
                                passes.merged_confidence(base))

    def test_a_nonsense_confidence_cannot_poison_the_group(self):
        got = passes.merged_confidence([0.6, float("nan")])
        self.assertTrue(math.isfinite(got))


# --- the grade -------------------------------------------------------------

class GradeLadder(unittest.TestCase):
    def grade(self, issues, findings=None):
        return adjust(issues, findings)[0].grade

    def test_sixteen_spotless_photos_are_excellent_not_unknown(self):
        """The regression that started this. Both halves of it.

        `max(..., default=-1)` against a dict with no `-1` key returned
        `unknown` for a truck with nothing wrong, and finding one cosmetic
        scuff upgraded it to `good`. The grade was non-monotone at the clean
        end and exaggerated at the dirty one; it was the same bug.
        """
        self.assertEqual(self.grade([]), "excellent")

    def test_finding_a_scuff_can_never_improve_a_grade(self):
        clean = self.grade([])
        scuffed = self.grade([issue(component="paint_finish", severity="cosmetic",
                                    impact="none", confidence=0.6,
                                    observation="light scuff on the lower door skin")])
        self.assertLessEqual(condition.GRADE_RANK[scuffed],
                             condition.GRADE_RANK[clean])

    def test_a_scuffed_fuel_tank_strap_does_not_grade_the_whole_truck_fair(self):
        # `moderate` on its own used to do it. The model's own stated price
        # impact now has to be medium or better.
        self.assertEqual(self.grade([issue(component="fuel_tank", severity="moderate",
                                           impact="low", confidence=0.7)]), "good")

    def test_an_uncorroborated_low_confidence_moderate_does_not_either(self):
        self.assertEqual(self.grade([issue(severity="moderate", impact="medium",
                                           confidence=0.6)]), "good")

    def test_a_confident_moderate_does(self):
        self.assertEqual(self.grade([issue(severity="moderate", impact="medium",
                                           confidence=0.9)]), "fair")

    def test_a_corroborated_moderate_does_too(self):
        self.assertEqual(self.grade([issue(severity="moderate", impact="medium",
                                           confidence=0.5, also_seen_in=[3, 7])]),
                         "fair")

    def test_a_confident_major_is_poor(self):
        self.assertEqual(self.grade([issue(component="chassis_frame", severity="major",
                                           impact="high", confidence=0.95)]), "poor")

    def test_sixteen_paraphrases_of_one_moderate_are_still_only_fair(self):
        flat = [issue(i, observation=DISTINCT_TIRE[0], severity="moderate",
                      impact="medium", confidence=0.9) for i in range(16)]
        self.assertEqual(self.grade(passes.merge_duplicates(flat, [])), "fair")

    def test_thin_coverage_is_ungradeable_rather_than_good(self):
        roll = condition.rollup([], [PhotoFinding(photo_id=0, view="exterior_front_34",
                                                  strengths=["clean"])])
        self.assertEqual(roll.grade, "unknown")
        self.assertIn("photographed legibly", roll.grade_reason)

    def test_an_illegible_set_earns_no_coverage_at_all(self):
        roll = condition.rollup([], photos(legible=False))
        self.assertEqual(roll.coverage, 0.0)
        self.assertEqual(roll.grade, "unknown")

    def test_a_failed_call_is_not_a_legible_photo(self):
        roll = condition.rollup([], [PhotoFinding(photo_id=i, view=v, error="boom")
                                     for i, v in enumerate(ALL_VIEWS)])
        self.assertEqual(roll.coverage, 0.0)

    def test_the_reason_always_names_the_term_that_decided_it(self):
        for issues in ([], [issue()], [issue(severity="major", impact="high",
                                             confidence=0.95)]):
            roll, _ = adjust(issues)
            self.assertTrue(roll.grade_reason.startswith(roll.grade + ":"),
                            roll.grade_reason)

    def test_an_ungraded_finding_is_invisible_to_the_ladder(self):
        self.assertEqual(self.grade([issue(severity="major", impact="high",
                                           confidence=1.0, ungraded=True)]),
                         "excellent")


class CoverageAndMerit(unittest.TestCase):
    """Absence of evidence is not evidence of excellence, in the arithmetic."""

    def test_merit_can_never_exceed_coverage(self):
        rng = random.Random(11)
        for _ in range(200):
            n = rng.randint(0, 16)
            picked = rng.sample(ALL_VIEWS, min(n, len(set(ALL_VIEWS))) if n else 0) or []
            findings = [PhotoFinding(photo_id=i, view=v, legible=rng.random() > 0.3,
                                     strengths=["sound"] if rng.random() > 0.4 else [])
                        for i, v in enumerate(picked)]
            roll = condition.rollup([], findings)
            self.assertLessEqual(roll.merit, roll.coverage + 1e-9)

    def test_a_photo_nobody_affirmed_anything_about_earns_half_credit(self):
        roll = condition.rollup([], photos(strengths=False))
        self.assertEqual(roll.coverage, 1.0)
        self.assertAlmostEqual(roll.merit, 0.5, places=6)

    def test_a_subsystem_with_any_finding_on_it_stops_being_affirmed(self):
        # Even a zero-price-impact cosmetic one. It is still something somebody
        # saw, so that panel stops being one the truck is paid for.
        clean = condition.rollup([], photos())
        scuffed = condition.rollup([issue(component="paint_finish", severity="cosmetic",
                                          impact="none")], photos())
        self.assertLess(scuffed.merit, clean.merit)

    def test_a_close_ups_only_set_cannot_earn_a_premium(self):
        # demo/closeups_only: tire, interior, dash, engine and no exterior at
        # all - the case whose whole point is that it cannot be priced.
        findings = [PhotoFinding(photo_id=i, view=v, strengths=["sound"])
                    for i, v in enumerate(["tire_wheel", "interior_cab",
                                           "dashboard_odometer", "engine_bay"])]
        roll, adj = adjust([], findings)
        self.assertLess(roll.coverage, condition.C_MIN)
        self.assertEqual(adj.multiplier, 1.0)


# --- the properties --------------------------------------------------------

COMPONENTS = list(prompts.COMPONENTS)
SEVERITIES = list(prompts.SEVERITIES)
IMPACTS = list(prompts.IMPACTS)
# The measured worst case for sixteen UNMERGED restatements of one finding,
# swept over the whole (severity, impact, confidence) grid. Before this change
# the same sweep peaked at 6.0 pp and the sixteen-paraphrase case sat pinned at
# the cap. See `test_sixteen_restatements_cost_a_bounded_amount`.
UNMERGED_DUPLICATE_TOLERANCE_PP = 2.5
# What a SECOND WITNESS is worth, which is all a duplicate is allowed to move.
# A copy never adds a finding - that is asserted separately, and it is the half
# of the bug that mattered - but corroboration does legitimately raise the
# confidence on the surviving one, and confidence is a factor in the weight.
# Measured worst case over the whole enum: 0.7 pp, on a major/high finding.
DUPLICATE_TOLERANCE_PP = 1.0


def random_issue(rng, photo_id=None):
    return issue(photo_id=rng.randrange(16) if photo_id is None else photo_id,
                 component=rng.choice(COMPONENTS),
                 severity=rng.choice(SEVERITIES),
                 impact=rng.choice(IMPACTS),
                 confidence=round(rng.random(), 2),
                 observation=" ".join(rng.sample(DISTINCT_TIRE, 2)))


class Properties(unittest.TestCase):

    def test_a_copy_of_a_finding_is_never_a_second_finding(self):
        """Count cannot drive money. This is the half of the bug that mattered.

        Sixteen per-photo calls describing one worn drive tire used to cost
        sixteen times one worn drive tire: -9.4%, pinned at the cap, against
        -5.5% for a single statement of it.
        """
        rng = random.Random(3)
        for _ in range(300):
            base = [random_issue(rng) for _ in range(rng.randint(1, 6))]
            before = passes.merge_duplicates(clone(base), [])
            after = passes.merge_duplicates(clone(base) + [twin_of(rng, base)], [])
            self.assertEqual(len(after), len(before))

    def test_a_duplicate_moves_the_price_only_by_what_a_witness_is_worth(self):
        """The residue after count is gone: corroboration.

        A second frame showing the same defect really is stronger evidence than
        one, so it raises the surviving finding's confidence - and confidence is
        a factor in the weight. That is the only channel left, and it is small.
        """
        rng = random.Random(3)
        for _ in range(200):
            # Distinct families, so the base cannot merge with itself and the
            # twin is isolated from the severity vote, which is a different
            # mechanism with its own tests.
            base = [issue(photo_id=i, component=condition.FAMILIES[f][1][0],
                          severity=rng.choice(SEVERITIES), impact=rng.choice(IMPACTS),
                          confidence=round(rng.random(), 2),
                          observation=DISTINCT_TIRE[i % len(DISTINCT_TIRE)])
                    for i, f in enumerate(rng.sample(list(condition.FAMILIES),
                                                     rng.randint(1, 5)))]
            before = adjust(passes.merge_duplicates(clone(base), []))[1]
            after = adjust(passes.merge_duplicates(
                clone(base) + [twin_of(rng, base, offset=8)], []))[1]
            self.assertLessEqual(abs(after.pct - before.pct), DUPLICATE_TOLERANCE_PP,
                                 f"{before.pct} -> {after.pct}")

    def test_sixteen_restatements_cost_a_bounded_amount_even_unmerged(self):
        """The merge is not the only thing standing between us and the old bug.

        If every paraphrase escapes the merge, per-family saturation still caps
        what they can cost. Measured over the whole enum the worst case is
        about 2.1 pp; it used to be 6.0.
        """
        worst = 0.0
        for severity in SEVERITIES:
            for impact in IMPACTS:
                for conf in (0.3, 0.6, 0.9, 1.0):
                    one = [issue(0, severity=severity, impact=impact, confidence=conf)]
                    many = [issue(i, severity=severity, impact=impact, confidence=conf,
                                  observation=DISTINCT_TIRE[i % len(DISTINCT_TIRE)])
                            for i in range(16)]
                    delta = abs(adjust(many)[1].pct - adjust(one)[1].pct)
                    worst = max(worst, delta)
        self.assertLessEqual(worst, UNMERGED_DUPLICATE_TOLERANCE_PP, worst)

    def test_raising_a_severity_never_raises_the_grade_or_the_multiplier(self):
        rng = random.Random(5)
        order = {s: i for i, s in enumerate(SEVERITIES)}
        for _ in range(200):
            base = [random_issue(rng) for _ in range(rng.randint(1, 6))]
            idx = rng.randrange(len(base))
            if order[base[idx].severity] == len(SEVERITIES) - 1:
                continue
            before_roll, before = adjust(clone(base))
            raised = clone(base)
            raised[idx].severity = SEVERITIES[order[base[idx].severity] + 1]
            after_roll, after = adjust(raised)
            self.assertLessEqual(after.multiplier, before.multiplier + 1e-9)
            self.assertLessEqual(rank(after_roll.grade), rank(before_roll.grade))

    def test_adding_a_finding_never_raises_the_grade_or_the_multiplier(self):
        """The non-monotonicity at the clean end, as one executable line.

        One cosmetic scuff used to take a truck from `unknown` to `good`.
        """
        rng = random.Random(7)
        for _ in range(200):
            base = [random_issue(rng) for _ in range(rng.randint(0, 5))]
            before_roll, before = adjust(clone(base))
            after_roll, after = adjust(clone(base) + [random_issue(rng)])
            self.assertLessEqual(after.multiplier, before.multiplier + 1e-9)
            self.assertLessEqual(rank(after_roll.grade), rank(before_roll.grade))

    def test_the_cap_holds_for_anything_at_all(self):
        rng = random.Random(13)
        cases = {
            "five hundred majors": [issue(i % 16, component=rng.choice(COMPONENTS),
                                          severity="major", impact="high",
                                          confidence=1.0,
                                          observation=f"a distinct fault number {i}")
                                    for i in range(500)],
            "nothing at all": [],
            "junk enums": [issue(severity="catastrophic", impact="ruinous",
                                 confidence=2.5)],
            "not a number": [issue(confidence=float("nan")),
                             issue(confidence=float("inf")),
                             issue(confidence=None)],
            "ungraded": [issue(ungraded=True, severity="major", impact="high")],
        }
        for name, issues in cases.items():
            with self.subTest(name):
                for findings in (photos(), photos(strengths=False), [], photos(legible=False)):
                    _, adj = adjust(issues, findings)
                    self.assertTrue(math.isfinite(adj.multiplier))
                    self.assertLessEqual(abs(adj.pct), adj.cap_pct + 1e-9,
                                         f"{name}: {adj.pct} vs {adj.cap_pct}")

    def test_removing_a_photo_never_increases_the_multiplier(self):
        """"Absence of evidence", executable.

        Holding the findings fixed - because dropping the frame that showed the
        cracked rail is a different question - taking a photograph away can only
        cost coverage and merit, and must never buy a better number.
        """
        rng = random.Random(17)
        for _ in range(120):
            issues = [random_issue(rng) for _ in range(rng.randint(0, 3))]
            findings = photos(rng.randint(4, 16))
            _, full = adjust(clone(issues), findings)
            for i in range(len(findings)):
                fewer = findings[:i] + findings[i + 1:]
                _, short = adjust(clone(issues), fewer)
                self.assertLessEqual(short.multiplier, full.multiplier + 1e-9)

    def test_a_premium_exists_and_only_an_excellent_truck_gets_one(self):
        rng = random.Random(19)
        seen_premium = False
        for _ in range(300):
            issues = [random_issue(rng) for _ in range(rng.randint(0, 4))]
            findings = photos(rng.randint(1, 16),
                              strengths=rng.random() > 0.3,
                              legible=rng.random() > 0.2)
            roll, adj = adjust(clone(issues), findings)
            if adj.multiplier > 1.0:
                seen_premium = True
                self.assertEqual(roll.grade, "excellent")
                self.assertGreaterEqual(roll.coverage, condition.C_MIN)
        self.assertTrue(seen_premium, "the adjustment is one-directional again")

    def test_the_grade_and_the_price_agree(self):
        """The cross-check that did not exist at all.

        `poor` and `fair` both mean real wear and both must price below the
        comparable baseline; `excellent` means better than average and must
        price at or above it. In the middle, `good` means average, which is
        exactly where the baseline is.
        """
        rng = random.Random(23)
        seen = set()
        for _ in range(400):
            issues = [random_issue(rng) for _ in range(rng.randint(0, 6))]
            findings = photos(rng.randint(1, 16),
                              strengths=rng.random() > 0.3,
                              legible=rng.random() > 0.2)
            roll, adj = adjust(clone(issues), findings)
            seen.add(roll.grade)
            if roll.grade == "poor":
                self.assertLess(adj.multiplier, 1.0)
            elif roll.grade == "fair":
                self.assertLess(adj.multiplier, 1.0)
            elif roll.grade == "excellent":
                self.assertGreaterEqual(adj.multiplier, 1.0)
            else:
                self.assertLessEqual(adj.multiplier, 1.0)
        self.assertTrue({"poor", "fair", "good", "excellent"} <= seen, seen)

    def test_two_trucks_with_the_same_grade_are_ordered_by_their_wear(self):
        """The middle of the range, where the two outputs used to cross over.

        Ten minor findings across ten subsystems graded `good` at -1.8% while
        one moderate fuel-tank strap graded `fair` at -0.5%: the cheaper truck
        got the worse grade. Within a grade the more worn truck now always
        prices lower.
        """
        rng = random.Random(29)
        by_grade: dict[str, list] = {}
        for _ in range(300):
            issues = [random_issue(rng) for _ in range(rng.randint(0, 5))]
            roll, adj = adjust(clone(issues))
            by_grade.setdefault(roll.grade, []).append((roll.demerit, adj.multiplier))
        for grade, rows in by_grade.items():
            if grade == "excellent":
                continue        # priced on merit, not on demerit, by design
            rows.sort()
            for (d1, m1), (d2, m2) in zip(rows, rows[1:]):
                self.assertLessEqual(m2, m1 + 1e-9, f"{grade}: {d1}->{d2}")


def twin_of(rng, base, offset: int = 1):
    """Another frame describing one of these findings in the same words."""
    original = rng.choice(base)
    return issue(photo_id=(original.photo_id + offset) % 16,
                 component=original.component, severity=original.severity,
                 impact=original.price_impact, confidence=original.confidence,
                 observation=original.observation)


def clone(issues):
    return [Issue(photo_id=i.photo_id, component=i.component,
                  observation=i.observation, severity=i.severity,
                  confidence=i.confidence, price_impact=i.price_impact,
                  also_seen_in=list(i.also_seen_in), ungraded=i.ungraded)
            for i in issues]


def rank(grade: str) -> int:
    # `unknown` is not a rung; it sits below every grade so "adding a finding
    # never raises the grade" is checkable across the whole space.
    return condition.GRADE_RANK.get(grade, -1)


# --- the band --------------------------------------------------------------

class ConditionBand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.pricing import load_model
        cls.model = load_model()

    def estimate(self, evidence=None, **kw):
        kw.setdefault("year", 2021)
        kw.setdefault("km", 300000)
        kw.setdefault("make", "FORD")
        return M.estimate(self.model, market="TR", evidence=evidence, **kw)

    def test_the_shipped_model_still_carries_the_measured_numbers(self):
        # Everything here is downstream of these four; if a refit moves them,
        # the cap and every basis string in the product move with it.
        c = self.model.calibration
        self.assertEqual(self.model.residual_std, CAP_LOG)
        self.assertEqual(c["r2_oof"], 0.8422)
        self.assertEqual(c["median_ape_oof"], 4.2)
        self.assertEqual(c["coverage_0.8"], 0.8027)
        self.assertEqual(c["coverage_n"], 958)

    def test_the_condition_band_is_never_narrower_than_the_comparable_one(self):
        """The mirror of the anchor's floor, and for the same reason.

        The measured 80.3% belongs to the unwidened hedonic interval. Nothing
        the photographs say has earned the right to claim a tighter one.
        """
        rng = random.Random(31)
        for _ in range(60):
            issues = [random_issue(rng) for _ in range(rng.randint(0, 4))]
            findings = photos(rng.randint(1, 16), legible=rng.random() > 0.3)
            roll = condition.rollup(issues, findings)
            ev = EvidenceReport(issues=issues, photo_findings=findings,
                                condition=roll, condition_grade=roll.grade,
                                condition_grade_model=rng.choice(
                                    ["", "poor", "fair", "good", "excellent"]))
            adj = M.condition_adjustment(ev, CAP_LOG)
            factor, _ = M.condition_widening(self.model, ev, adj)
            self.assertGreaterEqual(factor, M.CONDITION_WIDENING_FLOOR)
            self.assertLessEqual(factor, M.CONDITION_WIDENING_CAP)

    def test_the_baseline_band_keeps_the_unwidened_spec_factor(self):
        thin = [PhotoFinding(photo_id=0, view="tire_wheel", strengths=["sound"])]
        roll = condition.rollup([], thin)
        ev = EvidenceReport(issues=[], photo_findings=thin, condition=roll,
                            condition_grade=roll.grade)
        bare = self.estimate()
        read = self.estimate(evidence=ev)
        # Thin coverage widens the condition band and leaves the measured one
        # exactly where it was.
        self.assertEqual(bare.baseline_low, read.baseline_low)
        self.assertEqual(bare.baseline_high, read.baseline_high)
        self.assertGreater(math.log(read.high / read.low),
                           math.log(read.baseline_high / read.baseline_low))

    def test_a_full_clean_read_does_not_widen_the_condition_band(self):
        roll = condition.rollup([], photos())
        ev = EvidenceReport(issues=[], photo_findings=photos(), condition=roll,
                            condition_grade=roll.grade)
        adj = M.condition_adjustment(ev, CAP_LOG)
        self.assertEqual(M.condition_widening(self.model, ev, adj)[0], 1.0)
        e = self.estimate(evidence=ev)
        # In log space, because the adjustment shifts the whole band and a
        # shifted band is wider in lira without being wider in information.
        # Two places, not more: both ends are rounded to the nearest thousand.
        self.assertAlmostEqual(math.log(e.high / e.low)
                               / math.log(e.baseline_high / e.baseline_low),
                               1.0, places=2)

    def test_the_two_graders_disagreeing_widens_the_band(self):
        findings = photos()
        roll = condition.rollup([], findings)
        agree = EvidenceReport(issues=[], photo_findings=findings, condition=roll,
                               condition_grade=roll.grade,
                               condition_grade_model=roll.grade)
        clash = EvidenceReport(issues=[], photo_findings=findings, condition=roll,
                               condition_grade=roll.grade,
                               condition_grade_model="poor")
        a = M.condition_widening(self.model, agree,
                                 M.condition_adjustment(agree, CAP_LOG))[0]
        b, why = M.condition_widening(self.model, clash,
                                      M.condition_adjustment(clash, CAP_LOG))
        self.assertGreater(b, a)
        self.assertTrue(any("synthesis pass graded" in w for w in why))

    def test_no_evidence_means_the_two_bands_are_the_same_band(self):
        e = self.estimate()
        self.assertEqual(e.low, e.baseline_low)
        self.assertEqual(e.high, e.baseline_high)

    def test_the_cap_and_the_weights_are_labelled_differently(self):
        adj = M.condition_adjustment(None, CAP_LOG)
        self.assertIn("measured", adj.cap_basis)
        self.assertIn("assumed", adj.weights_basis)
        self.assertNotIn("assumption", adj.cap_basis)


if __name__ == "__main__":
    unittest.main()
