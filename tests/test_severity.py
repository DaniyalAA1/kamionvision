"""What stops the appraisal exaggerating, and what is allowed to change a severity.

The bug these cover, measured on live code before any of this existed: a
scuffed fuel-tank strap reported "moderate" graded a whole truck "fair" while
costing 0.5% of the price; sixteen paraphrases of one worn drive tire cost an
extra 3.9 points of price over one statement of it; and a close-up call was
never told how far the truck had run, so it graded every consumable against a
new one.

Everything here is offline. No API keys, no images, no network.
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from app.evidence import calibration, passes, prompts, sampling
from app.evidence import stage as stage_module
from app.schema import EvidenceReport, Issue, Magnitude, PhotoCheck, PhotoFinding


def _check(photo_id: int, name: str = "x.jpg", view: str = "tire_wheel") -> PhotoCheck:
    check = PhotoCheck(photo_id=photo_id, path=f"/tmp/{name}", filename=name, usable=True)
    check.view = view
    return check


def _issue(photo_id: int = 0, component: str = "drive_tires", severity: str = "moderate",
           observation: str = "the outer shoulder is worn below the centre ribs",
           **kw) -> Issue:
    return Issue(photo_id=photo_id, component=component, observation=observation,
                 severity=severity, confidence=kw.pop("confidence", 1.0),
                 price_impact=kw.pop("price_impact", "medium"), **kw)


def _closeup_json(observations=(), **kw) -> str:
    body = {"shows": "a tire", "legible": True, "odometer_km": None,
            "observations": list(observations), "strengths": ["tread stands proud"],
            "cannot_tell": [], "confidence": 0.6}
    body.update(kw)
    return json.dumps(body)


def _observation(severity: str = "moderate", component: str = "drive_tires",
                 observation: str = "outer shoulder worn below the centre ribs",
                 **kw) -> dict:
    entry = {"component": component, "observation": observation, "severity": severity,
             "confidence": 0.8, "price_impact": "medium", "box": None}
    entry.update(kw)
    return entry


class _Client:
    """A backend that answers with whatever it was handed, and keeps the prompts."""

    supports_structured_output = False
    name = "fake"

    def __init__(self, replies=("{}",), fail_on=()):
        self.replies = list(replies)
        self.fail_on = set(fail_on)
        self.prompts: list[str] = []
        self.calls = 0

    def complete(self, prompt, images, **kw):
        from app.vlm.base import VLMError, VLMResponse

        self.calls += 1
        self.prompts.append(prompt)
        if images and images[0].name in self.fail_on:
            raise VLMError("provider said no")
        text = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        return VLMResponse(text=text, backend="fake", model="fake")


# --- the baseline the close-up never had ----------------------------------

class ExpectationLine(unittest.TestCase):
    """The truck's own age and distance, and the one digit that may not travel."""

    def test_a_stated_distance_is_a_band_and_never_the_figure(self):
        line = passes.expectation_line({"year": 2021, "km": 164374.0}, today=2026)
        self.assertIn("100,000-200,000 km", line)
        for rendering in ("164374", "164,374", "164374.0"):
            self.assertNotIn(rendering, line)

    def test_the_declared_distance_never_reaches_a_closeup_prompt(self):
        # The hazard this exists to prevent: a close-up call told the seller's
        # exact distance can echo it back as `odometer_km`, and
        # `reconcile._check_odometer` - the only cross-check that can catch a
        # seller understating the distance - would then be comparing the
        # seller's number against itself. The rehearsed `odometer_lie` case
        # (declared 420,000 against a real 164,374) rests on this.
        declared = {"year": 2021, "km": 420000.0}
        prompt = prompts.closeup_prompt(
            view="dashboard_odometer", view_pretty="dashboard odometer",
            vehicle="A Ford F-MAX.", cropped=False, soft=False,
            expectation=passes.expectation_line(declared, today=2026),
            band=prompts.wear_band(420000))
        for rendering in ("420000", "420,000", "420000.0"):
            self.assertNotIn(rendering, prompt)
        self.assertIn('Do NOT use this band to fill in "odometer_km"', prompt)

    def test_that_check_fails_when_the_band_is_replaced_by_the_figure(self):
        # The mutation check. A test that asserts a number is absent passes
        # just as happily when the line it is checking is empty, so break the
        # thing on purpose and confirm the assertion notices.
        def leaky(declared, today=None):
            return f"The seller states {int(declared['km'])} km."

        original, passes.expectation_line = passes.expectation_line, leaky
        try:
            prompt = prompts.closeup_prompt(
                view="dashboard_odometer", view_pretty="dashboard odometer",
                vehicle="A Ford F-MAX.", cropped=False, soft=False,
                expectation=passes.expectation_line({"km": 420000.0}), band="mid")
            self.assertIn("420000", prompt)
        finally:
            passes.expectation_line = original

    def test_the_unknown_case_does_not_assume_an_old_truck(self):
        # The case that will actually run on judging day: a folder of photos
        # and nothing typed in. The null bias has to be neutral, because
        # "assume the worst" is the bug.
        line = passes.expectation_line(None)
        self.assertIn("Do not assume it is either new or old", line)
        self.assertIn("cannot_tell", line)

    def test_a_high_distance_truck_is_judged_against_the_work_it_has_done(self):
        line = passes.expectation_line({"year": 2018, "km": 900000}, today=2026)
        self.assertIn("worked harder than most", line)
        self.assertIn("is on schedule and is not a finding", line)

    def test_a_low_distance_truck_raises_the_bar_instead(self):
        line = passes.expectation_line({"year": 2014, "km": 120000}, today=2026)
        self.assertIn("worked far less than most", line)
        self.assertIn("genuine finding on this truck", line)

    def test_a_year_with_no_distance_says_the_spread_is_not_this_truck(self):
        line = passes.expectation_line({"year": 2021}, today=2026)
        self.assertIn("did not state a distance", line)
        self.assertIn("spread of stock offered for sale", line)

    def test_the_band_widens_past_half_a_million(self):
        self.assertEqual(passes.km_band(164_374), (100_000, 200_000))
        self.assertEqual(passes.km_band(499_999), (400_000, 500_000))
        self.assertEqual(passes.km_band(900_000), (750_000, 1_000_000))

    def test_the_wear_rows_that_reach_the_prompt_are_the_ones_for_that_view(self):
        prompt = prompts.closeup_prompt(
            view="tire_wheel", view_pretty="tire wheel", vehicle="A truck.",
            cropped=False, soft=False, expectation="", band="very_high")
        self.assertIn("retreads on the drive axle are ordinary commercial practice",
                      prompt)
        self.assertNotIn("seat foam and bolsters", prompt)

    def test_no_wear_row_is_injected_when_no_distance_was_stated(self):
        prompt = prompts.closeup_prompt(
            view="tire_wheel", view_pretty="tire wheel", vehicle="A truck.",
            cropped=False, soft=False, expectation="", band=None)
        self.assertNotIn("on schedule looks like", prompt)


class PromptZones(unittest.TestCase):
    """Invariant, then per-appraisal, then per-photo - in that order."""

    def test_the_invariant_prefix_is_byte_identical_across_views(self):
        # With the rubric in the prompt and each photo read CLOSEUP_SAMPLES
        # times, this ordering is the difference between paying for the rubric
        # 48 times and paying for it once on any backend that caches prefixes.
        a = prompts.closeup_prompt(view="tire_wheel", view_pretty="tire wheel",
                                   vehicle="A Ford.", cropped=False, soft=False,
                                   expectation="Five years old.", band="mid")
        b = prompts.closeup_prompt(view="engine_bay", view_pretty="engine bay",
                                   vehicle="A MAN.", cropped=True, soft=True,
                                   expectation="Ten years old.", band=None)
        shared = prompts.CLOSEUP_INVARIANT
        self.assertGreater(len(shared), 4000)
        self.assertTrue(a.startswith(shared))
        self.assertTrue(b.startswith(shared))
        self.assertEqual(a[:len(shared)], b[:len(shared)])

    def test_the_rubric_and_the_examples_are_in_the_invariant_zone(self):
        self.assertIn(prompts.SEVERITY_RUBRIC, prompts.CLOSEUP_INVARIANT)
        self.assertIn(prompts.WORKED_EXAMPLES, prompts.CLOSEUP_INVARIANT)

    def test_the_vehicle_and_the_expectation_come_after_the_invariant(self):
        prompt = prompts.closeup_prompt(
            view="tire_wheel", view_pretty="tire wheel", vehicle="A Ford F-MAX.",
            cropped=False, soft=False, expectation="It is five years old.", band="mid")
        self.assertGreater(prompt.index("A Ford F-MAX."), len(prompts.CLOSEUP_INVARIANT) - 1)
        self.assertLess(prompt.index("It is five years old."),
                        prompt.index("Work through each of these"))

    def test_the_close_up_asks_for_strengths_by_name(self):
        # 76 checklist items across eleven views named a failure mode and not
        # one asked the model to confirm a component was sound. `strengths` was
        # in the schema, rendered on three surfaces, and prompted for by
        # nothing - a model asked only what is wrong answers only what is wrong.
        prompt = prompts.closeup_prompt(
            view="fifth_wheel", view_pretty="fifth wheel", vehicle="A truck.",
            cropped=False, soft=False)
        self.assertIn('put what you can confirm into "strengths"', prompt)
        for item in prompts.VIEW_CONFIRMATIONS["fifth_wheel"]:
            self.assertIn(item, prompt)


class NoPriceInPrompts(unittest.TestCase):
    """The VLM never sees or emits a price - including the seller's own."""

    def test_the_asking_price_never_reaches_the_identity_pass(self):
        declared = {"year": 2021, "km": 164374.0, "asking_price": 2350000.0}
        context = passes.identity_context(declared, [_check(0), _check(1)])
        self.assertIn("year=2021", context)
        for rendering in ("2350000", "2,350,000", "2350000.0", "asking"):
            self.assertNotIn(rendering, context)

    def test_the_asking_price_never_reaches_the_expectation_line(self):
        line = passes.expectation_line({"year": 2021, "km": 164374.0,
                                        "asking_price": 2350000.0}, today=2026)
        self.assertNotIn("2350000", line)
        self.assertNotIn("2,350,000", line)

    def test_no_evidence_schema_carries_a_cost_field(self):
        blob = json.dumps([prompts.closeup_schema(), prompts.CALIBRATION_SCHEMA,
                           prompts.synthesis_schema()])
        for word in ("price\": {\"type\": \"number", "cost", "lira", "TRY", "USD"):
            self.assertNotIn(word, blob)


# --- one photograph, read k times -----------------------------------------

class SampleCombination(unittest.TestCase):
    """`Issue.confidence` stops being a self-report and becomes an agreement rate."""

    def _samples(self, severities, confidences=None):
        out = []
        for i, severity in enumerate(severities):
            finding = PhotoFinding(photo_id=3, view="tire_wheel", shows="a tire")
            if severity:
                finding.issues.append(_issue(
                    photo_id=3, severity=severity,
                    confidence=(confidences or [0.9] * len(severities))[i]))
            out.append(finding)
        return out

    def test_three_of_three_takes_the_median(self):
        merged, corrections = sampling.combine_samples(
            self._samples(["minor", "moderate", "major"]))
        self.assertEqual(len(merged.issues), 1)
        self.assertEqual(merged.issues[0].severity, "moderate")
        self.assertEqual(merged.issues[0].confidence, 1.0)
        self.assertTrue(merged.issues[0].corroborated)
        self.assertEqual([c.kind for c in corrections], ["sample_disagreement"])

    def test_unanimous_samples_raise_no_correction(self):
        merged, corrections = sampling.combine_samples(
            self._samples(["moderate", "moderate", "moderate"]))
        self.assertEqual(merged.issues[0].severity, "moderate")
        self.assertEqual(corrections, [])

    def test_two_of_three_takes_the_lower_and_says_they_disagreed(self):
        merged, corrections = sampling.combine_samples(
            self._samples(["major", "moderate", None]))
        issue = merged.issues[0]
        self.assertEqual(issue.severity, "moderate")
        self.assertEqual(issue.confidence, round(2 / 3, 3))
        self.assertTrue(issue.corroborated)
        self.assertEqual([c.kind for c in corrections], ["sample_disagreement"])

    def test_one_of_three_is_kept_uncorroborated_and_cannot_be_major(self):
        # Kept, not deleted: one read of a real defect is still evidence. But
        # one read is not corroboration, and "major" is the level that tells a
        # buyer to spend money before the truck earns.
        merged, corrections = sampling.combine_samples(
            self._samples(["major", None, None]))
        issue = merged.issues[0]
        self.assertEqual(issue.severity, "moderate")
        self.assertFalse(issue.corroborated)
        self.assertEqual(issue.confidence, round(1 / 3, 3))
        self.assertEqual([c.kind for c in corrections], ["uncorroborated_finding"])
        self.assertEqual(corrections[0].before, "major")
        self.assertEqual(corrections[0].after, "moderate")

    def test_the_models_own_number_is_kept_where_it_can_be_checked(self):
        merged, _ = sampling.combine_samples(
            self._samples(["moderate"] * 3, [0.9, 0.6, 0.9]))
        self.assertEqual(merged.issues[0].self_confidence, 0.8)
        self.assertEqual(merged.issues[0].confidence, 1.0)
        self.assertEqual(merged.issues[0].severity_votes,
                         ["moderate", "moderate", "moderate"])

    def test_one_sample_degrades_to_todays_behaviour(self):
        # A photo whose other two reads both failed is a provider failure, not
        # evidence about the truck: the survivor is not marked uncorroborated.
        merged, corrections = sampling.combine_samples(self._samples(["major"]))
        self.assertEqual(merged.issues[0].severity, "major")
        self.assertTrue(merged.issues[0].corroborated)
        self.assertEqual(corrections, [])

    def test_the_samples_vote_on_the_odometer(self):
        samples = self._samples([None, None, None])
        samples[0].odometer_km, samples[1].odometer_km = 164374, 164374
        samples[2].odometer_km = 764374
        merged, _ = sampling.combine_samples(samples)
        self.assertEqual(merged.odometer_km, 164374)
        self.assertTrue(any("disagreed" in e for e in merged.sample_errors))

    def test_strengths_from_every_sample_survive_deduplication(self):
        samples = self._samples([None, None])
        samples[0].strengths = ["the tread stands well proud of the wear bars"]
        samples[1].strengths = ["the tread stands well proud of the wear bars",
                                "the rim is straight and undamaged"]
        merged, _ = sampling.combine_samples(samples)
        self.assertEqual(len(merged.strengths), 2)


class SamplingFailure(unittest.TestCase):
    """A failed sample costs a vote; a failed photo costs a photo."""

    def test_a_photo_survives_losing_a_sample(self):
        client = _Client(replies=[_closeup_json([_observation()])])
        with tempfile.TemporaryDirectory() as tmp:
            finding, _, _ = sampling.closeup_consensus(
                [client], _check(0), "A truck.", tmpdir=Path(tmp),
                max_tokens=100, samples=3)
        self.assertEqual(finding.samples, 3)
        self.assertEqual(len(finding.issues), 1)

    def test_every_sample_failing_raises_so_the_photo_is_recorded_lost(self):
        client = _Client(fail_on={"0.jpg"})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception):
                sampling.closeup_consensus([client], _check(0, "0.jpg"), "A truck.",
                                           tmpdir=Path(tmp), max_tokens=100, samples=3)

    def test_a_sample_falls_back_to_the_next_backend_and_says_so(self):
        dead = _Client(fail_on={"0.jpg"})
        dead.name = "dead"
        alive = _Client(replies=[_closeup_json()])
        with tempfile.TemporaryDirectory() as tmp:
            finding, _, fallbacks = sampling.closeup_consensus(
                [dead, alive], _check(0, "0.jpg"), "A truck.", tmpdir=Path(tmp),
                max_tokens=100, samples=2)
        self.assertEqual(finding.samples, 2)
        self.assertTrue(all("retried on fake" in f for f in fallbacks))
        self.assertEqual(len(fallbacks), 2)

    def test_an_unparseable_sample_is_repaired_rather_than_lost(self):
        client = _Client(replies=["not json at all", _closeup_json()])
        with tempfile.TemporaryDirectory() as tmp:
            finding, _, _ = sampling.closeup_consensus(
                [client], _check(0), "A truck.", tmpdir=Path(tmp), max_tokens=100,
                samples=1, repair=stage_module._repair)
        self.assertEqual(finding.samples, 1)
        self.assertTrue(any("repaired" in e for e in finding.sample_errors))


# --- pass D ----------------------------------------------------------------

class MagnitudeParsing(unittest.TestCase):
    def test_the_four_absolute_fields_survive(self):
        mag = calibration.parse_magnitude({
            "extent": "local", "state": "end_of_life", "consumable": True,
            "blocks_use": "no"})
        self.assertEqual((mag.extent, mag.state, mag.consumable, mag.blocks_use),
                         ("local", "end_of_life", True, "no"))

    def test_a_value_outside_its_enum_is_dropped_not_rounded(self):
        mag = calibration.parse_magnitude({"extent": "enormous", "state": "end_of_life",
                                           "consumable": False, "blocks_use": "no"})
        self.assertEqual(mag.extent, "")
        self.assertEqual(mag.state, "end_of_life")

    def test_nothing_readable_is_nothing(self):
        self.assertIsNone(calibration.parse_magnitude(None))
        self.assertIsNone(calibration.parse_magnitude({"extent": "huge"}))

    def test_the_close_up_parser_binds_the_magnitude_to_the_finding(self):
        text = _closeup_json([_observation(magnitude={
            "extent": "whole_component", "state": "end_of_life",
            "consumable": True, "blocks_use": "maybe"})])
        finding = passes.parse_closeup(text, _check(4), cropped=False)
        self.assertEqual(finding.issues[0].magnitude.extent, "whole_component")


class CalibrationLicence(unittest.TestCase):
    """Lower freely, raise by one, raise to major only with corroboration."""

    def _apply(self, issues, revisions, **kw):
        data = {"revisions": revisions, "worst_finding": kw.get("worst"),
                "calibration_note": ""}
        return calibration.apply_revisions(issues, data)

    def test_lowering_is_free(self):
        issues = [_issue(severity="major", price_impact="high")]
        corrections, warnings = self._apply(issues, [
            {"finding": 0, "severity": "cosmetic", "price_impact": "none",
             "reason": "it is a scuff"}])
        self.assertEqual(issues[0].severity, "cosmetic")
        self.assertEqual(issues[0].price_impact, "none")
        self.assertEqual(issues[0].severity_provisional, "major")
        self.assertEqual([c.kind for c in corrections], ["severity_calibrated"])
        self.assertEqual(corrections[0].before, "major/high")
        self.assertEqual(corrections[0].after, "cosmetic/none")
        self.assertEqual(warnings, [])

    def test_a_raise_of_one_level_is_allowed(self):
        issues = [_issue(severity="minor", also_seen_in=[2, 5])]
        self._apply(issues, [{"finding": 0, "severity": "moderate",
                              "price_impact": "medium", "reason": "both sides"}])
        self.assertEqual(issues[0].severity, "moderate")

    def test_a_raise_of_two_levels_is_clamped_and_written_down(self):
        issues = [_issue(severity="cosmetic", price_impact="none", also_seen_in=[1])]
        corrections, _ = self._apply(issues, [
            {"finding": 0, "severity": "major", "price_impact": "high",
             "reason": "worse than it looked"}])
        self.assertEqual(issues[0].severity, "minor")
        self.assertEqual(issues[0].price_impact, "low")
        kinds = [c.kind for c in corrections]
        self.assertIn("severity_raise_clamped", kinds)
        self.assertIn("severity_calibrated", kinds)

    def test_major_needs_a_second_photograph(self):
        issues = [_issue(severity="moderate", also_seen_in=[])]
        corrections, _ = self._apply(issues, [
            {"finding": 0, "severity": "major", "price_impact": "high",
             "reason": "it is bad"}])
        self.assertEqual(issues[0].severity, "moderate")
        self.assertEqual([c.kind for c in corrections], ["severity_raise_clamped"])

    def test_a_refused_raise_cannot_come_back_through_the_price_impact(self):
        """The licence has to bind both axes, because the price multiplies them.

        `condition_adjustment` weighs a finding as
        `SEVERITY_WEIGHT[severity] * IMPACT_WEIGHT[price_impact] * confidence`.
        Refusing a raise to `major` for want of corroboration while granting
        `medium -> high` on the same revision let the identical claim through on
        the other axis and doubled the weight anyway. Assert the weight, not the
        labels: the labels are a proxy and this is the quantity that reaches the
        seller's number.
        """
        from app.pricing.model import SEVERITY_WEIGHT, IMPACT_WEIGHT

        def weight(issue):
            return SEVERITY_WEIGHT[issue.severity] * IMPACT_WEIGHT[issue.price_impact]

        issues = [_issue(severity="moderate", price_impact="medium", also_seen_in=[])]
        before = weight(issues[0])
        self._apply(issues, [{"finding": 0, "severity": "major",
                              "price_impact": "high", "reason": "it is bad"}])
        self.assertEqual(weight(issues[0]), before)

    def test_impact_still_rises_when_the_severity_it_rode_in_on_did(self):
        """The coupling is "impact may not outrun severity", not "impact is frozen"."""
        issues = [_issue(severity="cosmetic", price_impact="none", also_seen_in=[1])]
        self._apply(issues, [{"finding": 0, "severity": "major",
                              "price_impact": "high", "reason": "worse than it looked"}])
        self.assertEqual(issues[0].severity, "minor")     # clamped one level
        self.assertEqual(issues[0].price_impact, "low")   # followed it one level

    def test_lowering_the_impact_is_always_free(self):
        issues = [_issue(severity="moderate", price_impact="high", also_seen_in=[])]
        self._apply(issues, [{"finding": 0, "severity": "minor",
                              "price_impact": "none", "reason": "on schedule"}])
        self.assertEqual((issues[0].severity, issues[0].price_impact), ("minor", "none"))

    def test_major_is_allowed_when_the_defect_was_seen_twice(self):
        issues = [_issue(severity="moderate", also_seen_in=[7])]
        self._apply(issues, [{"finding": 0, "severity": "major",
                              "price_impact": "high", "reason": "both frames"}])
        self.assertEqual(issues[0].severity, "major")

    def test_an_uncorroborated_finding_cannot_be_raised_to_major(self):
        issues = [_issue(severity="moderate", also_seen_in=[7], corroborated=False)]
        self._apply(issues, [{"finding": 0, "severity": "major",
                              "price_impact": "high", "reason": "both frames"}])
        self.assertEqual(issues[0].severity, "moderate")

    def test_a_finding_with_no_decision_keeps_its_level_and_is_reported(self):
        issues = [_issue(severity="moderate"), _issue(photo_id=1, severity="minor",
                                                      observation="a kerbed rim")]
        corrections, warnings = self._apply(issues, [
            {"finding": 0, "severity": "minor", "price_impact": "low", "reason": "x"}])
        self.assertEqual(issues[1].severity, "minor")
        self.assertEqual(len(warnings), 1)
        self.assertIn("1", warnings[0])

    def test_pass_d_cannot_create_delete_or_rebind_a_finding(self):
        # The photo_id binding is the whole trust story of this system, and
        # pass D is text-only precisely so it cannot touch it.
        issues = [_issue(photo_id=0), _issue(photo_id=4, component="fifth_wheel",
                                             observation="the plate is dry")]
        before_ids = [i.photo_id for i in issues]
        self._apply(issues, [
            {"finding": 0, "severity": "minor", "price_impact": "low", "reason": "a"},
            {"finding": 1, "severity": "minor", "price_impact": "low", "reason": "b"},
            {"finding": 9, "severity": "major", "price_impact": "high", "reason": "c"},
            {"finding": -3, "severity": "major", "price_impact": "high", "reason": "d"},
            {"finding": 0, "severity": "major", "price_impact": "high", "reason": "e"},
        ])
        self.assertEqual(len(issues), 2)
        self.assertEqual([i.photo_id for i in issues], before_ids)
        self.assertEqual(issues[0].severity, "minor")   # the repeat index is ignored

    def test_a_downgraded_finding_is_still_shown(self):
        issues = [_issue(severity="major", price_impact="high")]
        self._apply(issues, [{"finding": 0, "severity": "cosmetic",
                              "price_impact": "none", "reason": "ordinary wear"}])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity_reason, "ordinary wear")

    def test_the_worst_finding_is_named_in_words(self):
        issues = [_issue(), _issue(photo_id=2, component="wheels_rims",
                                   observation="the rim is kerbed")]
        self.assertIn("wheels rims", calibration.worst_note(issues, 1))
        self.assertEqual(calibration.worst_note(issues, 9), "")
        self.assertEqual(calibration.worst_note(issues, None), "")


class CalibrationPrompt(unittest.TestCase):
    def test_every_finding_is_numbered_with_its_corroboration(self):
        issues = [_issue(also_seen_in=[3, 4], severity_votes=["moderate", "moderate"])]
        block = calibration.findings_block(issues, {"drive_tires": 6}, {0: 3})
        self.assertIn("[0] drive_tires", block)
        self.assertIn("seen in 3 of 6 frames that show this component", block)
        self.assertIn("reported by 2 of 3 reads", block)

    def test_the_prompt_demands_one_entry_per_finding(self):
        text = prompts.calibration_prompt(
            vehicle="A Ford.", expectation="Five years old.", n_photos=16,
            coverage="", strengths="  - tread proud", findings="  [0] x", last=0)
        self.assertIn("EXACTLY one entry per finding", text)
        self.assertIn("Re-decide EVERY finding", text)
        self.assertIn(prompts.SEVERITY_RUBRIC, text)

    def test_a_truck_with_nothing_confirmed_sound_says_so_honestly(self):
        text = prompts.calibration_prompt(
            vehicle="A Ford.", expectation="", n_photos=16, coverage="",
            strengths="", findings="  [0] x", last=0)
        self.assertIn("thin coverage, not as evidence against the truck", text)

    def test_the_strengths_a_photo_confirmed_are_folded_once(self):
        findings = [PhotoFinding(photo_id=i, view="tire_wheel",
                                 strengths=["the tread stands proud of the wear bars"])
                    for i in range(4)]
        findings[3].strengths.append("the rim is straight and true")
        self.assertEqual(len(calibration.confirmed_sound(findings)), 2)


class CalibrationPosture(unittest.TestCase):
    """What a failed pass D costs, and what it is allowed to move."""

    def _report(self, grade="poor", severity="major"):
        report = EvidenceReport()
        report.issues = [_issue(severity=severity)]
        report.condition_grade = grade
        report.photos_read = 16
        return report

    def test_a_failed_calibration_leaves_every_severity_alone(self):
        report = self._report()
        before = [(i.photo_id, i.severity, i.price_impact) for i in report.issues]
        stage_module._calibrate([_Client(fail_on=())], _Client(replies=["not json"]),
                                report, [], vehicle_line="A truck.", expectation="")
        self.assertEqual([(i.photo_id, i.severity, i.price_impact)
                          for i in report.issues], before)
        self.assertTrue(any("calibration pass" in w for w in report.parse_warnings))

    def test_the_grade_is_computed_after_calibration_not_patched_up_after(self):
        """`_relax_grade` is gone, and the ordering is why.

        It existed because pass D used to run AFTER the rollup had already
        turned severity into a grade, so a truck whose worst finding had just
        been calibrated down from major to moderate could still be handed to a
        seller graded "poor" on the strength of a finding that no longer existed
        at that level. It patched that up one-way, and only for the grade - the
        PRICE still read the calibrated severities, so the two could disagree.

        `stage.run` now calls `_calibrate` between `merge_duplicates` and
        `condition.rollup`, so the grade and the multiplier are both functions
        of the calibrated list by construction and there is nothing to patch.
        """
        import inspect
        src = inspect.getsource(stage_module.run)
        i_merge = src.index("merge_duplicates")
        i_cal = src.index("_calibrate(")
        i_roll = src.index("condition_stage.rollup")
        self.assertLess(i_merge, i_cal, "pass D must run after the merge")
        self.assertLess(i_cal, i_roll, "pass D must run before the rollup")
        self.assertFalse(hasattr(stage_module, "_relax_grade"),
                         "the ordering fix makes the patch-up unnecessary")

    def test_calibrating_a_severity_down_moves_the_grade_with_it(self):
        """The property `_relax_grade` was approximating, now structural."""
        from app import condition as condition_stage
        issues = [_issue(severity="major", price_impact="high", also_seen_in=[2])]
        findings = [PhotoFinding(photo_id=0, view="tire_wheel", shows="a tire",
                                 legible=True)]
        before = condition_stage.rollup(issues, findings).demerit
        calibration.apply_revisions(issues, {"revisions": [
            {"finding": 0, "severity": "minor", "price_impact": "low",
             "reason": "on schedule for the distance"}]})
        after = condition_stage.rollup(issues, findings).demerit
        self.assertEqual(issues[0].severity, "minor")
        # Asserted on the demerit rather than the grade: the grade also carries
        # coverage reasoning, and a one-photo fixture is correctly `unknown`
        # either side. The demerit is the quantity that reaches the price.
        self.assertLess(after, before)


class FrameDenominators(unittest.TestCase):
    """The count that turns "seen once" into evidence rather than a shrug."""

    def test_a_component_is_counted_in_the_frames_that_should_show_it(self):
        findings = [PhotoFinding(photo_id=0, view="tire_wheel"),
                    PhotoFinding(photo_id=1, view="tire_wheel"),
                    PhotoFinding(photo_id=2, view="engine_bay"),
                    PhotoFinding(photo_id=3, view="tire_wheel", error="lost")]
        counts = passes.component_frame_counts(findings)
        self.assertEqual(counts["drive_tires"], 2)
        self.assertEqual(counts["engine_bay"], 1)
        self.assertNotIn("bunk_sleeper", counts)


if __name__ == "__main__":
    unittest.main()
