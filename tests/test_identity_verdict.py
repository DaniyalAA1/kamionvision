"""The identity verdict: which witness wins, and what the band pays for it.

The cases that matter here are the ones where a witness should NOT get a vote.
A head that has never seen a Scania still names a class at high confidence; a
WMI table is silent about manufacturers nobody verified. Both of those have to
read as "unknown" and neither may dispute a badge the model can plainly read -
the same rule `reconcile` already enforces for the head, applied to all four.
"""
import unittest

from app import identity
from app.config import (IDENTITY_DISPUTED_WIDENING, IDENTITY_UNKNOWN_WIDENING,
                        WMI_CONFLICT_WIDENING)
from app.schema import PerceptionReport, VehicleRead


def vehicle(**kw) -> VehicleRead:
    return VehicleRead(**kw)


def head(brand, conf) -> PerceptionReport:
    report = PerceptionReport()
    report.brand, report.brand_conf = brand, conf
    return report


class Witnesses(unittest.TestCase):
    def test_a_silent_set_produces_no_witnesses(self):
        self.assertEqual(identity.collect(vehicle()), [])

    def test_the_identity_pass_prefers_the_measured_agreement_rate(self):
        # `confidence` is the model's opinion of itself; `identity_agreement` is
        # a measured rate across samples. The measurement wins where it exists.
        got = identity.collect(vehicle(make="FORD", confidence=0.99,
                                       identity_agreement=0.67))
        self.assertEqual(got, [["identity_pass", "FORD", 0.67]])

    def test_it_falls_back_to_the_self_report_when_nothing_was_sampled(self):
        got = identity.collect(vehicle(make="FORD", confidence=0.82))
        self.assertEqual(got, [["identity_pass", "FORD", 0.82]])

    def test_a_badge_transcription_names_a_brand_without_a_parsed_make(self):
        got = identity.collect(vehicle(badge_text=["F-MAX", "FORD TRUCKS"]))
        self.assertEqual(got, [["badge", "FORD", 0.9]])

    def test_the_head_may_not_dispute_a_brand_it_was_never_trained_on(self):
        # The corpus has no Scania. The head names one anyway, confidently.
        got = identity.collect(vehicle(make="SCANIA"), head("FORD", 0.91),
                               head_classes=["FORD", "MAN"])
        self.assertEqual([w[0] for w in got], ["identity_pass"])

    def test_the_head_votes_about_brands_it_does_know(self):
        got = identity.collect(vehicle(make="FORD"), head("FORD", 0.91),
                               head_classes=["FORD", "MAN"])
        self.assertEqual([w[0] for w in got], ["identity_pass", "head"])

    def test_an_unconfident_head_does_not_vote(self):
        got = identity.collect(vehicle(make="FORD"), head("MAN", 0.41),
                               head_classes=["FORD", "MAN"])
        self.assertEqual([w[0] for w in got], ["identity_pass"])


class Verdict(unittest.TestCase):
    def test_nothing_legible_is_the_reask_path(self):
        got = identity.decide(vehicle())
        self.assertEqual(got.status, "unknown")
        self.assertEqual(got.widening, IDENTITY_UNKNOWN_WIDENING)
        self.assertTrue(got.reask)
        self.assertIn("assumption", got.reason)

    def test_two_agreeing_reads_are_confirmed_and_cost_nothing(self):
        got = identity.decide(vehicle(make="FORD", identity_agreement=1.0,
                                      wmi_brand="FORD"))
        self.assertEqual(got.status, "confirmed")
        self.assertEqual(got.make, "FORD")
        self.assertEqual(got.widening, 1.0)
        self.assertEqual(got.agreed, ["identity_pass", "wmi"])

    def test_one_read_alone_is_probable_and_still_costs_nothing(self):
        # A single uncontradicted witness is not a reason to widen - it is a
        # reason to ask for a second one, which is what `reask` carries.
        got = identity.decide(vehicle(make="FORD", identity_agreement=1.0))
        self.assertEqual(got.status, "probable")
        self.assertEqual(got.widening, 1.0)
        self.assertTrue(got.reask)

    def test_a_plate_disagreeing_with_a_badge_widens_and_says_so(self):
        got = identity.decide(vehicle(make="MERCEDES-BENZ", identity_agreement=1.0,
                                      wmi_brand="FORD"))
        self.assertEqual(got.status, "disputed")
        self.assertEqual(got.widening, WMI_CONFLICT_WIDENING)
        self.assertIn("disagree", got.reason)
        self.assertIn("assumption", got.reason)

    def test_a_dispute_not_involving_the_plate_uses_the_other_multiplier(self):
        got = identity.decide(vehicle(make="FORD", identity_agreement=1.0,
                                      badge_text=["MAN", "TGX"]),
                              head("MAN", 0.95), head_classes=["FORD", "MAN"])
        self.assertEqual(got.status, "disputed")
        self.assertEqual(got.widening, IDENTITY_DISPUTED_WIDENING)

    def test_the_winner_is_weighted_not_counted(self):
        # One VIN outranks one badge read: the plate is stamped into the chassis
        # and arithmetically checked, the badge is bodywork.
        got = identity.decide(vehicle(make="MAN", identity_agreement=0.7,
                                      wmi_brand="FORD"))
        self.assertEqual(got.make, "FORD")

    def test_the_canonical_model_travels_with_the_verdict(self):
        got = identity.decide(vehicle(make="FORD", model="F Max",
                                      model_canonical="F-MAX",
                                      identity_agreement=1.0))
        self.assertEqual(got.model, "F-MAX")


class OneWideningPerFact(unittest.TestCase):
    """The compounding bug this module exists to prevent."""

    def test_superseded_per_rule_widenings_are_dropped(self):
        existing = [
            ["identity disputed between the badge read and the head", 1.25],
            ["the chassis-plate WMI says FORD", 1.20],
            ["two views are missing, so the band widens 1.12x", 1.12],
        ]
        got = identity.decide(vehicle(make="MERCEDES-BENZ", identity_agreement=1.0,
                                      wmi_brand="FORD"))
        merged = identity.merge_widening(got, existing)
        # The unrelated coverage widening survives; both identity ones are gone
        # and exactly one replaces them.
        self.assertEqual(len(merged), 2)
        self.assertIn("two views are missing", merged[0][0])
        self.assertEqual(merged[1][1], WMI_CONFLICT_WIDENING)

    def test_a_confirmed_verdict_adds_no_multiplier(self):
        got = identity.decide(vehicle(make="FORD", identity_agreement=1.0,
                                      wmi_brand="FORD"))
        merged = identity.merge_widening(got, [["unrelated", 1.12]])
        self.assertEqual(merged, [["unrelated", 1.12]])

    def test_merging_into_an_empty_list_is_safe(self):
        got = identity.decide(vehicle())
        self.assertEqual(len(identity.merge_widening(got, [])), 1)
        self.assertEqual(len(identity.merge_widening(got, None)), 1)


if __name__ == "__main__":
    unittest.main()
