"""Tests for pricing enhancements: salvage value floor and asking verdict bands."""
from __future__ import annotations

import unittest

from app.config import SALVAGE_VALUE_TRY, SALVAGE_VALUE_USD
from app.pricing import load_model
from app.pricing.model import estimate, judge_asking_price
from app.schema import AskingVerdict, PriceEstimate


class TestPricingEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_salvage_value_floor_tr(self):
        """Extreme age and mileage in TR market should not drop below salvage value."""
        # A 1970 truck with 4,000,000 km
        est = estimate(self.model, year=1970, km=4_000_000, make="Ford", market="TR")
        self.assertGreaterEqual(est.point, SALVAGE_VALUE_TRY)
        self.assertGreaterEqual(est.baseline_point, SALVAGE_VALUE_TRY)
        self.assertGreaterEqual(est.low, round(SALVAGE_VALUE_TRY * 0.85, -3))

    def test_salvage_value_floor_usd(self):
        """Extreme age and mileage in international market should not drop below salvage USD."""
        est = estimate(self.model, year=1970, km=4_000_000, make="Ford", market="DE")
        self.assertGreaterEqual(est.point, SALVAGE_VALUE_USD)
        self.assertGreaterEqual(est.baseline_point, SALVAGE_VALUE_USD)
        self.assertGreaterEqual(est.low, round(SALVAGE_VALUE_USD * 0.85, -3))

    def test_judge_asking_price_both_bands_inside(self):
        """Asking price inside both baseline market band and condition-adjusted band."""
        est = PriceEstimate(
            currency="TRY",
            point=1_500_000,
            low=1_350_000,
            high=1_650_000,
            baseline_point=1_500_000,
            baseline_low=1_350_000,
            baseline_high=1_650_000,
        )
        verdict = judge_asking_price(1_500_000, est)
        self.assertTrue(verdict.inside_comparable_band)
        self.assertTrue(verdict.inside_estimate_band)
        self.assertEqual(verdict.label, "in line with the market")
        self.assertNotIn("Exceeds the condition-adjusted valuation", verdict.summary)

    def test_judge_asking_price_in_market_band_but_exceeds_condition(self):
        """Asking price inside market comparable band, but truck has severe wear so condition band is lower."""
        # Market for this year/km is 1.4M - 1.8M, but severe wear adjusted this specific truck to 1.1M - 1.35M
        est = PriceEstimate(
            currency="TRY",
            point=1_200_000,
            low=1_100_000,
            high=1_350_000,
            baseline_point=1_600_000,
            baseline_low=1_400_000,
            baseline_high=1_800_000,
        )
        # Seller asks 1,500,000 (inside baseline 1.4M-1.8M, but above 1.35M high condition estimate)
        verdict = judge_asking_price(1_500_000, est)
        self.assertTrue(verdict.inside_comparable_band)
        self.assertFalse(verdict.inside_estimate_band)
        self.assertEqual(verdict.label, "in line with the market")
        self.assertIn("Exceeds the condition-adjusted valuation for this vehicle's specific observed wear", verdict.summary)

    def test_judge_asking_price_above_market(self):
        """Asking price well above comparable band."""
        est = PriceEstimate(
            currency="TRY",
            point=1_500_000,
            low=1_350_000,
            high=1_650_000,
            baseline_point=1_500_000,
            baseline_low=1_350_000,
            baseline_high=1_650_000,
        )
        verdict = judge_asking_price(2_200_000, est)
        self.assertFalse(verdict.inside_comparable_band)
        self.assertFalse(verdict.inside_estimate_band)
        self.assertEqual(verdict.label, "above the market")


if __name__ == "__main__":
    unittest.main()
