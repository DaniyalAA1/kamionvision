"""Shared budget arithmetic, imported by every suite.

Lives apart from `suites/__init__.py` so a suite can import it without the
package importing the suite back.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app import config


# Smoke reads each photo ONCE. Its job is "does the machinery work", not "what
# is the variance", and at CLOSEUP_SAMPLES=3 a faithful smoke tier costs 1,083
# calls - which is a decision, not a smoke test. The cost is that a one-sample
# run cannot apply the >=2/3 quorum rule, so its suites say so in their caveats
# and `--diff` refuses to treat a smoke run and a standard run as comparable.
SAMPLES_BY_TIER = {"smoke": 1}


def samples_for(tier: str) -> int:
    return int(SAMPLES_BY_TIER.get(tier, config.CLOSEUP_SAMPLES))


def calls_per_vehicle(photos: int | None = None, samples: int | None = None) -> int:
    """What one end-to-end appraisal costs in vision calls, today.

    The harness design's cost table says 18 - one identity call, sixteen
    close-ups and one synthesis. That predates `config.CLOSEUP_SAMPLES`, which
    reads every photo three times, and the set-aware calibration pass. Computed
    from config rather than written down, so the budget cannot quietly go stale
    the way the design's table did.
    """
    photos = config.MAX_EVIDENCE_PHOTOS if photos is None else photos
    samples = config.CLOSEUP_SAMPLES if samples is None else samples
    identity, synthesis, calibration = 1, 1, 1
    return identity + photos * samples + synthesis + calibration


@dataclass
class Budget:
    """A suite's planned cost when its keys cannot be resolved ahead of time.

    `twin_fp` returns a richer object because it forces its own prompts and can
    therefore key every call offline. A suite that has to run the gate, pass A
    and a crop decision first cannot, so it reports a total and declares the
    cache status unknown. Unknown is reported as UNCACHED, which over-estimates
    the bill - the safe direction for a number whose job is to stop a run.
    """
    name: str
    tier: str
    seed: int
    n_calls: int = 0
    units: int = 0
    unit: str = "vehicle"
    keys_resolvable: bool = False
    params: dict = field(default_factory=dict)
    note: str = ""
    calls: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    samples: int = 1

    def to_dict(self) -> dict:
        return {"name": self.name, "tier": self.tier, "seed": self.seed,
                "calls": self.n_calls, "units": self.units, "unit": self.unit,
                "keys_resolvable": self.keys_resolvable, "params": self.params,
                "note": self.note}


class NotImplementedSuite(NotImplementedError):
    """Raised by a stubbed suite. Distinct so the CLI can skip rather than abort."""
