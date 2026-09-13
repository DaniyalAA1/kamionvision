"""Pass A, read more than once - and the badge crop that now stands beside it.

Every close-up photo has been read `CLOSEUP_SAMPLES` times since the fan-out
landed, and `Issue.confidence` became a measured agreement rate rather than a
self-report. The call that decides what the truck *is* was still one draw from
an unmeasured distribution, and it is the most load-bearing call in the run:
`make` picks the brand column in the price model, `model` picks the anchor row,
`body_type` can stop the pricing stage and `same_vehicle` can stop the
valuation outright.

What these cover, in the order the run performs them:

  * the majority vote over samples, and the one field that is NOT symmetric -
    `same_vehicle` needs a quorum to go false, because one dissenting read
    must not be able to halt an appraisal
  * `identity_agreement` as a measured quantity, sitting next to `confidence`
    which stays the model's self-report
  * pass A's own photo selection, which prefers the views identity actually
    lives in, and the index mapping that selection must not break
  * the badge read: a second witness, recorded and never allowed to adjudicate

Everything here is offline. No API keys, no network, and the only images are
solid-colour JPEGs written into a temporary directory.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from app import modelspec
from app.config import (BADGE_CROP_BOTTOM, BADGE_CROP_TOP,
                        IDENTITY_IMAGE_LONG_EDGE, IDENTITY_PHOTOS,
                        IDENTITY_QUORUM, IDENTITY_SAMPLES)
from app.evidence import passes, prompts, sampling
from app.evidence import stage as stage_module
from app.schema import GateReport, PhotoCheck, VehicleRead
from app.vlm.base import VLMError, VLMResponse


# --- the fixtures ----------------------------------------------------------

def _check(photo_id: int, view: str = "exterior_front_34", quality: float = 0.8,
           *, path: str | None = None, box=None) -> PhotoCheck:
    name = f"{photo_id:03d}.jpg"
    check = PhotoCheck(photo_id=photo_id, path=path or f"/tmp/{name}", filename=name,
                       width=1600, height=1200, usable=True)
    check.view, check.view_conf, check.capture_quality = view, 0.8, quality
    check.subject_box = list(box) if box else None
    return check


def _gate(photos: list[PhotoCheck]) -> GateReport:
    return GateReport(photos=photos, usable_photo_ids=[c.photo_id for c in photos])


def _identity_json(**kw) -> str:
    body = {"make": "Ford Trucks", "model": "F-MAX", "body_type": "tractor_unit",
            "cab_type": "high sleeper", "axle_config": "4x2",
            "approx_year_range": "2018-2022", "badges_seen": ["F-MAX"],
            "confidence": 0.9, "same_vehicle": True, "vehicle_mismatch": ""}
    body.update(kw)
    return json.dumps(body)


def _badge_json(**kw) -> str:
    body = {"badge_text": ["FORD TRUCKS", "F-MAX", "500"], "make": "Ford Trucks",
            "model": "F-MAX", "trim_or_power": "500", "legible": True,
            "confidence": 0.9}
    body.update(kw)
    return json.dumps(body)


def _closeup_json(**kw) -> str:
    body = {"shows": "the front of a tractor unit", "legible": True,
            "odometer_km": None, "observations": [], "strengths": ["paint is even"],
            "cannot_tell": [], "confidence": 0.7}
    body.update(kw)
    return json.dumps(body)


def _textonly_json() -> str:
    """One object that both text-only parsers can read.

    `parse_synthesis` and `parse_calibration` each take the keys they know and
    ignore the rest, so one reply serves passes C and D and the router does not
    have to tell those two calls apart.
    """
    return json.dumps({
        "condition_summary": {}, "condition_grade": "good", "coverage_gaps": [],
        "headline": "a tidy tractor unit", "confidence": 0.7, "duplicates": [],
        "revisions": [], "worst_finding": None, "calibration_note": ""})


class _Client:
    """A backend that answers from a script and remembers how it was called."""

    supports_structured_output = False

    def __init__(self, replies=("{}",), name="fake", fail_on=(), fail_always=False):
        self.replies = list(replies)
        self.name = name
        self.fail_on = set(fail_on)        # 1-based call ordinals that raise
        self.fail_always = fail_always
        self.prompts: list[str] = []
        self.images: list[list[Path]] = []
        self.kwargs: list[dict] = []
        self.calls = 0

    def complete(self, prompt, images, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        self.images.append(list(images))
        self.kwargs.append(dict(kw))
        if self.fail_always or self.calls in self.fail_on:
            raise VLMError(f"{self.name} said no")
        return VLMResponse(text=self.replies[min(self.calls - 1, len(self.replies) - 1)],
                           backend=self.name, model=f"{self.name}-1")


class _Router(_Client):
    """One client that answers every pass in a whole-stage run.

    Routed on what the call actually is rather than on a counter, because the
    close-ups run concurrently and arrive in whatever order the pool finishes.
    """

    def __init__(self, identity=None, badge=None, badge_fails=False, **kw):
        super().__init__(**kw)
        self.identity = list(identity or [_identity_json()])
        self.badge = badge or _badge_json()
        self.badge_fails = badge_fails
        self.identity_calls = 0
        self.badge_calls = 0

    def complete(self, prompt, images, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        self.images.append(list(images))
        self.kwargs.append(dict(kw))
        if self.fail_always:
            raise VLMError(f"{self.name} said no")
        if images and images[0].name.startswith("badge_"):
            self.badge_calls += 1
            if self.badge_fails:
                raise VLMError("the badge call fell over")
            return VLMResponse(text=self.badge, backend=self.name, model="r-1")
        if "same_vehicle" in prompt:
            reply = self.identity[min(self.identity_calls, len(self.identity) - 1)]
            self.identity_calls += 1
            return VLMResponse(text=reply, backend=self.name, model="r-1")
        text = _closeup_json() if images else _textonly_json()
        return VLMResponse(text=text, backend=self.name, model="r-1")


def _badge_prompt_bridge():
    """`prompts.badge_prompt` belongs to another agent; bridge it if it is late.

    The contract is fixed - keyword-only `make` and `model`, one string out -
    so a stand-in that honours it exercises exactly the same wiring.
    """
    if hasattr(prompts, "badge_prompt"):
        return unittest.mock.patch.object(prompts, "badge_prompt",
                                          wraps=prompts.badge_prompt)
    return unittest.mock.patch.object(
        prompts, "badge_prompt", create=True,
        new=lambda *, make=None, model=None: 'Return "badge_text" as JSON.')


# --- item 6: the identity pass, sampled ------------------------------------

class IdentityConsensus(unittest.TestCase):
    """k reads of the identity call, combined the way the close-ups already are."""

    def setUp(self):
        self.selected = [_check(0), _check(1, "exterior_side")]

    def _run(self, chain, **kw):
        return sampling.identity_consensus(chain, self.selected, {"year": 2021},
                                           max_tokens=800, **kw)

    def test_the_majority_answer_wins(self):
        client = _Client([_identity_json(), _identity_json(),
                          _identity_json(make="MAN", model="TGX")])
        read = self._run([client], samples=3)
        self.assertEqual(client.calls, 3)
        self.assertEqual(read.vehicle.make, "Ford Trucks")
        self.assertEqual(read.vehicle.model, "F-MAX")
        self.assertEqual(read.vehicle.identity_samples, 3)

    def test_a_minority_answer_is_not_silently_dropped(self):
        # Changing your mind is allowed; doing it where nobody can see is not.
        client = _Client([_identity_json(), _identity_json(),
                          _identity_json(make="MAN", model="TGX")])
        read = self._run([client], samples=3)
        kinds = {c.kind for c in read.corrections}
        self.assertIn("identity_disagreement", kinds)
        detail = " ".join(c.detail + c.before for c in read.corrections)
        self.assertIn("MAN", detail)

    def test_every_sample_agreeing_records_nothing(self):
        client = _Client([_identity_json()] * 3)
        read = self._run([client], samples=3)
        self.assertEqual(read.corrections, [])
        self.assertEqual(read.vehicle.identity_agreement, 1.0)

    def test_identity_agreement_is_measured_and_confidence_is_not(self):
        # The separation is the whole point: `identity_agreement` is a measured
        # quantity about the claim, `confidence` stays the model's self-report.
        client = _Client([_identity_json(confidence=0.95),
                          _identity_json(confidence=0.95),
                          _identity_json(make="MAN", confidence=0.95)])
        read = self._run([client], samples=3)
        self.assertAlmostEqual(read.vehicle.identity_agreement, 2 / 3, places=2)
        self.assertAlmostEqual(read.vehicle.confidence, 0.95, places=2)

    def test_model_canonical_is_folded_through_the_spec_card(self):
        client = _Client([_identity_json(model="F Max 500")] * 3)
        read = self._run([client], samples=3)
        self.assertEqual(read.vehicle.model, "F Max 500")
        self.assertEqual(read.vehicle.model_canonical,
                         modelspec.normalise_model("Ford Trucks", "F Max 500"))

    def test_a_read_that_named_nothing_lowers_the_agreement_without_winning(self):
        # Two samples could not read a make and one could. Deleting the only
        # read that named the truck would cost the brand column outright; the
        # honest outcome is to keep it and measure the agreement low, which is
        # what `identity.collect` then treats as a weak witness.
        client = _Client([_identity_json(make=None, model=None),
                          _identity_json(make=None, model=None),
                          _identity_json()])
        read = self._run([client], samples=3)
        self.assertEqual(read.vehicle.make, "Ford Trucks")
        self.assertAlmostEqual(read.vehicle.identity_agreement, 1 / 3, places=2)


class SameVehicleIsAsymmetric(unittest.TestCase):
    """False stops the valuation, so it takes more than one read to get there."""

    def setUp(self):
        self.selected = [_check(0), _check(1, "exterior_side")]

    def _run(self, replies, **kw):
        client = _Client(replies)
        return client, sampling.identity_consensus(
            [client], self.selected, None, max_tokens=800, samples=3, **kw)

    def test_one_dissenting_sample_cannot_halt_an_appraisal(self):
        _, read = self._run([_identity_json(), _identity_json(),
                             _identity_json(same_vehicle=False,
                                            vehicle_mismatch="plate differs")])
        self.assertTrue(read.same_vehicle)

    def test_and_it_says_so_rather_than_swallowing_the_doubt(self):
        _, read = self._run([_identity_json(), _identity_json(),
                             _identity_json(same_vehicle=False,
                                            vehicle_mismatch="plate differs")])
        kinds = {c.kind for c in read.corrections}
        self.assertIn("same_vehicle_disagreement", kinds)
        self.assertIn("plate differs", read.vehicle_mismatch)

    def test_a_quorum_of_samples_can_halt_it(self):
        _, read = self._run([_identity_json(same_vehicle=False,
                                            vehicle_mismatch="two plates"),
                             _identity_json(same_vehicle=False,
                                            vehicle_mismatch="two plates"),
                             _identity_json()])
        self.assertFalse(read.same_vehicle)
        self.assertIn("two plates", read.vehicle_mismatch)

    def test_the_quorum_falls_back_to_one_when_only_one_sample_survived(self):
        # With nothing to corroborate against, the single read stands - which
        # is exactly what the unsampled pass did. Reporting a provider failure
        # as evidence about the truck would be the worse error.
        client = _Client([_identity_json(same_vehicle=False,
                                         vehicle_mismatch="two plates")],
                         fail_on=(2, 3))
        read = sampling.identity_consensus([client], self.selected, None,
                                           max_tokens=800, samples=3)
        self.assertEqual(read.vehicle.identity_samples, 1)
        self.assertFalse(read.same_vehicle)


class IdentityFailurePosture(unittest.TestCase):
    """A failed sample costs one sample. Every sample failing costs the run."""

    def setUp(self):
        self.selected = [_check(0)]

    def test_one_failed_sample_costs_one_sample(self):
        client = _Client([_identity_json()], fail_on=(2,))
        read = sampling.identity_consensus([client], self.selected, None,
                                           max_tokens=800, samples=3)
        self.assertEqual(client.calls, 3)
        self.assertEqual(read.vehicle.identity_samples, 2)
        self.assertEqual(read.vehicle.make, "Ford Trucks")
        self.assertTrue(any("said no" in w for w in read.warnings))

    def test_every_sample_failing_raises(self):
        client = _Client(fail_always=True)
        with self.assertRaises(VLMError):
            sampling.identity_consensus([client], self.selected, None,
                                        max_tokens=800, samples=3)

    def test_all_samples_go_to_the_same_backend(self):
        # Mixing backends would confound sample-to-sample disagreement with a
        # difference between two models, which is the measurement itself.
        first = _Client([_identity_json()] * 3, name="alpha")
        second = _Client([_identity_json()] * 3, name="beta")
        read = sampling.identity_consensus([first, second], self.selected, None,
                                           max_tokens=800, samples=3)
        self.assertEqual(first.calls, 3)
        self.assertEqual(second.calls, 0)
        self.assertEqual(read.backend, "alpha")
        self.assertEqual(read.fallbacks, [])

    def test_a_backend_that_dies_mid_sample_is_fallen_back_from_out_loud(self):
        first = _Client([_identity_json()] * 3, name="alpha", fail_on=(2,))
        second = _Client([_identity_json()] * 3, name="beta")
        read = sampling.identity_consensus([first, second], self.selected, None,
                                           max_tokens=800, samples=3)
        self.assertEqual(read.vehicle.identity_samples, 3)
        self.assertEqual(second.calls, 1)
        self.assertTrue(any("beta" in f for f in read.fallbacks))

    def test_every_sample_is_recorded_as_its_own_call(self):
        client = _Client([_identity_json()] * 3)
        read = sampling.identity_consensus([client], self.selected, None,
                                           max_tokens=800, samples=3)
        self.assertEqual(len(read.calls), 3)
        self.assertTrue(all(str(name).startswith("identity") for name, _ in read.calls))


# --- item 7: pass A's own photo selection ----------------------------------

class IdentityPhotoSelection(unittest.TestCase):
    """Identity lives in the front three-quarter, the side profile and the badge."""

    def test_whole_vehicle_frames_come_first(self):
        photos = [_check(i, "tire_wheel", quality=0.95) for i in range(12)]
        photos.append(_check(50, "exterior_front_34", quality=0.3))
        photos.append(_check(51, "exterior_side", quality=0.2))
        picked = stage_module.select_identity_photos(_gate(photos), limit=4)
        self.assertEqual([c.photo_id for c in picked[:2]], [50, 51])

    def test_a_side_profile_is_not_crowded_out_by_ten_front_threequarters(self):
        # The only view that can honestly settle 4x2 against 6x2.
        photos = [_check(i, "exterior_front_34", quality=0.9) for i in range(10)]
        photos.append(_check(99, "exterior_side", quality=0.4))
        picked = stage_module.select_identity_photos(_gate(photos), limit=4)
        self.assertIn(99, [c.photo_id for c in picked])

    def test_best_capture_quality_first_within_a_view(self):
        photos = [_check(0, "exterior_front_34", quality=0.2),
                  _check(1, "exterior_front_34", quality=0.9)]
        picked = stage_module.select_identity_photos(_gate(photos), limit=2)
        self.assertEqual([c.photo_id for c in picked], [1, 0])

    def test_a_set_with_no_whole_vehicle_frame_falls_back_rather_than_sending_nothing(self):
        photos = [_check(i, v) for i, v in enumerate(
            ["tire_wheel", "dashboard_odometer", "interior_cab", "engine_bay"])]
        gate = _gate(photos)
        self.assertEqual([c.photo_id for c in stage_module.select_identity_photos(gate, limit=4)],
                         [c.photo_id for c in stage_module.select_photos(gate, limit=4)])

    def test_unusable_frames_are_never_sent(self):
        good, bad = _check(0), _check(1)
        bad.usable = False
        picked = stage_module.select_identity_photos(_gate([good, bad]))
        self.assertEqual([c.photo_id for c in picked], [0])

    def test_it_caps_at_the_configured_count(self):
        photos = [_check(i, "exterior_front_34") for i in range(IDENTITY_PHOTOS + 5)]
        self.assertEqual(len(stage_module.select_identity_photos(_gate(photos))),
                         IDENTITY_PHOTOS)

    def test_the_remaining_slots_still_carry_the_rest_of_the_set(self):
        # `same_vehicle` is the reason: a padded listing whose odd frame is an
        # interior shot of a tidier truck has to be able to reach pass A.
        photos = [_check(0, "exterior_front_34"), _check(1, "interior_cab"),
                  _check(2, "dashboard_odometer")]
        picked = stage_module.select_identity_photos(_gate(photos), limit=8)
        self.assertEqual({c.photo_id for c in picked}, {0, 1, 2})

    def test_an_empty_gate_selects_nothing(self):
        self.assertEqual(stage_module.select_identity_photos(_gate([])), [])


class IdentityIndexMapping(unittest.TestCase):
    """The pass-A photo index is load-bearing, and pass A no longer sees pass B's set."""

    def test_the_context_describes_the_photos_that_are_actually_sent(self):
        photos = [_check(0, "tire_wheel"), _check(7, "exterior_front_34"),
                  _check(9, "exterior_side")]
        picked = stage_module.select_identity_photos(_gate(photos), limit=2)
        context = passes.identity_context(None, picked)
        self.assertIn("looking at 2 photos", context)
        self.assertIn("photo_id 0 to 1", context)
        self.assertIn("photo_id 0: exterior front 34", context)
        self.assertIn("photo_id 1: exterior side", context)
        self.assertNotIn("photo_id 2", context)
        self.assertNotIn("tire wheel", context)

    def test_a_mismatch_naming_a_photo_names_the_frame_the_reader_can_open(self):
        # The index the model answers with is its position in the images it was
        # sent - that is what the backends label them. `pipeline.pricing_blocker`
        # puts this string in front of a seller, so it has to point at the
        # photograph they actually uploaded.
        text = _identity_json(same_vehicle=False,
                              vehicle_mismatch="photo 1 shows a different plate")
        _, _, mismatch = passes.parse_identity(text, photo_ids=[7, 3, 9])
        self.assertIn("photo 3", mismatch)
        self.assertNotIn("photo 1", mismatch)

    def test_an_index_nobody_sent_is_left_exactly_as_it_came_back(self):
        text = _identity_json(vehicle_mismatch="photo 8 is blurred")
        _, _, mismatch = passes.parse_identity(text, photo_ids=[7, 3])
        self.assertEqual(mismatch, "photo 8 is blurred")

    def test_without_a_mapping_the_text_is_untouched(self):
        text = _identity_json(vehicle_mismatch="photo 1 shows a different plate")
        _, _, mismatch = passes.parse_identity(text)
        self.assertEqual(mismatch, "photo 1 shows a different plate")

    def test_the_identity_call_asks_for_the_higher_resolution(self):
        # A model badge is small in frame and 1024 px across a whole tractor
        # leaves "F-MAX 500" a few pixels tall. This pins the ASK; whether the
        # provider gets 1536 px depends on the backend taking the keyword.
        client = _Client([_identity_json()])
        passes.identity(client, [_check(0)], None, max_tokens=100)
        self.assertEqual(client.kwargs[0].get("long_edge"), IDENTITY_IMAGE_LONG_EDGE)

    def test_a_backend_that_cannot_take_it_is_not_broken_by_the_ask(self):
        # The three shipped backends encode at EVIDENCE_IMAGE_LONG_EDGE and
        # have no per-call override, so the keyword is dropped rather than
        # raising a TypeError halfway through a live demo. The day
        # `vlm.base.complete` grows `long_edge`, this pass starts sending 1536
        # with no change here.
        class Strict:
            supports_structured_output = False
            name = "strict"

            def __init__(self):
                self.seen = None

            def complete(self, prompt, images, *, system="", max_tokens=100,
                         json_schema=None, effort=None):
                self.seen = (system, max_tokens, effort)
                return VLMResponse(text=_identity_json(), backend="strict", model="s")

        client = Strict()
        vehicle, _, _ = passes.parse_identity(
            passes.identity(client, [_check(0)], None, max_tokens=100).text)
        self.assertEqual(vehicle.make, "Ford Trucks")
        self.assertIsNotNone(client.seen)

    def test_the_run_sends_pass_a_its_own_set_and_numbers_it_from_zero(self):
        photos = [_check(i, "tire_wheel") for i in range(6)]
        photos.append(_check(20, "exterior_front_34"))
        router = _Router()
        with unittest.mock.patch.object(stage_module.vlm, "resolve_chain",
                                        return_value=[router]):
            stage_module.run(_gate(photos), None, limit=4)
        identity_prompts = [p for p in router.prompts if "same_vehicle" in p]
        self.assertTrue(identity_prompts)
        sent = [i for i, p in enumerate(router.prompts) if "same_vehicle" in p][0]
        self.assertIn("photo_id 0: exterior front 34", identity_prompts[0])
        self.assertGreaterEqual(len(router.images[sent]), 1)


# --- item 1: what passes B, C and D know about the truck -------------------

class VehicleLineReadsTheSpecCard(unittest.TestCase):

    def _vehicle(self, **kw):
        base = dict(make="Ford Trucks", model="F-MAX", cab_type="high sleeper",
                    axle_config="4x2", approx_year_range="2018-2022")
        base.update(kw)
        return VehicleRead(**base)

    def test_it_is_byte_identical_when_the_card_has_nothing(self):
        with unittest.mock.patch.object(modelspec, "spec_lines", return_value=[]):
            line = passes.vehicle_line(self._vehicle())
        self.assertEqual(
            line,
            "The vehicle has been identified from the full set as a Ford Trucks F-MAX "
            "(high sleeper, 4x2, 2018-2022).")

    def test_an_unreadable_truck_still_says_so_and_nothing_else(self):
        with unittest.mock.patch.object(modelspec, "spec_lines", return_value=[]):
            line = passes.vehicle_line(VehicleRead())
        self.assertEqual(line, "The make and model could not be read from the photos.")

    def test_the_card_is_appended_behind_the_sentence_that_was_always_there(self):
        with unittest.mock.patch.object(modelspec, "spec_lines",
                                        return_value=["Cab: high sleeper.",
                                                      "Driveline: 12.7 litre."]):
            line = passes.vehicle_line(self._vehicle())
        self.assertTrue(line.startswith("The vehicle has been identified"))
        self.assertIn("Cab: high sleeper.", line)
        self.assertIn("Driveline: 12.7 litre.", line)

    def test_one_verbose_card_row_cannot_balloon_every_close_up_call(self):
        # This line is read on ~48 close-up calls plus the synthesis and the
        # calibration. A card row is hand-written prose and the shipped F-MAX
        # row's `cab` and `driveline` fields are each a paragraph - right for
        # the identity prompt, which reads them once, and not for this.
        with unittest.mock.patch.object(modelspec, "spec_lines",
                                        return_value=["Driveline: " + "x " * 900]):
            line = passes.vehicle_line(self._vehicle())
        body = line.splitlines()[1]
        self.assertLessEqual(len(body), passes.SPEC_LINE_CHARS)
        self.assertTrue(body.endswith("..."))

    def test_the_shipped_card_fits_the_budget(self):
        # Not a mock: the real card, through the real lookup. A row that grows
        # past the budget is clipped rather than silently paid for fifty times.
        line = passes.vehicle_line(self._vehicle(generation="f-max-2018"))
        for body in line.splitlines()[1:]:
            self.assertLessEqual(len(body), passes.SPEC_LINE_CHARS)

    def test_the_generation_travels_with_the_lookup(self):
        seen = {}

        def record(make, model, *, generation=None):
            seen.update(make=make, model=model, generation=generation)
            return []

        with unittest.mock.patch.object(modelspec, "spec_lines", side_effect=record):
            passes.vehicle_line(self._vehicle(generation="f-max-2018"))
        self.assertEqual(seen["generation"], "f-max-2018")


class WeakPointsReachTheRightPhoto(unittest.TestCase):
    """A close-up of a tire has no business being told about the AdBlue tank."""

    def _vehicle(self):
        return VehicleRead(make="Ford Trucks", model="F-MAX")

    def test_the_filter_is_the_mapping_the_prompt_already_uses(self):
        # One view-to-component mapping, not two that have to be kept in step.
        seen = {}

        def record(make, model, components=None):
            seen["components"] = list(components or [])
            return []

        with unittest.mock.patch.object(modelspec, "weak_points", side_effect=record):
            passes.weak_points_for_view(self._vehicle(), "tire_wheel")
        self.assertEqual(seen["components"], list(prompts.view_components("tire_wheel")))
        self.assertIn("drive_tires", seen["components"])
        self.assertNotIn("engine_bay", seen["components"])

    def test_an_unknown_view_asks_for_nothing_rather_than_everything(self):
        # `modelspec.weak_points` reads an empty filter as no filter at all, so
        # an unrecognised view must not reach it: it would come back with the
        # whole card, including the parts this frame cannot possibly show.
        with unittest.mock.patch.object(modelspec, "weak_points") as card:
            self.assertEqual(passes.weak_points_for_view(self._vehicle(), "not_a_view"), [])
        card.assert_not_called()

    def test_no_vehicle_means_no_lookup(self):
        self.assertEqual(passes.weak_points_for_view(None, "tire_wheel"), [])

    def test_the_closeup_call_hands_them_to_the_prompt(self):
        seen = {}

        def fake_prompt(*, view, view_pretty, vehicle, cropped, soft,
                        expectation="", band=None, spec_lines=(), weak_points=()):
            seen.update(view=view, weak_points=list(weak_points))
            return "PROMPT"

        client = _Client([_closeup_json()])
        with unittest.mock.patch.object(prompts, "closeup_prompt", fake_prompt):
            passes.closeup(client, _check(0, "tire_wheel"), "A Ford F-MAX.",
                           max_tokens=100,
                           weak_points=[("drive_tires", "the original fitment cups early")])
        self.assertEqual(seen["weak_points"],
                         [("drive_tires", "the original fitment cups early")])

    def test_a_prompt_that_does_not_take_them_yet_is_still_callable(self):
        # The prompts module is another agent's file. Threading a keyword it
        # has not grown yet must not take the appraisal down with it.
        def old_prompt(*, view, view_pretty, vehicle, cropped, soft,
                       expectation="", band=None):
            return "PROMPT"

        client = _Client([_closeup_json()])
        with unittest.mock.patch.object(prompts, "closeup_prompt", old_prompt):
            finding = passes.closeup(client, _check(0, "tire_wheel"), "A Ford F-MAX.",
                                     max_tokens=100, weak_points=[("drive_tires", "x")])
        self.assertEqual(finding.photo_id, 0)


# --- item 4: the badge read ------------------------------------------------

class BadgeCrop(unittest.TestCase):

    def setUp(self):
        from PIL import Image

        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "front.jpg"
        Image.new("RGB", (1600, 1200), (90, 30, 30)).save(self.path)
        self.addCleanup(self.tmp.cleanup)

    def _check(self):
        return _check(4, "exterior_front_34", path=str(self.path),
                      box=[200, 100, 1000, 900])

    def test_the_band_is_cut_from_the_subject_box_not_the_frame(self):
        rect = passes.badge_crop_rect(self._check())
        self.assertIsNotNone(rect)
        x1, y1, x2, y2 = rect
        box_top, box_height = 100, 800
        self.assertLessEqual(y1, box_top + BADGE_CROP_TOP * box_height)
        self.assertGreaterEqual(y2, box_top + BADGE_CROP_BOTTOM * box_height)
        # And it stops well short of the wheels, which is the entire point.
        self.assertLess(y2 - y1, box_height)
        self.assertLess(x1, 200)
        self.assertGreater(x2, 1000)

    def test_a_frame_with_no_subject_box_has_no_badge_band(self):
        self.assertIsNone(passes.badge_crop_rect(_check(0)))

    def test_it_writes_a_file_named_for_the_photo_it_came_from(self):
        out = passes.write_badge_crop(self._check(), Path(self.tmp.name))
        self.assertIsNotNone(out)
        self.assertEqual(out.name, "badge_4.jpg")
        self.assertTrue(out.is_file())

    def test_the_best_badge_frame_is_a_whole_vehicle_one_with_a_subject(self):
        photos = [_check(0, "tire_wheel", quality=0.99, box=[0, 0, 1600, 1200]),
                  _check(1, "exterior_rear", quality=0.9, box=[200, 100, 1000, 900]),
                  _check(2, "exterior_front_34", quality=0.5, box=[200, 100, 1000, 900])]
        best = passes.best_badge_frame(_gate(photos))
        self.assertEqual(best.photo_id, 2)

    def test_a_set_with_no_subject_box_has_no_badge_frame(self):
        self.assertIsNone(passes.best_badge_frame(_gate([_check(0)])))


class BadgeReadInTheRun(unittest.TestCase):
    """A second witness, recorded faithfully and never allowed to adjudicate."""

    def setUp(self):
        from PIL import Image

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.photos = []
        for i, view in enumerate(["exterior_front_34", "exterior_side", "tire_wheel"]):
            path = Path(self.tmp.name) / f"{i:03d}.jpg"
            Image.new("RGB", (1600, 1200), (40 + i, 40, 40)).save(path)
            self.photos.append(_check(i, view, path=str(path),
                                      box=[200, 100, 1000, 900]))

    def _run(self, router):
        with _badge_prompt_bridge(), \
             unittest.mock.patch.object(stage_module.vlm, "resolve_chain",
                                        return_value=[router]):
            return stage_module.run(_gate(self.photos), None, limit=3)

    def test_the_badge_text_and_its_photo_are_recorded(self):
        router = _Router()
        report = self._run(router)
        self.assertEqual(router.badge_calls, 1)
        self.assertEqual(report.vehicle.badge_text, ["FORD TRUCKS", "F-MAX", "500"])
        self.assertEqual(report.vehicle.badge_photo_id, 0)

    def test_it_never_overwrites_the_make_or_the_model(self):
        # The badge is a witness. `app/identity.py` is the only thing that
        # adjudicates between witnesses, and it needs both readings intact.
        router = _Router(badge=_badge_json(make="MAN", model="TGX"))
        report = self._run(router)
        self.assertEqual(report.vehicle.make, "Ford Trucks")
        self.assertEqual(report.vehicle.model, "F-MAX")
        self.assertEqual(report.vehicle.badge_make, "MAN")
        self.assertEqual(report.vehicle.badge_model, "TGX")

    def test_a_failed_badge_call_costs_the_badge_and_not_the_appraisal(self):
        router = _Router(badge_fails=True)
        report = self._run(router)
        self.assertEqual(report.vehicle.make, "Ford Trucks")
        self.assertEqual(report.vehicle.badge_text, [])
        self.assertTrue(any("badge" in w.lower() for w in report.parse_warnings))
        self.assertTrue(any("badge" in str(name) for name, _ in report.calls))

    def test_the_call_is_in_the_trace_like_every_other_pass(self):
        report = self._run(_Router())
        names = [str(name) for name, _ in report.calls]
        self.assertIn("badge", names)
        self.assertEqual(len([n for n in names if n.startswith("identity")]),
                         IDENTITY_SAMPLES)

    def test_turning_it_off_removes_the_call_and_nothing_else(self):
        router = _Router()
        with unittest.mock.patch.object(stage_module, "BADGE_READ", False):
            with _badge_prompt_bridge(), \
                 unittest.mock.patch.object(stage_module.vlm, "resolve_chain",
                                            return_value=[router]):
                report = stage_module.run(_gate(self.photos), None, limit=3)
        self.assertEqual(router.badge_calls, 0)
        self.assertEqual(report.vehicle.make, "Ford Trucks")


class TheRunStillHoldsTogether(unittest.TestCase):
    """The invariants the rest of the pipeline rests on, through a sampled pass A."""

    def setUp(self):
        self.photos = [_check(i, v) for i, v in enumerate(
            ["exterior_front_34", "tire_wheel", "dashboard_odometer"])]

    def _run(self, router, **kw):
        with unittest.mock.patch.object(stage_module.vlm, "resolve_chain",
                                        return_value=[router]):
            return stage_module.run(_gate(self.photos), None, limit=3, **kw)

    def test_on_photo_still_fires_once_per_photo(self):
        seen = []
        report = self._run(_Router(), on_photo=seen.append)
        self.assertEqual(len(seen), report.photos_read + report.photos_failed)
        self.assertEqual(sorted(f.photo_id for f in seen), [0, 1, 2])

    def test_the_identity_pass_is_sampled_and_the_report_says_how_often(self):
        router = _Router(identity=[_identity_json()] * IDENTITY_SAMPLES)
        report = self._run(router)
        self.assertEqual(router.identity_calls, IDENTITY_SAMPLES)
        self.assertEqual(report.vehicle.identity_samples, IDENTITY_SAMPLES)

    def test_a_split_pass_a_reaches_the_report_as_a_correction(self):
        router = _Router(identity=[_identity_json(), _identity_json(),
                                   _identity_json(make="MAN")])
        report = self._run(router)
        self.assertIn("identity_disagreement", {c.kind for c in report.corrections})

    def test_the_quorum_is_what_the_config_says(self):
        self.assertGreaterEqual(IDENTITY_SAMPLES, 2)
        self.assertGreaterEqual(IDENTITY_QUORUM, 2)
        self.assertLessEqual(IDENTITY_QUORUM, IDENTITY_SAMPLES)


if __name__ == "__main__":
    unittest.main()
