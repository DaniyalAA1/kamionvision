"""What the prompts know about a SPECIFIC truck, as opposed to trucks.

Three things are covered here, and each one exists because of a measured or
structural failure in what came before it:

  * The identity pass answered `model` as free text, so "F Max", "FMAX" and
    "F-MAX 500" were three different answers to one question and only one of
    them joined against the anchor row. The closed list from
    `modelspec.vocabulary` is the fix, and it has to disappear again when the
    card does - a reference file that is absent must leave the pipeline exactly
    as it ran before the file existed.
  * `approx_year_range` was free text nothing could check. `generation` is
    chosen from a closed list of visual markers and `year_evidence` says what
    in the photographs supports the year - and the generation may NEVER be
    derived from the year the seller typed, because the year is the thing the
    generation is used to check. A year-derived generation agrees with the year
    by construction and catches nothing.
  * The badge pass is a second, independent witness to what the truck says it
    is. Independent means it is not told what pass A concluded: a badge read
    that has been handed the answer corroborates nothing.

Everything here is offline. No API keys, no images, no network.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from app import modelspec
from app.evidence import prompts

# Currency in any of the shapes it could reach a prompt in. Deliberately not
# the WORD "price": several prompts legitimately tell the model not to estimate
# one, and a check that fired on its own warning would be turned off in a day.
CURRENCY = re.compile(r"(₺|\$|€|£|\bTRY\b|\bUSD\b|\bEUR\b|\bTL\b|\blira\b|\bfiyat\b)",
                      re.IGNORECASE)


@contextlib.contextmanager
def _card(payload: dict | None):
    """Run the block against a synthetic model card, or against none at all.

    The real `data/reference/models_tr.json` carries no generations and no weak
    points yet, so the composition those feed can only be exercised against a
    card written here. `_reset()` on both sides because `modelspec.reference()`
    caches globally and a temp card left in that cache would follow this module
    into every test that runs after it.
    """
    original = modelspec.MODEL_SPECS
    modelspec._reset()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "models_tr.json"
        if payload is not None:
            path.write_text(json.dumps(payload), encoding="utf-8")
        modelspec.MODEL_SPECS = path
        try:
            yield
        finally:
            modelspec.MODEL_SPECS = original
            modelspec._reset()


def _synthetic() -> dict:
    return {
        "as_of": "2026-09-13",
        "models": [
            {
                "brand": "FORD",
                "model": "F-MAX",
                "aliases": ["FMAX", "F MAX"],
                "segment": "long-haul tractor unit",
                "cab": "high sleeper, flat floor",
                "driveline": "12.7 litre six, automated 12-speed",
                "axle_configs": ["4x2"],
                "generations": [
                    {"id": "f-max-2018", "years": "2018-2022",
                     "visual_markers": ["one-piece grille bar",
                                        "single blade daytime running lamp"]},
                    {"id": "f-max-2023", "years": "2023-present",
                     "visual_markers": ["split grille with a raised centre",
                                        "stepped mirror housings"]},
                ],
                "known_weak_points": [
                    {"component": "drive_tires",
                     "note": "the drive axle scrubs its outer shoulders when the "
                             "rear air suspension has sagged"},
                    {"component": "adblue_tank",
                     "note": "the tank strap frets against the mounting and the "
                             "seam weeps below it"},
                ],
                "identity_tells": ["the door script reads F-MAX in a single line",
                                   "the sun visor is body coloured, not black"],
                "sources": [{"url": "https://example.invalid/f-max",
                             "date": "2026-09-13", "source_type": "oem_official"}],
            },
            {
                "brand": "MAN",
                "model": "TGX",
                "aliases": ["MAN TGX"],
                "axle_configs": ["4x2", "6x2"],
                "generations": [],
                "known_weak_points": [],
                "identity_tells": [],
                "sources": [],
            },
        ],
    }


# --- item 2: the closed model list ----------------------------------------

class IdentitySchema(unittest.TestCase):
    def test_the_model_field_is_a_closed_list_when_the_card_has_models(self):
        with _card(_synthetic()):
            model = prompts.identity_schema()["properties"]["model"]
        self.assertEqual(model["type"], ["string", "null"])
        self.assertEqual(model["enum"], ["F-MAX", "TGX", "other", None])

    def test_the_enum_narrows_to_one_brand_when_the_make_is_known(self):
        with _card(_synthetic()):
            model = prompts.identity_schema("FORD")["properties"]["model"]
        self.assertEqual(model["enum"], ["F-MAX", "other", None])
        self.assertNotIn("TGX", model["enum"])

    def test_a_brand_alias_narrows_the_enum_the_same_way(self):
        # `normalise_brand` folds "Ford Trucks" to FORD. Pass A answers with
        # whatever is on the grille, and the grille says "FORD TRUCKS".
        with _card(_synthetic()):
            self.assertEqual(prompts.identity_schema("Ford Trucks")
                             ["properties"]["model"]["enum"],
                             ["F-MAX", "other", None])

    def test_an_unknown_brand_leaves_the_field_free_text(self):
        # A judge's truck may be a brand this file has never heard of. An empty
        # enum would be an unanswerable schema; free text is the honest one.
        with _card(_synthetic()):
            model = prompts.identity_schema("SCANIA")["properties"]["model"]
        self.assertEqual(model, {"type": ["string", "null"]})

    def test_the_field_stays_free_text_when_the_card_is_missing(self):
        # The degrade path. Absent `models_tr.json`, pass A must run exactly as
        # it ran before this file existed.
        with _card(None):
            model = prompts.identity_schema()["properties"]["model"]
        self.assertEqual(model, {"type": ["string", "null"]})
        self.assertNotIn("enum", model)

    def test_the_schema_stays_strict_either_way(self):
        # Structured output in strict mode requires every property to be
        # required; `test_offline.JsonSchema.test_strict_shape` pins this for
        # the constant and it has to hold for the new shapes too.
        with _card(_synthetic()):
            schemas = [prompts.identity_schema(), prompts.identity_schema("FORD")]
        with _card(None):
            schemas.append(prompts.identity_schema())
        for schema in schemas:
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_the_constant_still_answers_everything_it_used_to(self):
        was = ["make", "model", "body_type", "cab_type", "axle_config",
               "approx_year_range", "badges_seen", "confidence", "same_vehicle",
               "vehicle_mismatch"]
        schema = prompts.IDENTITY_SCHEMA
        for key in was:
            self.assertIn(key, schema["properties"])
            self.assertIn(key, schema["required"])
        self.assertEqual(schema, prompts.identity_schema())

    def test_the_constant_tracks_the_card_rather_than_the_import(self):
        # It is resolved through a module __getattr__ and not frozen at import.
        # Importing `prompts` must not read a file off disk: a malformed card
        # would then take down `app.cli doctor`, which is the command you run
        # to find out that the card is malformed.
        with _card(None):
            self.assertNotIn("enum", prompts.IDENTITY_SCHEMA["properties"]["model"])
        with _card(_synthetic()):
            self.assertIn("enum", prompts.IDENTITY_SCHEMA["properties"]["model"])

    def test_the_generation_read_is_in_the_schema_and_required(self):
        schema = prompts.identity_schema()
        self.assertEqual(schema["properties"]["generation"], {"type": ["string", "null"]})
        self.assertEqual(schema["properties"]["generation_conf"], {"type": "number"})
        self.assertEqual(schema["properties"]["year_evidence"], {"type": "string"})
        for key in ("generation", "generation_conf", "year_evidence"):
            self.assertIn(key, schema["required"])

    def test_approx_year_range_survives_the_generation_read(self):
        # `passes.parse_identity` and `VehicleRead` both still read it, so the
        # generation is an addition and not a replacement.
        self.assertIn("approx_year_range", prompts.identity_schema()["properties"])

    def test_the_generation_is_not_itself_a_closed_list(self):
        # A generation list is a property of (brand, model) and the schema is
        # built before the model is known - pass A is the call that decides it.
        # The closed list travels in the prompt, where it can be conditioned on
        # the model the caller already has.
        with _card(_synthetic()):
            self.assertNotIn("enum", prompts.identity_schema("FORD")
                             ["properties"]["generation"])

    def test_no_identity_schema_carries_a_price_field(self):
        with _card(_synthetic()):
            blob = json.dumps([prompts.identity_schema(),
                               prompts.identity_schema("FORD"),
                               prompts.BADGE_SCHEMA])
        for word in ("price", "cost", "lira", "TRY", "USD", "asking"):
            self.assertNotIn(word, blob)


# --- item 2 and 3: what the identity prompt is told ------------------------

class IdentityPrompt(unittest.TestCase):
    CONTEXT = "You are looking at 8 photos of what should be one vehicle."

    def test_it_reads_identically_to_the_template_when_the_card_is_empty(self):
        # The degrade path again, and the reason it is asserted on the whole
        # string rather than on a marker: a composition that leaves an empty
        # heading, a doubled blank line or a dangling "one of:" behind is a
        # prompt regression nobody would notice by eye.
        with _card(None):
            for make, model in ((None, None), ("SCANIA", "R-SERIES"), ("FORD", None)):
                self.assertEqual(
                    prompts.identity_prompt(context=self.CONTEXT, make=make, model=model),
                    prompts.IDENTITY_PROMPT.format(context=self.CONTEXT))

    def test_the_context_still_reaches_the_prompt(self):
        with _card(_synthetic()):
            prompt = prompts.identity_prompt(context=self.CONTEXT, make="FORD")
        self.assertIn(self.CONTEXT, prompt)
        self.assertTrue(prompt.rstrip().endswith("Return the JSON object now."))

    def test_the_closed_model_list_reaches_the_prompt_with_its_escape(self):
        with _card(_synthetic()):
            prompt = prompts.identity_prompt(context=self.CONTEXT)
        self.assertIn("F-MAX", prompt)
        self.assertIn("TGX", prompt)
        self.assertIn('"other"', prompt)

    def test_the_list_is_narrowed_when_the_make_is_known(self):
        with _card(_synthetic()):
            prompt = prompts.identity_prompt(context=self.CONTEXT, make="FORD")
        self.assertNotIn("TGX", prompt)

    def test_the_generations_arrive_with_the_markers_that_separate_them(self):
        with _card(_synthetic()):
            prompt = prompts.identity_prompt(context=self.CONTEXT,
                                             make="FORD", model="F-MAX")
        for gen in ("f-max-2018", "f-max-2023"):
            self.assertIn(gen, prompt)
        self.assertIn("one-piece grille bar", prompt)
        self.assertIn("split grille with a raised centre", prompt)

    def test_a_generation_list_needs_the_model_and_not_only_the_make(self):
        # Generations belong to a (brand, model) pair. With the make alone
        # there is nothing to list, and listing every Ford generation would be
        # telling the model what to look for on a truck it has not identified.
        with _card(_synthetic()):
            prompt = prompts.identity_prompt(context=self.CONTEXT, make="FORD")
        self.assertNotIn("f-max-2018", prompt)

    def test_the_heading_names_the_model_the_card_matched_on(self):
        # The caller may be holding the raw vision answer. `card()` finds the
        # row through its aliases, so a heading built from what was typed would
        # name a model the reference does not have.
        with _card(_synthetic()):
            prompt = prompts.identity_prompt(context=self.CONTEXT,
                                             make="ford trucks", model="F Max")
        self.assertIn("The F-MAX has these generations", prompt)
        self.assertNotIn("F MAX", prompt)

    def test_the_identity_tells_reach_the_prompt(self):
        with _card(_synthetic()):
            prompt = prompts.identity_prompt(context=self.CONTEXT,
                                             make="FORD", model="F-MAX")
        self.assertIn("the sun visor is body coloured", prompt)

    def test_the_generation_is_read_off_the_markers_and_never_off_the_year(self):
        # The whole of item 3. A generation derived from the declared year
        # makes every downstream cross-check against that year circular: it
        # would agree by construction and catch nothing. The instruction has to
        # be there card or no card, because `generation` is in the schema card
        # or no card.
        for payload in (None, _synthetic()):
            with _card(payload):
                prompt = prompts.identity_prompt(context=self.CONTEXT,
                                                 make="FORD", model="F-MAX")
            low = prompt.lower()
            self.assertIn("never", low)
            self.assertIn("year the seller", low)
            self.assertIn("year_evidence", prompt)
            self.assertIn("generation_conf", prompt)

    def test_the_prompt_asks_what_in_the_photos_supports_the_year(self):
        with _card(None):
            prompt = prompts.identity_prompt(context=self.CONTEXT)
        self.assertIn("year_evidence", prompt)
        self.assertIn("approx_year_range", prompt)

    def test_no_currency_reaches_the_identity_prompt(self):
        with _card(_synthetic()):
            prompt = prompts.identity_prompt(context=self.CONTEXT,
                                             make="FORD", model="F-MAX")
        self.assertIsNone(CURRENCY.search(prompt))


# --- item 4: the badge read, as an independent witness ---------------------

class BadgePass(unittest.TestCase):
    def test_the_schema_is_strict_and_carries_the_six_answers(self):
        schema = prompts.BADGE_SCHEMA
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(set(schema["properties"]),
                         {"badge_text", "make", "model", "trim_or_power",
                          "legible", "confidence"})
        self.assertEqual(schema["properties"]["badge_text"],
                         {"type": "array", "items": {"type": "string"}})
        self.assertEqual(schema["properties"]["legible"], {"type": "boolean"})
        self.assertEqual(schema["properties"]["confidence"], {"type": "number"})

    def test_the_badge_model_is_not_a_closed_list(self):
        # A transcription has to be able to disagree with the vocabulary. If
        # the enum forced the badge read onto the same closed list pass A
        # answers from, the two would agree on a truck neither had read.
        with _card(_synthetic()):
            self.assertNotIn("enum", prompts.BADGE_SCHEMA["properties"]["model"])
            self.assertNotIn("enum", prompts.BADGE_SCHEMA["properties"]["make"])

    def test_it_says_it_is_looking_at_a_crop(self):
        prompt = prompts.badge_prompt()
        self.assertIn("CROP", prompt)

    def test_it_asks_for_literal_transcription(self):
        low = prompts.badge_prompt().lower()
        self.assertIn("verbatim", low)
        self.assertIn("transcribe", low)

    def test_it_forbids_guessing_the_model_from_the_styling(self):
        low = prompts.badge_prompt().lower()
        self.assertIn("cab", low)
        self.assertIn("grille", low)
        self.assertTrue(re.search(r"do not identify the model", low))

    def test_it_is_never_told_what_the_other_passes_concluded(self):
        # The parameters exist so the prompt COULD name a brand's badge
        # conventions. It does not, and that is the point: make and model are
        # the two answers this call is a second witness to, and a witness who
        # has been told the answer is not a second reading of anything.
        # `reconcile` compares this read against pass A's, and the comparison
        # is only worth running while the two are independent.
        told = prompts.badge_prompt(make="FORD", model="F-MAX")
        self.assertEqual(told, prompts.badge_prompt())
        for leak in ("FORD", "Ford", "F-MAX", "F-Max"):
            self.assertNotIn(leak, told)

    def test_it_refuses_the_windscreen_sticker(self):
        # The crop is the upper band of the cab, which on a dealer lot includes
        # the windscreen - and a windscreen carries an asking figure and a
        # telephone number. A transcription pass pointed at that band is the
        # one place in this system where a currency figure could walk back IN
        # through the model's own answer, and `badge_text` is displayed.
        low = prompts.badge_prompt().lower()
        self.assertIn("windscreen", low)
        self.assertIn("telephone", low)

    def test_no_currency_reaches_the_badge_prompt(self):
        self.assertIsNone(CURRENCY.search(prompts.badge_prompt()))
        self.assertIsNone(CURRENCY.search(prompts.badge_prompt(make="FORD",
                                                               model="F-MAX")))


# --- item 9: the model-conditioned close-up -------------------------------

class CloseupDefaultsUnchanged(unittest.TestCase):
    """The two new parameters are inert at their defaults, to the byte.

    These digests were taken from the close-up prompt as it stood before
    `spec_lines` and `weak_points` existed. A deliberate edit to the rubric, a
    checklist or the assembly will break them, and the fix is then to re-record
    them here with the date and the reason - what they are here to catch is an
    edit nobody meant to make, which is what adding two optional parameters to
    a three-zone prompt assembly usually is.
    """

    BASELINE = {
        "8ca2aa646c974390a22d07dae0ed2cc8fde154da2cdda20ed905fdccac169abf": dict(
            view="tire_wheel", view_pretty="tire wheel", vehicle="A Ford Trucks F-MAX.",
            cropped=False, soft=False, expectation="It is five years old.", band="mid"),
        "0eb48cae8e8bfb578c92bcfc41ef6b2aa9498440681be0735a990b79bab4adba": dict(
            view="engine_bay", view_pretty="engine bay", vehicle="A MAN TGX.",
            cropped=True, soft=True, expectation="", band=None),
        "45bee90b7dc2577d7910605e53f9ba9fa04e8e828eea001cd04c6b8b5859ecfa": dict(
            view="dashboard_odometer", view_pretty="dashboard odometer",
            vehicle="A Ford Trucks F-MAX.", cropped=False, soft=False,
            expectation="It has covered 100,000-200,000 km.", band="low"),
        "5f323bffa8b872d70caa546ead765d910bbee42e37c7cbdb9c6078959db3ac3d": dict(
            view="fifth_wheel", view_pretty="fifth wheel", vehicle="A truck.",
            cropped=False, soft=False),
    }

    def test_the_default_prompt_is_byte_identical_to_the_baseline(self):
        for digest, kwargs in self.BASELINE.items():
            out = prompts.closeup_prompt(**kwargs)
            self.assertEqual(hashlib.sha256(out.encode()).hexdigest(), digest,
                             f"{kwargs['view']}: the default close-up prompt moved")

    def test_empty_arguments_are_the_same_as_no_arguments(self):
        for kwargs in self.BASELINE.values():
            base = prompts.closeup_prompt(**kwargs)
            for empty in ((), [], None):
                self.assertEqual(base, prompts.closeup_prompt(
                    **kwargs, spec_lines=empty, weak_points=empty))


class CloseupModelAware(unittest.TestCase):
    SPEC = ["Segment: long-haul tractor unit.", "Cab: high sleeper, flat floor."]
    WEAK = [("drive_tires", "the drive axle scrubs its outer shoulders when the rear "
                            "air suspension has sagged"),
            ("adblue_tank", "the tank strap frets against the mounting and the seam "
                            "weeps below it")]

    def _tire(self, **kw) -> str:
        return prompts.closeup_prompt(
            view="tire_wheel", view_pretty="tire wheel", vehicle="A Ford Trucks F-MAX.",
            cropped=False, soft=False, expectation="It is five years old.",
            band="mid", **kw)

    def test_the_spec_lines_land_in_the_per_appraisal_zone(self):
        # Between the invariant prefix and the per-photo zone: they are the
        # same for all sixteen photos of one truck, so they belong with the
        # vehicle line and not in the part that changes per frame.
        prompt = self._tire(spec_lines=self.SPEC)
        for line in self.SPEC:
            self.assertIn(line, prompt)
            self.assertGreater(prompt.index(line), len(prompts.CLOSEUP_INVARIANT) - 1)
            self.assertLess(prompt.index(line), prompt.index("Work through each of these"))
        self.assertLess(prompt.index("A Ford Trucks F-MAX."), prompt.index(self.SPEC[0]))

    def test_the_invariant_prefix_survives_both_new_parameters(self):
        # The prefix-caching invariant: each photo is read CLOSEUP_SAMPLES
        # times, so anything that moves into the shared prefix is paid for once
        # and anything that breaks it is paid for 48 times.
        a = self._tire(spec_lines=self.SPEC, weak_points=self.WEAK)
        b = prompts.closeup_prompt(view="engine_bay", view_pretty="engine bay",
                                   vehicle="A MAN TGX.", cropped=True, soft=True,
                                   spec_lines=["Cab: day cab."], weak_points=self.WEAK)
        self.assertTrue(a.startswith(prompts.CLOSEUP_INVARIANT))
        self.assertTrue(b.startswith(prompts.CLOSEUP_INVARIANT))

    def test_weak_points_are_filtered_to_the_components_in_this_view(self):
        # A tire close-up has no business being told about this model's AdBlue
        # tank: it cannot see one, and a prior it cannot check is a prior it
        # can only report on faith.
        prompt = self._tire(weak_points=self.WEAK)
        self.assertIn("scrubs its outer shoulders", prompt)
        self.assertNotIn("the tank strap frets", prompt)

        engine = prompts.closeup_prompt(
            view="engine_bay", view_pretty="engine bay", vehicle="A Ford Trucks F-MAX.",
            cropped=False, soft=False, weak_points=self.WEAK)
        self.assertIn("the tank strap frets", engine)
        self.assertNotIn("scrubs its outer shoulders", engine)

    def test_a_weak_point_outside_the_component_enum_is_dropped(self):
        # A typo in the reference card must not invent a component id. The
        # close-up is asked to answer inside `COMPONENTS` and nothing else.
        prompt = self._tire(weak_points=[("drive_tyres", "British spelling, no such id")])
        self.assertNotIn("British spelling", prompt)

    def test_the_report_only_if_visible_hedge_travels_with_them(self):
        # The failure this wording exists to fight: a model told a component is
        # a known weak point reports it whether or not the frame shows one,
        # which is the same over-reporting SEVERITY_RUBRIC was written against
        # - only now with a reference card behind it, which reads to a buyer
        # like corroboration.
        prompt = self._tire(weak_points=self.WEAK)
        self.assertIn("ONLY if you can see", prompt)
        low = prompt.lower()
        self.assertIn("known weak point", low)
        self.assertIn("this model", low)
        self.assertIn("not an observation about this truck", low)

    def test_no_weak_point_block_survives_when_nothing_applies(self):
        # An empty heading is worse than no heading: it tells the model a list
        # was meant to be there and invites it to fill one in.
        prompt = self._tire(weak_points=[("adblue_tank", "not visible from here")])
        self.assertNotIn("weak point", prompt.lower())

    def test_the_declared_distance_still_never_reaches_the_prompt(self):
        # The new zone is a new way for a figure to travel. The close-up gets a
        # band, never the number, or `reconcile`'s odometer rule compares the
        # seller's figure against itself.
        prompt = prompts.closeup_prompt(
            view="dashboard_odometer", view_pretty="dashboard odometer",
            vehicle="A Ford Trucks F-MAX.", cropped=False, soft=False,
            expectation="It has covered 400,000-500,000 km.", band="mid",
            spec_lines=["Segment: long-haul tractor unit."],
            weak_points=[("dashboard_instruments", "the cluster backlight dims")])
        for rendering in ("420000", "420,000", "420000.0"):
            self.assertNotIn(rendering, prompt)

    def test_no_currency_reaches_a_model_aware_closeup(self):
        self.assertIsNone(CURRENCY.search(self._tire(spec_lines=self.SPEC,
                                                     weak_points=self.WEAK)))

    def test_the_repair_bands_never_reach_a_model_aware_closeup(self):
        # `config.REPAIR_BANDS` is the lira reading of the four severity words
        # and it is for the README and the panel card only.
        from app.config import REPAIR_BANDS

        prompt = self._tire(spec_lines=self.SPEC, weak_points=self.WEAK)
        for low, high in REPAIR_BANDS.values():
            for bound in (low, high):
                if bound:
                    self.assertNotIn(f"{bound:,}", prompt)
                    self.assertNotIn(str(bound), prompt)


class ViewComponents(unittest.TestCase):
    """The filter the weak points are run through is the real vocabulary."""

    def test_every_view_maps_onto_components_that_exist(self):
        from app import vision

        for view in vision.VIEW_LABELS:
            for component in prompts.view_components(view):
                self.assertIn(component, prompts.COMPONENTS)

    def test_the_views_between_them_cover_every_component(self):
        # If a component belonged to no view, a weak point naming it could
        # never reach any close-up call and the card entry would be dead text.
        seen = set()
        for view in prompts.VIEW_FAMILIES:
            seen.update(prompts.view_components(view))
        self.assertEqual(seen, set(prompts.COMPONENTS))

    def test_an_unknown_view_has_no_components_rather_than_all_of_them(self):
        self.assertEqual(prompts.view_components("not_a_view"), ())


if __name__ == "__main__":
    unittest.main()
