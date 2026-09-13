"""Offline tests for the eval harness. No vision calls, no network, no torch.

A new file rather than additions to `test_offline.py` on purpose: that file is
being edited by other workstreams right now and a merge conflict in the test
suite is the one place a conflict can silently delete a check.

What is worth testing here is narrow and specific. The harness is a measuring
instrument, so its failure mode is not an exception - it is a number that looks
fine and is wrong. Every test below pins a property whose violation would
produce a plausible number:

  * a cache key that collides across fields would serve one photo's answer for
    another and report a false-positive rate of zero;
  * a key that ignores `repeat_index` would collapse three samples into one and
    measure exactly zero variance;
  * a prompt that differs between the halves of a twin pair would make the
    whole suite a comparison of prompts;
  * a matcher that lets one original finding absorb two twin paraphrases would
    under-count false positives by exactly the amount being looked for;
  * a diff with no noise floor would report resampling as progress.
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from app import config
from app.evidence import passes, prompts
from app.vlm.base import VLMResponse

from eval import cache as evcache
from eval import cases, replay, scorecard
from eval.cases import TwinPair
from eval.suites import DEFAULT_ORDER, SUITES, base as suite_base
from eval.suites import demo_gate, distribution, monotonic, retest, twin_fp


def _pair(index: int = 0, view: str = "tire_wheel", **kw) -> TwinPair:
    defaults = dict(listing_id="14190", image_index=index, view=view, market="TR",
                    make="Ford", model_name="F-MAX", year=2021, km=164374.0,
                    quality_bucket="good", severity=0.5, severity_band="medium",
                    degradations=["motion_blur"],
                    original_rel="images/x/14190/000.jpg",
                    twin_rel="images/degraded/x/14190/000_deg.jpg")
    defaults.update(kw)
    return TwinPair(**defaults)


def _obs(sample: int, component: str, text: str, severity: str = "minor",
         impact: str = "low", confidence: float = 0.8) -> twin_fp.Obs:
    return twin_fp.Obs(sample=sample, component=component,
                       family=twin_fp.family_of(component), observation=text,
                       severity=severity, impact=impact, confidence=confidence)


class FakeBackend:
    """Counts calls and returns canned text. Never touches a network."""
    name = "fake"
    model = "fake-model"
    supports_structured_output = True

    def __init__(self, text: str = '{"shows": "a tire", "observations": []}'):
        self.text = text
        self.calls = 0

    def probe(self):
        from app.vlm.base import BackendStatus
        return BackendStatus(name=self.name, ready=True, model=self.model)

    def complete(self, prompt, images, *, system="", max_tokens=4096,
                 json_schema=None, effort=None):
        self.calls += 1
        return VLMResponse(text=self.text, backend=self.name, model=self.model,
                           elapsed_s=0.01)


# --- the cache key ---------------------------------------------------------

class CacheKey(unittest.TestCase):
    BASE = dict(image_digests=["aaa"], prompt="describe this tire",
                system="you are an appraiser", model_id="gpt-5.6-sol",
                effort="high", schema=None, repeat_index=0)

    def key(self, **over):
        return evcache.cache_key(**{**self.BASE, **over})

    def test_stable(self):
        self.assertEqual(self.key(), self.key())

    def test_fields_are_length_prefixed(self):
        # Without prefixing, prompt+system concatenation makes these identical,
        # and the cache would serve one photo's answer for a different question.
        a = self.key(prompt="ab", system="c")
        b = self.key(prompt="a", system="bc")
        self.assertNotEqual(a, b)

    def test_repeat_index_separates_samples(self):
        # config.CLOSEUP_SAMPLES reads each photo more than once on purpose. A
        # key that ignored this would collapse them and measure zero variance.
        self.assertNotEqual(self.key(repeat_index=0), self.key(repeat_index=1))

    def test_effort_is_keyed(self):
        # effort became per-call after the design's key list was written.
        self.assertNotEqual(self.key(effort="low"), self.key(effort="high"))

    def test_prompt_change_busts_the_key(self):
        self.assertNotEqual(self.key(), self.key(prompt="describe this tyre"))

    def test_image_change_busts_the_key(self):
        self.assertNotEqual(self.key(), self.key(image_digests=["bbb"]))
        self.assertNotEqual(self.key(), self.key(image_digests=["aaa", "bbb"]))

    def test_schema_is_keyed(self):
        self.assertNotEqual(self.key(), self.key(schema={"type": "object"}))

    def test_model_is_keyed(self):
        self.assertNotEqual(self.key(), self.key(model_id="claude-opus-5"))


class CacheStore(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="eval-cache-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.cache = evcache.ResponseCache(self.root)

    def test_round_trip(self):
        self.cache.put("m", "k" * 8, {"text": "hello"})
        self.assertEqual(self.cache.get("m", "k" * 8)["text"], "hello")
        self.assertTrue(self.cache.has("m", "k" * 8))
        self.assertIsNone(self.cache.get("m", "missing"))

    def test_corrupt_entry_is_a_miss_not_a_crash(self):
        path = self.cache.path_for("m", "bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.cache.get("m", "bad"))

    def test_model_ids_are_isolated(self):
        self.cache.put("a", "k", {"text": "one"})
        self.assertIsNone(self.cache.get("b", "k"))


class CachedBackendBehaviour(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="eval-cache-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.cache = evcache.ResponseCache(self.root)
        self.inner = FakeBackend()
        self.client = evcache.CachedBackend(self.inner, cache=self.cache,
                                            model_id="fake-model")
        self.image = self.root / "photo.jpg"
        from PIL import Image
        Image.new("RGB", (64, 48), (120, 120, 120)).save(self.image)

    def test_second_call_is_served_from_disk(self):
        first = self.client.complete("p", [self.image], system="s", effort="high")
        second = self.client.complete("p", [self.image], system="s", effort="high")
        self.assertEqual(first.text, second.text)
        self.assertEqual(self.inner.calls, 1, "the second call reached the provider")
        self.assertEqual(self.cache.stats.new, 1)
        self.assertEqual(self.cache.stats.cached, 1)
        self.assertEqual(self.cache.stats.hit_rate, 0.5)

    def test_repeat_index_costs_a_second_call(self):
        self.client.complete("p", [self.image], repeat_index=0)
        self.client.complete("p", [self.image], repeat_index=1)
        self.assertEqual(self.inner.calls, 2)

    def test_stores_raw_text_and_not_a_parsed_object(self):
        self.client.complete("p", [self.image])
        entry = json.loads(next(self.root.rglob("*.json")).read_text())
        self.assertIn("text", entry)
        self.assertEqual(entry["text"], self.inner.text)
        self.assertNotIn("issues", entry)
        self.assertNotIn("observations", entry)

    def test_max_tokens_is_recorded_but_not_keyed(self):
        self.client.complete("p", [self.image], max_tokens=1000)
        self.client.complete("p", [self.image], max_tokens=4000)
        self.assertEqual(self.inner.calls, 1, "max_tokens must not bust a 15 MB cache")
        entry = json.loads(next(self.root.rglob("*.json")).read_text())
        self.assertEqual(entry["max_tokens"], 1000)

    def test_offline_client_raises_rather_than_calling(self):
        offline = evcache.CachedBackend(None, cache=self.cache, model_id="fake-model")
        with self.assertRaises(evcache.CacheMiss):
            offline.complete("never seen", [self.image])

    def test_offline_client_serves_what_is_there(self):
        self.client.complete("p", [self.image], effort="high")
        offline = evcache.CachedBackend(None, cache=self.cache, model_id="fake-model")
        self.assertEqual(offline.complete("p", [self.image], effort="high").text,
                         self.inner.text)

    def test_image_digest_is_deterministic(self):
        self.assertEqual(evcache.image_digest(self.image),
                         evcache.image_digest(self.image))

    def test_provider_failure_is_counted_not_cached(self):
        class Broken(FakeBackend):
            def complete(self, *a, **k):
                raise RuntimeError("500")
        client = evcache.CachedBackend(Broken(), cache=self.cache, model_id="m")
        with self.assertRaises(RuntimeError):
            client.complete("p", [self.image])
        self.assertEqual(self.cache.stats.failed, 1)
        self.assertEqual(self.cache.stats.new, 0)


# --- the scorecard ---------------------------------------------------------

class Fingerprint(unittest.TestCase):
    def test_four_files_hashed_separately(self):
        root = Path(tempfile.mkdtemp(prefix="eval-fp-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for rel in scorecard.FINGERPRINT_FILES.values():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original", encoding="utf-8")
        before = scorecard.code_fingerprint(root)
        (root / scorecard.FINGERPRINT_FILES["prompts_sha"]).write_text("edited")
        after = scorecard.code_fingerprint(root)
        moved = [k for k in before if before[k] != after[k]]
        self.assertEqual(moved, ["prompts_sha"],
                         "a diff must be able to say WHICH workstream moved")

    def test_absent_file_is_absent_not_an_error(self):
        root = Path(tempfile.mkdtemp(prefix="eval-fp-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        self.assertEqual(scorecard.code_fingerprint(root)["condition_sha"], "absent")

    def test_condition_module_is_optional_today(self):
        # app/condition.py is the other workstream's file and may or may not
        # exist; either way the fingerprint must be a string.
        self.assertIsInstance(scorecard.code_fingerprint()["condition_sha"], str)


class Intervals(unittest.TestCase):
    def test_wilson_does_not_collapse_at_zero(self):
        lo, hi = scorecard.wilson_ci(0, 8)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.2, "0 of 8 is not 'confidently zero'")

    def test_wilson_narrows_with_n(self):
        small = scorecard.wilson_ci(1, 10)
        large = scorecard.wilson_ci(10, 100)
        self.assertLess(large[1] - large[0], small[1] - small[0])

    def test_bootstrap_brackets_the_statistic(self):
        units = list(range(1, 21))
        ci = scorecard.bootstrap_ci(units, lambda xs: sum(xs) / len(xs), draws=500)
        self.assertLess(ci[0], 10.5)
        self.assertGreater(ci[1], 10.5)

    def test_bootstrap_declines_below_two_units(self):
        self.assertIsNone(scorecard.bootstrap_ci([1], lambda xs: xs[0]))
        self.assertIsNone(scorecard.bootstrap_ci([], lambda xs: 0))

    def test_bootstrap_is_seeded(self):
        units = list(range(30))
        stat = (lambda xs: sum(xs) / len(xs))
        self.assertEqual(scorecard.bootstrap_ci(units, stat, seed=3),
                         scorecard.bootstrap_ci(units, stat, seed=3))


class NoiseFloor(unittest.TestCase):
    def test_delta_inside_the_before_interval_is_noise(self):
        noise, _ = scorecard.is_noise({"value": 0.30, "ci": [0.18, 0.44]},
                                      {"value": 0.40, "ci": [0.28, 0.52]})
        self.assertTrue(noise)

    def test_separated_intervals_are_a_real_move(self):
        noise, _ = scorecard.is_noise({"value": 0.50, "ci": [0.45, 0.55]},
                                      {"value": 0.10, "ci": [0.05, 0.15]})
        self.assertFalse(noise)

    def test_no_interval_anywhere_is_noise(self):
        noise, reason = scorecard.is_noise({"value": 0.5}, {"value": 0.1})
        self.assertTrue(noise)
        self.assertIn("interval", reason)

    def test_overlapping_intervals_are_noise_even_when_the_value_is_outside(self):
        noise, _ = scorecard.is_noise({"value": 0.50, "ci": [0.40, 0.60]},
                                      {"value": 0.30, "ci": [0.25, 0.45]})
        self.assertTrue(noise)


class ScorecardArtifact(unittest.TestCase):
    def _card(self, value=0.31, ci=(0.18, 0.44)):
        card = scorecard.Scorecard.start(
            tier="standard", seed=7,
            model={"backend": "openai", "id": "gpt-5.6-sol", "effort": "high"})
        result = scorecard.SuiteResult(name="twin_fp")
        result.metrics.append(scorecard.Metric(
            name="severity_inflation", value=value, ci=list(ci), n=40, headline=True))
        result.gates.append(scorecard.Gate(name="twin_severity_inflation", value=value,
                                           threshold=0.15))
        card.add(result)
        card.calls = {"new": 1472, "cached": 2140, "failed": 3, "hit_rate": 0.59}
        return card

    def test_schema_has_the_required_keys(self):
        d = self._card().to_dict()
        for key in ("schema", "run_id", "git", "model", "code_fingerprint", "tier",
                    "seed", "calls", "suites", "gates", "notes"):
            self.assertIn(key, d)
        self.assertEqual(sorted(d["code_fingerprint"]),
                         sorted(scorecard.FINGERPRINT_FILES))

    def test_gate_records_pass_and_fail(self):
        self.assertIs(self._card(value=0.31).to_dict()["gates"][0]["pass"], False)
        self.assertIs(self._card(value=0.05).to_dict()["gates"][0]["pass"], True)

    def test_calls_and_hit_rate_are_visible_in_the_markdown(self):
        md = self._card().markdown()
        self.assertIn("1472 new", md)
        self.assertIn("hit rate 59%", md)

    def test_the_bench_note_travels_with_every_artifact(self):
        self.assertIn("truck_vocab_share", "".join(self._card().notes))
        self.assertIn("no ground truth", "".join(self._card().notes))

    def test_write_then_load(self):
        root = Path(tempfile.mkdtemp(prefix="eval-run-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        out = self._card().write(root)
        self.assertTrue((out / "scorecard.json").exists())
        self.assertTrue((out / "scorecard.md").exists())
        self.assertEqual(scorecard.load(out)["tier"], "standard")

    def test_diff_marks_noise_and_movement(self):
        before, after = self._card(0.31).to_dict(), self._card(0.35).to_dict()
        rows = scorecard.diff(before, after)
        self.assertTrue(all(r.within_noise for r in rows))
        text = scorecard.render_diff(before, after, rows)
        self.assertIn("within noise", text)

    def test_diff_flags_a_tier_mismatch(self):
        before = self._card().to_dict()
        after = self._card().to_dict()
        after["tier"] = "smoke"
        text = scorecard.render_diff(before, after, scorecard.diff(before, after))
        self.assertIn("tiers differ", text)

    def test_diff_names_the_workstream_that_moved(self):
        before = self._card().to_dict()
        after = self._card().to_dict()
        after["code_fingerprint"]["prompts_sha"] = "deadbeef"
        text = scorecard.render_diff(before, after, scorecard.diff(before, after))
        self.assertIn("prompts", text)


# --- sampling --------------------------------------------------------------

class Allocation(unittest.TestCase):
    def test_sums_to_the_target(self):
        got = cases.allocate(40, {"TR": 84, "US": 116})
        self.assertEqual(sum(got.values()), 40)

    def test_never_exceeds_a_stratum(self):
        got = cases.allocate(10, {"a": 2, "b": 50})
        self.assertLessEqual(got["a"], 2)
        self.assertEqual(sum(got.values()), 10)

    def test_deterministic_regardless_of_dict_order(self):
        a = cases.allocate(7, {"x": 3, "y": 5, "z": 11})
        b = cases.allocate(7, {"z": 11, "y": 5, "x": 3})
        self.assertEqual(a, b)

    def test_degenerate_inputs(self):
        self.assertEqual(cases.allocate(0, {"a": 1}), {"a": 0})
        self.assertEqual(cases.allocate(5, {}), {})


class CorpusSampling(unittest.TestCase):
    def test_vehicle_table_matches_the_corpus(self):
        v = cases.vehicles()
        self.assertEqual(len(v), 200)
        self.assertEqual(dict(v.market.value_counts()), {"US": 116, "TR": 84})

    def test_sampling_is_deterministic(self):
        self.assertEqual(cases.sample_vehicles(20, seed=7),
                         cases.sample_vehicles(20, seed=7))
        self.assertNotEqual(cases.sample_vehicles(20, seed=7),
                            cases.sample_vehicles(20, seed=8))

    def test_sampling_is_stratified_by_market(self):
        picked = set(cases.sample_vehicles(40, seed=7))
        v = cases.vehicles().set_index("listing_id")
        markets = v.loc[sorted(picked), "market"]
        self.assertGreater((markets == "TR").sum(), 5)
        self.assertGreater((markets == "US").sum(), 5)

    def test_every_sampled_vehicle_has_enough_views(self):
        v = cases.vehicles().set_index("listing_id")
        for listing_id in cases.sample_vehicles(30, seed=11):
            self.assertGreaterEqual(v.loc[listing_id, "n_views"], cases.MIN_VIEWS)

    def test_twin_pairs_pair_on_listing_and_index(self):
        pairs = cases.twin_pairs(["14190"])
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertIn(f"/{pair.image_index:03d}", pair.original_rel)
            self.assertIn("degraded", pair.twin_rel)

    def test_clean_path_confirms_the_pairing(self):
        images = cases.load_images()
        degraded = images[images.variant == "degraded"].set_index(
            ["listing_id", "image_index"])
        original = images[images.variant == "original"].set_index(
            ["listing_id", "image_index"])
        shared = original.index.intersection(degraded.index)
        self.assertEqual(len(shared), 3729)
        agree = (degraded.clean_path.reindex(shared) == original.path.reindex(shared))
        self.assertTrue(agree.all(), "clean_path must corroborate the index pairing")

    def test_twin_sample_is_deterministic_and_sized(self):
        a = cases.sample_twin_pairs(10, 4, seed=7, require_files=False)
        b = cases.sample_twin_pairs(10, 4, seed=7, require_files=False)
        self.assertEqual([p.unit for p in a], [p.unit for p in b])
        self.assertLessEqual(len({p.listing_id for p in a}), 10)
        for _, group in _group(a).items():
            self.assertLessEqual(len(group), 4)

    def test_forced_views_are_taken_first(self):
        pairs = cases.sample_twin_pairs(6, 3, seed=7, require_files=False)
        for listing_id, group in _group(pairs).items():
            available = {p.view for p in cases.twin_pairs([listing_id])}
            wanted = [v for v in cases.FORCE_VIEWS if v in available][:3]
            self.assertTrue(set(wanted).issubset({p.view for p in group}),
                            f"{listing_id} missed a forced view")

    def test_describe_reports_the_strata(self):
        described = cases.describe(cases.sample_twin_pairs(6, 3, seed=7,
                                                           require_files=False))
        for key in ("pairs", "vehicles", "by_market", "by_view", "by_severity_band"):
            self.assertIn(key, described)


def _group(pairs):
    out = {}
    for pair in pairs:
        out.setdefault(pair.listing_id, []).append(pair)
    return out


# --- the twin suite --------------------------------------------------------

class PromptIsHeldIdentical(unittest.TestCase):
    """The central confound control. If this fails the suite measures prompts."""

    def test_both_halves_get_the_same_prompt(self):
        pair = _pair()
        self.assertEqual(twin_fp.build_prompt(pair), twin_fp.build_prompt(pair))

    def test_prompt_uses_the_originals_view_not_the_twins(self):
        original_view = twin_fp.build_prompt(_pair(view="tire_wheel"))
        drifted_view = twin_fp.build_prompt(_pair(view="chassis_undercarriage"))
        self.assertNotEqual(original_view, drifted_view)
        # `plan` builds ONE prompt per pair and reuses it for both halves, so a
        # view tag that shifts under degradation cannot reach the twin's call.
        plan = twin_fp.plan("smoke", seed=7, model_id="m", require_files=False,
                            resolve_keys=False)
        self.assertTrue(plan.pairs)
        self.assertIn("view=original", plan.params["prompt_held"])

    def test_the_soft_focus_line_is_off_on_both_halves(self):
        soft = prompts.closeup_prompt(view="tire_wheel", view_pretty="tire wheel",
                                      vehicle="x", cropped=False, soft=True)
        held = prompts.closeup_prompt(view="tire_wheel", view_pretty="tire wheel",
                                      vehicle="x", cropped=False, soft=False)
        self.assertNotEqual(soft, held)
        self.assertNotIn(_soft_only_text(soft, held), twin_fp.build_prompt(_pair()))

    def test_crop_is_off_on_both_halves(self):
        self.assertIn("cropped=False", twin_fp.plan(
            "smoke", seed=7, model_id="m", require_files=False,
            resolve_keys=False).params["prompt_held"])


def _soft_only_text(soft: str, held: str) -> str:
    """The sentence the soft flag adds, so a test can assert it is absent."""
    extra = [line for line in soft.splitlines() if line and line not in held]
    return extra[0] if extra else "\x00never appears\x00"


class MatchingRules(unittest.TestCase):
    def test_threshold_is_below_same_defect_so_the_rate_is_a_lower_bound(self):
        self.assertLess(twin_fp.TWIN_MATCH, passes.SAME_DEFECT)

    def test_quorum_is_two_thirds(self):
        self.assertEqual(twin_fp.quorum_for(3), 2)
        self.assertEqual(twin_fp.quorum_for(1), 1)
        self.assertEqual(twin_fp.quorum_for(5), 4)

    def test_paraphrases_of_one_defect_cluster(self):
        obs = [_obs(0, "drive_tires", "shoulder wear across the outer ribs of the drive tire"),
               _obs(1, "drive_tires", "outer ribs of the drive tire show shoulder wear"),
               _obs(2, "drive_tires", "shoulder wear visible across outer drive tire ribs")]
        clusters = twin_fp.cluster_observations(obs)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].support, 3)

    def test_different_components_never_cluster(self):
        obs = [_obs(0, "drive_tires", "shoulder wear across the outer ribs"),
               _obs(0, "steer_tires", "shoulder wear across the outer ribs")]
        self.assertEqual(len(twin_fp.cluster_observations(obs)), 2)

    def test_support_counts_samples_not_members(self):
        obs = [_obs(0, "drive_tires", "shoulder wear across the outer ribs"),
               _obs(0, "drive_tires", "shoulder wear across outer ribs again")]
        self.assertEqual(twin_fp.cluster_observations(obs)[0].support, 1)

    def test_matching_is_one_to_one(self):
        # One original finding must not absorb two twin paraphrases: that is
        # exactly the amount by which false positives would be under-counted.
        twin = twin_fp.cluster_observations([
            _obs(0, "drive_tires", "shoulder wear across the outer ribs of the drive tire")])
        twin += twin_fp.cluster_observations([
            _obs(1, "drive_tires", "shoulder wear across the outer ribs of drive tires")])
        original = twin_fp.cluster_observations([
            _obs(0, "drive_tires", "shoulder wear across the outer ribs of the drive tire")])
        matched = twin_fp.greedy_match(twin, original)
        self.assertEqual(len(matched), 1)

    def test_matches_any_counts_distinct_samples(self):
        cluster = twin_fp.cluster_observations([
            _obs(0, "drive_tires", "shoulder wear across the outer ribs")])[0]
        others = [_obs(0, "drive_tires", "shoulder wear across the outer ribs"),
                  _obs(1, "drive_tires", "shoulder wear across the outer ribs"),
                  _obs(1, "drive_tires", "shoulder wear on the outer ribs")]
        self.assertEqual(twin_fp.matches_any(cluster, others), 2)

    def test_severity_rank_uses_the_products_own_table(self):
        cluster = twin_fp.cluster_observations([
            _obs(0, "drive_tires", "shoulder wear across the outer ribs", "major"),
            _obs(1, "drive_tires", "shoulder wear across the outer ribs", "minor"),
            _obs(2, "drive_tires", "shoulder wear across the outer ribs", "minor")])[0]
        self.assertEqual(cluster.severity_rank, 1.0)

    def test_family_falls_back_to_the_component_id(self):
        # app/condition.py belongs to another workstream. Until it lands the
        # partition is component identity, which is STRICTER, and the artifact
        # has to say which was in force.
        self.assertIn(twin_fp.family_partition_name(),
                      ("app.condition", "component-identity"))
        self.assertTrue(twin_fp.family_of("drive_tires"))


class TwinScoring(unittest.TestCase):
    """End to end through a synthetic cache: no images, no provider, no torch."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="eval-score-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.cache = evcache.ResponseCache(self.root)
        self.model = "fake-model"

    def _plan(self, pairs, samples=3):
        return twin_fp.Plan(name="twin_fp", tier="standard", seed=7, samples=samples,
                            pairs=pairs, calls=[], params={})

    def _stock(self, pair, variant, sample, observations):
        key = f"{pair.unit}-{variant}-{sample}"
        body = {"shows": "a photo", "legible": True, "confidence": 0.8,
                "observations": observations}
        self.cache.put(self.model, key, {"text": json.dumps(body)})
        return {"unit": pair.unit, "key": key, "variant": variant, "sample": sample,
                "view": pair.view, "listing_id": pair.listing_id,
                "image_index": pair.image_index, "ok": True, "error": ""}

    @staticmethod
    def _finding(component, text, severity="minor", impact="low", confidence=0.8):
        return {"component": component, "observation": text, "severity": severity,
                "price_impact": impact, "confidence": confidence}

    def _metric(self, result, name):
        return next(m for m in result.metrics if m.name == name)

    def test_a_twin_only_finding_is_a_false_positive(self):
        pairs = [_pair(index=i, listing_id=f"veh{i}") for i in range(4)]
        records = []
        real = self._finding("steer_tires", "even tread wear across the steer tire ribs")
        fabricated = self._finding("chassis_frame",
                                   "heavy corrosion pitting along the chassis rail web")
        for pair in pairs:
            for sample in range(3):
                records.append(self._stock(pair, "original", sample, [real]))
                records.append(self._stock(pair, "degraded", sample, [real, fabricated]))
        result = twin_fp.score(records, self._plan(pairs), self.cache, self.model)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(self._metric(result, "twin_fp_rate").value, 0.5)
        self.assertAlmostEqual(self._metric(result, "twin_only_per_photo").value, 1.0)
        self.assertAlmostEqual(self._metric(result, "original_only_rate").value, 0.0)

    def test_a_finding_seen_once_on_the_original_is_not_a_false_positive(self):
        # The strict rule is ZERO supporting original samples, not zero
        # surviving original clusters - one mention on the clean photo is enough
        # to say the model did not invent it.
        pairs = [_pair(index=0, listing_id="veh0")]
        text = "heavy corrosion pitting along the chassis rail web"
        records = [self._stock(pairs[0], "original", 0, [self._finding("chassis_frame", text)])]
        records += [self._stock(pairs[0], "original", s, []) for s in (1, 2)]
        records += [self._stock(pairs[0], "degraded", s,
                                [self._finding("chassis_frame", text)]) for s in range(3)]
        result = twin_fp.score(records, self._plan(pairs), self.cache, self.model)
        self.assertEqual(self._metric(result, "twin_only_per_photo").value, 0.0)

    def test_a_single_sample_finding_does_not_meet_the_quorum(self):
        pairs = [_pair(index=0, listing_id="veh0")]
        fabricated = self._finding("chassis_frame", "heavy corrosion along the rail web")
        records = [self._stock(pairs[0], "original", s, []) for s in range(3)]
        records.append(self._stock(pairs[0], "degraded", 0, [fabricated]))
        records += [self._stock(pairs[0], "degraded", s, []) for s in (1, 2)]
        result = twin_fp.score(records, self._plan(pairs), self.cache, self.model)
        self.assertEqual(self._metric(result, "twin_only_per_photo").value, 0.0)

    def test_severity_inflation_is_signed_and_paired(self):
        pairs = [_pair(index=i, listing_id=f"veh{i}") for i in range(5)]
        text = "shoulder wear across the outer ribs of the drive tire"
        records = []
        for pair in pairs:
            for sample in range(3):
                records.append(self._stock(pair, "original", sample,
                                           [self._finding("drive_tires", text, "minor")]))
                records.append(self._stock(pair, "degraded", sample,
                                           [self._finding("drive_tires", text, "moderate")]))
        result = twin_fp.score(records, self._plan(pairs), self.cache, self.model)
        inflation = self._metric(result, "severity_inflation")
        self.assertAlmostEqual(inflation.value, 1.0)
        self.assertIsNotNone(inflation.ci)
        self.assertTrue(inflation.headline)
        gate = next(g for g in result.gates if g.name == "twin_severity_inflation")
        self.assertFalse(gate.passed)

    def test_identical_halves_score_zero_inflation(self):
        pairs = [_pair(index=i, listing_id=f"veh{i}") for i in range(4)]
        text = "shoulder wear across the outer ribs of the drive tire"
        records = []
        for pair in pairs:
            for sample in range(3):
                for variant in ("original", "degraded"):
                    records.append(self._stock(pair, variant, sample,
                                               [self._finding("drive_tires", text)]))
        result = twin_fp.score(records, self._plan(pairs), self.cache, self.model)
        self.assertEqual(self._metric(result, "severity_inflation").value, 0.0)
        self.assertEqual(self._metric(result, "twin_fp_rate").value, 0.0)
        gate = next(g for g in result.gates if g.name == "twin_severity_inflation")
        self.assertTrue(gate.passed)

    def test_the_caveats_travel_in_the_artifact(self):
        pairs = [_pair(index=0, listing_id="veh0")]
        records = [self._stock(pairs[0], v, s, []) for v in ("original", "degraded")
                   for s in range(3)]
        result = twin_fp.score(records, self._plan(pairs), self.cache, self.model)
        blob = json.dumps(result.to_dict())
        self.assertIn("LESS information", blob)
        self.assertIn("LOWER BOUND", blob)
        self.assertIn("REAL pixels", blob)

    def test_a_one_sample_run_says_its_quorum_degenerated(self):
        pairs = [_pair(index=0, listing_id="veh0")]
        records = [self._stock(pairs[0], v, 0, []) for v in ("original", "degraded")]
        result = twin_fp.score(records, self._plan(pairs, samples=1), self.cache,
                               self.model)
        self.assertIn("quorum", " ".join(result.caveats))

    def test_no_usable_samples_is_skipped_not_zero(self):
        pairs = [_pair(index=0, listing_id="veh0")]
        result = twin_fp.score([], self._plan(pairs), self.cache, self.model)
        self.assertEqual(result.status, "skipped")

    def test_truck_vocab_share_is_a_diagnostic_only(self):
        pairs = [_pair(index=i, listing_id=f"veh{i}") for i in range(3)]
        records = []
        for pair in pairs:
            for sample in range(3):
                for variant in ("original", "degraded"):
                    records.append(self._stock(pair, variant, sample, [
                        self._finding("drive_tires", "shoulder wear on the outer ribs")]))
        result = twin_fp.score(records, self._plan(pairs), self.cache, self.model)
        metric = self._metric(result, "truck_vocab_share")
        self.assertEqual(metric.basis, scorecard.DIAGNOSTIC)
        self.assertIn("bench.py", metric.note)


# --- planning, budgets, stubs ----------------------------------------------

class Budgets(unittest.TestCase):
    def test_calls_per_vehicle_tracks_config_not_the_design(self):
        expected = 1 + config.MAX_EVIDENCE_PHOTOS * config.CLOSEUP_SAMPLES + 1 + 1
        self.assertEqual(suite_base.calls_per_vehicle(), expected)
        self.assertNotEqual(suite_base.calls_per_vehicle(), 18,
                            "the design's 18 predates CLOSEUP_SAMPLES")

    def test_smoke_reads_each_photo_once(self):
        self.assertEqual(suite_base.samples_for("smoke"), 1)
        self.assertEqual(suite_base.samples_for("standard"), config.CLOSEUP_SAMPLES)

    def test_every_suite_plans_at_every_tier(self):
        for name in DEFAULT_ORDER:
            for tier in ("smoke", "standard", "full"):
                plan = SUITES[name].plan(tier, seed=7, model_id="m",
                                         require_files=False, **_cheap(name))
                self.assertGreaterEqual(plan.n_calls, 0, f"{name}/{tier}")
                self.assertIn("name", plan.to_dict())

    def test_budgets_grow_with_the_tier(self):
        smoke = twin_fp.plan("smoke", seed=7, model_id="m", require_files=False,
                             resolve_keys=False)
        standard = twin_fp.plan("standard", seed=7, model_id="m", require_files=False,
                                resolve_keys=False)
        self.assertLess(smoke.n_calls, standard.n_calls)

    def test_twin_plan_calls_are_pairs_times_halves_times_samples(self):
        plan = twin_fp.plan("standard", seed=7, model_id="m", require_files=False,
                            resolve_keys=False)
        self.assertEqual(plan.n_calls, len(plan.pairs) * 2 * plan.samples)

    def test_twin_plan_resolves_every_key_offline(self):
        # Deliberately tiny: resolving a key re-encodes the JPEG, and the
        # property under test is uniqueness, which four photographs show as
        # well as twelve hundred do.
        plan = twin_fp.plan("standard", seed=7, model_id="m", n_vehicles=2,
                            per_vehicle=2)
        self.assertTrue(plan.calls)
        self.assertTrue(all(c.key for c in plan.calls))
        self.assertEqual(len({c.key for c in plan.calls}), len(plan.calls),
                         "two planned calls must never share a key")

    def test_panel_skips_until_reference_exists(self):
        from eval.suites import panel
        plan = SUITES["panel"].plan("smoke", seed=7, model_id="m")
        if not panel.available():
            self.assertEqual(SUITES["panel"].run(plan, None), [])
            result = SUITES["panel"].score([], plan, None, "m")
            self.assertEqual(result.status, "skipped")

    def test_implemented_suites_skip_empty_offline_inputs(self):
        for name in ("retest", "monotonic", "distribution"):
            plan = SUITES[name].plan("smoke", seed=7, model_id="m")
            result = SUITES[name].score([], plan, evcache.ResponseCache(self.id()), "m")
            self.assertEqual(result.status, "skipped", name)

    def test_shared_corpus_suites_budget_nothing_extra(self):
        for name in ("monotonic", "panel"):
            self.assertEqual(SUITES[name].plan("standard", seed=7, model_id="m").n_calls, 0)


def _cheap(name: str) -> dict:
    """Plan sizes that do not re-encode 1,200 JPEGs just to count them."""
    return {"resolve_keys": False} if name == "twin_fp" else {}


class DemoGateScoring(unittest.TestCase):
    @staticmethod
    def _record(case_id, appraisal, expect="ok|ok_with_requests"):
        return {"case_id": case_id, "expected": expect, "status": appraisal["status"],
                "missing": False, "appraisal": appraisal}

    @staticmethod
    def _appraisal(*, status="ok", grade="good", multiplier=1.0, coverage=0.9,
                   prices=True, widened=False, correction=""):
        low, high = ((80.0, 120.0) if widened else (90.0, 110.0))
        return {
            "status": status,
            "evidence": {"condition_grade": grade,
                         "condition": {"coverage": coverage}},
            "reconcile": {"corrections": ([{"kind": correction}] if correction else [])},
            "price": ({"ok": True, "point": 100.0, "low": low, "high": high,
                       "baseline_low": 90.0, "baseline_high": 110.0,
                       "adjustment": {"multiplier": multiplier}} if prices else None),
        }

    def test_all_documented_checks_pass(self):
        records = [
            self._record("tr_clean", self._appraisal()),
            self._record("tr_phone", self._appraisal(grade="excellent", multiplier=0.98)),
            self._record("closeups_only",
                         self._appraisal(status="need_more_photos", grade="unknown",
                                         coverage=0.4, prices=False),
                         expect="need_more_photos"),
            self._record("odometer_lie",
                         self._appraisal(widened=True, correction="odometer_conflict")),
            self._record("unseen_brand", self._appraisal(widened=True)),
        ]
        result = demo_gate.score(records, demo_gate.plan("smoke"), None, "m")
        self.assertEqual(result.status, "ok")
        self.assertTrue(all(g.passed for g in result.gates))

    def test_missing_fixture_is_skipped_with_a_note(self):
        records = [{"case_id": "tr_clean", "expected": "ok", "missing": True}]
        result = demo_gate.score(records, demo_gate.plan("smoke"), None, "m")
        self.assertEqual(result.status, "skipped")
        self.assertIn("missing", " ".join(result.caveats).lower())


class RemainingSuiteScoring(unittest.TestCase):
    @staticmethod
    def _corpus_record(index, demerit, grade, model_grade, *, market="TR",
                       quality="dealer"):
        return {
            "listing_id": f"v{index}", "source_key": "tr_truckmarket",
            "market": market, "make": "Ford", "year": 2024 - index,
            "km": 100_000 + index * 50_000, "quality_bucket": quality,
            "capture_quality": 0.95 - index * 0.03,
            "appraisal": {
                "evidence": {
                    "condition_grade": grade,
                    "condition_grade_model": model_grade,
                    "condition": {"demerit": demerit},
                }
            },
        }

    def test_distribution_returns_histogram_prior_and_signed_bias(self):
        records = [
            self._corpus_record(0, 0.1, "excellent", "good"),
            self._corpus_record(1, 0.4, "good", "good"),
            self._corpus_record(2, 2.0, "fair", "poor"),
            self._corpus_record(3, 6.5, "poor", "fair"),
        ]
        result = distribution.score(
            records, distribution.plan("smoke", model_id="m"), None, "m")
        metrics = {m.name: m for m in result.metrics}
        self.assertEqual(result.status, "ok")
        self.assertEqual(metrics["grade_good"].value, 0.25)
        self.assertEqual(metrics["grade_agreement_model_vs_rollup"].value, 0.25)
        self.assertEqual(metrics["model_grade_bias"].value, -0.25)
        self.assertIn("market", result.strata)
        self.assertEqual(len(result.gates), 2)

    def test_monotonic_scores_distribution_records_and_prints_caveat(self):
        records = [
            self._corpus_record(i, float(i), grade, grade)
            for i, grade in enumerate(("excellent", "good", "fair", "poor", "poor"))
        ]
        plan = monotonic.plan("smoke", seed=3, model_id="m")
        plan.params["null_draws"] = 50
        self.assertIs(monotonic.run(plan, None, distribution_records=records)[0],
                      records[0])
        result = monotonic.score(records, plan, None, "m")
        metrics = {m.name: m for m in result.metrics}
        self.assertEqual(metrics["rho_km"].value, 1.0)
        self.assertIn("NEGATIVE", " ".join(result.caveats))
        self.assertEqual(result.detail["shared_with"], "distribution")

    def test_retest_scores_cached_photo_and_vehicle_repeats(self):
        root = Path(tempfile.mkdtemp(prefix="eval-retest-"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        cache = evcache.ResponseCache(root)
        finding = {
            "component": "drive_tires",
            "observation": "shoulder wear across the outer ribs",
            "severity": "minor", "price_impact": "low", "confidence": 0.8,
        }
        records = []
        for repeat, severity in enumerate(("minor", "moderate", "minor")):
            body = {"shows": "tire", "legible": True, "confidence": 0.8,
                    "observations": [{**finding, "severity": severity}]}
            key = f"photo-{repeat}"
            cache.put("m", key, {"text": json.dumps(body)})
            records.append({
                "tier": "photo", "unit": "v:000", "listing_id": "v",
                "image_index": 0, "view": "tire_wheel", "repeat": repeat,
                "key": key, "ok": True,
            })
        for repeat, (grade, model_grade, multiplier) in enumerate((
                ("good", "good", 0.99), ("good", "fair", 0.97), ("fair", "fair", 0.95))):
            records.append({
                "tier": "vehicle", "unit": "v", "repeat": repeat, "ok": True,
                "appraisal": {
                    "evidence": {"condition_grade": grade,
                                 "condition_grade_model": model_grade},
                    "price": {"adjustment": {"multiplier": multiplier}},
                },
            })
        plan = retest.plan("smoke", seed=7, model_id="m")
        plan.params.update(photo_repeats=3, vehicle_repeats=3)
        result = retest.score(records, plan, cache, "m")
        metrics = {m.name: m for m in result.metrics}
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(metrics["severity_disagreement"].value, 2 / 3, places=4)
        self.assertAlmostEqual(metrics["grade_instability"].value, 2 / 3, places=4)
        self.assertGreater(metrics["multiplier_sd_log"].value, 0)
        self.assertEqual(result.detail["matching"], "twin_fp.cluster_observations")

    def test_panel_scores_pipeline_against_consensus_and_loo_ceiling(self):
        from eval.suites import panel
        records = [
            {
                "listing_id": "v0", "split": "test",
                "panel": {
                    "listing_id": "v0", "grade": "good",
                    "grade_votes": ["good", "good", "fair"],
                    "findings": [{
                        "family": "drive_tires",
                        "severity": "moderate",
                        "impact": "medium",
                        "observation": "shoulder wear across the outer ribs",
                        "votes": 2, "n_panelists": 3,
                    }],
                },
                "appraisal": {
                    "evidence": {
                        "condition_grade": "good",
                        "issues": [{
                            "component": "drive_tires",
                            "family": "drive_tires",
                            "severity": "minor",
                            "observation": "shoulder wear across the outer ribs",
                        }],
                    }
                },
            },
            {
                "listing_id": "v1", "split": "test",
                "panel": {
                    "listing_id": "v1", "grade": "fair",
                    "grade_votes": ["fair", "fair", "poor"],
                    "findings": [],
                },
                "appraisal": {
                    "evidence": {"condition_grade": "good", "issues": []},
                },
            },
        ]
        result = panel.score(records, panel.plan("smoke", model_id="m"), None, "m")
        metrics = {m.name: m for m in result.metrics}
        self.assertEqual(result.status, "ok")
        self.assertEqual(metrics["grade_exact"].value, 0.5)
        self.assertEqual(metrics["grade_within_one"].value, 1.0)
        self.assertEqual(metrics["grade_bias"].value, 0.5)
        self.assertEqual(metrics["finding_recall"].value, 1.0)
        self.assertEqual(metrics["severity_mae"].value, 1.0)
        self.assertIsNotNone(result.detail["ceiling"]["grade_exact"])


class ReplayAndSweep(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="eval-replay-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_parse_values(self):
        self.assertEqual(replay.parse_values("0.2,0.3"), [0.2, 0.3])
        self.assertEqual(replay.parse_values("0.1:0.3:0.1"), [0.1, 0.2, 0.3])

    def test_sweep_restores_the_constant_even_when_it_raises(self):
        before = twin_fp.TWIN_MATCH
        with self.assertRaises(Exception):
            replay.sweep(self.root / "nope", "eval.suites.twin_fp.TWIN_MATCH",
                         [0.1, 0.2])
        self.assertEqual(twin_fp.TWIN_MATCH, before,
                         "a failed sweep must not leave the product retuned")

    def test_sweep_rejects_an_unknown_attribute(self):
        with self.assertRaises(ValueError):
            replay._resolve("eval.suites.twin_fp.NOT_A_CONSTANT")

    def test_round_trip_rebuilds_the_plan_from_the_index(self):
        pairs = [_pair(index=0, listing_id="veh0")]
        plan = twin_fp.Plan(name="twin_fp", tier="standard", seed=7, samples=3,
                            pairs=pairs, calls=[], params={})
        run_dir = self.root / "run"
        replay.write_raw(run_dir, "twin_fp", plan, [{"unit": pairs[0].unit}])
        blob = json.loads(replay.raw_path(run_dir, "twin_fp").read_text())
        rebuilt = replay._rebuild_plan("twin_fp", blob)
        self.assertEqual(rebuilt.samples, 3)
        self.assertEqual([p.unit for p in rebuilt.pairs], [pairs[0].unit])
        self.assertEqual(rebuilt.pairs[0].degradations, ["motion_blur"])


class CommandLine(unittest.TestCase):
    @staticmethod
    def _main(argv):
        from eval.__main__ import main
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(argv)
        return code, out.getvalue()

    def test_estimate_needs_no_credentials(self):
        code, text = self._main(["--estimate", "--tier", "smoke", "--suite",
                                 "demo_gate", "--model", "test-model"])
        self.assertEqual(code, 0)
        self.assertIn("vision calls", text)

    def test_a_big_run_refuses_without_yes(self):
        code, text = self._main(["--tier", "full", "--suite", "distribution",
                                 "--model", "test-model"])
        self.assertEqual(code, 2)
        self.assertIn("refusing to start", text)

    def test_unknown_suite_is_rejected(self):
        from eval.__main__ import chosen_suites
        with self.assertRaises(SystemExit):
            chosen_suites(["not_a_suite"])

    def test_model_id_resolves_from_config_without_probing(self):
        from eval.__main__ import default_model_id
        self.assertEqual(default_model_id("openai"), config.OPENAI_MODEL)
        self.assertEqual(default_model_id("anthropic"), config.ANTHROPIC_MODEL)


class EndToEndOnFakeBackend(unittest.TestCase):
    """plan -> run -> write_raw -> score -> replay, on real corpus pixels.

    The only test that exercises `image_digest` against files on disk and the
    agreement between the key the plan computed and the key the client used.
    Two vehicles, two photos, one sample: eight calls, all fake.
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="eval-e2e-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.cache = evcache.ResponseCache(self.root / "cache")
        self.inner = FakeBackend(json.dumps({
            "shows": "a drive tire", "legible": True, "confidence": 0.7,
            "observations": [{"component": "drive_tires",
                              "observation": "shoulder wear across the outer ribs",
                              "severity": "minor", "price_impact": "low",
                              "confidence": 0.7}]}))
        self.client = evcache.CachedBackend(self.inner, cache=self.cache,
                                            model_id="fake-model")

    def _plan(self):
        return twin_fp.plan("standard", seed=7, model_id="fake-model", effort="high",
                            samples=1, n_vehicles=2, per_vehicle=2)

    def test_run_then_score_then_replay(self):
        plan = self._plan()
        if not plan.pairs or not all(p.exists() for p in plan.pairs):
            self.skipTest("corpus images are not on this disk (data/images/ is gitignored)")
        records = twin_fp.run(plan, self.client, concurrency=4)
        self.assertEqual(len(records), plan.n_calls)
        self.assertTrue(all(r["ok"] for r in records), [r["error"] for r in records])
        self.assertTrue(all(r["planned_key_matched"] for r in records),
                        "the plan and the client must derive the same key, or "
                        "--estimate is measuring a cache nobody writes to")
        self.assertEqual(self.inner.calls, plan.n_calls)

        result = twin_fp.score(records, plan, self.cache, "fake-model")
        self.assertEqual(result.status, "ok")
        # Identical canned text on both halves: no fabrication, no inflation.
        by_name = {m.name: m.value for m in result.metrics}
        self.assertEqual(by_name["twin_fp_rate"], 0.0)
        self.assertEqual(by_name["severity_inflation"], 0.0)

        run_dir = self.root / "runs" / "run-a"
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)
        replay.write_raw(run_dir, "twin_fp", plan, records)
        card = scorecard.Scorecard.start(tier="standard", seed=7,
                                         model={"backend": "fake", "id": "fake-model"})
        card.add(result)
        card.calls = self.cache.stats.to_dict()
        card.run_id = "run-a"
        card.write(self.root / "runs")

        before_calls = self.inner.calls
        replayed = replay.replay(run_dir, cache=self.cache, suites=["twin_fp"])
        self.assertEqual(self.inner.calls, before_calls, "a replay must cost nothing")
        self.assertEqual(replayed.calls["new"], 0)
        replayed_metrics = {m["name"]: m["value"]
                            for m in replayed.to_dict()["suites"]["twin_fp"]["metrics"]}
        self.assertEqual(replayed_metrics["twin_fp_rate"], by_name["twin_fp_rate"])

    def test_a_second_run_is_entirely_cached(self):
        plan = self._plan()
        if not plan.pairs or not all(p.exists() for p in plan.pairs):
            self.skipTest("corpus images are not on this disk")
        twin_fp.run(plan, self.client, concurrency=4)
        first = self.inner.calls
        twin_fp.run(plan, self.client, concurrency=4)
        self.assertEqual(self.inner.calls, first, "identical plan, zero new calls")
        self.assertEqual(self.cache.stats.cached, first)

    def test_a_prompt_change_busts_the_cache_visibly(self):
        plan = self._plan()
        if not plan.pairs or not all(p.exists() for p in plan.pairs):
            self.skipTest("corpus images are not on this disk")
        twin_fp.run(plan, self.client, concurrency=4)
        # CLOSEUP_PROMPT became CLOSEUP_INVARIANT when the prompt was split into
        # invariant / per-appraisal / per-photo zones for prefix caching.
        first = self.inner.calls
        original = prompts.CLOSEUP_INVARIANT
        try:
            prompts.CLOSEUP_INVARIANT = original + "\nAn added instruction.\n"
            twin_fp.run(plan, self.client, concurrency=4)
        finally:
            prompts.CLOSEUP_INVARIANT = original
        self.assertEqual(self.inner.calls, 2 * first,
                         "a prompt edit must show up as a real bill, not a silent hit")
        self.assertEqual(self.cache.stats.cached, 0)


class Layout(unittest.TestCase):
    def test_the_cache_is_ignored_and_the_runs_are_not(self):
        import subprocess
        from app.config import REPO

        def ignored(rel: str) -> bool:
            return subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO).returncode == 0

        self.assertTrue(ignored("eval/cache/m/abc.json"),
                        "the response cache is rebuildable and large")
        self.assertFalse(ignored("eval/runs/x/scorecard.json"),
                         "scorecards are the record and must be committed")


if __name__ == "__main__":
    unittest.main()
