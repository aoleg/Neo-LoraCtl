"""Offline harness for loractl_core — no torch, no Forge, stdlib only.

Simulates the pinned build's OnlineLoRAPatch shape (Forge Neo 92b55e1b) and a
merge consumer, then exercises masks, curves, classification, filtering, and
per-step rewriting. The API-shape tests are the loud-failure guard the plan
requires: if Forge churns the patch layout again, phase-1 collection refuses
to adopt patches and these tests document what "correct" looked like.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import loractl_core as core


class FakeOnlineLoRAPatch:
    """Mirror of backend/patcher/base.py::OnlineLoRAPatch at 92b55e1b."""

    def __init__(self, key, patch_tuple):
        self.key = key
        self.patch = [patch_tuple]

    def __call__(self, weight):
        # Fake merge: consume the CURRENT tuple strength, like
        # merge_lora_to_weight re-reading self.patch per forward.
        strength = self.patch[0][0]
        return weight + strength

    @staticmethod
    def make(key, strength):
        return FakeOnlineLoRAPatch(key, (strength, object(), 1.0, None, None))


def krea2_keys():
    """Representative patch keys in the real format model_lora_keys_unet
    produces for Krea 2 (model state-dict keys)."""
    keys = []
    for i in (0, 5, 13, 27):
        keys.append(f"diffusion_model.blocks.{i}.attn.qkv.weight")
        keys.append(f"diffusion_model.blocks.{i}.mlp.0.weight")
    keys += [
        "diffusion_model.txtfusion.layerwise_blocks.0.attn.qkv.weight",
        "diffusion_model.txtfusion.refiner_blocks.1.mlp.0.weight",
        "diffusion_model.txtfusion.projector.weight",
        "diffusion_model.first.weight",
        "diffusion_model.last.linear.weight",
        "diffusion_model.tmlp.0.weight",
        "diffusion_model.txtmlp.1.weight",
        "diffusion_model.tproj.1.weight",
    ]
    return keys


class TestApiShape(unittest.TestCase):
    def test_valid_object_passes(self):
        obj = FakeOnlineLoRAPatch.make("k", 1.0)
        self.assertIsNone(core.validate_patch_object(obj))

    def test_six_tuple_rejected(self):
        obj = FakeOnlineLoRAPatch("k", (1.0, object(), 1.0, None, None, True))
        self.assertIn("arity", core.validate_patch_object(obj))

    def test_missing_patch_list_rejected(self):
        class Legacy:  # pre-2026-07 style object without .patch
            pass
        self.assertIsNotNone(core.validate_patch_object(Legacy()))

    def test_non_numeric_strength_rejected(self):
        obj = FakeOnlineLoRAPatch("k", (None, object(), 1.0, None, None))
        self.assertIsNotNone(core.validate_patch_object(obj))


class TestBlockMask(unittest.TestCase):
    COUNT = 28

    def test_full_and_zero_contrast_are_identity(self):
        for preset in core.BLOCK_PRESETS:
            for modifier in core.MODIFIERS:
                mask = core.build_block_mask(self.COUNT, "FULL", modifier, 1.0)
                self.assertEqual(mask, [1.0] * self.COUNT)
                mask = core.build_block_mask(self.COUNT, preset, modifier, 0.0)
                self.assertEqual(mask, [1.0] * self.COUNT)

    def test_bounds_for_all_combinations(self):
        for preset in ("COMPOSITION", "CHARACTER", "STYLE"):
            for modifier in core.MODIFIERS:
                for contrast in (0.25, 0.5, 1.0):
                    mask = core.build_block_mask(self.COUNT, preset, modifier, contrast)
                    self.assertEqual(len(mask), self.COUNT)
                    for v in mask:
                        self.assertGreaterEqual(v, 1.0 - contrast - 1e-9)
                        self.assertLessEqual(v, 1.0 + 1e-9)

    def test_emphasize_composition_shape(self):
        mask = core.build_block_mask(self.COUNT, "COMPOSITION", "Emphasize", 1.0)
        self.assertAlmostEqual(mask[0], 1.0, places=6)   # inside zone
        self.assertAlmostEqual(mask[27], 0.0, places=6)  # deep outside
        self.assertLess(mask[12], mask[7])               # falls across shoulder

    def test_suppress_mirrors_emphasize(self):
        for preset in ("COMPOSITION", "CHARACTER", "STYLE"):
            emph = core.build_block_mask(self.COUNT, preset, "Emphasize", 0.8)
            supp = core.build_block_mask(self.COUNT, preset, "Suppress", 0.8)
            for e, s in zip(emph, supp):
                self.assertAlmostEqual(e + s, 2.0 - 0.8, places=6)

    def test_character_zone_is_interior(self):
        mask = core.build_block_mask(self.COUNT, "CHARACTER", "Emphasize", 1.0)
        self.assertAlmostEqual(mask[14], 1.0, places=3)  # zone center
        self.assertAlmostEqual(mask[0], 0.0, places=6)
        self.assertAlmostEqual(mask[27], 0.0, places=6)

    def test_style_zone_is_tail(self):
        mask = core.build_block_mask(self.COUNT, "STYLE", "Emphasize", 1.0)
        self.assertAlmostEqual(mask[27], 1.0, places=6)
        self.assertAlmostEqual(mask[0], 0.0, places=6)


class TestKeyClassification(unittest.TestCase):
    def test_detect_arch(self):
        self.assertEqual(core.detect_arch(krea2_keys()), core.ARCH_FLAT)
        self.assertEqual(core.detect_arch(["diffusion_model.double_blocks.3.img_mod.lin.weight"]), core.ARCH_FLUX)
        self.assertEqual(core.detect_arch(["diffusion_model.input_blocks.4.1.proj_in.weight"]), core.ARCH_SD)
        self.assertEqual(core.detect_arch(["diffusion_model.first.weight"]), core.ARCH_UNKNOWN)

    def test_krea2_blocks_and_exclusions(self):
        arch = core.ARCH_FLAT
        self.assertEqual(core.classify_key("diffusion_model.blocks.0.attn.qkv.weight", arch), 0)
        self.assertEqual(core.classify_key("diffusion_model.blocks.27.mlp.0.weight", arch), 27)
        # The collision that broke lora-block-weight-neo's mapping:
        self.assertIsNone(core.classify_key("diffusion_model.txtfusion.layerwise_blocks.0.attn.qkv.weight", arch))
        self.assertIsNone(core.classify_key("diffusion_model.txtfusion.refiner_blocks.1.mlp.0.weight", arch))
        for key in ("diffusion_model.first.weight", "diffusion_model.last.linear.weight",
                    "diffusion_model.tmlp.0.weight", "diffusion_model.txtmlp.1.weight",
                    "diffusion_model.tproj.1.weight"):
            self.assertIsNone(core.classify_key(key, arch), key)

    def test_bias_keys_classify_too(self):
        self.assertEqual(core.classify_key("diffusion_model.blocks.5.attn.qkv.bias", core.ARCH_FLAT), 5)

    def test_flux_indexing(self):
        arch = core.ARCH_FLUX
        self.assertEqual(core.classify_key("diffusion_model.double_blocks.18.img_attn.qkv.weight", arch, num_double=19), 18)
        self.assertEqual(core.classify_key("diffusion_model.single_blocks.0.linear1.weight", arch, num_double=19), 19)
        self.assertEqual(core.classify_key("diffusion_model.single_blocks.37.linear1.weight", arch, num_double=19), 56)

    def test_sd_indexing(self):
        arch = core.ARCH_SD
        self.assertEqual(core.classify_key("diffusion_model.input_blocks.0.0.weight", arch), 1)
        self.assertEqual(core.classify_key("diffusion_model.middle_block.1.proj_in.weight", arch), 13)
        self.assertEqual(core.classify_key("diffusion_model.output_blocks.11.1.proj_out.weight", arch), 25)
        self.assertEqual(core.classify_key("diffusion_model.time_embed.0.weight", arch), 0)

    def test_infer_block_layout(self):
        count, num_double = core.infer_block_layout(krea2_keys(), core.ARCH_FLAT)
        self.assertEqual((count, num_double), (28, 0))
        flux = ["diffusion_model.double_blocks.7.a.weight", "diffusion_model.single_blocks.23.b.weight"]
        self.assertEqual(core.infer_block_layout(flux, core.ARCH_FLUX), (32, 8))
        self.assertEqual(core.infer_block_layout([], core.ARCH_SD), (26, 0))


class TestTimeCurve(unittest.TestCase):
    def test_flat_and_zero_contrast(self):
        self.assertEqual(core.TimeCurve("FLAT", "Emphasize", 1.0).factor(0.5), 1.0)
        self.assertEqual(core.TimeCurve("COMPOSITION", "Emphasize", 0.0).factor(0.5), 1.0)

    def test_composition_emphasize(self):
        curve = core.TimeCurve("COMPOSITION", "Emphasize", 1.0)
        self.assertAlmostEqual(curve.factor(1.0), 1.0, places=6)    # start of run
        self.assertAlmostEqual(curve.factor(0.98), 1.0, places=6)
        self.assertAlmostEqual(curve.factor(0.90), 0.5, places=6)   # boundary midpoint
        self.assertAlmostEqual(curve.factor(0.20), 0.0, places=6)   # late steps
        self.assertAlmostEqual(curve.factor(0.0), 0.0, places=6)

    def test_detail_emphasize(self):
        curve = core.TimeCurve("DETAIL", "Emphasize", 1.0)
        self.assertAlmostEqual(curve.factor(0.10), 1.0, places=6)
        self.assertAlmostEqual(curve.factor(0.50), 0.5, places=6)
        self.assertAlmostEqual(curve.factor(0.95), 0.0, places=6)

    def test_character_is_interior(self):
        curve = core.TimeCurve("CHARACTER", "Emphasize", 1.0)
        self.assertAlmostEqual(curve.factor(0.70), 1.0, places=6)
        self.assertAlmostEqual(curve.factor(1.0), 0.0, places=6)
        self.assertAlmostEqual(curve.factor(0.10), 0.0, places=6)

    def test_suppress_mirrors_emphasize(self):
        e = core.TimeCurve("CHARACTER", "Emphasize", 0.6)
        s = core.TimeCurve("CHARACTER", "Suppress", 0.6)
        for sig in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
            self.assertAlmostEqual(e.factor(sig) + s.factor(sig), 2.0 - 0.6, places=6)

    def test_adjacent_zones_partition(self):
        """COMPOSITION + CHARACTER + DETAIL windows sum to 1 at every sigma
        (same boundaries and transition width), so emphasize presets tile the
        run without gaps or double coverage."""
        curves = [core.TimeCurve(p, "Emphasize", 1.0) for p in ("COMPOSITION", "CHARACTER", "DETAIL")]
        for sig in (0.05, 0.35, 0.5, 0.62, 0.88, 0.9, 0.93, 1.0):
            total = sum(c.factor(sig) for c in curves)
            self.assertAlmostEqual(total, 1.0, places=6, msg=f"sigma={sig}")

    def test_bounds(self):
        for preset in ("COMPOSITION", "CHARACTER", "DETAIL"):
            for modifier in core.MODIFIERS:
                curve = core.TimeCurve(preset, modifier, 0.7)
                for sig in (0.0, 0.2, 0.5, 0.9, 1.0):
                    f = curve.factor(sig)
                    self.assertGreaterEqual(f, 0.3 - 1e-9)
                    self.assertLessEqual(f, 1.0 + 1e-9)

    def test_normalize_sigma(self):
        self.assertAlmostEqual(core.normalize_sigma(0.9, 1.0), 0.9)       # flow
        self.assertAlmostEqual(core.normalize_sigma(14.6, 14.6), 1.0)     # epsilon
        self.assertAlmostEqual(core.normalize_sigma(7.3, 14.6), 0.5)


class TestFilter(unittest.TestCase):
    PATTERNS = core.parse_patterns(core.DEFAULT_EXCLUDE_PATTERNS)

    def test_default_excludes_accelerators(self):
        for name in ("Krea2_Turbo_v1.safetensors", "sdxl_lightning_4step.safetensors",
                     "Hyper-SD15.safetensors", "LCM_lora.safetensors", "dmd2_fp16.safetensors"):
            self.assertFalse(core.lora_is_scheduled(f"C:/loras/{name}", self.PATTERNS, "exclude"), name)

    def test_default_schedules_normal_loras(self):
        self.assertTrue(core.lora_is_scheduled("/x/my_character_v3.safetensors", self.PATTERNS, "exclude"))

    def test_include_mode(self):
        self.assertTrue(core.lora_is_scheduled("my_character.safetensors", ["character"], "include"))
        self.assertFalse(core.lora_is_scheduled("my_style.safetensors", ["character"], "include"))
        self.assertFalse(core.lora_is_scheduled("anything.safetensors", [], "include"))

    def test_case_insensitive(self):
        self.assertFalse(core.lora_is_scheduled("KREA-TURBO.safetensors",
                                                core.parse_patterns("Turbo"), "exclude"))


class TestScheduleSet(unittest.TestCase):
    def _wrapper_dict_with_two_loras(self):
        """Simulate the stock loader adding a baked-era LoRA first (not ours),
        then our scheduled LoRA, into weight_wrapper_patches."""
        wrappers = {
            "diffusion_model.blocks.0.attn.qkv.weight": [FakeOnlineLoRAPatch.make("a", 0.7)],
        }
        before = core.ScheduleSet.snapshot_counts(wrappers)
        wrappers["diffusion_model.blocks.0.attn.qkv.weight"].append(FakeOnlineLoRAPatch.make("b", 0.5))
        wrappers["diffusion_model.blocks.20.mlp.0.weight"] = [FakeOnlineLoRAPatch.make("b", 0.5)]
        wrappers["diffusion_model.txtfusion.projector.weight"] = [FakeOnlineLoRAPatch.make("b", 0.5)]
        return wrappers, before

    def _block_factor(self, key):
        idx = core.classify_key(key, core.ARCH_FLAT)
        mask = core.build_block_mask(28, "STYLE", "Emphasize", 1.0)
        return 1.0 if idx is None else mask[idx]

    def test_collect_only_new_objects(self):
        wrappers, before = self._wrapper_dict_with_two_loras()
        sched = core.ScheduleSet()
        errors = sched.collect(before, wrappers, self._block_factor)
        self.assertEqual(errors, [])
        self.assertEqual(len(sched.entries), 3)
        untouched = wrappers["diffusion_model.blocks.0.attn.qkv.weight"][0]
        self.assertNotIn(untouched, [e.obj for e in sched.entries])

    def test_rewrite_reaches_fake_merge(self):
        wrappers, before = self._wrapper_dict_with_two_loras()
        sched = core.ScheduleSet()
        sched.collect(before, wrappers, self._block_factor)
        sched.apply_time_factor(0.25)
        # blocks.0 is deep outside STYLE zone -> block factor 0 -> strength 0.
        ours_b0 = wrappers["diffusion_model.blocks.0.attn.qkv.weight"][1]
        self.assertAlmostEqual(ours_b0(0.0), 0.0, places=6)
        # blocks.20 is inside STYLE zone (factor ~1) -> 0.5 * 1 * 0.25.
        ours_b20 = wrappers["diffusion_model.blocks.20.mlp.0.weight"][0]
        self.assertAlmostEqual(ours_b20(0.0), 0.125, places=3)
        # Non-block key: block factor fixed at 1.0.
        ours_txt = wrappers["diffusion_model.txtfusion.projector.weight"][0]
        self.assertAlmostEqual(ours_txt(0.0), 0.125, places=6)
        # Foreign object untouched.
        self.assertAlmostEqual(wrappers["diffusion_model.blocks.0.attn.qkv.weight"][0](0.0), 0.7, places=6)

    def test_base_strength_survives_many_steps(self):
        wrappers, before = self._wrapper_dict_with_two_loras()
        sched = core.ScheduleSet()
        sched.collect(before, wrappers, self._block_factor)
        for tf in (0.1, 0.9, 0.0, 0.5, 1.0):
            sched.apply_time_factor(tf)
        ours_b20 = wrappers["diffusion_model.blocks.20.mlp.0.weight"][0]
        self.assertAlmostEqual(ours_b20(0.0), 0.5 * self._block_factor("diffusion_model.blocks.20.mlp.0.weight"), places=3)

    def test_collect_refuses_on_api_drift(self):
        wrappers = {"k": [FakeOnlineLoRAPatch("k", (1.0, object(), 1.0, None, None, True))]}
        sched = core.ScheduleSet()
        errors = sched.collect({}, wrappers, lambda key: 1.0)
        self.assertEqual(len(errors), 1)
        self.assertEqual(sched.entries, [])

    def test_flat_identity_oracle(self):
        """Core-level version of the phase-1 oracle: FULL blocks + FLAT time
        must leave every consumed strength exactly at its base."""
        wrappers, before = self._wrapper_dict_with_two_loras()
        sched = core.ScheduleSet()
        sched.collect(before, wrappers, lambda key: 1.0)  # FULL mask
        curve = core.TimeCurve("FLAT", "Emphasize", 1.0)
        for sigma in (1.0, 0.9567, 0.9045, 0.8403, 0.3109, 0.0):
            sched.apply_time_factor(curve.factor(core.normalize_sigma(sigma, 1.0)))
        for key, objs in wrappers.items():
            for obj in objs:
                self.assertIn(round(obj.patch[0][0], 6), (0.7, 0.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
