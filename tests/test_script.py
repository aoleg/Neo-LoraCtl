"""Offline harness for scripts/neo_loractl.py — a mini-Forge.

Fakes modules/gradio/networks with the semantics that matter on the pinned
build (92b55e1b): dict.copy() clone sharing, wrapper-list rebinding on add,
load_networks hash caching, and the real hook firing order:

    process -> before_process_batch -> [extra_networks.activate]
    -> process_batch -> process_before_every_sampling -> on_cfg per step
    -> postprocess
"""

import importlib.util
import os
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import loractl_core as core


# ---------------------------------------------------------------------------
# Fake gradio
# ---------------------------------------------------------------------------

class _FakeComponent:
    def __init__(self, *a, **kw):
        self.kwargs = kw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_gradio():
    gr = types.ModuleType("gradio")
    for name in ("Accordion", "Row", "Group", "Checkbox", "Dropdown", "Radio",
                 "Slider", "Textbox", "Markdown", "HTML", "Button"):
        setattr(gr, name, _FakeComponent)
    return gr


# ---------------------------------------------------------------------------
# Fake modules package
# ---------------------------------------------------------------------------

_cfg_callbacks = []


def _fake_modules():
    modules = types.ModuleType("modules")
    modules.__path__ = []

    scripts_mod = types.ModuleType("modules.scripts")

    class Script:
        pass

    scripts_mod.Script = Script
    scripts_mod.AlwaysVisible = object()
    scripts_mod.basedir = lambda: REPO

    shared_mod = types.ModuleType("modules.shared")
    shared_mod.sd_model = None

    callbacks_mod = types.ModuleType("modules.script_callbacks")
    callbacks_mod.on_cfg_denoiser = _cfg_callbacks.append

    class CFGDenoiserParams:
        def __init__(self, sigma):
            self.sigma = sigma

    callbacks_mod.CFGDenoiserParams = CFGDenoiserParams

    modules.scripts = scripts_mod
    modules.shared = shared_mod
    modules.script_callbacks = callbacks_mod
    return modules, scripts_mod, shared_mod, callbacks_mod


# ---------------------------------------------------------------------------
# Fake Forge patcher / networks (pinned-build semantics)
# ---------------------------------------------------------------------------

class FakeOnlineLoRAPatch:
    def __init__(self, key, patch_tuple):
        self.key = key
        self.patch = [patch_tuple]

    def __call__(self, weight):
        return weight + self.patch[0][0]


class FakePatcher:
    def __init__(self):
        self.patches = {}
        self.weight_wrapper_patches = {}

    def clone(self):
        n = FakePatcher()
        n.patches = {k: v[:] for k, v in self.patches.items()}
        n.weight_wrapper_patches = self.weight_wrapper_patches.copy()  # shares lists, like Forge
        return n

    def add_patches(self, keys, strength, online_mode, tuple_len=5):
        for key in keys:
            if tuple_len == 5:
                t = (strength, object(), 1.0, None, None)
            else:
                t = (strength, object(), 1.0, None, None, True)
            if online_mode:
                obj = FakeOnlineLoRAPatch(key, t)
                self.weight_wrapper_patches[key] = self.weight_wrapper_patches.get(key, []) + [obj]
            else:
                cur = self.patches.pop(key, [])
                cur.append(t)
                self.patches[key] = cur


LORA_KEYS = {
    "charA.safetensors": ["diffusion_model.blocks.0.attn.qkv.weight",
                          "diffusion_model.blocks.20.mlp.0.weight",
                          "diffusion_model.txtfusion.projector.weight"],
    "styleB.safetensors": ["diffusion_model.blocks.5.attn.qkv.weight",
                           "diffusion_model.blocks.27.mlp.0.weight"],
    "krea_turbo.safetensors": ["diffusion_model.blocks.1.attn.qkv.weight"],
}


def _fake_networks(tuple_len=5):
    networks = types.ModuleType("networks")
    networks.calls = []

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                             filename="default", online_mode=False):
        networks.calls.append({"filename": filename, "online_mode": online_mode,
                               "strength_clip": strength_clip})
        new_model = model.clone()
        new_model.add_patches(LORA_KEYS[os.path.basename(filename)], strength_model,
                              online_mode, tuple_len=tuple_len)
        return new_model, clip

    networks.load_lora_for_models = load_lora_for_models
    return networks


class FakeForgeObjects:
    def __init__(self, unet):
        self.unet = unet


class FakeSDModel:
    def __init__(self):
        self.current_lora_hash = None
        self.forge_objects = FakeForgeObjects(FakePatcher())
        self.forge_objects_original = FakeForgeObjects(FakePatcher())


class FakeSampler:
    def __init__(self, schedule):
        self._schedule = schedule
        self.get_sigmas = lambda p, steps: list(self._schedule)


KREA_SCHEDULE = [1.0, 0.9567, 0.9045, 0.8403, 0.7595, 0.6546, 0.5128, 0.3109, 0.0]


class FakeP:
    def __init__(self, sd_model, schedule=KREA_SCHEDULE):
        self.sd_model = sd_model
        self.sampler = FakeSampler(schedule)
        self.extra_generation_params = {}


# ---------------------------------------------------------------------------
# Environment bootstrap + generation driver
# ---------------------------------------------------------------------------

def _load_script(networks_mod):
    for name in list(sys.modules):
        if name in ("gradio", "networks", "modules", "modules.scripts",
                    "modules.shared", "modules.script_callbacks", "neo_loractl"):
            del sys.modules[name]
    _cfg_callbacks.clear()
    sys.modules["gradio"] = _fake_gradio()
    modules, scripts_mod, shared_mod, callbacks_mod = _fake_modules()
    sys.modules["modules"] = modules
    sys.modules["modules.scripts"] = scripts_mod
    sys.modules["modules.shared"] = shared_mod
    sys.modules["modules.script_callbacks"] = callbacks_mod
    sys.modules["networks"] = networks_mod

    path = os.path.join(REPO, "scripts", "neo_loractl.py")
    spec = importlib.util.spec_from_file_location("neo_loractl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["neo_loractl"] = mod
    spec.loader.exec_module(mod)
    return mod


UI_DEFAULTS = dict(enabled=True, time_preset="FLAT", time_modifier="Emphasize",
                   time_contrast=1.0, te_enabled=True, filter_mode="exclude",
                   filter_patterns=core.DEFAULT_EXCLUDE_PATTERNS, debug=False)


class Harness:
    """Drives fake generations through the real hook order."""

    def __init__(self, tuple_len=5):
        self.networks = _fake_networks(tuple_len=tuple_len)
        self.mod = _load_script(self.networks)
        self.script = self.mod.NeoLoraCtlScript()
        self.sd_model = FakeSDModel()
        sys.modules["modules.shared"].sd_model = self.sd_model

    def activate(self, p, lora_files, strengths=None):
        """Mini load_networks: hash caching + reset-to-original + loader loop."""
        strengths = strengths or [0.5] * len(lora_files)
        sig = str(list(zip(lora_files, strengths)))
        if self.sd_model.current_lora_hash == sig:
            return
        self.sd_model.current_lora_hash = sig
        self.sd_model.forge_objects.unet = self.sd_model.forge_objects_original.unet.clone()
        for f, s in zip(lora_files, strengths):
            result = sys.modules["networks"].load_lora_for_models(
                self.sd_model.forge_objects.unet, object(), {}, s, s, filename=f)
            if result is not None:
                self.sd_model.forge_objects.unet = result[0]

    def generate(self, lora_files, strengths=None, ui=None, steps_schedule=KREA_SCHEDULE):
        ui_args = dict(UI_DEFAULTS)
        ui_args.update(ui or {})
        p = FakeP(self.sd_model, schedule=steps_schedule)
        s = self.script
        s.process(p, **ui_args)
        s.before_process_batch(p)
        self.activate(p, lora_files, strengths)
        s.process_batch(p)
        s.process_before_every_sampling(p)
        sigmas = p.sampler.get_sigmas(p, len(steps_schedule) - 1)
        Params = sys.modules["modules.script_callbacks"].CFGDenoiserParams
        strengths_per_step = []
        for sigma in sigmas[:-1]:
            for cb in _cfg_callbacks:
                cb(Params(sigma))
            strengths_per_step.append(self.snapshot_strengths())
        s.postprocess(p, object())
        return p, strengths_per_step

    def snapshot_strengths(self):
        out = {}
        for key, objs in self.sd_model.forge_objects.unet.weight_wrapper_patches.items():
            for i, obj in enumerate(objs):
                out[f"{key}#{i}"] = obj.patch[0][0]
        return out

    def cls(self):
        return self.mod.NeoLoraCtlScript


class TestInterceptionRouting(unittest.TestCase):
    def test_disabled_passthrough(self):
        h = Harness()
        h.generate(["charA.safetensors"], ui={"enabled": False})
        self.assertEqual(h.networks.calls[0]["online_mode"], False)
        self.assertEqual(len(h.cls().sched.entries), 0)

    def test_scheduled_forced_online(self):
        h = Harness()
        h.generate(["charA.safetensors"])
        self.assertEqual(h.networks.calls[0]["online_mode"], True)
        self.assertEqual(len(h.cls().sched.entries), 3)
        self.assertEqual(h.cls().scheduled_files, ["charA.safetensors"])

    def test_turbo_excluded_by_default(self):
        h = Harness()
        h.generate(["krea_turbo.safetensors", "charA.safetensors"])
        by_file = {os.path.basename(c["filename"]): c for c in h.networks.calls}
        self.assertEqual(by_file["krea_turbo.safetensors"]["online_mode"], False)
        self.assertEqual(by_file["charA.safetensors"]["online_mode"], True)
        self.assertEqual(h.cls().scheduled_files, ["charA.safetensors"])

    def test_te_toggle_zeroes_clip_strength(self):
        h = Harness()
        h.generate(["charA.safetensors"], strengths=[0.8], ui={"te_enabled": False})
        self.assertEqual(h.networks.calls[0]["strength_clip"], 0.0)
        h2 = Harness()
        h2.generate(["charA.safetensors"], strengths=[0.8])
        self.assertEqual(h2.networks.calls[0]["strength_clip"], 0.8)

    def test_infotext_written(self):
        h = Harness()
        p, _ = h.generate(["charA.safetensors"],
                          ui={"time_preset": "COMPOSITION", "time_contrast": 0.5})
        self.assertEqual(p.extra_generation_params["LoraCtl time"],
                         "COMPOSITION/Emphasize/0.5")
        self.assertIn("LoraCtl filter", p.extra_generation_params)


class TestScheduling(unittest.TestCase):
    def test_flat_identity_oracle(self):
        """FLAT preset: every step leaves every strength at its base."""
        h = Harness()
        _, per_step = h.generate(["charA.safetensors"], strengths=[0.5])
        for step in per_step:
            for name, s in step.items():
                self.assertAlmostEqual(s, 0.5, places=9, msg=name)

    def test_composition_emphasize_drives_strengths(self):
        h = Harness()
        _, per_step = h.generate(["charA.safetensors"], strengths=[0.5],
                                 ui={"time_preset": "COMPOSITION", "time_contrast": 1.0})
        first, last = per_step[0], per_step[-1]
        for name in first:
            self.assertAlmostEqual(first[name], 0.5, places=6)   # sigma 1.0 in zone
            self.assertAlmostEqual(last[name], 0.0, places=6)    # sigma 0.3109 far out
        mid = per_step[2]  # sigma 0.9045, past the 0.90 boundary midpoint
        for name in mid:
            self.assertTrue(0.0 < mid[name] < 0.5, f"{name}={mid[name]}")

    def test_factor_matches_core_curve(self):
        h = Harness()
        _, per_step = h.generate(["charA.safetensors"], strengths=[0.5],
                                 ui={"time_preset": "DETAIL", "time_modifier": "Suppress",
                                     "time_contrast": 0.6})
        curve = core.TimeCurve("DETAIL", "Suppress", 0.6)
        for sigma, step in zip(KREA_SCHEDULE[:-1], per_step):
            expected = 0.5 * curve.factor(core.normalize_sigma(sigma, 1.0))
            for name, s in step.items():
                self.assertAlmostEqual(s, expected, places=9, msg=f"sigma={sigma} {name}")

    def test_postprocess_parks_at_base(self):
        h = Harness()
        h.generate(["charA.safetensors"], strengths=[0.5],
                   ui={"time_preset": "COMPOSITION", "time_contrast": 1.0})
        for name, s in h.snapshot_strengths().items():
            self.assertAlmostEqual(s, 0.5, places=9, msg=name)

    def test_epsilon_schedule_normalization(self):
        """SDXL-style schedule: sigma0 ~ 14.6; boundaries scale with it."""
        sched = [14.6, 9.0, 5.0, 2.5, 1.0, 0.3, 0.0]
        h = Harness()
        _, per_step = h.generate(["charA.safetensors"], strengths=[0.5],
                                 ui={"time_preset": "COMPOSITION", "time_contrast": 1.0},
                                 steps_schedule=sched)
        self.assertAlmostEqual(h.cls().sigma0, 14.6)
        first = per_step[0]
        for s in first.values():
            self.assertAlmostEqual(s, 0.5, places=6)  # 14.6/14.6 = 1.0, in zone
        last = per_step[-1]
        for s in last.values():
            self.assertAlmostEqual(s, 0.0, places=6)


class TestLifecycle(unittest.TestCase):
    def test_entries_persist_across_cached_generations(self):
        h = Harness()
        h.generate(["charA.safetensors"])
        entries_first = list(h.cls().sched.entries)
        h.generate(["charA.safetensors"])  # same hash -> no reload
        self.assertEqual(len(h.networks.calls), 1)  # loader ran once
        self.assertEqual([id(e.obj) for e in h.cls().sched.entries],
                         [id(e.obj) for e in entries_first])

    def test_new_cycle_replaces_entries(self):
        h = Harness()
        h.generate(["charA.safetensors"])
        old_objs = [e.obj for e in h.cls().sched.entries]  # strong refs: no id reuse
        h.generate(["styleB.safetensors"])  # different hash -> reload
        new_objs = [e.obj for e in h.cls().sched.entries]
        self.assertEqual(len(new_objs), 2)
        for new in new_objs:
            self.assertFalse(any(new is old for old in old_objs))
        self.assertEqual(h.cls().scheduled_files, ["styleB.safetensors"])

    def test_membership_change_busts_hash(self):
        h = Harness()
        h.generate(["charA.safetensors"])
        self.assertIsNotNone(h.sd_model.current_lora_hash)
        h.generate(["charA.safetensors"], ui={"filter_mode": "include",
                                              "filter_patterns": "nothing"})
        # Loader ran again (hash was nulled), and charA is now unscheduled.
        self.assertEqual(len(h.networks.calls), 2)
        self.assertEqual(h.networks.calls[1]["online_mode"], False)
        self.assertEqual(len(h.cls().sched.entries), 0)

    def test_disable_reverts_to_baked(self):
        h = Harness()
        h.generate(["charA.safetensors"])
        h.generate(["charA.safetensors"], ui={"enabled": False})
        self.assertEqual(h.networks.calls[-1]["online_mode"], False)
        unet = h.sd_model.forge_objects.unet
        self.assertEqual(unet.weight_wrapper_patches, {})
        self.assertEqual(len(unet.patches), 3)

    def test_stale_entries_dropped_and_hash_busted(self):
        h = Harness()
        h.generate(["charA.safetensors"])
        self.assertEqual(len(h.cls().sched.entries), 3)
        # Simulate a checkpoint swap: fresh unet, wrappers gone, hash kept
        # artificially to prove OUR staleness path busts it.
        h.sd_model.forge_objects.unet = FakePatcher()
        h.sd_model.forge_objects_original.unet = FakePatcher()
        kept_hash = h.sd_model.current_lora_hash
        h.generate(["charA.safetensors"])
        self.assertEqual(len(h.networks.calls), 2)  # reloaded despite kept hash
        self.assertEqual(len(h.cls().sched.entries), 3)  # recollected fresh


class TestApiDrift(unittest.TestCase):
    def test_six_tuple_loader_refused_but_safe(self):
        h = Harness(tuple_len=6)
        _, per_step = h.generate(["charA.safetensors"], strengths=[0.5],
                                 ui={"time_preset": "COMPOSITION", "time_contrast": 1.0})
        cls = h.cls()
        self.assertEqual(len(cls.sched.entries), 0)
        self.assertEqual(cls.scheduled_files, [])
        # The LoRA is still applied (online, base strength) — never rewritten.
        for step in per_step:
            for s in step.values():
                self.assertAlmostEqual(s, 0.5, places=9)


class TestSigmaCapture(unittest.TestCase):
    def test_wrap_is_read_only_and_guarded(self):
        h = Harness()
        p, _ = h.generate(["charA.safetensors"])
        self.assertEqual(h.cls().sigma0, 1.0)
        self.assertEqual(h.cls().captured_schedule, KREA_SCHEDULE)
        wrapped = p.sampler.get_sigmas
        self.assertTrue(getattr(wrapped, "_loractl_wrapped", False))
        h.script.process_before_every_sampling(p)  # second call must not re-wrap
        self.assertIs(p.sampler.get_sigmas, wrapped)
        self.assertEqual(wrapped(p, 8), KREA_SCHEDULE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
