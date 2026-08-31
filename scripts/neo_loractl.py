"""Neo-LoraCtl — block- and timestep-aware LoRA strength control for Forge Neo.

Phase 1: time axis (sigma-zone presets) driving prompt-loaded LoRAs through
per-key online patches. Block axis lands in phase 2; loractl_core.py already
carries its math.

Mechanism (Forge Neo commit 92b55e1b, see docs/PLAN.md):
- networks.load_lora_for_models is intercepted; LoRAs passing the
  include/exclude filter are loaded with online_mode=True, which turns each
  of their patches into an OnlineLoRAPatch object in weight_wrapper_patches.
- New objects are adopted by snapshot-diff (core.ScheduleSet.collect), which
  refuses on any API-shape drift.
- An on_cfg_denoiser callback rewrites the adopted objects' strengths every
  step: base * block_factor * time_factor(sigma / schedule_sigma0).
- p.sampler.get_sigmas is wrapped READ-ONLY to capture the full schedule head
  for sigma normalization; the hires pass lands on the curve tail naturally.
"""

import importlib.util
import os
import sys

import gradio as gr

from modules import scripts, shared
from modules.script_callbacks import on_cfg_denoiser

# Load the framework-agnostic core. It MUST be registered in sys.modules
# before exec: its dataclasses resolve annotations via
# sys.modules.get(cls.__module__), which crashes on an unregistered module.
_core_path = os.path.join(scripts.basedir(), "loractl_core.py")
_spec = importlib.util.spec_from_file_location("neo_loractl_core", _core_path)
core = importlib.util.module_from_spec(_spec)
sys.modules["neo_loractl_core"] = core
_spec.loader.exec_module(core)

TAG = "[LoraCtl]"

_original_load_lora_for_models = None


def _log(msg):
    try:
        print(f"{TAG} {msg}")
    except UnicodeEncodeError:
        print(f"{TAG} {msg.encode('ascii', 'replace').decode('ascii')}")


def _null_lora_hash(p):
    """Force the stock loader to rebuild LoRAs on the next activation."""
    for model in (getattr(p, "sd_model", None), getattr(shared, "sd_model", None)):
        if model is not None and hasattr(model, "current_lora_hash"):
            model.current_lora_hash = None


def _install_interception():
    global _original_load_lora_for_models
    if _original_load_lora_for_models is not None:
        return True
    try:
        import networks
    except ImportError:
        _log("ERROR: builtin sd_forge_lora 'networks' module not importable; extension inactive")
        return False
    _original_load_lora_for_models = networks.load_lora_for_models
    networks.load_lora_for_models = _intercepted_load_lora_for_models
    _log("installed load_lora_for_models interception")
    return True


def _intercepted_load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                                      filename="default", online_mode=False):
    cls = NeoLoraCtlScript
    if not cls.enabled or not core.lora_is_scheduled(filename, cls.patterns, cls.filter_mode):
        return _original_load_lora_for_models(model, clip, lora, strength_model,
                                              strength_clip, filename, online_mode)

    if not cls.cycle_open:
        # First scheduled LoRA of a fresh load cycle: the stock loader has
        # just reset forge_objects to originals, so prior entries are stale.
        cls.sched.clear()
        cls.scheduled_files = []
        cls.cycle_open = True

    before = core.ScheduleSet.snapshot_counts(getattr(model, "weight_wrapper_patches", {}))
    sc = 0.0 if not cls.te_enabled else strength_clip
    result = _original_load_lora_for_models(model, clip, lora, strength_model,
                                            sc, filename, True)
    if result is None:
        return result
    new_model, new_clip = result

    wrappers = getattr(new_model, "weight_wrapper_patches", None)
    if wrappers is None:
        _log(f"ERROR: model has no weight_wrapper_patches (Forge API drift?); "
             f"'{os.path.basename(filename)}' left unscheduled")
        return new_model, new_clip

    errors = cls.sched.collect(before, wrappers, cls.block_factor_for_key)
    if errors:
        _log(f"ERROR: refusing to schedule '{os.path.basename(filename)}' "
             f"(Forge API drift?): {errors[0]}"
             + (f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""))
        # The LoRA stays online at its base strength — same output as a plain
        # <lora:> tag, just applied on-the-fly. Safe failure mode.
    else:
        cls.scheduled_files.append(os.path.basename(filename))
        if cls.debug:
            _log(f"scheduled '{os.path.basename(filename)}' "
                 f"({len(cls.sched.entries)} total entries)")
    return new_model, new_clip


class NeoLoraCtlScript(scripts.Script):
    # --- config (set in process() from UI args) ---
    enabled: bool = False
    te_enabled: bool = True
    patterns: list = []
    filter_mode: str = "exclude"
    time_curve = core.TimeCurve()
    debug: bool = False

    # --- runtime state ---
    sched = core.ScheduleSet()
    scheduled_files: list = []
    cycle_open: bool = False
    membership_sig = None
    sigma0: float | None = None
    captured_schedule: list = []

    # --- per-generation stats (reset in process(), reported in postprocess) ---
    cb_count: int = 0
    factor_min: float = 1.0
    factor_max: float = 0.0
    diag_done: bool = False

    @classmethod
    def block_factor_for_key(cls, key):
        return 1.0  # phase 2: arch classification + block mask

    def title(self):
        return "Neo-LoraCtl"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Neo-LoraCtl", open=False):
            enabled = gr.Checkbox(label="Enable", value=False)
            gr.Markdown("Schedules the strength of prompt-loaded `<lora:...>` "
                        "networks over the sampling run (by sigma).")
            with gr.Row():
                time_preset = gr.Dropdown(choices=list(core.TIME_PRESETS), value="FLAT",
                                          label="Timestep preset")
                time_modifier = gr.Radio(choices=list(core.MODIFIERS), value="Emphasize",
                                         label="Modifier")
                time_contrast = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Contrast")
            te_enabled = gr.Checkbox(
                label="Apply text-encoder half of scheduled LoRAs (at prompt strength)",
                value=True)
            with gr.Row():
                filter_mode = gr.Radio(choices=["exclude", "include"], value="exclude",
                                       label="Filter mode")
                filter_patterns = gr.Textbox(
                    value=core.DEFAULT_EXCLUDE_PATTERNS, label="Filter patterns",
                    placeholder="comma-separated name substrings")
            gr.Markdown("*Accelerator LoRAs (turbo/lightning/...) must stay excluded — "
                        "scheduling them breaks distilled checkpoints.*")
            debug = gr.Checkbox(label="Debug logging", value=False)
        return [enabled, time_preset, time_modifier, time_contrast,
                te_enabled, filter_mode, filter_patterns, debug]

    def process(self, p, enabled=False, time_preset="FLAT", time_modifier="Emphasize",
                time_contrast=0.7, te_enabled=True, filter_mode="exclude",
                filter_patterns="", debug=False, *args):
        cls = NeoLoraCtlScript
        cls.enabled = bool(enabled) and _install_interception()
        cls.debug = bool(debug)
        cls.cb_count = 0
        cls.factor_min = 1.0
        cls.factor_max = 0.0
        cls.diag_done = False
        cls.sigma0 = None
        cls.captured_schedule = []
        if not cls.enabled:
            return
        cls.te_enabled = bool(te_enabled)
        cls.patterns = core.parse_patterns(filter_patterns)
        cls.filter_mode = filter_mode if filter_mode in ("exclude", "include") else "exclude"
        cls.time_curve = core.TimeCurve(
            preset=time_preset if time_preset in core.TIME_PRESETS else "FLAT",
            modifier=time_modifier if time_modifier in core.MODIFIERS else "Emphasize",
            contrast=float(time_contrast),
        )
        p.extra_generation_params["LoraCtl time"] = (
            f"{cls.time_curve.preset}/{cls.time_curve.modifier}/{cls.time_curve.contrast:g}")
        p.extra_generation_params["LoraCtl TE"] = "on" if cls.te_enabled else "off"
        p.extra_generation_params["LoraCtl filter"] = (
            f"{cls.filter_mode}:{','.join(cls.patterns)}")

    def before_process_batch(self, p, *args, **kwargs):
        cls = NeoLoraCtlScript
        cls.cycle_open = False

        sig = ((True, tuple(cls.patterns), cls.filter_mode, cls.te_enabled)
               if cls.enabled else (False,))
        if sig != cls.membership_sig:
            cls.membership_sig = sig
            cls.sched.clear()
            cls.scheduled_files = []
            _null_lora_hash(p)
            if cls.debug:
                _log("membership config changed; forcing LoRA reload")
            return

        if not cls.enabled or not cls.sched.entries:
            return

        # Staleness check: entries must reference objects still present in the
        # current unet's wrapper dict (checkpoint switches drop them; the
        # stock loader's own hash also resets then, but belt-and-braces).
        unet = getattr(getattr(getattr(p, "sd_model", None), "forge_objects", None), "unet", None)
        wrappers = getattr(unet, "weight_wrapper_patches", None)
        if wrappers is None:
            return
        live = {id(obj) for objs in wrappers.values() for obj in objs}
        kept = [e for e in cls.sched.entries if id(e.obj) in live]
        if len(kept) != len(cls.sched.entries):
            cls.sched.entries = kept
            cls.sched.last_factor = None
            if not kept:
                cls.scheduled_files = []
                _null_lora_hash(p)
                if cls.debug:
                    _log("scheduled entries went stale; forcing LoRA reload")

    def process_batch(self, p, *args, **kwargs):
        cls = NeoLoraCtlScript
        cls.cycle_open = False
        if cls.enabled and cls.debug:
            _log(f"active: {len(cls.scheduled_files)} scheduled LoRA(s) "
                 f"{cls.scheduled_files}, {len(cls.sched.entries)} patch entries, "
                 f"curve={cls.time_curve.preset}/{cls.time_curve.modifier}"
                 f"/{cls.time_curve.contrast:g}")

    def process_before_every_sampling(self, p, *args, **kwargs):
        cls = NeoLoraCtlScript
        if not cls.enabled:
            return
        sampler = getattr(p, "sampler", None)
        original = getattr(sampler, "get_sigmas", None)
        if sampler is None or not callable(original):
            return
        if getattr(original, "_loractl_wrapped", False):
            return

        def get_sigmas(processing, steps):
            sigmas = original(processing, steps)
            try:
                schedule = [float(s) for s in sigmas]
                cls.sigma0 = schedule[0]
                cls.captured_schedule = schedule
                if cls.debug:
                    lo, hi = cls.time_curve.lo_boundary, cls.time_curve.hi_boundary
                    _log(f"schedule captured ({len(schedule) - 1} steps): "
                         + ", ".join(f"{s:.4f}" for s in schedule))
                    _log(f"zone boundaries at sigma {hi * cls.sigma0:.4f} / "
                         f"{lo * cls.sigma0:.4f} (normalized {hi} / {lo})")
            except Exception as e:
                _log(f"schedule capture failed (non-fatal): {e}")
            return sigmas

        get_sigmas._loractl_wrapped = True
        sampler.get_sigmas = get_sigmas

    @classmethod
    def on_cfg(cls, params):
        if not cls.enabled or not cls.sched.entries:
            return
        try:
            sigma = params.sigma
            sigma = float(sigma.max()) if hasattr(sigma, "max") else float(sigma)
        except Exception:
            return
        if cls.sigma0 is None:
            # No schedule captured (unexpected sampler); the first callback
            # sigma is the schedule head for txt2img, and exactly 1.0 on flow
            # models either way.
            cls.sigma0 = sigma
            _log(f"no captured schedule; normalizing by first-step sigma {sigma:.4f}")
        factor = cls.time_curve.factor(core.normalize_sigma(sigma, cls.sigma0))
        cls.sched.apply_time_factor(factor)
        cls.cb_count += 1
        cls.factor_min = min(cls.factor_min, factor)
        cls.factor_max = max(cls.factor_max, factor)
        if not cls.diag_done:
            cls.diag_done = True
            if cls.debug:
                _log(f"diagnostic: step 0 sigma={sigma:.4f} sigma0={cls.sigma0:.4f} "
                     f"factor={factor:.4f} entries={len(cls.sched.entries)}")

    def postprocess(self, p, processed, *args):
        cls = NeoLoraCtlScript
        cls.cycle_open = False
        if cls.enabled and cls.sched.entries:
            # Park strengths at base * block_factor between generations.
            cls.sched.apply_time_factor(1.0)
        if cls.enabled and cls.debug:
            if cls.cb_count == 0:
                _log("summary: callback never invoked (no scheduled LoRA in prompt, "
                     "or a dispatch problem)")
            else:
                _log(f"summary: {cls.cb_count} callback invocations, "
                     f"factor range [{cls.factor_min:.4f}, {cls.factor_max:.4f}]")
        # Entries deliberately persist across generations: the stock loader
        # reuses its LoRA application when the hash is unchanged, and our
        # entries reference the same live OnlineLoRAPatch objects. Staleness
        # is handled in before_process_batch.


on_cfg_denoiser(NeoLoraCtlScript.on_cfg)
