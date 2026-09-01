"""Neo-LoraCtl — block- and timestep-aware LoRA strength control for Forge Neo.

Two symmetric axes, multiplied per patch key per sampling step:
  effective = base * block_factor(key) * time_factor(sigma / schedule_sigma0)

Mechanism (Forge Neo commit 92b55e1b, see docs/PLAN.md and docs/MECHANISM.md):
- networks.load_lora_for_models is intercepted; LoRAs passing the
  include/exclude filter are loaded with online_mode=True, which turns each
  of their patches into an OnlineLoRAPatch object in weight_wrapper_patches.
- New objects are adopted by snapshot-diff (core.ScheduleSet.collect), which
  refuses on any API-shape drift. Block factors bind at adoption and rebind
  in-place when the block preset changes (no LoRA reload needed).
- An on_cfg_denoiser callback rewrites the adopted objects' strengths every
  step. p.sampler.get_sigmas is wrapped READ-ONLY to capture the schedule
  head for sigma normalization; the hires pass lands on the curve tail
  naturally.
"""

import importlib.util
import os
import sys

import gradio as gr

from modules import scripts, shared
from modules.script_callbacks import on_before_ui, on_cfg_denoiser

# Load the framework-agnostic core. It MUST be registered in sys.modules
# before exec: its dataclasses resolve annotations via
# sys.modules.get(cls.__module__), which crashes on an unregistered module.
_core_path = os.path.join(scripts.basedir(), "loractl_core.py")
_spec = importlib.util.spec_from_file_location("neo_loractl_core", _core_path)
core = importlib.util.module_from_spec(_spec)
sys.modules["neo_loractl_core"] = core
_spec.loader.exec_module(core)

TAG = "[LoraCtl]"
DEV_MODE = os.environ.get("LORACTL_DEV", "") == "1"

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

    cls.ensure_block_context(new_model, wrappers)
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
    # --- config (set in process() from UI args + XYZ overrides) ---
    enabled: bool = False
    te_enabled: bool = True
    patterns: list = []
    filter_mode: str = "exclude"
    time_curve = core.TimeCurve()
    block_preset: str = "FULL"
    block_modifier: str = "Emphasize"
    block_contrast: float = 0.0
    block_boost: float = 1.0
    dev_mask_text: str = ""
    debug: bool = False

    # --- block context (bound to the current model, built lazily) ---
    block_sig = None
    block_arch: str = core.ARCH_UNKNOWN
    block_count: int = 0
    num_double: int = 0
    block_mask = None       # list[float] | None (None until built)
    arch_logged: bool = False

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
    factors_seen: list = []
    diag_done: bool = False

    # ------------------------------------------------------------------
    # Block axis
    # ------------------------------------------------------------------

    @classmethod
    def ensure_block_context(cls, model, wrappers):
        """Bind arch/count to the live model (introspection first, patch-key
        fallback) and build the mask. Cheap no-op once built."""
        if cls.block_mask is not None:
            return
        if cls.block_count == 0:
            dm = getattr(getattr(model, "model", None), "diffusion_model", None)
            if dm is not None and hasattr(dm, "double_blocks"):
                cls.block_arch = core.ARCH_FLUX
                cls.num_double = len(dm.double_blocks)
                cls.block_count = cls.num_double + len(getattr(dm, "single_blocks", []))
            elif dm is not None and hasattr(dm, "input_blocks"):
                cls.block_arch = core.ARCH_SD
                cls.block_count = 26
            elif dm is not None and hasattr(dm, "blocks"):
                cls.block_arch = core.ARCH_FLAT
                cls.block_count = len(dm.blocks)
            else:
                cls.block_arch = core.detect_arch(wrappers.keys())
                if cls.block_arch != core.ARCH_UNKNOWN:
                    cls.block_count, cls.num_double = core.infer_block_layout(
                        wrappers.keys(), cls.block_arch)
            if not cls.arch_logged:
                cls.arch_logged = True
                if cls.block_arch == core.ARCH_UNKNOWN:
                    _log("block axis: unknown architecture; block presets inactive "
                         "(factor 1.0 everywhere)")
                elif cls.debug:
                    _log(f"block axis: arch={cls.block_arch} blocks={cls.block_count}"
                         + (f" (double={cls.num_double})" if cls.num_double else ""))
        cls.block_mask = cls.build_block_mask()

    @classmethod
    def build_block_mask(cls):
        if cls.block_arch == core.ARCH_UNKNOWN or cls.block_count < 1:
            return []
        if DEV_MODE and cls.dev_mask_text.strip():
            override = core.parse_mask_override(cls.dev_mask_text, cls.block_count)
            if override is not None:
                _log(f"DEV: explicit block mask in effect ({cls.block_count} values)")
                return override
            _log(f"DEV: block mask override ignored (need exactly "
                 f"{cls.block_count} numeric values)")
        return core.build_block_mask(cls.block_count, cls.block_preset,
                                     cls.block_modifier, cls.block_contrast,
                                     cls.block_boost)

    @classmethod
    def block_factor_for_key(cls, key):
        if not cls.block_mask:
            return 1.0
        idx = core.classify_key(key, cls.block_arch, cls.num_double)
        if idx is None or idx >= len(cls.block_mask):
            return 1.0
        return cls.block_mask[idx]

    # ------------------------------------------------------------------
    # Script plumbing
    # ------------------------------------------------------------------

    def title(self):
        return "Neo-LoraCtl"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Neo-LoraCtl", open=False):
            enabled = gr.Checkbox(label="Enable", value=False)
            gr.Markdown("Focuses prompt-loaded `<lora:...>` networks on what you "
                        "want from them: pick **where** in the model (blocks) and "
                        "**when** in the run (timesteps) each LoRA applies.")
            gr.Markdown("**Blocks** — what the LoRA is allowed to shape")
            with gr.Row():
                block_preset = gr.Dropdown(choices=list(core.BLOCK_PRESETS), value="FULL",
                                           label="Block preset")
                block_modifier = gr.Radio(choices=list(core.MODIFIERS), value="Emphasize",
                                          label="Modifier")
                block_contrast = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Contrast")
                block_boost = gr.Slider(0.25, 2.0, value=1.0, step=0.05,
                                        label="Boost (Emphasize)")
            gr.Markdown("**Timesteps** — when in the run the LoRA applies")
            with gr.Row():
                time_preset = gr.Dropdown(choices=list(core.TIME_PRESETS), value="FLAT",
                                          label="Timestep preset")
                time_modifier = gr.Radio(choices=list(core.MODIFIERS), value="Emphasize",
                                         label="Modifier")
                time_contrast = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Contrast")
                time_boost = gr.Slider(0.25, 2.0, value=1.0, step=0.05,
                                       label="Boost (Emphasize)")
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
            with gr.Row(visible=DEV_MODE):
                dev_mask = gr.Textbox(value="", label="DEV: explicit block mask",
                                      placeholder="one value per block, comma-separated")
                dev_bounds = gr.Textbox(value="", label="DEV: time boundaries hi,lo",
                                        placeholder=f"{core.DEFAULT_TIME_HI_BOUNDARY},"
                                                    f"{core.DEFAULT_TIME_LO_BOUNDARY}")
        return [enabled, block_preset, block_modifier, block_contrast, block_boost,
                time_preset, time_modifier, time_contrast, time_boost,
                te_enabled, filter_mode, filter_patterns, debug, dev_mask, dev_bounds]

    def process(self, p, enabled=False, block_preset="FULL", block_modifier="Emphasize",
                block_contrast=0.7, block_boost=1.0, time_preset="FLAT",
                time_modifier="Emphasize", time_contrast=0.7, time_boost=1.0,
                te_enabled=True, filter_mode="exclude",
                filter_patterns="", debug=False, dev_mask="", dev_bounds="", *args):
        cls = NeoLoraCtlScript
        cls.enabled = bool(enabled) and _install_interception()
        cls.debug = bool(debug)
        cls.cb_count = 0
        cls.factor_min = 1.0
        cls.factor_max = 0.0
        cls.factors_seen = []
        cls.diag_done = False
        cls.sigma0 = None
        cls.captured_schedule = []
        if not cls.enabled:
            return

        # XYZ grid overrides (apply_field sets these on the per-cell p).
        block_preset = getattr(p, "loractl_xyz_block_preset", block_preset)
        block_modifier = getattr(p, "loractl_xyz_block_modifier", block_modifier)
        block_contrast = getattr(p, "loractl_xyz_block_contrast", block_contrast)
        block_boost = getattr(p, "loractl_xyz_block_boost", block_boost)
        time_preset = getattr(p, "loractl_xyz_time_preset", time_preset)
        time_modifier = getattr(p, "loractl_xyz_time_modifier", time_modifier)
        time_contrast = getattr(p, "loractl_xyz_time_contrast", time_contrast)
        time_boost = getattr(p, "loractl_xyz_time_boost", time_boost)
        time_hi = getattr(p, "loractl_xyz_time_hi", None)
        time_lo = getattr(p, "loractl_xyz_time_lo", None)

        cls.te_enabled = bool(te_enabled)
        cls.patterns = core.parse_patterns(filter_patterns)
        cls.filter_mode = filter_mode if filter_mode in ("exclude", "include") else "exclude"

        hi = core.DEFAULT_TIME_HI_BOUNDARY
        lo = core.DEFAULT_TIME_LO_BOUNDARY
        if DEV_MODE and str(dev_bounds).strip():
            parsed = core.parse_mask_override(dev_bounds, 2)
            if parsed is not None:
                hi, lo = parsed
            else:
                _log("DEV: time boundaries override ignored (need 'hi,lo')")
        if time_hi is not None:
            hi = float(time_hi)
        if time_lo is not None:
            lo = float(time_lo)
        if not (0.0 <= lo < hi <= 1.0):
            _log(f"invalid time boundaries hi={hi} lo={lo}; using defaults")
            hi, lo = core.DEFAULT_TIME_HI_BOUNDARY, core.DEFAULT_TIME_LO_BOUNDARY

        cls.time_curve = core.TimeCurve(
            preset=time_preset if time_preset in core.TIME_PRESETS else "FLAT",
            modifier=time_modifier if time_modifier in core.MODIFIERS else "Emphasize",
            contrast=float(time_contrast),
            boost=float(time_boost),
            hi_boundary=hi, lo_boundary=lo,
        )

        block_sig = (str(block_preset), str(block_modifier), float(block_contrast),
                     float(block_boost), str(dev_mask))
        if block_sig != cls.block_sig:
            cls.block_sig = block_sig
            cls.block_preset = block_preset if block_preset in core.BLOCK_PRESETS else "FULL"
            cls.block_modifier = (block_modifier if block_modifier in core.MODIFIERS
                                  else "Emphasize")
            cls.block_contrast = float(block_contrast)
            cls.block_boost = float(block_boost)
            cls.dev_mask_text = str(dev_mask)
            cls.block_mask = None  # rebuild lazily (count may be unknown yet)
            if cls.sched.entries and cls.block_count:
                # Entries persist across generations; rebind without a reload.
                cls.block_mask = cls.build_block_mask()
                cls.sched.rebind_block_factors(cls.block_factor_for_key)
                if cls.debug:
                    _log("block config changed; rebound factors on "
                         f"{len(cls.sched.entries)} entries")

        block_desc = f"{cls.block_preset}/{cls.block_modifier}/{cls.block_contrast:g}"
        if cls.block_modifier == "Emphasize":
            block_desc += f"/x{cls.block_boost:g}"
        time_desc = (f"{cls.time_curve.preset}/{cls.time_curve.modifier}"
                     f"/{cls.time_curve.contrast:g}")
        if cls.time_curve.modifier == "Emphasize":
            time_desc += f"/x{cls.time_curve.boost:g}"
        p.extra_generation_params["LoraCtl blocks"] = block_desc
        p.extra_generation_params["LoraCtl time"] = time_desc
        if (hi, lo) != (core.DEFAULT_TIME_HI_BOUNDARY, core.DEFAULT_TIME_LO_BOUNDARY):
            p.extra_generation_params["LoraCtl bounds"] = f"{hi:g},{lo:g}"
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
                # Model changed under us: rebind block context too.
                cls.block_count = 0
                cls.num_double = 0
                cls.block_arch = core.ARCH_UNKNOWN
                cls.block_mask = None
                cls.arch_logged = False
                _null_lora_hash(p)
                if cls.debug:
                    _log("scheduled entries went stale; forcing LoRA reload")

    def process_batch(self, p, *args, **kwargs):
        cls = NeoLoraCtlScript
        cls.cycle_open = False
        if cls.enabled and cls.debug:
            _log(f"active: {len(cls.scheduled_files)} scheduled LoRA(s) "
                 f"{cls.scheduled_files}, {len(cls.sched.entries)} patch entries, "
                 f"blocks={cls.block_preset}/{cls.block_modifier}/{cls.block_contrast:g}, "
                 f"time={cls.time_curve.preset}/{cls.time_curve.modifier}"
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
                if cls.sigma0 > 0.0:
                    # Emphasize budget: conserve the mean over this pass's
                    # actual model-call steps (schedule[:-1]).
                    cls.time_curve.zone_mean = None
                    cls.time_curve.prepare([s / cls.sigma0 for s in schedule[:-1]])
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
        cls.factors_seen.append(factor)
        if not cls.diag_done:
            cls.diag_done = True
            if cls.debug:
                _log(f"diagnostic: step 0 sigma={sigma:.4f} sigma0={cls.sigma0:.4f} "
                     f"time_factor={factor:.4f} entries={len(cls.sched.entries)}")

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
                     f"time factor range [{cls.factor_min:.4f}, {cls.factor_max:.4f}]")
                _log("per-step time factors: "
                     + ", ".join(f"{f:.3f}" for f in cls.factors_seen))
                if cls.block_mask:
                    _log("block mask: "
                         + ", ".join(f"{v:.2f}" for v in cls.block_mask))
        # Entries deliberately persist across generations: the stock loader
        # reuses its LoRA application when the hash is unchanged, and our
        # entries reference the same live OnlineLoRAPatch objects. Staleness
        # is handled in before_process_batch.


# ---------------------------------------------------------------------------
# XYZ grid axes
# ---------------------------------------------------------------------------

def _register_xyz_axes():
    xyz = None
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid", "xyz_grid"):
            xyz = data.module
            break
    if xyz is None:
        return
    if any(getattr(opt, "label", "").startswith("(LoraCtl)") for opt in xyz.axis_options):
        return

    def choice(label, field, choices):
        return xyz.AxisOption(f"(LoraCtl) {label}", str, xyz.apply_field(field),
                              choices=lambda: list(choices))

    def number(label, field):
        return xyz.AxisOption(f"(LoraCtl) {label}", float, xyz.apply_field(field))

    xyz.axis_options.extend([
        choice("Block preset", "loractl_xyz_block_preset", core.BLOCK_PRESETS),
        choice("Block modifier", "loractl_xyz_block_modifier", core.MODIFIERS),
        number("Block contrast", "loractl_xyz_block_contrast"),
        number("Block boost", "loractl_xyz_block_boost"),
        choice("Time preset", "loractl_xyz_time_preset", core.TIME_PRESETS),
        choice("Time modifier", "loractl_xyz_time_modifier", core.MODIFIERS),
        number("Time contrast", "loractl_xyz_time_contrast"),
        number("Time boost", "loractl_xyz_time_boost"),
        number("Time hi boundary", "loractl_xyz_time_hi"),
        number("Time lo boundary", "loractl_xyz_time_lo"),
    ])
    _log("registered 10 XYZ grid axes")


on_before_ui(_register_xyz_axes)
on_cfg_denoiser(NeoLoraCtlScript.on_cfg)
