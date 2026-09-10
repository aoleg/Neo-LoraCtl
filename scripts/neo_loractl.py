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


def _rebake_keys(patcher, keys):
    """Recompute the given keys' live weights from the patcher's pristine
    backups plus whatever the patches dict currently says (our payloads
    included, at their current strength). Mirrors patch_weight_to_device
    (backend/patcher/base.py:447) except it sources from backup — the live
    weight is already merged and merging onto it would double-apply.

    A key with no backup is untouched: its patches are applied per-forward
    (lowvram path), so the dict change alone already took effect.
    Returns the number of keys rewritten."""
    import torch
    from backend import memory_management, utils as backend_utils
    from backend.float import stochastic_rounding
    from backend.patcher.base import get_key_weight
    from backend.patcher.lora import merge_lora_to_weight, string_to_seed

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    done = 0
    for key in keys:
        bk = patcher.backup.get(key)
        if bk is None:
            continue
        weight, set_func, convert_func = get_key_weight(patcher.model, key)
        remaining = [p for p in patcher.patches.get(key, []) if p[0] != 0.0]
        if not remaining:
            # Pure restore: mimic unpatch_model — the backup is the raw
            # original (quantized checkpoints included), set it back as-is.
            restored = bk.weight.to(weight.device, copy=True)
            if bk.inplace_update:
                backend_utils.copy_to_param(patcher.model, key, restored)
            else:
                backend_utils.set_attr(patcher.model, key, restored)
            done += 1
            continue
        temp_dtype = memory_management.lora_compute_dtype(weight.device)
        temp = memory_management.cast_to_device(bk.weight, weight.device,
                                                temp_dtype, copy=True)
        if convert_func is not None:
            temp = convert_func(temp, inplace=True)
        out = merge_lora_to_weight(remaining, temp, key)
        if set_func is None:
            out = stochastic_rounding(out, weight.dtype, seed=string_to_seed(key))
            if bk.inplace_update:
                backend_utils.copy_to_param(patcher.model, key, out)
            else:
                backend_utils.set_attr(patcher.model, key, out)
        else:
            set_func(out, inplace_update=bk.inplace_update, seed=string_to_seed(key))
        done += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return done


def _sv_lora_choices():
    try:
        import networks
        names = sorted({x.name for x in networks.available_network_aliases.values()})
    except Exception:
        names = []
    return ["None"] + core.sort_sv_choices(names)


def _sv_resolve_path(name):
    if not name or name == "None":
        return None
    try:
        import networks
        entry = (networks.available_network_aliases.get(name)
                 or networks.available_networks.get(name))
        if entry is not None and getattr(entry, "filename", None):
            return entry.filename
    except Exception:
        pass
    return None


def _builtin_ctl_mapping():
    """Forge >= c2ae52e5 ships 'LoRA Control Integrated' (prompt syntax
    <lora:name:[a:b]>), which drives patch strengths per step itself. Its
    class-level mapping (keyed by LoRA filename) tells us which LoRAs it
    owns this run; we must not fight it over the same objects."""
    for data in getattr(scripts, "scripts_data", []):
        if getattr(data.script_class, "__name__", "") == "LoRAControl":
            return getattr(data.script_class, "mapping", {})
    return {}


def _intercepted_load_lora_for_models(model, clip, lora, strength_model, strength_clip,
                                      filename="default", online_mode=False):
    cls = NeoLoraCtlScript
    if cls.enabled:
        # Track every prompt-loaded file this cycle (scheduled or not) so the
        # Seed Variance section can detect a collision with its selection.
        if not cls.prompt_cycle_open:
            cls.prompt_files = []
            cls.prompt_cycle_open = True
        cls.prompt_files.append(filename)
    if not cls.enabled or not core.lora_is_scheduled(filename, cls.patterns, cls.filter_mode):
        return _original_load_lora_for_models(model, clip, lora, strength_model,
                                              strength_clip, filename, online_mode)
    if filename in _builtin_ctl_mapping():
        if cls.debug:
            _log(f"'{os.path.basename(filename)}' uses <lora:...:[a:b]> syntax; "
                 "left to Forge's builtin LoRA Control")
        return _original_load_lora_for_models(model, clip, lora, strength_model,
                                              strength_clip, filename, online_mode)

    if not cls.cycle_open:
        # First scheduled LoRA of a fresh load cycle: the stock loader has
        # just reset forge_objects to originals, so prior entries are stale.
        cls.sched.clear()
        cls.scheduled_files = []
        cls.baked_files = []
        cls.cycle_open = True

    sc = 0.0 if not cls.te_enabled else strength_clip

    if cls.baked_mode_active() and not online_mode:
        # Compile: route BAKED with block factors pre-scaled into the tuple
        # strengths. The ordinary load path merges them once; the generation
        # then runs at full native speed with no per-step work at all.
        before = core.ScheduleSet.snapshot_counts(getattr(model, "patches", {}))
        result = _original_load_lora_for_models(model, clip, lora, strength_model,
                                                sc, filename, False)
        if result is None:
            return result
        new_model, new_clip = result
        cls.ensure_block_context(new_model, getattr(new_model, "patches", {}))
        count, errors = core.scale_new_baked_patches(
            before, getattr(new_model, "patches", {}), cls.block_factor_for_key)
        if errors:
            _log(f"ERROR: could not bake block mask for '{os.path.basename(filename)}' "
                 f"(Forge API drift?): {errors[0]}")
        else:
            cls.baked_files.append(os.path.basename(filename))
            if cls.debug:
                _log(f"compiled '{os.path.basename(filename)}': block mask baked "
                     f"into {count} patches, no runtime scheduling needed")
        return new_model, new_clip

    before = core.ScheduleSet.snapshot_counts(getattr(model, "weight_wrapper_patches", {}))
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
    compile_bake: bool = True
    time_curve = core.TimeCurve()

    # --- seed variance section ---
    sv_path = None
    sv_strength: float = 1.0
    sv_te: bool = False
    sv_handle = None                 # core.BakedLoraHandle | None
    sv_debaked: bool = False
    sv_warned_collision: bool = False

    # --- compile-mode bookkeeping ---
    baked_files: list = []
    prompt_files: list = []
    prompt_cycle_open: bool = False
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
    def baked_mode_active(cls):
        """Compile bakes every schedule that is static: block masks can be
        cooked into the weights whenever no time curve runs."""
        return (cls.compile_bake
                and (cls.time_curve.preset == "FLAT" or cls.time_curve.contrast <= 0.0))

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
            gr.Markdown("**Seed Variance** — a helper LoRA applied only during the "
                        "composition steps, to diversify results across seeds")
            with gr.Row():
                sv_lora = gr.Dropdown(choices=_sv_lora_choices(), value="None",
                                      label="Seed variance LoRA")
                sv_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05,
                                        label="Strength")
                sv_te = gr.Checkbox(label="Apply its text encoder", value=False)
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
            compile_bake = gr.Checkbox(
                label="Compile (bake static LoRA schedules into the weights for "
                      "full-speed generation)", value=True)
            debug = gr.Checkbox(label="Debug logging", value=False)
            with gr.Row(visible=DEV_MODE):
                dev_mask = gr.Textbox(value="", label="DEV: explicit block mask",
                                      placeholder="one value per block, comma-separated")
                dev_bounds = gr.Textbox(value="", label="DEV: time boundaries hi,lo",
                                        placeholder=f"{core.DEFAULT_TIME_HI_BOUNDARY},"
                                                    f"{core.DEFAULT_TIME_LO_BOUNDARY}")
        return [enabled, block_preset, block_modifier, block_contrast, block_boost,
                time_preset, time_modifier, time_contrast, time_boost,
                sv_lora, sv_strength, sv_te,
                te_enabled, filter_mode, filter_patterns, compile_bake,
                debug, dev_mask, dev_bounds]

    def process(self, p, enabled=False, block_preset="FULL", block_modifier="Emphasize",
                block_contrast=0.7, block_boost=1.0, time_preset="FLAT",
                time_modifier="Emphasize", time_contrast=0.7, time_boost=1.0,
                sv_lora="None", sv_strength=1.0, sv_te=False,
                te_enabled=True, filter_mode="exclude",
                filter_patterns="", compile_bake=True,
                debug=False, dev_mask="", dev_bounds="", *args):
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
        cls.compile_bake = bool(compile_bake)
        cls.sv_strength = float(sv_strength)
        cls.sv_te = bool(sv_te)
        cls.sv_warned_collision = False
        sv_name = str(sv_lora)
        cls.sv_path = _sv_resolve_path(sv_name)
        if sv_name not in ("", "None") and cls.sv_path is None:
            _log(f"seed variance LoRA '{sv_name}' not found; section inactive")

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
        p.extra_generation_params["LoraCtl mode"] = (
            "baked" if cls.baked_mode_active() else "online")
        if cls.sv_path:
            p.extra_generation_params["LoraCtl SV"] = (
                f"{os.path.splitext(os.path.basename(cls.sv_path))[0]}"
                f"/{cls.sv_strength:g}/TE {'on' if cls.sv_te else 'off'}")
        if (hi, lo) != (core.DEFAULT_TIME_HI_BOUNDARY, core.DEFAULT_TIME_LO_BOUNDARY):
            p.extra_generation_params["LoraCtl bounds"] = f"{hi:g},{lo:g}"
        p.extra_generation_params["LoraCtl TE"] = "on" if cls.te_enabled else "off"
        p.extra_generation_params["LoraCtl filter"] = (
            f"{cls.filter_mode}:{','.join(cls.patterns)}")

    def before_process_batch(self, p, *args, **kwargs):
        cls = NeoLoraCtlScript
        cls.cycle_open = False
        cls.prompt_cycle_open = False

        baked = cls.enabled and cls.baked_mode_active()
        sv_sig = ((cls.sv_path, cls.sv_strength, cls.sv_te)
                  if (cls.enabled and cls.sv_path) else None)
        sig = ((True, tuple(cls.patterns), cls.filter_mode, cls.te_enabled,
                baked, cls.block_sig if baked else None, sv_sig)
               if cls.enabled else (False,))
        if sig != cls.membership_sig:
            cls.membership_sig = sig
            cls.sched.clear()
            cls.scheduled_files = []
            cls.baked_files = []
            # A stock reload restores every key from backup and repatches, so
            # dropping the SV handle here is clean: its contribution is gone
            # from the rebuilt patches dict.
            cls.sv_handle = None
            cls.sv_debaked = False
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

    # ------------------------------------------------------------------
    # Seed Variance: bake at X, switch off (restore) below the composition
    # boundary, clean everything at end of job.
    # ------------------------------------------------------------------

    @classmethod
    def sv_apply(cls, p):
        if not cls.enabled or cls.sv_path is None:
            return
        unet = getattr(getattr(getattr(p, "sd_model", None), "forge_objects", None), "unet", None)
        patches = getattr(unet, "patches", None)
        if unet is None or patches is None:
            return
        sv_norm = os.path.normcase(os.path.normpath(cls.sv_path))
        if any(os.path.normcase(os.path.normpath(f)) == sv_norm for f in cls.prompt_files):
            if not cls.sv_warned_collision:
                cls.sv_warned_collision = True
                _log("seed variance LoRA is also in the prompt; the prompt "
                     "instance wins, section skipped")
            return
        if cls.sv_handle is not None and cls.sv_handle.present_in(patches):
            if cls.sv_debaked:
                # Next batch of the same job: bring the X phase back.
                affected = cls.sv_handle.set_strength(patches, cls.sv_strength)
                n = _rebake_keys(unet, affected)
                cls.sv_debaked = False
                if cls.debug:
                    _log(f"seed variance re-baked at {cls.sv_strength:g} ({n} keys)")
            return
        cls.sv_handle = None
        cls.sv_debaked = False
        try:
            import networks
            sd = networks.load_lora_state_dict(cls.sv_path)
        except Exception as e:
            _log(f"ERROR: cannot load seed variance LoRA: {e}")
            cls.sv_path = None
            return
        clip = p.sd_model.forge_objects.clip if cls.sv_te else None
        before = core.ScheduleSet.snapshot_counts(patches)
        result = _original_load_lora_for_models(
            unet, clip, sd, cls.sv_strength,
            cls.sv_strength if cls.sv_te else 0.0, cls.sv_path, False)
        del sd
        if result is None:
            _log("ERROR: seed variance LoRA failed to load")
            return
        new_unet, new_clip = result
        handle, errors = core.BakedLoraHandle.collect(
            before, getattr(new_unet, "patches", {}), cls.sv_strength)
        if handle is None or not handle.entries:
            _log("ERROR: seed variance adoption refused"
                 + (f" (Forge API drift?): {errors[0]}" if errors else
                    " (no patches matched this model)"))
            return  # new_unet not adopted; behavior stays unchanged
        p.sd_model.forge_objects.unet = new_unet
        if cls.sv_te and new_clip is not None:
            p.sd_model.forge_objects.clip = new_clip
        cls.sv_handle = handle
        if cls.debug:
            _log(f"seed variance '{os.path.basename(cls.sv_path)}' baked at "
                 f"{cls.sv_strength:g} ({len(handle.entries)} patches); switches "
                 f"off below normalized sigma {cls.time_curve.hi_boundary:g}")

    @classmethod
    def sv_cleanup(cls, p):
        handle = cls.sv_handle
        if handle is None:
            return
        unet = getattr(getattr(getattr(p, "sd_model", None), "forge_objects", None), "unet", None)
        patches = getattr(unet, "patches", None)
        if unet is None or patches is None or not handle.present_in(patches):
            cls.sv_handle = None
            cls.sv_debaked = False
            return
        if not cls.sv_debaked:
            # Interrupted before the boundary: restore now, unconditionally.
            affected = handle.set_strength(patches, 0.0)
            _rebake_keys(unet, affected)
        our_keys, emptied = handle.remove_from(patches)
        backup = getattr(unet, "backup", {})
        freed = 0
        for key in emptied:
            if backup.pop(key, None) is not None:
                freed += 1
        cls.sv_handle = None
        cls.sv_debaked = False
        if cls.debug:
            _log(f"seed variance cleaned up ({len(our_keys)} keys, "
                 f"{freed} exclusive backups freed)")

    def process_batch(self, p, *args, **kwargs):
        cls = NeoLoraCtlScript
        cls.cycle_open = False
        cls.prompt_cycle_open = False
        cls.sv_apply(p)
        if cls.enabled and cls.debug and cls.baked_files:
            _log(f"compiled (full speed): {cls.baked_files}")
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
        if not cls.enabled:
            return
        sv_armed = cls.sv_handle is not None and not cls.sv_debaked
        if not cls.sched.entries and not sv_armed:
            return
        try:
            sigma = params.sigma
            sigma = float(sigma.max()) if hasattr(sigma, "max") else float(sigma)
        except Exception:
            return
        if sv_armed:
            sigma0 = cls.sigma0 if cls.sigma0 else sigma
            if core.normalize_sigma(sigma, sigma0) < cls.time_curve.hi_boundary:
                unet = getattr(getattr(getattr(shared, "sd_model", None),
                                       "forge_objects", None), "unet", None)
                patches = getattr(unet, "patches", None)
                if unet is not None and patches is not None:
                    affected = cls.sv_handle.set_strength(patches, 0.0)
                    n = _rebake_keys(unet, affected)
                    cls.sv_debaked = True
                    if cls.debug:
                        _log(f"seed variance switched off at sigma {sigma:.4f} "
                             f"({n} keys restored)")
        if not cls.sched.entries:
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
        cls.prompt_cycle_open = False
        cls.sv_cleanup(p)
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
