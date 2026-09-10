"""Framework-agnostic core for Neo-LoraCtl.

Pure stdlib on purpose: no torch, no gradio, no Forge imports. The Forge
script (scripts/neo_loractl.py) owns all webui/backend interaction and calls
into this module for everything that can be tested offline: preset masks,
sigma curves, key classification, LoRA filtering, and per-step strength
rewriting of OnlineLoRAPatch-style objects.

Target builds: sd-webui-forge-classic, neo branch, 92b55e1b through
c2ae52e5+. Online patches are OnlineLoRAPatch objects held in
ModelPatcher.weight_wrapper_patches[key], each carrying one 5-element
payload (strength, adapter, strength_model, offset, function) in a
1-element list `obj.patch`. The payload is a tuple up to 92b55e1b and a
mutable list (plus an `obj.name` filename attribute) from c2ae52e5 on,
where upstream itself schedules strengths per step; both forms are
supported. See docs/PLAN.md and knowledge_loractl.md.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


PATCH_TUPLE_LEN = 5

BLOCK_PRESETS = ("FULL", "COMPOSITION", "CHARACTER", "STYLE")
TIME_PRESETS = ("FLAT", "COMPOSITION", "MIDRANGE", "DETAIL")

# Emphasize: mean-preserving redistribution — boosts the zone above the
# prompt strength and lowers the rest so the average stays exactly 1.
# Suppress: dims the zone, rest untouched. Isolate: zone at full strength,
# rest dimmed (the pre-redesign "Emphasize"; kept under an honest name for
# zone-only application of e.g. style LoRAs).
MODIFIERS = ("Emphasize", "Suppress", "Isolate")

# Hard cap on any strength factor (Emphasize can exceed 1; unbounded boosts
# overbake LoRAs). Floors clamp at 0.
MAX_FACTOR = 2.0

# Provisional Krea 2 defaults (normalized sigma). The COMPOSITION/rest
# boundary comes from the SPEED calibration (knowledge_speed.md §11); the
# CHARACTER/DETAIL boundary is a guess until phase 3.
DEFAULT_TIME_HI_BOUNDARY = 0.90
DEFAULT_TIME_LO_BOUNDARY = 0.50
DEFAULT_TIME_TRANSITION = 0.06

# Smoothstep shoulder width on the block axis, in blocks. Not user-exposed.
DEFAULT_BLOCK_SHOULDER = 2.5

DEFAULT_EXCLUDE_PATTERNS = "turbo, lightning, hyper, lcm, dmd"


# ---------------------------------------------------------------------------
# API-shape validation (the guard against Forge patcher churn)
# ---------------------------------------------------------------------------

def validate_patch_object(obj) -> str | None:
    """Return an error string if obj does not look like an OnlineLoRAPatch
    from the pinned build, else None."""
    patch = getattr(obj, "patch", None)
    if not isinstance(patch, list) or len(patch) < 1:
        return f"patch object has no 1+-element .patch list: {type(obj).__name__}"
    entry = patch[0]
    if not isinstance(entry, (tuple, list)) or len(entry) != PATCH_TUPLE_LEN:
        got = len(entry) if isinstance(entry, (tuple, list)) else type(entry).__name__
        return f"patch payload arity mismatch: expected {PATCH_TUPLE_LEN}, got {got}"
    if not isinstance(entry[0], (int, float)):
        return f"patch tuple strength is not a number: {type(entry[0]).__name__}"
    if not callable(obj):
        return f"patch object is not callable: {type(obj).__name__}"
    return None


# ---------------------------------------------------------------------------
# Smooth window primitives
# ---------------------------------------------------------------------------

def _smoothstep(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _rise(x: float, edge: float, width: float) -> float:
    """0 -> 1 smoothstep ramp straddling `edge` (0.5 exactly at the edge)."""
    if width <= 0.0:
        return 1.0 if x >= edge else 0.0
    return _smoothstep((x - (edge - width / 2.0)) / width)


def _window(x: float, lo: float | None, hi: float | None, width: float) -> float:
    """Smooth membership of x in [lo, hi]; None means the zone extends to the
    domain edge on that side. Adjacent zones with shared boundaries and equal
    widths sum to 1 everywhere."""
    w = 1.0
    if lo is not None:
        w *= _rise(x, lo, width)
    if hi is not None:
        w *= 1.0 - _rise(x, hi, width)
    return w


def _apply_modifier(w: float, modifier: str, contrast: float) -> float:
    """One-sided modifiers: map zone membership w to a factor in
    [1 - contrast, 1]. Emphasize is NOT handled here — it needs the zone's
    domain mean (see _emphasize_factor)."""
    c = min(max(contrast, 0.0), 1.0)
    if modifier == "Suppress":
        return 1.0 - c * w
    return 1.0 - c * (1.0 - w)  # Isolate


def emphasize_amplitude(contrast: float, boost: float) -> float:
    """Bell amplitude a = contrast * boost, clamped to [0, MAX_FACTOR]."""
    return min(max(contrast, 0.0) * max(boost, 0.0), MAX_FACTOR)


def _emphasize_factor(w: float, p: float, a: float) -> float:
    """Mean-preserving redistribution: factor = 1 + a*(w - p)/(1 - p), where
    p is the mean of the window over the evaluated domain. Peak 1+a in-zone,
    floor 1 - a*p/(1-p) outside; the domain mean is exactly 1 unless the
    [0, MAX_FACTOR] clamp binds. p ~ 1 (zone covers everything) degenerates
    to a flat 1.0 — a uniform boost would break conservation."""
    if a <= 0.0 or p >= 1.0 - 1e-6:
        return 1.0
    p = max(p, 0.0)
    factor = 1.0 + a * (w - p) / (1.0 - p)
    return min(max(factor, 0.0), MAX_FACTOR)


# ---------------------------------------------------------------------------
# Block axis
# ---------------------------------------------------------------------------

def block_zone(preset: str, count: int) -> tuple[float | None, float | None]:
    """Zone boundaries in block-index space (even thirds)."""
    third = count / 3.0
    if preset == "COMPOSITION":
        return None, third
    if preset == "CHARACTER":
        return third, 2.0 * third
    if preset == "STYLE":
        return 2.0 * third, None
    raise ValueError(f"unknown block preset: {preset}")


def build_block_mask(count: int, preset: str, modifier: str, contrast: float,
                     boost: float = 1.0,
                     shoulder: float = DEFAULT_BLOCK_SHOULDER) -> list[float]:
    """Per-block factor, evaluated at block centers (index + 0.5).

    Emphasize masks are mean-preserving over the `count` classified blocks
    (non-block keys sit outside the mask at 1.0 and outside the budget)."""
    if count < 1:
        raise ValueError("block count must be >= 1")
    if preset == "FULL" or contrast <= 0.0:
        return [1.0] * count
    lo, hi = block_zone(preset, count)
    windows = [_window(i + 0.5, lo, hi, shoulder) for i in range(count)]
    if modifier == "Emphasize":
        p = sum(windows) / count
        a = emphasize_amplitude(contrast, boost)
        return [_emphasize_factor(w, p, a) for w in windows]
    return [_apply_modifier(w, modifier, contrast) for w in windows]


# ---------------------------------------------------------------------------
# Key classification
# ---------------------------------------------------------------------------

_MODEL_PREFIX = "diffusion_model."

# Krea 2 (SingleStreamDiT) module prefixes that are NOT part of the 28-block
# stack; explicit so txtfusion.layerwise_blocks/refiner_blocks can never be
# misread as DiT blocks.
_FLAT_NON_BLOCK_PREFIXES = ("txtfusion.", "first", "last.", "tmlp.", "txtmlp.", "tproj.", "pe_embedder")

_RE_FLAT_BLOCK = re.compile(r"^blocks\.(\d+)\.")
_RE_FLUX_DOUBLE = re.compile(r"^double_blocks\.(\d+)\.")
_RE_FLUX_SINGLE = re.compile(r"^single_blocks\.(\d+)\.")
_RE_SD_INPUT = re.compile(r"^input_blocks\.(\d+)\.")
_RE_SD_MIDDLE = re.compile(r"^middle_block\.")
_RE_SD_OUTPUT = re.compile(r"^output_blocks\.(\d+)\.")

ARCH_FLAT = "flat"      # Krea 2, Anima, Wan-style flat DiT stacks
ARCH_FLUX = "flux"      # double/single stream
ARCH_SD = "sd"          # UNet in/mid/out (26-slot convention)
ARCH_UNKNOWN = "unknown"


def _strip_prefix(key: str) -> str:
    return key[len(_MODEL_PREFIX):] if key.startswith(_MODEL_PREFIX) else key


def detect_arch(keys) -> str:
    for key in keys:
        k = _strip_prefix(key)
        if _RE_FLUX_DOUBLE.match(k) or _RE_FLUX_SINGLE.match(k):
            return ARCH_FLUX
        if _RE_SD_INPUT.match(k) or _RE_SD_OUTPUT.match(k):
            return ARCH_SD
        if _RE_FLAT_BLOCK.match(k):
            return ARCH_FLAT
    return ARCH_UNKNOWN


def classify_key(key: str, arch: str, num_double: int = 0) -> int | None:
    """Map a patch key (model state-dict key) to its block index, or None for
    the non-block bucket (factor fixed at 1.0)."""
    k = _strip_prefix(key)
    if arch == ARCH_FLAT:
        if k.startswith(_FLAT_NON_BLOCK_PREFIXES):
            return None
        m = _RE_FLAT_BLOCK.match(k)
        return int(m.group(1)) if m else None
    if arch == ARCH_FLUX:
        m = _RE_FLUX_DOUBLE.match(k)
        if m:
            return int(m.group(1))
        m = _RE_FLUX_SINGLE.match(k)
        if m:
            return int(m.group(1)) + num_double
        return None
    if arch == ARCH_SD:
        m = _RE_SD_INPUT.match(k)
        if m:
            return int(m.group(1)) + 1
        if _RE_SD_MIDDLE.match(k):
            return 13
        m = _RE_SD_OUTPUT.match(k)
        if m:
            return int(m.group(1)) + 14
        return 0  # BASE slot in the 26-block convention
    return None


def infer_block_layout(keys, arch: str) -> tuple[int, int]:
    """(block_count, num_double) inferred from patch keys. For the SD family
    the count is the 26-slot convention. Prefer model introspection in the
    script (len(dm.blocks)); this is the offline/LoRA-side fallback."""
    if arch == ARCH_SD:
        return 26, 0
    d_max = s_max = flat_max = -1
    for key in keys:
        k = _strip_prefix(key)
        if arch == ARCH_FLUX:
            m = _RE_FLUX_DOUBLE.match(k)
            if m:
                d_max = max(d_max, int(m.group(1)))
            m = _RE_FLUX_SINGLE.match(k)
            if m:
                s_max = max(s_max, int(m.group(1)))
        elif arch == ARCH_FLAT:
            if k.startswith(_FLAT_NON_BLOCK_PREFIXES):
                continue
            m = _RE_FLAT_BLOCK.match(k)
            if m:
                flat_max = max(flat_max, int(m.group(1)))
    if arch == ARCH_FLUX:
        return d_max + 1 + s_max + 1, d_max + 1
    return flat_max + 1, 0


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------

@dataclass
class TimeCurve:
    preset: str = "FLAT"
    modifier: str = "Emphasize"
    contrast: float = 0.0
    boost: float = 1.0
    hi_boundary: float = DEFAULT_TIME_HI_BOUNDARY
    lo_boundary: float = DEFAULT_TIME_LO_BOUNDARY
    transition: float = DEFAULT_TIME_TRANSITION
    zone_mean: float | None = None  # p for Emphasize; set by prepare()

    def zone(self) -> tuple[float | None, float | None]:
        if self.preset == "COMPOSITION":
            return self.hi_boundary, None
        if self.preset == "MIDRANGE":
            return self.lo_boundary, self.hi_boundary
        if self.preset == "DETAIL":
            return None, self.lo_boundary
        raise ValueError(f"unknown time preset: {self.preset}")

    def _window_at(self, sigma_norm: float) -> float:
        lo, hi = self.zone()
        return _window(sigma_norm, lo, hi, self.transition)

    def prepare(self, step_sigmas_norm) -> None:
        """Fix the Emphasize budget to the actual run: p = mean window over
        the FULL schedule's model-call sigmas (normalized). Conserving over
        the full schedule (not an img2img/hires slice) keeps partial passes
        on the same absolute curve. No-op for other modifiers/presets."""
        if self.preset == "FLAT" or not step_sigmas_norm:
            return
        windows = [self._window_at(s) for s in step_sigmas_norm]
        self.zone_mean = sum(windows) / len(windows)

    def _default_zone_mean(self) -> float:
        # Fallback when no schedule was captured: uniform grid over (0, 1].
        n = 200
        return sum(self._window_at((i + 1) / n) for i in range(n)) / n

    def factor(self, sigma_norm: float) -> float:
        """Strength factor at a normalized sigma (sigma / full-schedule
        sigma[0]); high sigma = early in the run."""
        if self.preset == "FLAT" or self.contrast <= 0.0:
            return 1.0
        w = self._window_at(sigma_norm)
        if self.modifier == "Emphasize":
            if self.zone_mean is None:
                self.zone_mean = self._default_zone_mean()
            a = emphasize_amplitude(self.contrast, self.boost)
            return _emphasize_factor(w, self.zone_mean, a)
        return _apply_modifier(w, self.modifier, self.contrast)


def normalize_sigma(sigma: float, schedule_sigma0: float) -> float:
    """Normalize by the FULL schedule's first sigma so the same boundaries
    work on flow models (sigma0 == 1.0, identity) and epsilon models
    (sigma0 ~ 14.6). img2img/hires start mid-schedule and land mid-curve."""
    if schedule_sigma0 <= 0.0:
        return sigma
    return sigma / schedule_sigma0


# ---------------------------------------------------------------------------
# LoRA include/exclude filter
# ---------------------------------------------------------------------------

def parse_patterns(text: str) -> list[str]:
    return [p.strip().lower() for p in (text or "").split(",") if p.strip()]


def lora_is_scheduled(filename: str, patterns: list[str], mode: str) -> bool:
    """mode 'exclude': schedule unless a pattern matches the basename.
    mode 'include': schedule only if a pattern matches (empty list -> none)."""
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    matched = any(p in base for p in patterns)
    if mode == "include":
        return matched
    return not matched


# ---------------------------------------------------------------------------
# Per-step scheduling of OnlineLoRAPatch objects
# ---------------------------------------------------------------------------

def parse_mask_override(text: str, count: int) -> list[float] | None:
    """Dev-only: parse an explicit per-block mask ('1, 0.5, ...'). Returns
    None (with no side effects) unless exactly `count` finite values parse."""
    tokens = [t for t in re.split(r"[,;\s]+", (text or "").strip()) if t]
    if len(tokens) != count:
        return None
    try:
        values = [float(t) for t in tokens]
    except ValueError:
        return None
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):
        return None
    return values


def _payload_error(entry) -> str | None:
    if not isinstance(entry, (tuple, list)) or len(entry) != PATCH_TUPLE_LEN:
        got = len(entry) if isinstance(entry, (tuple, list)) else type(entry).__name__
        return f"payload arity mismatch: expected {PATCH_TUPLE_LEN}, got {got}"
    if not isinstance(entry[0], (int, float)):
        return f"payload strength is not a number: {type(entry[0]).__name__}"
    return None


def _set_payload_strength(container: list, index: int, strength: float) -> None:
    entry = container[index]
    if isinstance(entry, list):
        entry[0] = strength
    else:
        container[index] = (strength,) + tuple(entry[1:])


# ---------------------------------------------------------------------------
# Compile mode: scale freshly added BAKED patches by their block factor, once,
# before any weight is touched. The ordinary baked path then merges them and
# the generation runs at full native speed.
# ---------------------------------------------------------------------------

def scale_new_baked_patches(before_counts: dict[str, int], patches: dict,
                            factor_for_key) -> tuple[int, list[str]]:
    """Multiply the strength of every payload added since `before_counts` by
    factor_for_key(key). Validate-all-then-apply: on any drift error nothing
    is scaled. Returns (scaled_count, errors)."""
    targets = []
    errors = []
    for key, plist in patches.items():
        for i in range(before_counts.get(key, 0), len(plist)):
            err = _payload_error(plist[i])
            if err is not None:
                errors.append(f"{key}: {err}")
            else:
                targets.append((key, plist, i))
    if errors:
        return 0, errors
    for key, plist, i in targets:
        factor = float(factor_for_key(key))
        if factor != 1.0:
            _set_payload_strength(plist, i, float(plist[i][0]) * factor)
    return len(targets), []


# ---------------------------------------------------------------------------
# Seed Variance: a self-loaded LoRA baked at strength X, switched off (de-
# baked) when the run leaves the composition zone. This handle is the pure
# bookkeeping half; the weight surgery lives in the Forge script.
# ---------------------------------------------------------------------------

def sv_is_priority(name: str) -> bool:
    """Seed-variance LoRA name heuristic: both 'turbo' and 'sda', or a
    delimited 'sda' token."""
    low = name.lower()
    if "turbo" in low and "sda" in low:
        return True
    return re.search(r"(^|[-_ .])sda([-_ .]|$)", low) is not None


def sort_sv_choices(names) -> list[str]:
    pri = sorted((n for n in names if sv_is_priority(n)), key=str.lower)
    rest = sorted((n for n in names if not sv_is_priority(n)), key=str.lower)
    return pri + rest


@dataclass
class BakedLoraHandle:
    """Tracks one baked LoRA's payloads inside a patcher's `patches` dict by
    (key, adapter identity), so strengths can be swapped between X and 0 and
    the payloads removed cleanly at end of job."""
    entries: list = field(default_factory=list)  # (key, adapter object)
    base_strength: float = 1.0
    current_strength: float = 1.0

    @classmethod
    def collect(cls, before_counts: dict[str, int], patches: dict,
                base_strength: float):
        """Adopt payloads added since `before_counts`. Returns (handle, errors);
        on any error the handle is None and nothing is adopted."""
        entries = []
        errors = []
        for key, plist in patches.items():
            for i in range(before_counts.get(key, 0), len(plist)):
                err = _payload_error(plist[i])
                if err is not None:
                    errors.append(f"{key}: {err}")
                else:
                    entries.append((key, plist[i][1]))
        if errors:
            return None, errors
        return cls(entries=entries, base_strength=float(base_strength),
                   current_strength=float(base_strength)), []

    def _locate(self, patches: dict, key: str, adapter) -> tuple[list, int] | None:
        plist = patches.get(key)
        if not plist:
            return None
        for i, entry in enumerate(plist):
            if isinstance(entry, (tuple, list)) and len(entry) == PATCH_TUPLE_LEN \
                    and entry[1] is adapter:
                return plist, i
        return None

    def keys(self) -> list[str]:
        seen = []
        for key, _ in self.entries:
            if key not in seen:
                seen.append(key)
        return seen

    def present_in(self, patches: dict) -> bool:
        return bool(self.entries) and all(
            self._locate(patches, key, adapter) is not None
            for key, adapter in self.entries)

    def set_strength(self, patches: dict, strength: float) -> list[str]:
        """Write `strength` into every owned payload. Returns the affected
        keys (for the caller to re-bake their weights)."""
        affected = []
        for key, adapter in self.entries:
            loc = self._locate(patches, key, adapter)
            if loc is None:
                continue
            plist, i = loc
            _set_payload_strength(plist, i, strength)
            if key not in affected:
                affected.append(key)
        self.current_strength = float(strength)
        return affected

    def remove_from(self, patches: dict) -> tuple[list[str], list[str]]:
        """Delete owned payloads from the dict. Returns (our_keys,
        emptied_keys) where emptied keys had no other patches and were popped
        — their backups are exclusively ours and safe to drop once weights
        are restored."""
        our_keys = self.keys()
        emptied = []
        for key, adapter in self.entries:
            loc = self._locate(patches, key, adapter)
            if loc is None:
                continue
            plist, i = loc
            del plist[i]
            if not plist and key in patches:
                del patches[key]
                emptied.append(key)
        self.entries = []
        return our_keys, emptied


@dataclass
class ScheduledEntry:
    obj: object          # OnlineLoRAPatch-like: .patch = [5-tuple]
    base_strength: float
    block_factor: float
    key: str = ""


@dataclass
class ScheduleSet:
    entries: list[ScheduledEntry] = field(default_factory=list)
    last_factor: float | None = None

    @staticmethod
    def snapshot_counts(wrapper_patches: dict) -> dict[str, int]:
        return {key: len(objs) for key, objs in wrapper_patches.items()}

    def collect(self, before_counts: dict[str, int], wrapper_patches: dict,
                block_factor_for_key) -> list[str]:
        """Adopt every OnlineLoRAPatch added since `before_counts` was taken.
        Returns a list of error strings (empty on success); on any error no
        entry from this collection round is adopted."""
        errors = []
        adopted = []
        for key, objs in wrapper_patches.items():
            for obj in objs[before_counts.get(key, 0):]:
                err = validate_patch_object(obj)
                if err is not None:
                    errors.append(f"{key}: {err}")
                    continue
                adopted.append(ScheduledEntry(
                    obj=obj,
                    base_strength=float(obj.patch[0][0]),
                    block_factor=float(block_factor_for_key(key)),
                    key=key,
                ))
        if errors:
            return errors
        self.entries.extend(adopted)
        self.last_factor = None
        return []

    def apply_time_factor(self, time_factor: float) -> None:
        """Rewrite every owned patch payload's strength to
        base * block_factor * time_factor. Cheap no-op if unchanged.
        List payloads (Forge c2ae52e5+) are written in place; tuple
        payloads (older builds) are rebuilt."""
        if self.last_factor is not None and time_factor == self.last_factor:
            return
        self.last_factor = time_factor
        for e in self.entries:
            entry = e.obj.patch[0]
            strength = e.base_strength * e.block_factor * time_factor
            if isinstance(entry, list):
                entry[0] = strength
            else:
                e.obj.patch[0] = (strength,) + entry[1:]

    def rebind_block_factors(self, block_factor_for_key) -> None:
        """Recompute block factors for existing entries (block preset changed
        without a LoRA reload) and force a rewrite on the next apply."""
        for e in self.entries:
            e.block_factor = float(block_factor_for_key(e.key))
        self.last_factor = None

    def clear(self) -> None:
        self.entries.clear()
        self.last_factor = None
