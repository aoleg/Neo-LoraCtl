# Neo-LoraCtl — how it integrates with Forge Neo

Technical companion to [PLAN.md](PLAN.md). Target build: `sd-webui-forge-classic`, neo branch, commit `92b55e1b` (2026-09). The project knowledge base (`knowledge_loractl.md`, kept outside this repo) carries the full investigation history.

## The strength model

Every scheduled LoRA patch gets its strength recomputed per sampling step:

```
effective(key, sigma) = base_strength * block_factor(key) * time_factor(sigma / sigma0)
```

`base_strength` is the prompt-tag strength. `sigma0` is the first sigma of the sampler's full schedule, making the time axis resolution- and step-count-independent on Forge Neo (whose Flux-family schedules run a fixed shift) and best-effort-portable to epsilon models.

Each factor comes from one of three modifiers:

- **Suppress** dims the zone: factor in `[1 - contrast, 1]`, rest untouched.
- **Isolate** keeps only the zone: zone at 1.0, rest in `[1 - contrast, 1]` (the pre-redesign "Emphasize" — live calibration showed this mode crosses the likeness strength threshold on character LoRAs at any useful contrast, so it is reserved for deliberate zone-only application, e.g. style transfer).
- **Emphasize** is a mean-preserving redistribution: `factor = 1 + a*(w - p)/(1 - p)` with amplitude `a = contrast * boost` and `p` the window's mean over the evaluated domain — the zone rises above 1, the rest drops below, and the average stays exactly 1, so the prompt-tag strength acts as a conserved budget. `p` is computed over the classified blocks (block axis) and, once the schedule is captured, over the **full schedule's actual model-call steps** (time axis; `TimeCurve.prepare`) — conserving over the full schedule keeps img2img/hires slices on the same absolute curve. All factors clamp to `[0, 2]`; the clamp binding (`a > 1`) is the one case where conservation bends.

## Integration points

1. **Loader interception** (`networks.load_lora_for_models`, installed lazily on first enabled run). LoRAs passing the include/exclude filter are re-routed with `online_mode=True`. On this build that means their patches never enter `ModelPatcher.patches`; each becomes an `OnlineLoRAPatch(key, tuple)` object appended to `ModelPatcher.weight_wrapper_patches[key]` (`backend/patcher/base.py:417`). At model load these objects extend each module's `weight_function` list, and `backend/operations.py::get_weight_and_bias` re-runs them on a cloned weight at **every** forward — no caching. Rewriting `obj.patch[0]` therefore changes the LoRA's effect on the next model call.
2. **Adoption by snapshot-diff** (`loractl_core.ScheduleSet.collect`). Wrapper-dict counts are snapshotted around the original loader call; every newly appended object is validated against the pinned API shape (5-element tuple, `.patch` list, callable) and adopted with its base strength, patch key, and block factor. On any mismatch nothing is adopted and an error names the drift — the LoRA then applies at base strength like a plain tag.
3. **Per-step drive** (`on_cfg_denoiser`). Fires once per step inside `CFGDenoiser.forward`, before the model call. Reads the step sigma, evaluates the time curve, and rewrites all adopted tuples (a no-op when the factor is unchanged).
4. **Schedule capture** (`p.sampler.get_sigmas`, wrapped read-only in `process_before_every_sampling`). Captures `sigma0` and the full schedule for diagnostics. The hires pass constructs a fresh sampler, gets wrapped again, and its own schedule head keeps normalization consistent.
5. **Reload control** (`current_lora_hash`). The stock loader caches LoRA application by a target-list hash. Whenever scheduling *membership* changes (enable state, filter, TE toggle), the hash is nulled in `before_process_batch` — which runs right before `extra_networks.activate` (`modules/processing.py:962` vs `:970`). Preset/contrast changes never force a reload: block factors are rebound on the live entries and the time factor is per-step anyway.

## Lifecycle

- Entries persist across generations on purpose: with an unchanged hash the stock loader reuses its application and the same `OnlineLoRAPatch` objects stay live in the reused patcher.
- A reload cycle (prompt LoRA change, hires LoRA swap, forced reload) reopens collection: the first intercepted scheduled LoRA clears old entries, and each loader call in the cycle appends.
- A staleness check in `before_process_batch` drops entries whose objects vanished from the current unet's wrapper dict (checkpoint switch) and forces a reload when everything went stale; the block-axis context (architecture, block count) resets with it.
- `postprocess` parks strengths at `base * block_factor` so nothing is left mid-curve between generations.

## Block classification

Patch keys are model state-dict keys (`diffusion_model.<module>.weight`). Classification is anchored-prefix based, never bare substring:

| Architecture | Detection | Blocks |
|---|---|---|
| Krea 2 / flat DiT | `dm.blocks` (introspection) or `^blocks\.(\d+)\.` on keys | `len(dm.blocks)` (Krea 2: 28) |
| Flux | `dm.double_blocks` | double + single, single offset by `num_double` |
| SD / SDXL UNet | `dm.input_blocks` | 26-slot convention (`BASE, IN00-11, M00, OUT00-11`) |

Krea 2's text-fusion modules (`txtfusion.layerwise_blocks.N`, `.refiner_blocks.N`) and non-block modules (`first`, `last`, `tmlp`, `txtmlp`, `tproj`) are explicitly outside the block axis: their factor is pinned to 1.0. The anchored match makes the `layerwise_blocks` collision (which broke another extension's mapping) structurally impossible.

Zone masks are even thirds of the block index space with smoothstep shoulders (fixed 2.5-block width) evaluated at block centers; time zones use smoothstep ramps straddling the sigma boundaries (defaults 0.90 / 0.50, provisional pending calibration). Adjacent zone windows tile to exactly 1.0 (visible directly in Isolate at contrast 1).

## Dev mode

`LORACTL_DEV=1` in the environment reveals two extra fields: an explicit per-block mask (exactly `block_count` comma-separated values, replacing the preset mask) and explicit time boundaries (`hi,lo`). The XYZ axes `(LoraCtl) Time hi/lo boundary` expose the boundaries without dev mode — together these are the phase-3 calibration instruments.

## Failure modes, by design

- Forge API drift: adoption refuses, console error, LoRAs behave like plain tags.
- Unknown architecture: block presets inactive (factor 1.0), time axis unaffected, one console line.
- No schedule captured (exotic sampler): normalization falls back to the first callback sigma with a console line.
- Accelerator LoRAs: excluded by default filter patterns (`turbo, lightning, hyper, lcm, dmd`) — scheduling a distillation LoRA breaks the checkpoint's contract, so keep them excluded.

## Offline verification

`tests/test_core.py` covers the math and the adoption/rewrite engine; `tests/test_script.py` is a mini-Forge (fake `modules`/`gradio`/`networks` reproducing clone list-sharing, hash caching, and hook order) driving full fake generations; `tests/smoke_ui.py` builds the UI against the real gradio from the Forge venv. Run the first two with any Python 3.10+, the third with `<forge>/venv/Scripts/python.exe`.
