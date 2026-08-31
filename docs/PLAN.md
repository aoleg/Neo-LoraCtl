# Neo-LoraCtl — development plan

Two-axis LoRA strength control for Forge Neo (`sd-webui-forge-classic`, neo branch): per-block (spatial) and per-sigma (temporal) scaling of prompt-loaded LoRAs, preset-driven, targeting Krea 2 first.

This is the technical planning document. User-facing documentation lives in `README.md`. Findings and lessons learned go to `t:\claude\github\knowledge_loractl.md` as development proceeds.

## Design summary (decisions recorded 2026-09-01)

Effective per-key, per-step strength is a product: `user_strength × block_factor(key) × time_factor(sigma)`, with both factors in `[floor, 1.0]` — never above the prompt-tag strength.

| Decision | Resolution |
|---|---|
| Mechanism | `add_patches(online_mode=True)` for scheduled LoRAs, which routes each patch into an `OnlineLoRAPatch` object; per-step rewrite of those objects' patch tuples from an `on_cfg_denoiser` callback. No bypass injection, no Forge modification. Plan B if the patcher API churns again: activation-space injection via forward hooks (see krea-multi-lora notes in `knowledge_loractl.md`). |
| Scope | Prompt-loaded LoRAs (`<lora:name:s>`), via interception of `networks.load_lora_for_models` with snapshot-diff to attribute patches to files. |
| Filter | User-facing include/exclude list (name substrings, case-insensitive) with mode radio. Default: exclude `turbo, lightning, hyper, lcm, dmd`. |
| Text encoder | Untouched by default (stock loader applies TE at prompt strength). Single on/off toggle; off = TE half at strength 0. |
| Time axis | Three zones bounded in **sigma** (not step fraction): COMPOSITION / CHARACTER / DETAIL, plus FLAT. Smooth (smoothstep) transitions in sigma domain. |
| Block axis | Three zones over the block index: COMPOSITION / CHARACTER / STYLE, plus FULL. Krea 2: 28 flat blocks split in even thirds (0–8 / 9–18 / 19–27), smoothstep shoulders of fixed internal width (2–3 blocks). Non-block keys (embeddings, final layer, text-fusion `layerwise_blocks`/`refiner_blocks`) fixed at 1.0. |
| Modifier | Each axis: emphasize (zone at 1.0, rest attenuated to floor) or suppress (zone attenuated, rest at 1.0). |
| Knobs | One contrast slider per axis (0 = flat/no-op, 1 = hard mask; floor = 1 − contrast). Shoulder width and zone boundaries not user-exposed. |
| Hires pass | No special-casing and no toggle: the time factor is evaluated on actual sigma, so hires (starting at a denoise-strength-determined sigma) lands on the tail of the curve automatically. Block masks apply unchanged. |
| UI | One accordion, two symmetric clusters: Blocks [preset | modifier | contrast] and Timesteps [preset | modifier | contrast], plus enable, TE toggle, filter, debug toggle. No numeric per-block/per-step entry in the UI (a dev-only path exists for calibration). |
| Infotext | Every setting written to generation parameters from phase 1. |

## Ground truth carried in from prior projects

- **Target build: Forge Neo commit `92b55e1b` (2026-09-01, neo branch).** The patcher API changed again between `e1df9201` (2026-08) and this build — third layout in five months, see `knowledge_loractl.md` for the history. Current API: patch tuples are 5 elements `(strength, adapter, strength_model, offset, function)`; `add_patches(..., online_mode=True)` does not touch `self.patches` at all — each online patch becomes an `OnlineLoRAPatch(key, tuple)` object appended to `ModelPatcher.weight_wrapper_patches[key]` (`backend/patcher/base.py:100,417`). At load, these extend every affected module's `weight_function` list (`base.py:589-593`), and `get_weight_and_bias` applies them on **every** forward, both manual-cast and plain paths, on a cloned weight (`backend/operations.py:76-90,228`).
- Per-step scheduling therefore rewrites `OnlineLoRAPatch.patch[0] = (new_strength, adapter, sm, offset, fn)` on objects we own — `merge_lora_to_weight` re-reads the object's list per forward, no caching. Attribution is a snapshot-diff of `weight_wrapper_patches` around the intercepted stock-loader call. `add_patches` still bumps `patches_uuid`, so a fresh load installs the wrappers.
- Online and baked application can now coexist on one key (wrappers stack on top of merged weights); a LoRA loaded with `online_mode=True` is exclusively online by construction (its tuples never enter `patches`). Memory reservation follows the wrapper dict (`backend/sampling/sampling_function.py:377`).
- `current_lora_hash` must be nulled whenever our config changes, or Forge reuses the previous LoRA application (pattern from lora-block-weight-neo).
- On Forge Neo, Flux-family schedules run a **fixed** shift (Krea 2: `mu = 1.15`, effective 3.158, resolution-independent; `use_shift` is false for Krea) — so sigma-anchored zones transfer across step counts and resolutions (`knowledge_speed.md` §11, `knowledge_sigmas.md` §5.4).
- Default composition/detail boundary: sigma ≈ 0.90 for Krea 2 (SPEED calibration; Flux measured "perfect" at ≥ 0.9249 on Forge Neo). The CHARACTER/DETAIL boundary is provisional (≈ 0.5) until phase 3.
- Calibration method when needed: pin sigmas, probe with unambiguous details, binary-search, confirm on two resolutions and two step counts (`knowledge_speed.md` §3.1, §11.1).
- `ui-config.json`: set `do_not_save_to_config` on sliders whose ranges the extension defines (`knowledge_sigmas.md` §5.1) but **not** on dropdowns — the platform already guards stale dropdown values (`knowledge_speed.md` §11.3).
- Methodology: one-shot runtime diagnostics before any re-architecting; offline stub harness before any live run (`knowledge.md` §6, §9).

## Phases

### Phase 0 — scaffolding and offline mechanism validation

No live Forge required. Exit criterion: the offline harness is green.

- Repo layout: `scripts/neo_loractl.py` (Forge script), `loractl_core.py` (framework-agnostic math: masks, curves, sigma interpolation, key classification), `tests/` (stub harness), `docs/`.
- Stub harness faking `modules.scripts`, `modules.script_callbacks`, `modules.shared`, the patcher's 6-tuple `patches` dict, and a fake `WeightPatch` consumer — exercising: tuple rewrite by identity, mask generation (all presets × modifiers × contrast values), sigma-zone interpolation, include/exclude filtering, and the contrast-0 no-op identity.
- Key-classification tables for Krea 2 (28 blocks + explicit non-block bucket) validated against key names dumped from `model_lora_keys_unet` structure; generic flat-`blocks.N` fallback.

### Phase 1 — core mechanism + time axis, live

Exit criteria: oracle tests pass live; time presets produce visibly correct behavior on Krea 2.

- Interception of `networks.load_lora_for_models` (snapshot-diff patch attribution), per-key `online_mode=True` for scheduled LoRAs, filter with default excludes, `current_lora_hash` bust on config change.
- Read-only `p.sampler.get_sigmas` wrap for schedule capture; `on_cfg_denoiser` per-step rewrite; state as class attributes with explicit reset (knowledge.md §2 skeleton).
- Time presets FLAT/COMPOSITION/CHARACTER/DETAIL with emphasize/suppress and contrast; TE toggle; infotext.
- One-shot diagnostics behind a debug toggle: captured schedule, resolved zone sigmas, first-step factor, per-key online-patch count, and a postprocess summary distinguishing "never invoked" / "invoked, no effect" / "exception".
- Oracles: (a) disabled == FLAT @ contrast 0 == plain `<lora:x:s>` run, near-pixel-identical; (b) Turbo LoRA in prompt remains bit-identical to stock when the default filter excludes it.
- Live matrix: txt2img / img2img / hires; Krea 2 primary, one SDXL-class model secondary; global on-the-fly option both off and on.

### Phase 2 — block axis, XYZ, docs

Exit criteria: combined masks verified live; XYZ grids usable; README shipped.

- Block masks (thirds, smoothstep shoulders, emphasize/suppress, contrast) folded as static per-key factors into the per-step rewrite; block-axis oracle (contrast 0 == FULL == phase-1 behavior).
- Architectures: Krea 2 first-class; generic flat-DiT fallback; SD/SDXL (in/mid/out) and Flux (double/single) best-effort using lora-block-weight-neo's mapping ideas rebuilt on explicit key classification (no bare `blocks\.(\d+)\.` regex — it collides with Krea's text-fusion keys).
- XYZ script axes for preset / modifier / contrast on both axes (also the phase-3 instrument).
- User-focused `README.md` (what the presets do, when to use which, the filter, one screenshot); technical docs in `docs/` (mechanism, key classification, calibration notes).
- Dev-only numeric mask/curve entry (hidden or config-gated) for calibration use.

### Phase 3 — Krea-2 calibration campaign (spare-time, open-ended)

Exit criteria: none hard; defaults promoted as data accumulates.

- Fixtures: one known character LoRA + one known style LoRA with unambiguous probes (identity features, style signatures; the negative-weight-suppression lesson applies — probe something the base model does not render unbidden).
- Time-zone boundaries: seed-locked binary search on the CHARACTER/DETAIL sigma boundary, two resolutions × two step counts, per `knowledge_speed.md` §3.1.
- Block zones: contiguous-range ablation grids via XYZ against the even-thirds default; adjust boundaries/shoulders only on repeatable evidence; the 0–8 = composition prior gets tested first.
- Promote measured defaults, flag provisional ones in docs, and record everything (including failed hypotheses) in `knowledge_loractl.md`.

## Cross-phase rules

- Source projects (`ComfyUI-SigmaSync-LoRA`, `lora-block-weight-neo`, `krea-multi-lora`, `ComfyUI-SPEED`, `sd-webui-forge-classic`) are read-only references.
- Every phase ends with a `knowledge_loractl.md` update and a commit.
- Pin and record the Forge Neo commit developed against; patch internals have churned twice already (2026-07, 2026-08) and will again. The offline harness must assert the live API shape (tuple arity, `weight_wrapper_patches` presence) so drift fails loudly, not silently.
