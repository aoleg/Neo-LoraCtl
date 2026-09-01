# Neo-LoraCtl

Fine-grained LoRA control for **Stable Diffusion WebUI Forge Neo**: decide **where** in the model (blocks) and **when** in the sampling run (timesteps) your prompt's `<lora:...>` networks apply — with simple presets instead of number grids.

Built for **Krea 2** first, where character LoRAs often drag in unwanted style and style LoRAs distort characters. SD 1.5 / SDXL and Flux checkpoints are supported on a best-effort basis; unknown architectures fall back gracefully (timestep control still works, block control switches off).

## Status

In development. The mechanism is fully built and offline-tested; live calibration of the preset defaults on Krea 2 is ongoing. Treat preset boundaries as provisional.

## Installation

Copy or clone this folder into `extensions/` inside your Forge Neo install and restart the UI. No extra Python packages are needed.

## Usage

Add LoRAs to your prompt as usual (`<lora:my_character:0.8>`), open the **Neo-LoraCtl** accordion, tick **Enable**, and pick a preset on either axis (or both).

### Blocks — what the LoRA is allowed to shape

| Preset | Focus |
|---|---|
| `FULL` | No block filtering (default). |
| `COMPOSITION` | Early blocks: layout, poses, macro geometry. |
| `CHARACTER` | Middle blocks: subject identity, faces, concepts. |
| `STYLE` | Late blocks: textures, coloring, lighting, rendering style. |

### Timesteps — when in the run the LoRA applies

| Preset | Focus |
|---|---|
| `FLAT` | No time scheduling (default). |
| `COMPOSITION` | Early steps (high sigma), where the image layout forms. |
| `CHARACTER` | Middle steps, where subjects take shape. |
| `DETAIL` | Late steps, where fine detail and style are rendered. |

### Modifier and Contrast

Each axis has a modifier and a contrast slider:

- **Emphasize**: the LoRA runs at full prompt strength inside the chosen zone and is attenuated outside it.
- **Suppress**: the LoRA is attenuated inside the zone and runs at full strength everywhere else.
- **Contrast** sets how deep the attenuation goes: `0` = no effect at all, `1` = fully off in the attenuated region. Values around `0.6-0.8` are a good start; hard masking (`1.0`) can degrade LoRAs whose blocks depend on each other.

The two axes multiply, so `CHARACTER` blocks x `CHARACTER` timesteps narrows a LoRA down hard, and a strength never exceeds what your `<lora:...:s>` tag asked for.

Typical recipes: a character LoRA that drags its training style into everything — blocks `STYLE` + `Suppress`; a style LoRA that deforms faces — blocks `CHARACTER` + `Suppress`, or timesteps `DETAIL` + `Emphasize`.

### Filter

Some LoRAs must not be scheduled at all — above all **accelerator LoRAs** (Turbo, Lightning, Hyper, LCM, DMD), which are part of the checkpoint's distillation and break when their strength changes mid-run. The filter excludes them by default via name matching; edit the pattern list, or switch to `include` mode to schedule only the LoRAs you name.

### Text encoder

The text-encoder half of a LoRA is applied once, at prompt encoding, so it cannot be scheduled; it runs at your prompt-tag strength. Untick the TE checkbox to drop it entirely (useful when testing what the model half does on its own).

### Hires fix and img2img

No settings needed: scheduling is anchored to the sampler's actual noise level, so the hires pass and img2img automatically land on the late part of the timestep curve, matching what those passes really do.

### XYZ grid

Eight axes are registered under `(LoraCtl) ...` — both presets, modifiers, contrasts, and the two timestep zone boundaries — so you can sweep any of them systematically.

### Reproducibility

All settings are written into the generation's infotext (`LoraCtl blocks`, `LoraCtl time`, `LoraCtl TE`, `LoraCtl filter`).

## Notes and caveats

- Scheduled LoRAs are applied **on-the-fly** (never merged into the weights), which costs some speed on the affected layers — the same trade-off as Forge's own "Patch LoRAs on-the-fly" option, applied only to the LoRAs you schedule.
- Enable **Debug logging** to see the captured sigma schedule, the zone boundaries, which LoRAs were scheduled, and a per-run summary — include that output in any bug report.
- The extension pins a specific Forge Neo internal API and refuses (loudly, in the console) rather than misbehaving if Forge changes it; your LoRAs then still apply at plain prompt strength.

## Technical documentation

See [docs/MECHANISM.md](docs/MECHANISM.md) for how the extension integrates with Forge Neo, and [docs/PLAN.md](docs/PLAN.md) for the development plan.
