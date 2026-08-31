"""UI smoke test: build the script's gradio UI with the REAL gradio from the
Forge venv (fake modules, real components). Run with the venv python:

    <forge>/venv/Scripts/python.exe tests/smoke_ui.py

Catches gradio API misuse the fake-gradio harness cannot see."""

import importlib.util
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

modules = types.ModuleType("modules")
modules.__path__ = []
scripts_mod = types.ModuleType("modules.scripts")
scripts_mod.Script = type("Script", (), {})
scripts_mod.AlwaysVisible = object()
scripts_mod.basedir = lambda: REPO
shared_mod = types.ModuleType("modules.shared")
callbacks_mod = types.ModuleType("modules.script_callbacks")
callbacks_mod.on_cfg_denoiser = lambda fn: None
modules.scripts = scripts_mod
modules.shared = shared_mod
modules.script_callbacks = callbacks_mod
for name, mod in (("modules", modules), ("modules.scripts", scripts_mod),
                  ("modules.shared", shared_mod), ("modules.script_callbacks", callbacks_mod)):
    sys.modules[name] = mod

import gradio as gr

spec = importlib.util.spec_from_file_location(
    "neo_loractl", os.path.join(REPO, "scripts", "neo_loractl.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

with gr.Blocks():
    components = mod.NeoLoraCtlScript().ui(False)

assert len(components) == 8, f"expected 8 components, got {len(components)}"
print(f"OK: gradio {gr.__version__}, {len(components)} components built")
