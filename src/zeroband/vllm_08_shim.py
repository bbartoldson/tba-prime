"""
Shim to expose process_weights_after_loading for vLLM 0.8.x.

Import this **before** any other tba-prime or vllm code runs.
"""

import importlib

# 1) Import the utils module (creates it if absent).
utils_mod = importlib.import_module("vllm.model_executor.model_loader.utils")

# 2) Try to grab the public symbol. If it already exists (vLLM ≥0.9)
# do nothing; otherwise pull in the private one from loader.py.
if not hasattr(utils_mod, "process_weights_after_loading"):
    loader_mod = importlib.import_module("vllm.model_executor.model_loader.loader")
    utils_mod.process_weights_after_loading = (
        loader_mod._process_weights_after_loading  # noqa: SLF001
    )
