"""Inspect upstream imports inside the approved project container."""

import importlib.util
import json
import sys

names = ("torch", "nestconfig", "fvcore", "mamba_ssm", "fla", "triton",
         "einops", "wandb", "yaml", "pandas", "numpy")
report = {name: importlib.util.find_spec(name) is not None for name in names}
report["python"] = sys.version
print(json.dumps(report, indent=2))
