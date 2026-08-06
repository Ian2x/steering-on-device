#!/usr/bin/env python3
"""Remove machine-specific paths from the committed adapter metadata."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "LoRA" / "adapter" / "adapter_config.json"
config = json.loads(PATH.read_text())
config["model"] = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
config["model_revision"] = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
config["data"] = "LoRA/data"
config["adapter_path"] = "LoRA/adapter"
PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
print(f"normalized {PATH}")

training_log = ROOT / "LoRA" / "results" / "training.log"
if training_log.exists():
    training_log.write_text(training_log.read_text().replace(str(ROOT), "."))
    print(f"normalized {training_log}")
