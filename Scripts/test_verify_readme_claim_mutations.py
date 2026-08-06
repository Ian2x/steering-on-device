#!/usr/bin/env python3
"""Prove the README verifier rejects claim mutations it previously missed."""

from __future__ import annotations

import subprocess
import tempfile
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
VERIFY = ROOT / "Scripts/verify_readme_claims.py"
FINAL = json.loads((ROOT / "docs/final-demo-run.json").read_text())


def must_reject(name: str, before: str, after: str) -> None:
    if before not in README:
        raise AssertionError(f"mutation source missing for {name}: {before!r}")
    with tempfile.TemporaryDirectory(prefix="steerdemo-readme-mutation-") as directory:
        candidate = Path(directory) / "README.md"
        candidate.write_text(README.replace(before, after, 1))
        result = subprocess.run(
            ["python3", str(VERIFY), "--readme", str(candidate)],
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        raise AssertionError(f"verifier accepted {name} mutation")
    print(f"PASS rejected mutation: {name}")


memory_mib = 10 * round(
    max(FINAL[p]["residentMemoryBytes"] for p in ("baseline", "steered", "actAdd"))
    / 2**20
    / 10
)
must_reject("memory", f"about {memory_mib} MB peak resident memory", "about 999 MB peak resident memory")
must_reject("test count", "has 12 tests", "has 11 tests")
must_reject(
    "Qwen revision",
    "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
    "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f0",
)
must_reject(
    "on-device rho",
    "on-device result was **`rho = 11.51541576343149`**",
    "on-device result was **`rho = 0.9586776859504132`**",
)
must_reject("ActAdd layer", "block **3** was therefore selected", "block **11** was therefore selected")
must_reject("LoRA result", "`0/9` before training to `9/9` after training", "`9/9` before training to `9/9` after training")
must_reject("warm-up count", "untimed one-token same-prompt warm-up", "untimed 16-token same-prompt warm-up")
must_reject("rho zero rows", "including 3 zero logit-bias shifts", "including 2 zero logit-bias shifts")
must_reject("rho token cap", "a maximum of 64 generated tokens", "64 generated tokens")
