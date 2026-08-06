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
    max(FINAL[p]["residentMemoryBytes"] for p in ("baseline", "steered")) / 2**20 / 10
)
must_reject("memory", f"about {memory_mib} MB resident memory", "about 999 MB resident memory")
must_reject("test count", "has ten tests", "has nine tests")
must_reject(
    "Qwen revision",
    "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
    "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f0",
)
