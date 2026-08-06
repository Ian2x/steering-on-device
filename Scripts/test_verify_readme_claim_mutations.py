#!/usr/bin/env python3
"""Adversarially mutate claims that the earlier verifier did not recompute."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
VERIFY = ROOT / "Scripts/verify_readme_claims.py"


def must_reject(name: str, before: str, after: str) -> None:
    if README.count(before) != 1:
        raise AssertionError(
            f"mutation source for {name!r} must occur exactly once; found {README.count(before)}"
        )
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
    print(f"PASS rejected adversarial mutation: {name}")


must_reject("audit percentage", "reported that a static logit-bias controller reproduced 95.9%", "reported that a static logit-bias controller reproduced 94.9%")
must_reject("audit exact ratio", "`rho = 0.9586776859504132`", "`rho = 0.9486776859504132`")
must_reject("hero baseline score", "topic scores are `-0.0338` and", "topic scores are `-0.1338` and")
must_reject("hero steered score", "and `0.3815`. Single-run", "and `0.4815`. Single-run")
must_reject("comparison temperature", "temperature `0.7`", "temperature `0.8`")
must_reject("sparse intervention range", "**2–18 biased steps**", "**2–17 biased steps**")
must_reject("dense first-step range", "**97.4%–99.9%**", "**96.4%–99.9%**")
must_reject("audit per-step KL", "`0.43523801873284795` nats per step", "`0.43523801873284790` nats per step")
must_reject("layer degeneracy case count", "in all four matched cases", "in all three matched cases")
must_reject("warm-up pane count", "before measuring all three panes", "before measuring either pane")
