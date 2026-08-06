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
must_reject("blocking packet count", "**180 Release packets**", "**179 Release packets**")
must_reject("blocking passing cells", "**2/15 layer/coefficient cells passed**", "**3/15 layer/coefficient cells passed**")
must_reject("blocking selected cell", "**block 10, coefficient 4**", "**block 8, coefficient 4**")
must_reject("blocking wedding median", "`+0.078512` for wedding", "`+0.088512` for wedding")
must_reject("Stage 3 unit count", "**4 prompts × 2 topics = 8 prompt-topic units**", "**8 prompts × 2 topics = 16 prompt-topic units**")
must_reject("Stage 3 residual NLL", "`1.984159` for the semantic residual arm", "`1.884159` for the semantic residual arm")
must_reject("Stage 3 NLL excess", "`0.119157` nat/token", "`0.019157` nat/token")
must_reject("Stage 3 ratio withholding", "No point ratio or ratio interval is reported.", "A point ratio is reported.")
must_reject("topic-specificity disclaimer", "or show **topic specificity**", "and also shows **topic specificity**")
must_reject("in-packet dense collapse count", "into step one in **8 of 8** runs", "into step one in **7 of 8** runs")
must_reject("in-packet sparse exception count", "the sparse logit bias did so in **1 of 8**", "the sparse logit bias did so in **0 of 8**")
must_reject("in-packet sparse exception packet", "(`wedding-study`, at 99.9998%)", "(`ocean-study`, at 99.9998%)")
must_reject("sanity median biased steps", "biased steps**, median 3, 5 of 6", "biased steps**, median 9, 5 of 6")
must_reject("hero post-hoc disclosure", "selected post hoc rather than drawn", "drawn uniformly at random")
must_reject("hero logit-gain rank", "rank **1/8** on logit-bias topic gain", "rank **4/8** on logit-bias topic gain")
must_reject("hero residual-gain rank", "**2/8** on residual topic gain", "**6/8** on residual topic gain")
must_reject("hero per-prompt KL", "`1.143989` nats/step", "`0.443989` nats/step")
