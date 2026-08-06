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
must_reject("dense first-step range", "**97.3%–99.9%**", "**96.3%–99.9%**")
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
must_reject("remediated residual slowdown", "**3.5×–6.6×**", "**1.5×–2.0×**")
must_reject("pre-remediation hero-timing scope", "this packet predates the residual remediation", "this packet already uses the residual remediation")

# Literals the earlier verifier left unguarded, now derived from their artifacts.
must_reject("blocking baseline NLL", "`0.983149` baseline", "`0.883149` baseline")
must_reject("blocking semantic NLL", "`1.096959` semantic", "`1.196959` semantic")
must_reject("blocking random NLL", "`0.916626` random", "`0.916627` random")
must_reject("blocking grid dimensions", "3 layers × 5 direct coefficients", "3 layers × 6 direct coefficients")
must_reject("blocking selected-cell length", "Median returned length was 32 tokens", "Median returned length was 30 tokens")
must_reject("frozen gate ordinals", "because gate 1 asks only", "because gate 3 asks only")
must_reject("retained blind calibration packets", "all 152 blind packets were retained", "all 151 blind packets were retained")
must_reject("app pass seed", "seed (`42`), temperature", "seed (`43`), temperature")
must_reject("preserved packet seed", "at seed 42, temperature", "at seed 7, temperature")
must_reject("audit citation year", "The steering audit (Wang, 2026)", "The steering audit (Wang, 2025)")
must_reject("second audit KL target occurrence", "`0.43523801873284795`-nat/step target", "`0.43523801873284790`-nat/step target")
must_reject("calibration continuation length", "target on shared fixed 64-token continuations", "target on shared fixed 32-token continuations")
must_reject("model quantization label", "Instruct (4-bit) through MLX Swift", "Instruct (8-bit) through MLX Swift")
must_reject("sanity KL cap", "an 8.0000-nat KL cap", "an 9.0000-nat KL cap")
must_reject("sanity run length", "These are real 64-token app runs", "These are real 32-token app runs")
must_reject("sanity bias strength", "ocean strength 12 spends", "ocean strength 13 spends")
must_reject("KL-cap table header", "Topic score (KL cap 4)", "Topic score (KL cap 5)")
must_reject("in-packet cumulative cap", "the cumulative 8-nat cap", "the cumulative 9-nat cap")
must_reject("first-step collapse threshold", "more than 97% of its cumulative cap", "more than 87% of its cumulative cap")
must_reject("pre-remediation packet block", "(block 3, nominal coefficient 12,", "(block 5, nominal coefficient 12,")
must_reject("pre-remediation packet length", "its 96-token length inflates", "its 64-token length inflates")
must_reject("remediated blocking packet count", "(180 blocking-control packets)", "(179 blocking-control packets)")
# The 64-token remediated range must cover both teacher-forced arms. Each of these
# mutations restores the semantic-only scope, which excluded the slowest packet.
must_reject("remediated 64-token packet scope", "(16 teacher-forced packets", "(8 teacher-forced packets")
must_reject("remediated 64-token speed range", "`49.8`–`64.7` tok/s", "`53.4`–`64.5` tok/s")
must_reject("remediated packet total", "all **196** remediated packets", "all **188** remediated packets")
must_reject("layer sweep blocks", "blocks 3, 7, 11, 15, 19, and 23 were swept", "blocks 3, 7, 11, 15, 19, and 21 were swept")
must_reject("layer sweep coefficient", "at a nominal coefficient of 12", "at a nominal coefficient of 14")
must_reject("layer sweep packet count", "All 24 Release packets are retained", "All 25 Release packets are retained")
must_reject("layer sweep degenerate pair", "Because blocks 3 and 19 produced the same text", "Because blocks 3 and 23 produced the same text")
must_reject("audit dissolution threshold", "`rho_lo < 0.9`", "`rho_lo < 0.8`")
must_reject("second observed judge shift", "the full `+0.4154` shift", "the full `+0.5154` shift")
must_reject("toy LoRA rank", "four-layer rank-8 LoRA", "four-layer rank-9 LoRA")

# Two Phase 6 protocols were edited after their runs. The disclosure of that, and
# each of its four load-bearing halves, is derived from git and mutation-tested.
must_reject(
    "post-run protocol amendment disclosed",
    "Committed is not the same as unedited.",
    "Every protocol stands exactly as first committed.",
)
must_reject("amending commit named", "in commit `df30259`", "in commit `df30258`")
must_reject(
    "amendment scope",
    "every other predeclared line survives byte-identically",
    "most other predeclared lines survive",
)
must_reject(
    "never-edited protocols",
    "have one commit each and were never edited",
    "have one commit each and were also amended",
)
must_reject(
    "which protocols were amended",
    "[`layer-sweep`](docs/phase6/layer-sweep/protocol.md)",
    "[`blocking-control`](docs/phase6/blocking-control/protocol.md)",
)
must_reject(
    "layer protocol immutability over-claim",
    "committed before its outcomes, then amended after them as disclosed above",
    "committed before its outcomes and never touched afterwards",
)

# Printed ranges must contain their own extremes. Each mutation here rounds one
# endpoint back to nearest, which is exactly the interval-shrinking defect the
# range_floor/range_ceiling renderers exist to prevent.
must_reject("dense first-step floor rounds down", "**97.3%–99.9%**", "**97.4%–99.9%**")
must_reject("blocking speed floor rounds down", "`61.4`–`86.7` tok/s", "`61.5`–`86.7` tok/s")
must_reject("remediated baseline ceiling rounds up", "`365.2` tok/s baseline", "`365.1` tok/s baseline")
must_reject("sparse other-seven ceiling rounds up", "0.02%–25.0% there", "0.02%–24.9% there")
must_reject("audit CI rounds outward", "95% CI 85.2%–107.2%", "95% CI 85.3%–107.1%")
must_reject("bootstrap interval rounds outward", "`[0.065654, 0.319530]`", "`[0.065655, 0.319529]`")

# The sweep itself: a brand-new number that no require() pins must not slip through,
# and an allowlisted literal is excused from artifact derivation, not from change.
must_reject(
    "unguarded number added to the README",
    "Prompts and generated text stay on the Mac.",
    "Prompts and generated text stay on the Mac. It sustains 999 tok/s.",
)
must_reject("allowlisted toolchain floor", "Xcode 26 or newer", "Xcode 27 or newer")
