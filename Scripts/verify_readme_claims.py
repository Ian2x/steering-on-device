#!/usr/bin/env python3
"""Verify README numeric claims that are backed by this repository's artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def require(fragment: str) -> None:
    if fragment not in README:
        raise AssertionError(f"README fragment not found: {fragment!r}")


final = json.loads((ROOT / "docs/final-demo-run.json").read_text())
assert final["baseline"]["tokenCount"] == final["steered"]["tokenCount"] == 96
assert f"{final['baseline']['tokensPerSecond']:.1f} baseline" in README
assert f"{final['steered']['tokensPerSecond']:.1f} steered" in README
assert abs(max(final[p]["residentMemoryBytes"] for p in ("baseline", "steered")) / 2**20 - 570) < 2
assert f"{final['cumulativeKL']:.4f} cumulative KL" in README
assert f"`{final['baseline']['topicScore']:.4f}` to `{final['steered']['topicScore']:.4f}`" in README
print("PASS final-run metrics: docs/final-demo-run.json")

rendered_table = subprocess.run(
    ["python3", str(ROOT / "Scripts/summarize_sanity.py")],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
assert rendered_table in README
print("PASS sanity table: docs/sanity-runs/*.json via Scripts/summarize_sanity.py")

for strength in (12, 14, 16):
    cap4 = json.loads(
        (ROOT / f"docs/negative-results/matched-kl4/wedding-strength{strength}.json").read_text()
    )
    cap8 = json.loads((ROOT / f"docs/sanity-runs/wedding-strength{strength}.json").read_text())
    assert cap4["steered"]["text"] == cap8["steered"]["text"]
    assert cap4["steered"]["topicScore"] == cap8["steered"]["topicScore"]
    require(
        f"| {strength} | yes | {cap4['steered']['topicScore']:.6f} | "
        f"{cap8['steered']['topicScore']:.6f} |"
    )
print("PASS KL-4/KL-8 plateau: docs/negative-results/matched-kl4 + docs/sanity-runs")

decomposition = json.loads((ROOT / "docs/judge-decomposition.json").read_text())
scores = decomposition["scores"]
for key in ("baseline", "steered", "prefix_plus_baseline", "steered_prefix_removed"):
    require(f"{scores[key]:.4f}")
require(f"{decomposition['prefix_only_shift']:.4f} / {decomposition['observed_shift']:.4f}")
require(f"{100 * decomposition['prefix_fraction_of_observed_shift']:.1f}%")
print("PASS judge decomposition: docs/judge-decomposition.json")

parity = json.loads((ROOT / "docs/coreml-parity.json").read_text())
metadata = parity["metadata"]
assert len(parity["rows"]) == metadata["validation_cases"] == 24
require(f"`{metadata['minimum_cosine']:.9f}`")
require(f"`{metadata['threshold']:.4f}` gate")
assert metadata["max_length"] == 128
assert metadata["compute_units"] == "ALL"
assert len(metadata["weight_sha256"]) == 64
print("PASS Core ML parity: docs/coreml-parity.json")

centroids = json.loads((ROOT / "Resources/CoreML/topic-centroids.json").read_text())
assert centroids["dimensions"] == 384
assert centroids["model_revision"] in README
package_bytes = sum(
    path.stat().st_size
    for path in (ROOT / "Resources/CoreML/TopicEncoder.mlpackage").rglob("*")
    if path.is_file()
)
assert round(package_bytes / 2**20) == 43
print("PASS Core ML dimensions, revision, and package size: Resources/CoreML")

test_count = sum(
    path.read_text().count("@Test")
    for path in (ROOT / "SteeringKit/Tests/SteeringKitTests").glob("*.swift")
)
assert test_count == 10
print("PASS test-count claim: SteeringKit/Tests/SteeringKitTests/*.swift")
