#!/usr/bin/env python3
"""Verify README claims against committed repository artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args()
    readme = args.readme.read_text()

    def require(fragment: str) -> None:
        check(fragment in readme, f"README fragment not found: {fragment!r}")

    final = json.loads((ROOT / "docs/final-demo-run.json").read_text())
    check(
        final["baseline"]["tokenCount"] == final["steered"]["tokenCount"] == 96,
        "final run is not a matched 96-token pair",
    )
    require("default 96-token")
    require(f"{final['baseline']['tokensPerSecond']:.1f} baseline")
    require(f"{final['steered']['tokensPerSecond']:.1f} steered")
    memory_mib = 10 * round(
        max(final[p]["residentMemoryBytes"] for p in ("baseline", "steered")) / 2**20 / 10
    )
    require(f"about {memory_mib} MB resident memory")
    require(f"{final['cumulativeKL']:.4f} cumulative KL")
    require(
        f"`{final['baseline']['topicScore']:.4f}` to "
        f"`{final['steered']['topicScore']:.4f}`"
    )
    check(final["buildConfiguration"] == "Release", "final run was not recorded in Release")
    require("Evidence runs use Release builds")
    print("PASS final-run metrics and Release provenance: docs/final-demo-run.json")

    rendered_table = subprocess.run(
        ["python3", str(ROOT / "Scripts/summarize_sanity.py")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(rendered_table)
    sanity_paths = sorted((ROOT / "docs/sanity-runs").glob("*.json"))
    check(len(sanity_paths) == 6, f"expected 6 sanity packets, found {len(sanity_paths)}")
    sanity = [json.loads(path.read_text()) for path in sanity_paths]
    check(all(row["buildConfiguration"] == "Release" for row in sanity), "a sanity run is not Release")
    history_lengths = [len(row["klHistory"]) for row in sanity]
    require(f"over {min(history_lengths)}–{max(history_lengths)} biased steps")
    print("PASS sanity table, packet count, KL-step range, and Release provenance")

    comparison_rows: list[str] = []
    for cap4_path in sorted((ROOT / "docs/negative-results/matched-kl4").glob("*.json")):
        cap8_path = ROOT / "docs/sanity-runs" / cap4_path.name
        check(cap8_path.exists(), f"missing cap-8 match for {cap4_path.name}")
        cap4 = json.loads(cap4_path.read_text())
        cap8 = json.loads(cap8_path.read_text())
        check(
            (cap4["lexicon"], cap4["biasStrength"])
            == (cap8["lexicon"], cap8["biasStrength"]),
            f"parameter mismatch for {cap4_path.name}",
        )
        identical = cap4["steered"]["text"] == cap8["steered"]["text"]
        row = (
            f"| {cap4['lexicon'].title()} | {cap4['biasStrength']:g} | "
            f"{'yes' if identical else 'no'} | {cap4['steered']['topicScore']:.6f} | "
            f"{cap8['steered']['topicScore']:.6f} |"
        )
        comparison_rows.append(row)
        require(row)
    check(len(comparison_rows) == 6, f"expected 6 KL-cap comparisons, found {len(comparison_rows)}")
    ocean = [row for row in sanity if row["lexicon"] == "ocean"]
    check(
        [row["steered"]["text"] == row["baseline"]["text"] for row in ocean]
        == [True, False, False],
        "ocean cap-8 threshold pattern changed",
    )
    require("the permitted cost changes sampled behavior for ocean")
    require("wedding plateaus by 4 nats")
    print("PASS all six KL-4/KL-8 comparisons and linked-directory conclusions")

    decomposition = json.loads((ROOT / "docs/judge-decomposition.json").read_text())
    scores = decomposition["scores"]
    for key in ("baseline", "steered", "prefix_plus_baseline", "steered_prefix_removed"):
        require(f"{scores[key]:.4f}")
    require(f"{decomposition['prefix_only_shift']:.4f} / {decomposition['observed_shift']:.4f}")
    require(f"{100 * decomposition['prefix_fraction_of_observed_shift']:.1f}%")
    print("PASS judge decomposition: docs/judge-decomposition.json")

    parity = json.loads((ROOT / "docs/coreml-parity.json").read_text())
    metadata = parity["metadata"]
    check(len(parity["rows"]) == metadata["validation_cases"] == 24, "Core ML case count changed")
    require("24 inputs")
    require(f"`{metadata['minimum_cosine']:.9f}`")
    require(f"`{metadata['threshold']:.4f}` gate")
    check(metadata["max_length"] == 128, "Core ML max length changed")
    require("fixed 128-token inputs")
    check(metadata["compute_units"] == "ALL", "Core ML compute units changed")
    check(len(metadata["weight_sha256"]) == 64, "Core ML weight digest is not SHA-256")
    print("PASS Core ML parity: docs/coreml-parity.json")

    centroids = json.loads((ROOT / "Resources/CoreML/topic-centroids.json").read_text())
    check(centroids["dimensions"] == 384, "Core ML output dimensions changed")
    require("384-dimensional embedding")
    require(centroids["model_revision"])
    package_bytes = sum(
        path.stat().st_size
        for path in (ROOT / "Resources/CoreML/TopicEncoder.mlpackage").rglob("*")
        if path.is_file()
    )
    package_mib = round(package_bytes / 2**20)
    require(f"{package_mib} MB in FP16")
    print("PASS Core ML dimensions, revision, and package size: Resources/CoreML")

    test_count = sum(
        path.read_text().count("@Test")
        for path in (ROOT / "SteeringKit/Tests/SteeringKitTests").glob("*.swift")
    )
    number_words = {10: "ten"}
    require(f"has {number_words.get(test_count, str(test_count))} tests")
    print("PASS test-count claim: SteeringKit/Tests/SteeringKitTests/*.swift")

    service_source = (ROOT / "SteerDemo/MLXGenerationService.swift").read_text()
    model_id = re.search(r'static let modelID = "([^"]+)"', service_source)
    model_revision = re.search(r'static let modelRevision = "([0-9a-f]+)"', service_source)
    check(model_id is not None and model_revision is not None, "could not parse pinned Qwen model")
    require(f"`{model_id.group(1)}`")
    require(f"`{model_revision.group(1)}`")
    print("PASS Qwen model ID and pinned revision: SteerDemo/MLXGenerationService.swift")

    audit = json.loads((ROOT / "docs/audit-reference.json").read_text())
    require(f"`rho = {audit['rho']['point']}`")
    require(f"`n = {audit['n_eval']}`")
    require(f"[{audit['rho']['ci_lo']}, {audit['rho']['ci_hi']}]")
    require(f"verdict is `{audit['verdict']['class']}`")
    require("95% CI 85.3%–107.1%")
    print("PASS upstream audit point, interval, n, and verdict: docs/audit-reference.json")

    require("Codex generated most of the implementation, tests, and documentation")
    require("directed, agent-assisted work")
    require("makes no claim of personal Swift implementation experience")
    print("PASS implementation-provenance disclosure")


if __name__ == "__main__":
    main()
