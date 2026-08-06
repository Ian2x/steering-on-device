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
        final["baseline"]["tokenCount"]
        == final["steered"]["tokenCount"]
        == final["actAdd"]["tokenCount"]
        == 96,
        "final run is not a matched 96-token triple",
    )
    require("default 96-token")
    require(f"{final['baseline']['tokensPerSecond']:.1f} baseline")
    require(f"{final['steered']['tokensPerSecond']:.1f} logit-bias")
    require(f"{final['actAdd']['tokensPerSecond']:.1f} ActAdd")
    memory_mib = 10 * round(
        max(final[p]["residentMemoryBytes"] for p in ("baseline", "steered", "actAdd"))
        / 2**20
        / 10
    )
    require(f"about {memory_mib} MB peak resident memory")
    check(final["cumulativeKL"] <= final["klBudget"] + 1e-6, "logit-bias KL exceeded cap")
    check(final["actAddCumulativeKL"] <= final["klBudget"] + 1e-6, "ActAdd KL exceeded cap")
    require(f"{final['cumulativeKL']:.4f} cumulative KL for each intervention")
    require(
        f"`{final['steered']['topicScore']:.4f}` under logit bias and "
        f"`{final['actAdd']['topicScore']:.4f}` under ActAdd"
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

    sweep = json.loads((ROOT / "docs/phase6/layer-sweep/summary.json").read_text())
    sweep_paths = sorted((ROOT / "docs/phase6/layer-sweep/runs").glob("*.json"))
    check(len(sweep_paths) == sweep["runCount"] == 24, "layer-sweep packet count changed")
    sweep_packets = [json.loads(path.read_text()) for path in sweep_paths]
    check(all(row["buildConfiguration"] == "Release" for row in sweep_packets), "a layer sweep packet is not Release")
    check(all(row["actAddCumulativeKL"] <= row["klBudget"] + 1e-6 for row in sweep_packets), "a layer sweep packet exceeds KL")
    check(sweep["selectedLayer"] == 3, "selected ActAdd layer changed")
    selected = next(row for row in sweep["perLayer"] if row["layer"] == 3)
    require("all 24 Release packets")
    require("Blocks 3 and 19 tied")
    require(f"`{selected['medianAbsoluteTopicShift']}`")
    require("block **3** was therefore selected")
    print("PASS preregistered 24-packet layer sweep and selected layer")

    rho = json.loads((ROOT / "docs/phase6/on-device-rho/summary.json").read_text())
    rho_paths = sorted((ROOT / "docs/phase6/on-device-rho/runs").glob("*.json"))
    check(len(rho_paths) == rho["n"] == 8, "on-device rho packet count changed")
    rho_packets = [json.loads(path.read_text()) for path in rho_paths]
    check(all(row["buildConfiguration"] == "Release" for row in rho_packets), "a rho packet is not Release")
    check(all(row["cumulativeKL"] <= row["klBudget"] + 1e-6 for row in rho_packets), "a rho logit pass exceeds KL")
    check(all(row["actAddCumulativeKL"] <= row["klBudget"] + 1e-6 for row in rho_packets), "a rho ActAdd pass exceeds KL")
    require(f"`n = {rho['n']}`")
    require(f"`{rho['logitBiasMeanShift']}`")
    require(f"`{rho['actAddMeanShift']}`")
    require(f"`rho = {rho['rho']}`")
    require(f"on-device result was **`rho = {rho['rho']}`**")
    require("No confidence interval is reported")
    check(rho["confidenceInterval"] is None, "small-n rho unexpectedly has a CI")
    print("PASS held-out on-device rho, n, KL caps, and no-CI claim")

    zero = json.loads((ROOT / "docs/phase6/coefficient-zero/report.json").read_text())
    check(zero["actAddCoefficient"] == 0, "coefficient-zero packet is not zero")
    check(zero["baseline"]["text"] == zero["actAdd"]["text"], "coefficient-zero text differs")
    check(zero["baseline"]["tokenCount"] == zero["actAdd"]["tokenCount"], "coefficient-zero count differs")
    require("identical baseline and ActAdd text and token count")
    print("PASS coefficient-zero Release packet identity")

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
    require(f"has {test_count} tests")
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

    before = json.loads((ROOT / "LoRA/results/before.json").read_text())
    after = json.loads((ROOT / "LoRA/results/after.json").read_text())
    adapter = ROOT / "LoRA/adapter/adapters.safetensors"
    check(before["n"] == after["n"] == 9, "LoRA evaluation n changed")
    check(before["exactMatches"] == 0 and after["exactMatches"] == 9, "LoRA exact-match result changed")
    check(adapter.stat().st_size > 0, "LoRA adapter is empty")
    require("**MLX Python**, not MLX Swift")
    require("120 optimizer steps on 36 toy codebook examples")
    require("`0/9` before training to `9/9` after training")
    require("3 MB adapter")
    print("PASS toy MLX Python LoRA adapter and held-out evaluation")

    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    vendored_revision = "9bff95ca5f0b9e8c021acc4d71a2bbe4a7441631"
    check(vendored_revision in notices, "vendored MLX revision missing from notices")
    require(f"`{vendored_revision}`")
    check((ROOT / "LICENSES/mlx-swift-examples-LICENSE.txt").exists(), "vendored MIT license missing")
    print("PASS vendored Qwen provenance and license")

    require("Codex generated most of the implementation, tests, and documentation")
    require("directed, agent-assisted work")
    require("makes no claim of personal Swift implementation experience")
    print("PASS implementation-provenance disclosure")


if __name__ == "__main__":
    main()
