#!/usr/bin/env python3
"""Recompute README claims from committed artifacts and require exact placement."""

from __future__ import annotations

import argparse
import json
import re
import statistics
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

    screenshot = ROOT / "docs/steerdemo.png"
    final_frame = ROOT / "docs/demo-frames/99-final.png"
    check(screenshot.read_bytes() == final_frame.read_bytes(), "hero screenshot differs from final demo frame")
    require_hero_alt = (
        "one preserved teacher-forced packet chosen post hoc for this screenshot, "
        "whose controller comparison was withheld by its NLL gate"
    )
    check(
        require_hero_alt in readme,
        "hero alt text does not disclose both the withheld comparison and the post-hoc selection",
    )
    render_script = (ROOT / "Scripts/render_preserved_demo.sh").read_text()
    hero_match = re.search(r'^report="\$root_dir/(\S+?)"', render_script, re.MULTILINE)
    check(hero_match is not None, "could not parse the hardcoded hero packet out of render_preserved_demo.sh")
    hero_path = Path(hero_match.group(1))
    check(
        hero_path.parent.as_posix() == "docs/phase6/teacher-forced-comparison/runs",
        "hero packet moved out of the preserved teacher-forced run set",
    )
    hero_name = hero_path.stem
    check(f"replaying {hero_name}," in readme, "hero alt text does not name the rendered packet")
    print("PASS hero screenshot matches final frame, discloses the NLL gate, and names the packet")

    def require(fragment: str, count: int = 1) -> None:
        actual = readme.count(fragment)
        check(
            actual == count,
            f"README fragment count for {fragment!r}: expected {count}, found {actual}",
        )

    def forbid(fragment: str) -> None:
        check(fragment not in readme, f"forbidden README fragment found: {fragment!r}")

    preserved_paths = sorted((ROOT / "docs/phase6/teacher-forced-comparison/runs").glob("*.json"))
    preserved_packets = [json.loads(path.read_text()) for path in preserved_paths]
    check(len(preserved_packets) == 8, "preserved teacher-forced run count changed")
    hero_rows = [
        {
            "name": path.stem,
            "logitGain": packet["steered"]["topicScore"] - packet["baseline"]["topicScore"],
            "residualGain": packet["actAdd"]["topicScore"] - packet["baseline"]["topicScore"],
            "teacherForcedLogit": packet["teacherForcedLogit"]["meanNatsPerStep"],
            "teacherForcedResidual": packet["teacherForcedActAdd"]["meanNatsPerStep"],
            "target": packet["teacherForcedTargetKL"],
        }
        for path, packet in zip(preserved_paths, preserved_packets)
    ]
    hero = next(row for row in hero_rows if row["name"] == hero_name)

    def hero_rank(key: str) -> int:
        ordered = sorted(hero_rows, key=lambda row: row[key], reverse=True)
        return next(index for index, row in enumerate(ordered, 1) if row["name"] == hero_name)

    check(
        hero_rank("teacherForcedLogit") == hero_rank("teacherForcedResidual") == 1,
        "hero packet is no longer the per-prompt KL extreme in both arms",
    )
    require(
        f"rank **{hero_rank('logitGain')}/{len(hero_rows)}** on logit-bias topic gain "
        f"(`{hero['logitGain']:+.6f}`)"
    )
    require(
        f"**{hero_rank('residualGain')}/{len(hero_rows)}** on residual topic gain "
        f"(`{hero['residualGain']:+.6f}`)"
    )
    require(
        f"**{hero_rank('teacherForcedLogit')}/{len(hero_rows)}** on per-prompt teacher-forced KL "
        "for both arms"
    )
    require(
        f"`{hero['teacherForcedLogit']:.6f}` nats/step, "
        f"`{hero['teacherForcedLogit'] / hero['target']:.2f}×` the "
        f"`{hero['target']:.5f}` target"
    )
    require("selected post hoc rather than drawn")
    print("PASS hero packet disclosed as post-hoc selected, with recomputed ranks")

    final = json.loads((ROOT / "docs/final-demo-run.json").read_text())
    check(
        final["baseline"]["tokenCount"]
        == final["steered"]["tokenCount"]
        == final["actAdd"]["tokenCount"]
        == 96,
        "final run is not a matched 96-token triple",
    )
    require("default 96-token")
    memory_mib = 10 * round(
        max(final[p]["residentMemoryBytes"] for p in ("baseline", "steered", "actAdd"))
        / 2**20
        / 10
    )
    require(f"about {memory_mib} MB peak resident memory")
    check(final["cumulativeKL"] <= final["klBudget"] + 1e-6, "logit-bias KL exceeded cap")
    check(final["actAddCumulativeKL"] <= final["klBudget"] + 1e-6, "residual-edit KL exceeded cap")
    require(f"`{final['cumulativeKL']:.4f}` cumulative KL for each intervention")
    require(f"`{final['baseline']['topicScore']:.4f}`")
    require(f"`{final['steered']['topicScore']:.4f}`")
    require(f"| Baseline | {final['baseline']['topicScore']:.4f} |")
    require(f"| Steered | {final['steered']['topicScore']:.4f} |")
    check(final["buildConfiguration"] == "Release", "final run was not recorded in Release")
    for pane in ("baseline", "steered", "actAdd"):
        forbid(f"{final[pane]['tokensPerSecond']:.1f} {pane}")
    require("Single-run token rates were removed from the headline")
    require("about 30× slower than baseline")
    legacy_slowdown = final["baseline"]["tokensPerSecond"] / final["actAdd"]["tokensPerSecond"]
    check(29 <= legacy_slowdown <= 32, "final-run residual slowdown is no longer about 30x")
    check(
        "actAddAppliedCoefficient" not in final and "actAddDirectionDiagnostics" not in final,
        "final-demo packet now records remediated residual fields; the pre-remediation note must be revisited",
    )
    require("this packet predates the residual remediation")
    print("PASS final-run memory, KL, hero topic scores, and scoped timing claims")

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
    require(f"**{min(history_lengths)}–{max(history_lengths)} biased steps**")
    sanity_median = statistics.median(history_lengths)
    sanity_short = sum(1 for value in history_lengths if value <= 5)
    require(
        f"**{min(history_lengths)}–{max(history_lengths)} biased steps**, "
        f"median {sanity_median:g}, {sanity_short} of {len(history_lengths)} at 5 or fewer"
    )
    check(sanity_median == 3, "sanity median biased-step count changed")
    check(sanity_short == 5, "sanity short-run count changed")
    require("the eighteen is a lone outlier, with a median of three and five of the six at five or fewer")
    require("six packets in this table are Release builds")
    require("untimed one-token same-prompt warm-up before measuring all three panes")
    print("PASS sanity table, exact 2–18 range with its median, packet count, and Release provenance")

    comparison_rows: list[str] = []
    cap4_paths = sorted((ROOT / "docs/negative-results/matched-kl4").glob("*.json"))
    for cap4_path in cap4_paths:
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
    check(all("buildConfiguration" not in row for row in map(json.loads, (p.read_text() for p in cap4_paths))), "historical KL-4 packet unexpectedly has build configuration")
    require("predate build-configuration recording and are not claimed as Release evidence")
    require("the permitted cost changes sampled behavior for ocean")
    require("wedding plateaus by 4 nats")
    print("PASS KL-4/KL-8 table and honest historical-build scope")

    analysis_path = ROOT / "docs/phase6/invalid-comparison-analysis.json"
    subprocess.run(
        ["python3", str(ROOT / "Scripts/analyze_invalid_phase6.py"), "--check", str(analysis_path)],
        check=True,
    )
    invalid = json.loads(analysis_path.read_text())
    comparison = invalid["comparison"]
    layer = invalid["layerSweep"]
    check(invalid["status"] == "invalidated", "Phase 6 analysis is not invalidated")
    check(comparison["runCount"] == 8, "invalid comparison packet count changed")
    check(comparison["promptCount"] == 4, "invalid comparison prompt count changed")
    check(comparison["distinctResidualEditOutputs"] == 4, "unexpected residual output count")
    check(comparison["crossLexiconIdenticalPromptCount"] == 4, "cross-lexicon identity no longer universal")
    check(comparison["allResidualEditOutputsStartWithWhenToken"], "a residual output no longer begins with the shared When token")
    low = 100 * comparison["firstStepFractionMinimum"]
    high = 100 * comparison["firstStepFractionMaximum"]
    require(f"**{low:.1f}%–{high:.1f}%**")
    require("byte-identical between the ocean and wedding directions in every case")
    require("The previous Phase 6 controller comparison is invalid.")
    require("**no conclusion about activation steering, the audit, or transfer**")
    check(layer["runCount"] == 24, "layer sweep packet count changed")
    check(layer["block3Vs19CaseCount"] == layer["byteIdenticalCaseCount"] == 4, "block 3/19 identity changed")
    check(layer["selectedLayer"] is None, "invalid sweep selected a layer")
    require("blocks 3 and 19 produced byte-identical residual-edit text in all four matched cases")
    rho_summary = json.loads((ROOT / "docs/phase6/on-device-rho/summary.json").read_text())
    sweep_summary = json.loads((ROOT / "docs/phase6/layer-sweep/summary.json").read_text())
    check(rho_summary["status"] == "invalidated" and rho_summary["ratio"] is None, "ratio summary was not invalidated")
    check(sweep_summary["status"] == "invalidated" and sweep_summary["selectedLayer"] is None, "layer summary was not invalidated")
    forbid("transfer failure")
    forbid("did not reproduce equivalence")
    forbid("materially disagrees")
    forbid("same cumulative KL cap")
    check(not re.search(r"rho\s*=\s*1\d(?:\.\d+)?", readme), "invalid two-digit app ratio remains")
    print("PASS invalid comparison, layer degeneracy, first-step fractions, and claim removal")

    blocking_path = ROOT / "docs/phase6/blocking-control/summary.json"
    blocking_before = blocking_path.read_bytes()
    subprocess.run(
        ["python3", str(ROOT / "Scripts/summarize_blocking_control.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    check(blocking_path.read_bytes() == blocking_before, "blocking summary is not reproducible")
    blocking = json.loads(blocking_before)
    blocking_packets = [
        json.loads(path.read_text())
        for path in sorted((ROOT / "docs/phase6/blocking-control/runs").glob("*.json"))
    ]
    check(len(blocking_packets) == blocking["packetCount"] == 180, "blocking packet count changed")
    check(all(row["buildConfiguration"] == "Release" for row in blocking_packets), "a blocking packet is not Release")
    check(all(row["actAddKLCapEnabled"] is False for row in blocking_packets), "a blocking packet enabled the cap")
    check(
        all(row["actAddCoefficient"] == row["actAddAppliedCoefficient"] for row in blocking_packets),
        "a blocking packet does not record the exact applied coefficient",
    )
    check(blocking["status"] == "pass", "blocking control no longer passes")
    check(blocking["passingCellCount"] == 2, "blocking passing-cell count changed")
    check(
        blocking["selectedCellByPredeclaredTieBreak"] == {"layer": 10, "coefficient": 4.0},
        "blocking selected cell changed",
    )
    selected = next(
        cell
        for cell in blocking["cells"]
        if cell["layer"] == 10 and cell["coefficient"] == 4.0
    )
    check(selected["passed"], "selected blocking cell no longer passes")
    check(not any(selected["crossLexiconIdentityByPrompt"]), "selected cell has cross-topic identity")
    require("**180 Release packets**")
    require("**2/15 layer/coefficient cells passed**")
    require("**block 10, coefficient 4**")
    for topic in ("wedding", "ocean"):
        row = selected["topics"][topic]
        require(f"`{row['semanticMedianShift']:+.6f}`")
        require(f"`{row['randomMedianShift']:+.6f}`")
    require("Median returned length was 32 tokens in every arm")
    require("repeated-trigram fraction was zero")
    check(
        not any(
            "crossTopic" in key or "offTarget" in key or "otherTopic" in key
            for cell in blocking["cells"]
            for topic_row in cell["topics"].values()
            for key in topic_row
        ),
        "blocking summary now records an off-target metric; the specificity disclaimer must be revisited",
    )
    require("or show **topic specificity**")
    require("no gate compared a direction's effect on the other topic's centroid")

    blocking_actadd = [row["actAdd"]["tokensPerSecond"] for row in blocking_packets]
    preserved_actadd = [row["actAdd"]["tokensPerSecond"] for row in preserved_packets]
    remediated_baseline = [
        row["baseline"]["tokensPerSecond"] for row in blocking_packets + preserved_packets
    ]
    remediated_slowdowns = [
        row["baseline"]["tokensPerSecond"] / row["actAdd"]["tokensPerSecond"]
        for row in blocking_packets + preserved_packets
    ]
    require(f"`{min(blocking_actadd):.1f}`–`{max(blocking_actadd):.1f}` tok/s over 32 tokens")
    require(f"`{min(preserved_actadd):.1f}`–`{max(preserved_actadd):.1f}` tok/s over 64 tokens")
    require(f"`{min(remediated_baseline):.1f}`–`{max(remediated_baseline):.1f}` tok/s baseline")
    require(f"**{min(remediated_slowdowns):.1f}×–{max(remediated_slowdowns):.1f}×**")
    check(
        max(remediated_slowdowns) < legacy_slowdown,
        "the remediated residual path is no longer faster than the pre-remediation packet",
    )
    print("PASS blocking-control grid, selected cell, specificity scope, and remediated cost claims")

    stage3_path = ROOT / "docs/phase6/teacher-forced-comparison/summary.json"
    stage3_before = stage3_path.read_bytes()
    subprocess.run(
        ["python3", str(ROOT / "Scripts/summarize_teacher_forced_comparison.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    check(stage3_path.read_bytes() == stage3_before, "teacher-forced summary is not reproducible")
    stage3 = json.loads(stage3_before)
    calibration_paths = sorted(
        (ROOT / "docs/phase6/teacher-forced-comparison/calibration-runs").glob("*.json")
    )
    comparison_paths = sorted(
        (ROOT / "docs/phase6/teacher-forced-comparison/runs").glob("*.json")
    )
    random_paths = sorted(
        (ROOT / "docs/phase6/teacher-forced-comparison/random-floor-runs").glob("*.json")
    )
    check(len(calibration_paths) == 304, "teacher-forced calibration packet count changed")
    check(len(comparison_paths) == len(random_paths) == 8, "teacher-forced output packet count changed")
    check(stage3["n"] == {"prompts": 4, "topics": 2, "promptTopicUnits": 8}, "Stage 3 n changed")
    check(stage3["klMatchPass"], "teacher-forced KL matching failed")
    check(stage3["directionDependencePass"], "Stage 3 direction dependence failed")
    check(all(row["passed"] for row in stage3["topics"].values()), "a Stage 3 topic/floor gate failed")
    check(stage3["lengthPass"] and stage3["repetitionPass"], "a Stage 3 degeneration gate changed")
    check(not stage3["nllPass"], "Stage 3 NLL gate unexpectedly passes")
    check(stage3["status"] == "invalid-comparison", "Stage 3 invalid status changed")
    check(stage3["ratioStatus"] == "withheld-validity-gate", "Stage 3 ratio status changed")
    check(stage3["ratio"] is None, "Stage 3 ratio must remain withheld")
    require("**4 prompts × 2 topics = 8 prompt-topic units**")
    achieved = [
        value
        for topic in stage3["achievedMeanTeacherForcedKL"].values()
        for value in topic.values()
    ]
    require(f"`{min(achieved):.9f}` to `{max(achieved):.9f}` nats/step")
    baseline_nll = stage3["medianBaseModelNLL"]["baseline"]
    residual_nll = stage3["medianBaseModelNLL"]["residual"]
    require(f"`{baseline_nll:.6f}` baseline")
    require(f"`{residual_nll:.6f}` for the semantic residual arm")
    require(f"`{residual_nll - baseline_nll - 1:.6f}` nat/token")
    interval = stage3["denominatorPromptClusterBootstrap95"]
    require(f"`[{interval[0]:.6f}, {interval[1]:.6f}]`")
    require("No point ratio or ratio interval is reported.")
    require("304 calibration packets, 16 output packets")
    failure = json.loads(
        (ROOT / "docs/phase6/teacher-forced-comparison/calibration-failure.json").read_text()
    )
    check(failure["status"] == "failed-monotonicity-gate", "original calibration failure changed")
    check(failure["topicOutputsObserved"] is False, "original calibration observed topic output")
    print("PASS teacher-forced calibration, invalid NLL gate, packet retention, and ratio withholding")

    rho_paths = sorted((ROOT / "docs/phase6/on-device-rho/runs").glob("*.json"))
    rho_packets = [json.loads(path.read_text()) for path in rho_paths]
    check(len(rho_packets) == 8, "comparison packet count changed")
    check(all(abs(row["temperature"] - 0.7) < 1e-6 for row in rho_packets), "comparison temperature changed")
    protocol = (ROOT / "docs/phase6/on-device-rho/protocol.md").read_text()
    check("temperature 0.7" in protocol, "precommitted protocol temperature changed")
    require("temperature `0.7`")
    check(all(row["buildConfiguration"] == "Release" for row in rho_packets), "a comparison packet is not Release")

    dense_shares, sparse_shares, dense_steps, sparse_steps = [], [], [], []
    for row in rho_packets:
        dense, sparse = row["actAddKLHistory"], row["klHistory"]
        dense_shares.append(dense[0]["perStep"] / dense[-1]["cumulative"])
        sparse_shares.append(sparse[0]["perStep"] / sparse[-1]["cumulative"])
        dense_steps.append(len(dense))
        sparse_steps.append(len(sparse))
    dense_collapsed = sum(1 for share in dense_shares if share > 0.97)
    sparse_collapsed = sum(1 for share in sparse_shares if share > 0.97)
    check(dense_collapsed == 8, "dense in-packet first-step collapse count changed")
    check(sparse_collapsed == 1, "sparse in-packet first-step collapse count changed")
    require(f"into step one in **{dense_collapsed} of {len(rho_packets)}** runs")
    require(f"the sparse logit bias did so in **{sparse_collapsed} of {len(rho_packets)}**")
    exception_name, exception_share = max(
        ((path.stem, share) for path, share in zip(rho_paths, sparse_shares)),
        key=lambda pair: pair[1],
    )
    require(f"(`{exception_name}`, at {100 * exception_share:.4f}%)")
    others = sorted(share for share in sparse_shares if share <= 0.97)
    require(f"only {100 * others[0]:.2f}%–{100 * others[-1]:.1f}% there in the other seven")
    require(
        f"spreading its cap over {min(sparse_steps)}–{max(sparse_steps)} biased steps "
        f"against the dense arm's {min(dense_steps)}–{max(dense_steps)}"
    )
    print("PASS comparison temperature, Release provenance, and matched in-packet first-step contrast")

    zero = json.loads((ROOT / "docs/phase6/coefficient-zero/report.json").read_text())
    check(zero["actAddCoefficient"] == 0, "coefficient-zero packet is not zero")
    check(zero["baseline"]["tokenIDs"] == zero["actAdd"]["tokenIDs"], "coefficient-zero packet token IDs differ")
    check(zero["baseline"]["text"].encode() == zero["actAdd"]["text"].encode(), "coefficient-zero packet bytes differ")
    check(zero["baseline"]["tokenCount"] == zero["actAdd"]["tokenCount"], "coefficient-zero packet counts differ")
    require("baseline and residual-pane token IDs, decoded text, and counts are equal")
    tests = (ROOT / "SteeringKit/Tests/SteeringKitTests/KLMeterTests.swift").read_text()
    check("actAddCoefficientZeroRoutesToBaselineClosure" in tests, "route test name is misleading")
    check("actAddCoefficientZeroUsesBitIdenticalBaselineRoute" not in tests, "old route test name remains")
    print("PASS real coefficient-zero packet equality and accurately named route unit test")

    decomposition = json.loads((ROOT / "docs/judge-decomposition.json").read_text())
    scores = decomposition["scores"]
    require(f"{scores['prefix_plus_baseline']:.4f}")
    require(f"{scores['steered_prefix_removed']:.4f}")
    require(f"{decomposition['prefix_only_shift']:.4f} / {decomposition['observed_shift']:.4f}")
    require(f"{100 * decomposition['prefix_fraction_of_observed_shift']:.1f}%")
    print("PASS judge decomposition and prominent 68.4% self-measurement")

    parity = json.loads((ROOT / "docs/coreml-parity.json").read_text())
    metadata = parity["metadata"]
    check(len(parity["rows"]) == metadata["validation_cases"] == 24, "Core ML case count changed")
    require("24 inputs")
    require(f"`{metadata['minimum_cosine']:.9f}`")
    require(f"`{metadata['threshold']:.4f}` gate")
    check(metadata["max_length"] == 128, "Core ML max length changed")
    require("fixed 128-token inputs")
    check(metadata["compute_units"] == "ALL", "Core ML compute units changed")
    check(len(metadata["weight_sha256"]) == 64, "Core ML digest is not SHA-256")
    centroids = json.loads((ROOT / "Resources/CoreML/topic-centroids.json").read_text())
    check(centroids["dimensions"] == 384, "Core ML dimensions changed")
    require("384-dimensional embedding")
    require(centroids["model_revision"])
    package_bytes = sum(
        path.stat().st_size
        for path in (ROOT / "Resources/CoreML/TopicEncoder.mlpackage").rglob("*")
        if path.is_file()
    )
    require(f"{round(package_bytes / 2**20)} MB in FP16")
    print("PASS Core ML parity, dimensions, revision, and package size")

    test_count = sum(
        path.read_text().count("@Test")
        for path in (ROOT / "SteeringKit/Tests/SteeringKitTests").glob("*.swift")
    )
    require(f"has {test_count} tests")
    service_source = (ROOT / "SteerDemo/MLXGenerationService.swift").read_text()
    model_id = re.search(r'static let modelID = "([^"]+)"', service_source)
    model_revision = re.search(r'static let modelRevision = "([0-9a-f]+)"', service_source)
    check(model_id is not None and model_revision is not None, "could not parse pinned Qwen model")
    require(f"`{model_id.group(1)}`")
    require(f"`{model_revision.group(1)}`")
    print("PASS test count and pinned Qwen identity")

    audit = json.loads((ROOT / "docs/audit-reference.json").read_text())
    point = audit["rho"]["point"]
    require(f"{100 * point:.1f}%")
    require(f"`rho = {point}`")
    require(f"`n = {audit['n_eval']}`")
    require(f"[{audit['rho']['ci_lo']}, {audit['rho']['ci_hi']}]")
    require(f"verdict is `{audit['verdict']['class']}`")
    require("95% CI 85.3%–107.1%")
    matched = audit["matched_protocol"]
    require(f"`{matched['mean_teacher_forced_kl_nats_per_step']}` nats per step")
    require(f"fixed {matched['continuation_steps']}-step continuation")
    require("prompt positions 0 through 6")
    require(f"layer {matched['residual_layer']} of {matched['transformer_layers']}")
    require("matched-norm random-direction floor")
    require("repetition, length, and NLL degeneracy gates")
    print("PASS recomputed audit percentage, exact reference, and actual matching protocol")

    before = json.loads((ROOT / "LoRA/results/before.json").read_text())
    after = json.loads((ROOT / "LoRA/results/after.json").read_text())
    adapter = ROOT / "LoRA/adapter/adapters.safetensors"
    check(before["n"] == after["n"] == 9, "LoRA evaluation n changed")
    check(before["exactMatches"] == 0 and after["exactMatches"] == 9, "LoRA result changed")
    check(adapter.stat().st_size > 0, "LoRA adapter is empty")
    check(after["adapter"] == "LoRA/adapter", "LoRA report leaked a local path")
    require("**MLX Python**, not MLX Swift")
    require("120 optimizer steps on 36 toy codebook examples")
    require("`0/9` before training to `9/9` after training")
    require("3 MB adapter")
    print("PASS toy MLX Python LoRA artifact")

    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    vendored_revision = "9bff95ca5f0b9e8c021acc4d71a2bbe4a7441631"
    check(vendored_revision in notices, "vendored revision missing from notices")
    require(f"`{vendored_revision}`")
    check((ROOT / "LICENSES/mlx-swift-examples-LICENSE.txt").exists(), "vendored license missing")
    require("Codex generated most of the implementation, tests, and documentation")
    require("directed, agent-assisted work")
    require("makes no claim of personal Swift implementation experience")
    print("PASS vendored license and implementation-provenance disclosure")


if __name__ == "__main__":
    main()
