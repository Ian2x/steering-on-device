#!/usr/bin/env python3
"""Recompute README claims from committed artifacts and require exact placement."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def git(*arguments: str) -> str:
    """Read this repository's own history. Claims about it must come from it."""
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments], check=True, capture_output=True, text=True
    ).stdout


# --- printed ranges must contain their own extremes -------------------------
#
# Rounding both endpoints of a range to nearest can shrink the printed interval
# past the data it summarises: a set whose minimum is 97.3621% renders a floor
# of "97.4%", an interval that excludes its own minimum. Round a range's lower
# bound down and its upper bound up instead, so reduced precision only ever
# widens the claim. Point estimates are unaffected and keep nearest-rounding.
#
# Decimal(repr(value)) is the shortest decimal that round-trips the float, which
# is the value the float stands for. Quantizing the raw binary expansion instead
# would shove an endpoint that already sits exactly on the printing grid outward
# by a whole quantum.


def _quantize(value: float, digits: int, rounding: str) -> str:
    quantized = Decimal(repr(value)).quantize(Decimal(1).scaleb(-digits), rounding=rounding)
    return f"{quantized:f}"


def range_floor(value: float, digits: int = 1) -> str:
    """Render a range's lower bound, rounding down so the interval contains it."""
    return _quantize(value, digits, ROUND_FLOOR)


def range_ceiling(value: float, digits: int = 1) -> str:
    """Render a range's upper bound, rounding up so the interval contains it."""
    return _quantize(value, digits, ROUND_CEILING)


# The share of an intervention's cumulative KL that has to land in step one for
# the README to call it collapsed. The README quotes this threshold, so it is
# rendered from here rather than restated.
FIRST_STEP_COLLAPSE_SHARE = 0.97

# The biased-step count at or below which the README calls a sanity run short.
SHORT_BIASED_RUN_STEPS = 5


# --- numeric-literal sweep -------------------------------------------------
#
# require() is exact-substring with an exact-count assertion, so pinning one
# rendered fragment does not guard the same number where it appears with a
# different suffix. The sweep below closes that structurally: it extracts every
# numeric literal from the README and demands that each one sit inside a span
# that some require() call actually pinned, or inside an allowlisted context
# with a stated reason. Coverage is decided by character position, never by
# re-matching the bare literal, so a second occurrence of an already-pinned
# number is still reported as unguarded.

# Regions whose digits are excluded from the sweep, by rule:
EXCLUDED_REGIONS = (
    # Fenced code blocks: reproduction commands and flags, not result claims.
    re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL),
    # Markdown links whose text is an inline code span: both halves are
    # repository paths, whose digits name files and directories rather than
    # assert anything.
    re.compile(r"\[`[^`]*`\]\([^)\s]*\)"),
    # Every other markdown link and image target. Prose link text and image alt
    # text are deliberately left in scope.
    re.compile(r"\]\([^)\s]*\)"),
)

# A numeric literal is a digit run with an optional decimal part that is not
# glued to a letter or an underscore, and does not begin immediately after a
# decimal point. That adjacency rule — and nothing else — is what removes model
# names, git revisions, and unit-suffixed identifiers (Qwen2.5, 0.5B, 4bit,
# all-MiniLM-L6-v2, FP16, arm64, 0d5a15e, a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3)
# without an allowlist entry. A trailing period is left outside the match so
# that a sentence-final decimal is still swept. Signs, percent signs, and
# multiplication crosses are outside the match but inside the pinned fragments,
# so they still have to line up.
NUMERIC_LITERAL = re.compile(r"(?<![0-9A-Za-z_.])[0-9]+(?:\.[0-9]+)?(?![0-9A-Za-z_])(?!\.[0-9])")

# Literals that no artifact can supply, keyed by the exact README context that
# contains them. Each key must occur exactly once in the README and stay short,
# so an entry cannot silently rot or quietly blanket a paragraph.
ALLOWED_NUMERIC_CONTEXTS = {
    "Xcode 26 or newer": "external toolchain floor chosen by the author; no artifact records it",
    "model-weight SHA-256": "digest algorithm name; the digest itself is checked for SHA-256 length",
    "Historical Phase 6 comparison invalidated": "project phase label, not a measurement",
    "historical Phase 6 protocol": "project phase label, not a measurement",
    "the stored 95% interval": "confidence level is not recorded in audit-reference.json; its bounds are pinned",
}
ALLOWED_CONTEXT_MAX_LENGTH = 120


def _mask(length: int, spans: list[tuple[int, int]]) -> bytearray:
    mask = bytearray(length)
    for start, end in spans:
        mask[start:end] = b"\x01" * (end - start)
    return mask


def _occurrences(haystack: str, needle: str) -> list[tuple[int, int]]:
    spans = []
    start = haystack.find(needle)
    while start != -1:
        spans.append((start, start + len(needle)))
        start = haystack.find(needle, start + 1)
    return spans


def sweep_numeric_literals(readme: str, fragments: list[str]) -> None:
    for context, reason in ALLOWED_NUMERIC_CONTEXTS.items():
        check(bool(reason), f"allowlisted numeric context {context!r} has no reason")
        check(
            len(context) <= ALLOWED_CONTEXT_MAX_LENGTH,
            f"allowlisted numeric context {context!r} is too broad "
            f"({len(context)} > {ALLOWED_CONTEXT_MAX_LENGTH} characters)",
        )
        check(
            any(character.isdigit() for character in context),
            f"allowlisted numeric context {context!r} contains no digit",
        )
        found = readme.count(context)
        check(found == 1, f"allowlisted numeric context {context!r}: expected 1, found {found}")

    covered = _mask(len(readme), [span for f in fragments for span in _occurrences(readme, f)])
    allowed = _mask(
        len(readme),
        [span for c in ALLOWED_NUMERIC_CONTEXTS for span in _occurrences(readme, c)],
    )
    excluded = _mask(
        len(readme),
        [m.span() for pattern in EXCLUDED_REGIONS for m in pattern.finditer(readme)],
    )

    unguarded: list[str] = []
    pinned = 0
    excused = 0
    for match in NUMERIC_LITERAL.finditer(readme):
        start, end = match.span()
        if any(excluded[start:end]):
            continue
        if all(covered[start:end]):
            pinned += 1
            continue
        if all(allowed[start:end]):
            excused += 1
            continue
        line = readme.count("\n", 0, start) + 1
        context = readme[max(0, start - 45) : end + 45].replace("\n", " ")
        unguarded.append(f"  line {line}: {match.group()!r} in ...{context}...")

    check(
        not unguarded,
        f"{len(unguarded)} README numeric literal(s) are neither pinned to an artifact "
        "nor allowlisted:\n" + "\n".join(unguarded),
    )
    print(
        f"PASS numeric-literal sweep: {pinned + excused} literals in scope, "
        f"{pinned} pinned to artifacts, {excused} excused by "
        f"{len(ALLOWED_NUMERIC_CONTEXTS)} allowlisted contexts"
    )


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

    required_fragments: list[str] = []

    def require(fragment: str, count: int = 1) -> None:
        actual = readme.count(fragment)
        check(
            actual == count,
            f"README fragment count for {fragment!r}: expected {count}, found {actual}",
        )
        required_fragments.append(fragment)

    def forbid(fragment: str) -> None:
        check(fragment not in readme, f"forbidden README fragment found: {fragment!r}")

    preserved_paths = sorted((ROOT / "docs/phase6/teacher-forced-comparison/runs").glob("*.json"))
    preserved_packets = [json.loads(path.read_text()) for path in preserved_paths]
    check(len(preserved_packets) == 8, "preserved teacher-forced run count changed")
    random_floor_paths = sorted(
        (ROOT / "docs/phase6/teacher-forced-comparison/random-floor-runs").glob("*.json")
    )
    random_floor_packets = [json.loads(path.read_text()) for path in random_floor_paths]
    check(len(random_floor_packets) == 8, "matched-random teacher-forced run count changed")
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
    require(f"default {final['baseline']['tokenCount']}-token")
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
    legacy_slowdown = final["baseline"]["tokensPerSecond"] / final["actAdd"]["tokensPerSecond"]
    check(29 <= legacy_slowdown <= 32, "final-run residual slowdown is no longer about 30x")
    # Rounded to the nearest ten, matching the "about ... MB" convention above.
    approximate_slowdown = 10 * round(legacy_slowdown / 10)
    require(f"about {approximate_slowdown}× slower than baseline")
    require(f"Do not carry {approximate_slowdown}× forward as the current cost")
    check(
        "actAddAppliedCoefficient" not in final and "actAddDirectionDiagnostics" not in final,
        "final-demo packet now records remediated residual fields; the pre-remediation note must be revisited",
    )
    require("this packet predates the residual remediation")
    require(
        f"(block {final['actAddLayer']}, nominal coefficient {final['actAddCoefficient']:g}, "
        "no recorded applied coefficient or direction diagnostics)"
    )
    require(f"its {final['baseline']['tokenCount']}-token length inflates a path")
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
    sanity_budgets = {row["klBudget"] for row in sanity}
    sanity_token_limits = {row["maxTokens"] for row in sanity}
    check(len(sanity_budgets) == 1, "sanity packets no longer share one KL cap")
    check(len(sanity_token_limits) == 1, "sanity packets no longer share one token limit")
    sanity_budget = sanity_budgets.pop()
    require(f"These are real {sanity_token_limits.pop()}-token app runs")
    require(f"fixed seed, temperature, and an {sanity_budget:.4f}-nat KL cap")
    history_lengths = [len(row["klHistory"]) for row in sanity]
    require(f"**{min(history_lengths)}–{max(history_lengths)} biased steps**")
    sanity_median = statistics.median(history_lengths)
    sanity_short = sum(1 for value in history_lengths if value <= SHORT_BIASED_RUN_STEPS)
    require(
        f"**{min(history_lengths)}–{max(history_lengths)} biased steps**, "
        f"median {sanity_median:g}, {sanity_short} of {len(history_lengths)} at "
        f"{SHORT_BIASED_RUN_STEPS} or fewer"
    )
    check(sanity_median == 3, "sanity median biased-step count changed")
    check(sanity_short == 5, "sanity short-run count changed")
    require("the eighteen is a lone outlier, with a median of three and five of the six at five or fewer")
    require("six packets in this table are Release builds")
    require("untimed one-token same-prompt warm-up before measuring all three panes")
    print("PASS sanity table, exact 2–18 range with its median, packet count, and Release provenance")

    comparison_rows: list[str] = []
    cap4_paths = sorted((ROOT / "docs/negative-results/matched-kl4").glob("*.json"))
    cap4_packets = [json.loads(path.read_text()) for path in cap4_paths]
    cap4_budgets = {row["klBudget"] for row in cap4_packets}
    check(len(cap4_budgets) == 1, "historical KL-4 packets no longer share one KL cap")
    cap4_budget = cap4_budgets.pop()
    check(cap4_budget * 2 == sanity_budget, "the KL-4/KL-8 pair is no longer a doubling")
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
    check(all("buildConfiguration" not in row for row in cap4_packets), "historical KL-4 packet unexpectedly has build configuration")
    require(f"including the KL-{cap4_budget:g} half of the comparison below")
    require("predate build-configuration recording and are not claimed as Release evidence")
    require(
        f"| Lexicon | Bias strength | Steered text identical? | Topic score (KL cap {cap4_budget:g}) "
        f"| Topic score (KL cap {sanity_budget:g}) |"
    )
    require(f"The KL-{cap4_budget:g} packets are in")
    require(f"the KL-{sanity_budget:g} packets are in")
    require("the permitted cost changes sampled behavior for ocean")
    require(f"wedding plateaus by {cap4_budget:g} nats")

    strengths = sorted({row["biasStrength"] for row in sanity})
    check(len(strengths) == 3, "the sanity sweep no longer tests three bias strengths")
    check(
        {row["biasStrength"] for row in cap4_packets} == set(strengths),
        "the KL-4 and KL-8 halves no longer test the same bias strengths",
    )
    require(
        f"ocean strength {strengths[0]:g} spends the same KL without crossing a sampled-token "
        f"threshold, while strengths {strengths[1]:g} and {strengths[2]:g} do"
    )
    require(
        f"a {cap4_budget:g}-nat cap never crosses a sampled-token threshold at the three tested "
        f"strengths; at {sanity_budget:g} nats, strengths {strengths[1]:g} and {strengths[2]:g} do"
    )
    ocean_gain = next(
        row["steered"]["topicScore"] - row["baseline"]["topicScore"]
        for row in sanity
        if row["lexicon"] == "ocean" and row["biasStrength"] == strengths[1]
    )
    require(f"moving the score by {ocean_gain:+.4f}")
    require(f"already saturated by {cap4_budget:g} nats")
    print("PASS KL-4/KL-8 table, matched strengths, cap labels, and honest historical-build scope")

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
    require(f"**{range_floor(low)}%–{range_ceiling(high)}%**")
    require("byte-identical between the ocean and wedding directions in every case")
    require("The previous Phase 6 controller comparison is invalid.")
    require("**no conclusion about activation steering, the audit, or transfer**")
    check(layer["runCount"] == 24, "layer sweep packet count changed")
    check(layer["block3Vs19CaseCount"] == layer["byteIdenticalCaseCount"] == 4, "block 3/19 identity changed")
    check(layer["selectedLayer"] is None, "invalid sweep selected a layer")

    sweep_path = ROOT / "docs/phase6/layer-sweep/summary.json"
    sweep_before = sweep_path.read_bytes()
    subprocess.run(
        ["python3", str(ROOT / "Scripts/summarize_actadd_layer_sweep.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    check(sweep_path.read_bytes() == sweep_before, "layer-sweep summary is not reproducible")
    sweep_results = (ROOT / "docs/phase6/layer-sweep/results.md").read_text()
    check(
        "The summary is still regenerable from its packets" in sweep_results,
        "the layer sweep no longer documents its summary's regenerability",
    )
    check(
        all(field not in sweep_before.decode() for field in ("perLayer", '"runs"')),
        "the layer-sweep summary regained a per-layer statistic; results.md must be revisited",
    )
    sweep_summary = json.loads(sweep_before)
    sweep_packets = [
        json.loads(path.read_text())
        for path in sorted((ROOT / "docs/phase6/layer-sweep/runs").glob("*.json"))
    ]
    check(len(sweep_packets) == sweep_summary["rawPacketCount"] == layer["runCount"], "layer sweep packet count changed")
    check(all(row["buildConfiguration"] == "Release" for row in sweep_packets), "a layer-sweep packet is not Release")
    sweep_layers = sorted({row["actAddLayer"] for row in sweep_packets})
    check(sweep_layers == sweep_summary["candidateLayers"], "layer-sweep summary no longer lists its packets' blocks")
    sweep_settings = {
        key: {row[key] for row in sweep_packets}
        for key in ("actAddCoefficient", "klBudget", "maxTokens")
    }
    check(
        all(len(values) == 1 for values in sweep_settings.values()),
        "layer-sweep packets no longer share one coefficient, cap, and token limit",
    )
    sweep_coefficient = sweep_settings["actAddCoefficient"].pop()
    sweep_budget = sweep_settings["klBudget"].pop()
    sweep_token_limit = sweep_settings["maxTokens"].pop()
    sweep_cases: dict[tuple[str, str], dict[int, str]] = {}
    for row in sweep_packets:
        sweep_cases.setdefault((row["prompt"], row["lexicon"]), {})[row["actAddLayer"]] = row["actAdd"]["text"]
    # "four matched cases" and "four neutral prompt-topic cases" below are the
    # word form of this count, already asserted as 4 through layerSweep above.
    check(len(sweep_cases) == layer["block3Vs19CaseCount"], "layer-sweep matched-case count changed")
    identical_pairs = [
        (low_block, high_block)
        for index, low_block in enumerate(sweep_layers)
        for high_block in sweep_layers[index + 1 :]
        if all(texts[low_block] == texts[high_block] for texts in sweep_cases.values())
    ]
    check(len(identical_pairs) == 1, f"layer-sweep degeneracy is no longer a single pair: {identical_pairs}")
    degenerate = identical_pairs[0]
    require(
        f"blocks {degenerate[0]} and {degenerate[1]} produced byte-identical residual-edit text "
        "in all four matched cases"
    )
    require(
        "blocks " + ", ".join(str(block) for block in sweep_layers[:-1])
        + f", and {sweep_layers[-1]} were swept across four neutral prompt-topic cases at a nominal "
        f"coefficient of {sweep_coefficient:g}, an {sweep_budget:g}-nat cumulative cap, "
        f"and at most {sweep_token_limit} generated tokens. "
        f"All {len(sweep_packets)} Release packets are retained"
    )
    require(f"Because blocks {degenerate[0]} and {degenerate[1]} produced the same text in every matched case")

    rho_summary = json.loads((ROOT / "docs/phase6/on-device-rho/summary.json").read_text())
    check(rho_summary["status"] == "invalidated" and rho_summary["ratio"] is None, "ratio summary was not invalidated")
    check(sweep_summary["status"] == "invalidated" and sweep_summary["selectedLayer"] is None, "layer summary was not invalidated")
    forbid("transfer failure")
    forbid("did not reproduce equivalence")
    forbid("materially disagrees")
    forbid("same cumulative KL cap")
    check(not re.search(r"rho\s*=\s*1\d(?:\.\d+)?", readme), "invalid two-digit app ratio remains")
    print("PASS invalid comparison, reproducible layer sweep, first-step fractions, and claim removal")

    # `git log -p -- docs/phase6/*/protocol.md` is the first command a hostile
    # reviewer runs, and it shows two protocols edited after their runs. The
    # README discloses that rather than leaving it to be discovered, so the split
    # between amended and never-edited protocols is derived from git here and the
    # disclosure cannot drift away from the history it describes.
    protocol_history = {}
    for protocol_path in sorted((ROOT / "docs/phase6").glob("*/protocol.md")):
        name = protocol_path.parent.name
        log = git("log", "--format=%H", "--", f"docs/phase6/{name}/protocol.md").split()
        check(bool(log), f"docs/phase6/{name}/protocol.md is not committed")
        protocol_history[name] = log
    amended = sorted(name for name, log in protocol_history.items() if len(log) > 1)
    unedited = sorted(name for name, log in protocol_history.items() if len(log) == 1)
    check(
        amended == ["layer-sweep", "on-device-rho"],
        f"the set of post-run-amended Phase 6 protocols changed: {amended}",
    )
    check(
        unedited == ["blocking-control", "teacher-forced-comparison"],
        f"the set of never-edited Phase 6 protocols changed: {unedited}",
    )
    check(
        all(len(protocol_history[name]) == 2 for name in amended),
        "an amended protocol gained a further edit; the README disclosure must be revisited",
    )
    amending = {protocol_history[name][0] for name in amended}
    check(len(amending) == 1, f"the amended protocols no longer share one amending commit: {amending}")
    amending_commit = amending.pop()
    amended_at = int(git("show", "-s", "--format=%ct", amending_commit).strip())
    for name in amended:
        relative = f"docs/phase6/{name}/protocol.md"
        original = git("show", f"{protocol_history[name][1]}:{relative}").splitlines()
        current = (ROOT / relative).read_text()
        # Every predeclared line but one survives byte-identically, and the one
        # that does not is the output-length shorthand. That is what carries the
        # "no gate, threshold, or selection rule changed" half of the disclosure.
        removed = [line for line in original if line not in current.splitlines()]
        check(
            len(removed) == 1,
            f"{relative}: the amendment dropped {len(removed)} predeclared lines, not one: {removed}",
        )
        check(
            re.search(r"[Oo]utput length: \d+ tokens|\d+ output tokens", removed[0]) is not None,
            f"{relative}: the dropped line is not the output-length shorthand: {removed[0]!r}",
        )
        check(
            "after the run" in current,
            f"{relative} no longer labels its amendment as post-run",
        )
        latest_packet = max(
            datetime.strptime(json.loads(path.read_text())["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            for path in (ROOT / f"docs/phase6/{name}/runs").glob("*.json")
        )
        check(
            latest_packet < amended_at,
            f"{relative} was amended before its last packet; the post-run wording is wrong",
        )
    require("Committed is not the same as unedited")
    require(f"were each edited once after their runs, in commit `{amending_commit[:7]}`")
    require("every other predeclared line survives byte-identically")
    require("have one commit each and were never edited")
    for name in amended + unedited:
        require(f"[`{name}`](docs/phase6/{name}/protocol.md)")
    require("amended after them as disclosed above", count=2)
    print("PASS post-run protocol amendments disclosed, scoped, and never-edited half verified")

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
    tie_break = blocking["selectedCellByPredeclaredTieBreak"]
    selected = next(
        cell
        for cell in blocking["cells"]
        if cell["layer"] == tie_break["layer"] and cell["coefficient"] == tie_break["coefficient"]
    )
    check(selected["passed"], "selected blocking cell no longer passes")
    check(not any(selected["crossLexiconIdentityByPrompt"]), "selected cell has cross-topic identity")

    grid = {
        "layers": len({row["actAddLayer"] for row in blocking_packets}),
        "coefficients": len({row["actAddCoefficient"] for row in blocking_packets}),
        "directions": len({row["actAddDirectionMode"] for row in blocking_packets}),
        "topics": len({row["lexicon"] for row in blocking_packets}),
        "prompts": len({row["prompt"] for row in blocking_packets}),
    }
    cells = grid["layers"] * grid["coefficients"]
    product = cells * grid["directions"] * grid["topics"] * grid["prompts"]
    check(product == len(blocking_packets), "the blocking grid is no longer full and balanced")
    require(
        f"{grid['layers']} layers × {grid['coefficients']} direct coefficients × "
        f"{grid['directions']} directions × {grid['topics']} topics × {grid['prompts']} neutral "
        f"prompts = **{len(blocking_packets)} Release packets**"
    )
    require("**180 Release packets**")
    require(f"**{blocking['passingCellCount']}/{cells} layer/coefficient cells passed**")
    require(f"**block {tie_break['layer']}, coefficient {tie_break['coefficient']:g}**")
    for topic in ("wedding", "ocean"):
        row = selected["topics"][topic]
        require(f"`{row['semanticMedianShift']:+.6f}`")
        require(f"`{row['randomMedianShift']:+.6f}`")
    selected_lengths = set(selected["medianLengths"].values())
    check(len(selected_lengths) == 1, "the selected cell's arms no longer share one median length")
    require(f"Median returned length was {selected_lengths.pop():g} tokens in every arm")
    check(
        set(selected["medianRepeatedTrigramFractions"].values()) == {0.0},
        "the selected cell's repeated-trigram fractions are no longer zero",
    )
    require("repeated-trigram fraction was zero")
    nll = selected["medianBaseModelNLL"]
    require(
        f"base-model mean token NLL was `{nll['baseline']:.6f}` baseline, "
        f"`{nll['semantic']:.6f}` semantic, and `{nll['random']:.6f}` random"
    )
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
    blocking_protocol = (ROOT / "docs/phase6/blocking-control/protocol.md").read_text()
    gates = re.findall(r"^(\d+)\. \*\*([^:*]+):\*\*", blocking_protocol, re.MULTILINE)
    check(
        [name for _, name in gates[:2]] == ["Direction dependence", "On-target movement"],
        "the frozen blocking protocol's first two gates are no longer direction dependence and on-target movement",
    )
    require(
        f"because gate {gates[0][0]} asks only that the wedding and ocean outputs be byte-different "
        f"and gate {gates[1][0]} scores every packet against its own selected topic centroid"
    )

    # Every remediated residual packet, not a favourable subset. The matched-random
    # teacher-forced arm is built by the identical construction at the same
    # calibrated coefficient, and generation cost cannot depend on whether the
    # injected matrix is semantic or random -- so scoping the 64-token set to its
    # semantic half would have been arbitrary. It was also the flattering half:
    # the slowest packet in the whole remediated corpus is a random-arm packet.
    # The 180-packet blocking set already contributes both arms, so this simply
    # treats the two run sets alike. "Remediated" is defined here exactly as the
    # pre-remediation note above defines its absence: the packet records both the
    # exact applied coefficient and the direction diagnostics.
    long_packets = preserved_packets + random_floor_packets
    remediated_packets = blocking_packets + long_packets
    check(
        all(
            "actAddAppliedCoefficient" in row and "actAddDirectionDiagnostics" in row
            for row in remediated_packets
        ),
        "a packet counted as remediated does not record its applied coefficient and diagnostics",
    )
    check(
        all(row["buildConfiguration"] == "Release" for row in long_packets),
        "a 64-token remediated packet is not Release",
    )
    blocking_actadd = [row["actAdd"]["tokensPerSecond"] for row in blocking_packets]
    long_actadd = [row["actAdd"]["tokensPerSecond"] for row in long_packets]
    remediated_baseline = [row["baseline"]["tokensPerSecond"] for row in remediated_packets]
    remediated_slowdowns = [
        row["baseline"]["tokensPerSecond"] / row["actAdd"]["tokensPerSecond"]
        for row in remediated_packets
    ]
    blocking_token_limits = {row["maxTokens"] for row in blocking_packets}
    long_token_limits = {row["maxTokens"] for row in long_packets}
    check(
        len(blocking_token_limits) == len(long_token_limits) == 1,
        "the remediated run sets no longer share one token limit each",
    )
    require(
        f"`{range_floor(min(blocking_actadd))}`–`{range_ceiling(max(blocking_actadd))}` tok/s over "
        f"{blocking_token_limits.pop()} tokens ({len(blocking_packets)} blocking-control packets)"
    )
    require(
        f"`{range_floor(min(long_actadd))}`–`{range_ceiling(max(long_actadd))}` tok/s over "
        f"{long_token_limits.pop()} tokens ({len(long_packets)} teacher-forced packets, "
        "semantic and matched-random alike)"
    )
    require(
        f"`{range_floor(min(remediated_baseline))}`–"
        f"`{range_ceiling(max(remediated_baseline))}` tok/s baseline"
    )
    require(
        f"Over all **{len(remediated_packets)}** remediated packets the per-packet slowdown is "
        f"**{range_floor(min(remediated_slowdowns))}×–"
        f"{range_ceiling(max(remediated_slowdowns))}×**"
    )
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
    comparison_paths = preserved_paths
    random_paths = random_floor_paths
    calibration_packets = [json.loads(path.read_text()) for path in calibration_paths]
    check(len(calibration_paths) == 304, "teacher-forced calibration packet count changed")
    check(len(comparison_paths) == len(random_paths) == 8, "teacher-forced output packet count changed")
    check(stage3["n"] == {"prompts": 4, "topics": 2, "promptTopicUnits": 8}, "Stage 3 n changed")
    calibration_summary = json.loads(
        (ROOT / "docs/phase6/teacher-forced-comparison/calibration-summary.json").read_text()
    )
    require(
        f"to the audit's `{calibration_summary['targetMeanTeacherForcedKL']}`-nat/step target on "
        f"shared fixed {calibration_summary['continuationSteps']}-token continuations"
    )
    check(stage3["klMatchPass"], "teacher-forced KL matching failed")
    check(stage3["directionDependencePass"], "Stage 3 direction dependence failed")
    check(all(row["passed"] for row in stage3["topics"].values()), "a Stage 3 topic/floor gate failed")
    check(stage3["lengthPass"] and stage3["repetitionPass"], "a Stage 3 degeneration gate changed")
    check(not stage3["nllPass"], "Stage 3 NLL gate unexpectedly passes")
    check(stage3["status"] == "invalid-comparison", "Stage 3 invalid status changed")
    check(stage3["ratioStatus"] == "withheld-validity-gate", "Stage 3 ratio status changed")
    check(stage3["ratio"] is None, "Stage 3 ratio must remain withheld")
    require(
        f"**{stage3['n']['prompts']} prompts × {stage3['n']['topics']} topics = "
        f"{stage3['n']['promptTopicUnits']} prompt-topic units**"
    )
    achieved = [
        value
        for topic in stage3["achievedMeanTeacherForcedKL"].values()
        for value in topic.values()
    ]
    require(f"`{range_floor(min(achieved), 9)}` to `{range_ceiling(max(achieved), 9)}` nats/step")
    baseline_nll = stage3["medianBaseModelNLL"]["baseline"]
    residual_nll = stage3["medianBaseModelNLL"]["residual"]
    require(f"`{baseline_nll:.6f}` baseline")
    require(f"`{residual_nll:.6f}` for the semantic residual arm")
    require(f"`{residual_nll - baseline_nll - 1:.6f}` nat/token")
    interval = stage3["denominatorPromptClusterBootstrap95"]
    require(f"`[{range_floor(interval[0], 6)}, {range_ceiling(interval[1], 6)}]`")
    require("No point ratio or ratio interval is reported.")
    require(
        f"{len(calibration_paths)} calibration packets, "
        f"{len(comparison_paths) + len(random_paths)} output packets"
    )
    failure = json.loads(
        (ROOT / "docs/phase6/teacher-forced-comparison/calibration-failure.json").read_text()
    )
    check(failure["status"] == "failed-monotonicity-gate", "original calibration failure changed")
    check(failure["topicOutputsObserved"] is False, "original calibration observed topic output")
    # The blind set is the topics the failed monotonic run had completed before
    # amendment-1 changed the selection rule; those packets are still on disk.
    blind = [row for row in calibration_packets if row["lexicon"] in failure["completedTopics"]]
    check(
        len(blind) == failure["packetCount"],
        f"retained blind calibration packets ({len(blind)}) no longer match the recorded "
        f"count ({failure['packetCount']})",
    )
    require(f"all {failure['packetCount']} blind packets were retained")
    print("PASS teacher-forced calibration, invalid NLL gate, blind-packet retention, and ratio withholding")

    rho_paths = sorted((ROOT / "docs/phase6/on-device-rho/runs").glob("*.json"))
    rho_packets = [json.loads(path.read_text()) for path in rho_paths]
    check(len(rho_packets) == 8, "comparison packet count changed")
    check(all(abs(row["temperature"] - 0.7) < 1e-6 for row in rho_packets), "comparison temperature changed")
    protocol = (ROOT / "docs/phase6/on-device-rho/protocol.md").read_text()
    check("temperature 0.7" in protocol, "precommitted protocol temperature changed")
    temperature = rho_packets[0]["temperature"]
    require(f"temperature `{temperature:.1f}`")
    check(all(row["buildConfiguration"] == "Release" for row in rho_packets), "a comparison packet is not Release")
    rho_budgets = {row["klBudget"] for row in rho_packets}
    rho_token_limits = {row["maxTokens"] for row in rho_packets}
    check(
        len(rho_budgets) == len(rho_token_limits) == 1,
        "comparison packets no longer share one cap and token limit",
    )
    rho_budget = rho_budgets.pop()
    require(f"the cumulative {rho_budget:g}-nat cap in all eight prompt-topic runs")
    require(f"under the same {rho_budget:g}-nat cap")

    # Every committed packet that records a seed must record the same one. The
    # historical KL-4 set predates seed recording and is the only exemption.
    zero = json.loads((ROOT / "docs/phase6/coefficient-zero/report.json").read_text())
    all_packets = (
        [final, zero]
        + sanity
        + cap4_packets
        + rho_packets
        + sweep_packets
        + blocking_packets
        + preserved_packets
        + calibration_packets
        + random_floor_packets
    )
    recorded_seeds = [row["seed"] for row in all_packets if "seed" in row]
    check(len(set(recorded_seeds)) == 1, f"committed packets disagree on the seed: {sorted(set(recorded_seeds))}")
    check(
        len(all_packets) - len(recorded_seeds) == len(cap4_packets),
        "a packet outside the historical KL-4 set omits its seed",
    )
    seed = recorded_seeds[0]
    require(f"chat template, seed (`{seed}`), temperature (`{temperature:.1f}`)")
    require(
        f"eight Release packets at seed {seed}, temperature `{temperature:.1f}`, "
        f"and at most {rho_token_limits.pop()} generated tokens"
    )

    dense_shares, sparse_shares, dense_steps, sparse_steps = [], [], [], []
    for row in rho_packets:
        dense, sparse = row["actAddKLHistory"], row["klHistory"]
        dense_shares.append(dense[0]["perStep"] / dense[-1]["cumulative"])
        sparse_shares.append(sparse[0]["perStep"] / sparse[-1]["cumulative"])
        dense_steps.append(len(dense))
        sparse_steps.append(len(sparse))
    dense_collapsed = sum(1 for share in dense_shares if share > FIRST_STEP_COLLAPSE_SHARE)
    sparse_collapsed = sum(1 for share in sparse_shares if share > FIRST_STEP_COLLAPSE_SHARE)
    check(dense_collapsed == 8, "dense in-packet first-step collapse count changed")
    check(sparse_collapsed == 1, "sparse in-packet first-step collapse count changed")
    require(f"put more than {100 * FIRST_STEP_COLLAPSE_SHARE:g}% of its cumulative cap")
    require(f"into step one in **{dense_collapsed} of {len(rho_packets)}** runs")
    require(f"the sparse logit bias did so in **{sparse_collapsed} of {len(rho_packets)}**")
    exception_name, exception_share = max(
        ((path.stem, share) for path, share in zip(rho_paths, sparse_shares)),
        key=lambda pair: pair[1],
    )
    require(f"(`{exception_name}`, at {100 * exception_share:.4f}%)")
    others = sorted(share for share in sparse_shares if share <= FIRST_STEP_COLLAPSE_SHARE)
    require(
        f"only {range_floor(100 * others[0], 2)}%–{range_ceiling(100 * others[-1])}% "
        "there in the other seven"
    )
    require(
        f"spreading its cap over {min(sparse_steps)}–{max(sparse_steps)} biased steps "
        f"against the dense arm's {min(dense_steps)}–{max(dense_steps)}"
    )
    print("PASS comparison temperature, Release provenance, and matched in-packet first-step contrast")

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
    require(f"the full `{decomposition['observed_shift']:+.4f}` shift")
    print("PASS judge decomposition and prominent 68.4% self-measurement")

    parity = json.loads((ROOT / "docs/coreml-parity.json").read_text())
    metadata = parity["metadata"]
    check(len(parity["rows"]) == metadata["validation_cases"] == 24, "Core ML case count changed")
    require(f"{metadata['validation_cases']} inputs")
    require(f"`{metadata['minimum_cosine']:.9f}`")
    require(f"`{metadata['threshold']:.4f}` gate")
    check(metadata["max_length"] == 128, "Core ML max length changed")
    require(f"fixed {metadata['max_length']}-token inputs")
    check(metadata["compute_units"] == "ALL", "Core ML compute units changed")
    check(len(metadata["weight_sha256"]) == 64, "Core ML digest is not SHA-256")
    centroids = json.loads((ROOT / "Resources/CoreML/topic-centroids.json").read_text())
    check(centroids["dimensions"] == 384, "Core ML dimensions changed")
    require(f"{centroids['dimensions']}-dimensional embedding")
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
    quantization = re.search(r"^(.*)-(\d+)bit$", model_id.group(1).split("/")[-1])
    check(quantization is not None, "the pinned model id no longer names its quantization")
    require(f"runs {quantization.group(1)} ({quantization.group(2)}-bit) through MLX Swift")
    require(f"and {quantization.group(2)}-bit MLX model")
    print("PASS test count, pinned Qwen identity, and quantization label")

    audit = json.loads((ROOT / "docs/audit-reference.json").read_text())
    point = audit["rho"]["point"]
    # The citation year is the year of the audit's own dated results directory.
    audit_year = re.search(r"/results/(\d{4})-\d{2}-\d{2}-", audit["source"])
    check(audit_year is not None, "audit source no longer carries a dated results directory")
    require(f"The steering audit (Wang, {audit_year.group(1)}) reported")
    require(f"{100 * point:.1f}%")
    require(f"`rho = {point}`")
    require(f"`n = {audit['n_eval']}`")
    require(f"[{audit['rho']['ci_lo']}, {audit['rho']['ci_hi']}]")
    require(f"verdict is `{audit['verdict']['class']}`")
    require(
        f"95% CI {range_floor(100 * audit['rho']['ci_lo'])}%–"
        f"{range_ceiling(100 * audit['rho']['ci_hi'])}%"
    )
    dissolution = re.search(r"rho_lo\s*>=\s*([0-9.]+)", audit["verdict"]["dissolved_rule"])
    check(dissolution is not None, "audit dissolution rule no longer names its rho_lo threshold")
    require(f"`rho_lo < {dissolution.group(1)}` fails the audit's dissolution rule")
    matched = audit["matched_protocol"]
    require(f"`{matched['mean_teacher_forced_kl_nats_per_step']}` nats per step")
    require(f"fixed {matched['continuation_steps']}-step continuation")
    positions = matched["prompt_positions"]
    check(positions == list(range(positions[0], positions[-1] + 1)), "audit prompt positions are no longer contiguous")
    require(f"prompt positions {positions[0]} through {positions[-1]}")
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
    lora_config = json.loads((ROOT / "LoRA/adapter/adapter_config.json").read_text())
    train_examples = len((ROOT / "LoRA/data/train.jsonl").read_text().splitlines())
    check(lora_config["num_layers"] == 4, "the toy LoRA is no longer four-layer")
    require("**MLX Python**, not MLX Swift")
    require(f"A four-layer rank-{lora_config['lora_parameters']['rank']} LoRA")
    require(f"{lora_config['iters']} optimizer steps on {train_examples} toy codebook examples")
    require(
        f"`{before['exactMatches']}/{before['n']}` before training to "
        f"`{after['exactMatches']}/{after['n']}` after training"
    )
    require(f"{round(adapter.stat().st_size / 2**20)} MB adapter")
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

    sweep_numeric_literals(readme, required_fragments)


if __name__ == "__main__":
    main()
