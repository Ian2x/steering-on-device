#!/usr/bin/env python3
"""Apply the predeclared blocking-control criteria to every committed packet."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "docs/phase6/blocking-control/runs"
OUTPUT = ROOT / "docs/phase6/blocking-control/summary.json"
LAYERS = (8, 10, 12)
COEFFICIENTS = (1.0, 2.0, 4.0, 8.0, 12.0)
LEXICONS = ("wedding", "ocean")
DIRECTIONS = ("semantic", "random-matched-norm")
PROMPTS = (
    "Describe a quiet morning routine in two short paragraphs.",
    "Describe a routine bus commute across a city in two short paragraphs.",
    "Explain how to make a cup of tea in two short paragraphs.",
)


def repeated_trigram_fraction(tokens: list[int]) -> float:
    grams = list(zip(tokens, tokens[1:], tokens[2:]))
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def median(values: list[float]) -> float:
    if not values:
        raise AssertionError("cannot take median of an empty cell")
    return statistics.median(values)


paths = sorted(RUN_DIR.glob("*.json"))
expected_count = len(LAYERS) * len(COEFFICIENTS) * len(LEXICONS) * len(DIRECTIONS) * len(PROMPTS)
if len(paths) != expected_count:
    raise SystemExit(f"Expected {expected_count} packets, found {len(paths)}")

rows = []
seen = set()
for path in paths:
    packet = json.loads(path.read_text())
    key = (
        packet["actAddDirectionMode"],
        packet["actAddLayer"],
        float(packet["actAddAppliedCoefficient"]),
        packet["lexicon"],
        packet["prompt"],
    )
    if key in seen:
        raise AssertionError(f"duplicate cell: {key}")
    seen.add(key)
    if packet["actAddKLCapEnabled"] is not False:
        raise AssertionError(f"KL cap unexpectedly enabled in {path}")
    if packet["actAddCoefficient"] != packet["actAddAppliedCoefficient"]:
        raise AssertionError(f"nominal/applied coefficient mismatch in {path}")
    diagnostics = packet["actAddDirectionDiagnostics"]
    if diagnostics["mode"] != packet["actAddDirectionMode"]:
        raise AssertionError(f"direction mode mismatch in {path}")
    if packet["actAddDirectionMode"] == "random-matched-norm" and not math.isclose(
        diagnostics["appliedMatrixNorm"],
        diagnostics["semanticMatrixNorm"],
        rel_tol=1e-4,
        abs_tol=1e-6,
    ):
        raise AssertionError(f"random direction norm mismatch in {path}")
    if len(packet["actAddKLHistory"]) != packet["actAdd"]["tokenCount"]:
        raise AssertionError(f"EOS/KL accounting mismatch in {path}")
    rows.append(
        {
            "packet": str(path.relative_to(ROOT)),
            "direction": packet["actAddDirectionMode"],
            "layer": packet["actAddLayer"],
            "coefficient": float(packet["actAddAppliedCoefficient"]),
            "lexicon": packet["lexicon"],
            "prompt": packet["prompt"],
            "shift": packet["actAdd"]["topicScore"] - packet["baseline"]["topicScore"],
            "baselineText": packet["baseline"]["text"],
            "residualText": packet["actAdd"]["text"],
            "baselineLength": packet["baseline"]["tokenCount"],
            "residualLength": packet["actAdd"]["tokenCount"],
            "baselineRepetition": repeated_trigram_fraction(packet["baseline"]["tokenIDs"]),
            "residualRepetition": repeated_trigram_fraction(packet["actAdd"]["tokenIDs"]),
            "baselineNLL": packet["baseline"]["baseModelNLL"],
            "residualNLL": packet["actAdd"]["baseModelNLL"],
        }
    )

expected = {
    (direction, layer, coefficient, lexicon, prompt)
    for direction in DIRECTIONS
    for layer in LAYERS
    for coefficient in COEFFICIENTS
    for lexicon in LEXICONS
    for prompt in PROMPTS
}
if seen != expected:
    raise AssertionError(f"grid mismatch: missing={expected-seen}, extra={seen-expected}")

cells = []
for layer in LAYERS:
    for coefficient in COEFFICIENTS:
        semantic = [r for r in rows if r["direction"] == "semantic" and r["layer"] == layer and r["coefficient"] == coefficient]
        random_rows = [r for r in rows if r["direction"] == "random-matched-norm" and r["layer"] == layer and r["coefficient"] == coefficient]
        identity = []
        for prompt in PROMPTS:
            pair = [r for r in semantic if r["prompt"] == prompt]
            if len(pair) != 2:
                raise AssertionError(f"missing semantic topic pair for {layer}/{coefficient}/{prompt}")
            identity.append(pair[0]["residualText"].encode() == pair[1]["residualText"].encode())

        topic_rows = {}
        topic_passes = []
        for lexicon in LEXICONS:
            semantic_topic = [r for r in semantic if r["lexicon"] == lexicon]
            random_topic = [r for r in random_rows if r["lexicon"] == lexicon]
            semantic_median = median([r["shift"] for r in semantic_topic])
            random_median = median([r["shift"] for r in random_topic])
            positive_count = sum(r["shift"] > 0 for r in semantic_topic)
            on_target = positive_count >= 2 and semantic_median >= 0.05
            above_floor = semantic_median - random_median >= 0.03
            topic_passes.append(on_target and above_floor)
            topic_rows[lexicon] = {
                "semanticShifts": [r["shift"] for r in semantic_topic],
                "semanticMedianShift": semantic_median,
                "semanticPositiveCount": positive_count,
                "randomShifts": [r["shift"] for r in random_topic],
                "randomMedianShift": random_median,
                "semanticMinusRandomMedian": semantic_median - random_median,
                "onTargetPass": on_target,
                "randomFloorPass": above_floor,
            }

        baseline_rows = semantic
        all_residual = semantic + random_rows
        length_pass = (
            median([r["baselineLength"] for r in baseline_rows]) >= 24
            and median([r["residualLength"] for r in semantic]) >= 24
            and median([r["residualLength"] for r in random_rows]) >= 24
        )
        repetition_pass = (
            median([r["baselineRepetition"] for r in baseline_rows]) <= 0.25
            and median([r["residualRepetition"] for r in semantic]) <= 0.25
            and median([r["residualRepetition"] for r in random_rows]) <= 0.25
        )
        all_nll = [r["baselineNLL"] for r in baseline_rows] + [r["residualNLL"] for r in all_residual]
        finite_nll = all(value is not None and math.isfinite(value) for value in all_nll)
        baseline_nll = median([r["baselineNLL"] for r in baseline_rows])
        semantic_nll = median([r["residualNLL"] for r in semantic])
        nll_pass = finite_nll and semantic_nll <= baseline_nll + 1.0
        direction_pass = not any(identity)
        passed = direction_pass and all(topic_passes) and length_pass and repetition_pass and nll_pass
        cells.append(
            {
                "layer": layer,
                "coefficient": coefficient,
                "passed": passed,
                "crossLexiconIdentityByPrompt": identity,
                "directionDependencePass": direction_pass,
                "topics": topic_rows,
                "medianLengths": {
                    "baseline": median([r["baselineLength"] for r in baseline_rows]),
                    "semantic": median([r["residualLength"] for r in semantic]),
                    "random": median([r["residualLength"] for r in random_rows]),
                },
                "lengthPass": length_pass,
                "medianRepeatedTrigramFractions": {
                    "baseline": median([r["baselineRepetition"] for r in baseline_rows]),
                    "semantic": median([r["residualRepetition"] for r in semantic]),
                    "random": median([r["residualRepetition"] for r in random_rows]),
                },
                "repetitionPass": repetition_pass,
                "medianBaseModelNLL": {
                    "baseline": baseline_nll,
                    "semantic": semantic_nll,
                    "random": median([r["residualNLL"] for r in random_rows]),
                },
                "nllPass": nll_pass,
            }
        )

passing = [cell for cell in cells if cell["passed"]]
selected = None
if passing:
    selected_cell = sorted(
        passing,
        key=lambda cell: (cell["coefficient"], abs(cell["layer"] - 10), cell["layer"]),
    )[0]
    selected = {"layer": selected_cell["layer"], "coefficient": selected_cell["coefficient"]}

semantic_diagnostics = []
for path in paths:
    packet = json.loads(path.read_text())
    if packet["actAddDirectionMode"] == "semantic":
        semantic_diagnostics.append(packet["actAddDirectionDiagnostics"])

summary = {
    "status": "pass" if passing else "fail",
    "protocol": "docs/phase6/blocking-control/protocol.md",
    "packetCount": len(rows),
    "passDefinition": "all predeclared direction, on-target, random-floor, length, repetition, and NLL gates",
    "passingCellCount": len(passing),
    "selectedCellByPredeclaredTieBreak": selected,
    "cells": cells,
    "directionDiagnostics": {
        "historicalFinalVectorNormRange": [
            min(row["historicalFinalVectorNorm"] for row in semantic_diagnostics),
            max(row["historicalFinalVectorNorm"] for row in semantic_diagnostics),
        ],
        "semanticMatrixNormRange": [
            min(row["semanticMatrixNorm"] for row in semantic_diagnostics),
            max(row["semanticMatrixNorm"] for row in semantic_diagnostics),
        ],
        "alignedPositionCounts": sorted(set(row["alignedPositionCount"] for row in semantic_diagnostics)),
        "alignment": sorted(set(row["alignment"] for row in semantic_diagnostics)),
        "injection": sorted(set(row["injection"] for row in semantic_diagnostics)),
    },
}
OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"blocking control {summary['status'].upper()}: {len(passing)}/{len(cells)} cells pass; wrote {OUTPUT}")
