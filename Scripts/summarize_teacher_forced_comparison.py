#!/usr/bin/env python3
"""Validate and summarize the frozen teacher-forced comparison."""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/phase6/teacher-forced-comparison"
CALIBRATION_DIR = BASE / "calibration-runs"
RUN_DIR = BASE / "runs"
RANDOM_DIR = BASE / "random-floor-runs"
OUTPUT = BASE / "summary.json"
TARGET = 0.43523801873284795
TOLERANCE = 0.002
PROMPTS = ("bookshelf", "rainwalk", "library", "lunch")
TOPICS = ("wedding", "ocean")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def median(values: list[float]) -> float:
    return statistics.median(values)


def repeated_trigram_fraction(tokens: list[int]) -> float:
    grams = list(zip(tokens, tokens[1:], tokens[2:]))
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


calibration = load(BASE / "calibration-summary.json")
if calibration["status"] != "pass" or calibration["targetMeanTeacherForcedKL"] != TARGET:
    raise AssertionError("calibration summary is not a pass at the frozen target")
calibration_paths = sorted(CALIBRATION_DIR.glob("*.json"))
if len(calibration_paths) != 2 * 2 * 19 * 4:
    raise AssertionError(f"expected 304 calibration packets, found {len(calibration_paths)}")
calibration_continuations: dict[str, tuple[int, ...]] = {}
for path in calibration_paths:
    packet = load(path)
    continuation = tuple(packet["teacherForcedContinuationTokenIDs"])
    if len(continuation) != 64:
        raise AssertionError(f"calibration continuation is not 64 steps in {path}")
    prompt = packet["prompt"]
    if prompt in calibration_continuations and calibration_continuations[prompt] != continuation:
        raise AssertionError(f"calibration candidate changed the continuation for {prompt}")
    calibration_continuations[prompt] = continuation

rows = []
continuations: dict[str, tuple[int, ...]] = {}
for topic in TOPICS:
    expected_strength = calibration["calibration"][topic]["logit"]["scalar"]
    expected_coefficient = calibration["calibration"][topic]["actadd"]["scalar"]
    semantic_topic = []
    random_topic = []
    for prompt_id in PROMPTS:
        packet = load(RUN_DIR / f"{topic}-{prompt_id}.json")
        random_packet = load(RANDOM_DIR / f"{topic}-{prompt_id}.json")
        if packet["buildConfiguration"] != "Release" or random_packet["buildConfiguration"] != "Release":
            raise AssertionError("non-Release comparison packet")
        if packet["stage3RunMode"] != "evaluate" or random_packet["stage3RunMode"] != "evaluate-random":
            raise AssertionError("Stage 3 mode mismatch")
        if packet["biasStrength"] != expected_strength:
            raise AssertionError("static-bias scalar mismatch")
        if packet["actAddAppliedCoefficient"] != expected_coefficient:
            raise AssertionError("semantic residual scalar mismatch")
        if random_packet["actAddAppliedCoefficient"] != expected_coefficient:
            raise AssertionError("random residual scalar mismatch")
        continuation = tuple(packet["teacherForcedContinuationTokenIDs"])
        if len(continuation) != 64 or tuple(random_packet["teacherForcedContinuationTokenIDs"]) != continuation:
            raise AssertionError("final packets do not share a 64-step continuation")
        if random_packet["baseline"]["tokenIDs"] != packet["baseline"]["tokenIDs"]:
            raise AssertionError("semantic and random packets have different baseline output")
        if prompt_id in continuations and continuations[prompt_id] != continuation:
            raise AssertionError("topic changed the fixed continuation")
        continuations[prompt_id] = continuation
        for field in ("teacherForcedLogit", "teacherForcedActAdd"):
            result = packet[field]
            if len(result["perStepNats"]) != 64 or not all(math.isfinite(x) for x in result["perStepNats"]):
                raise AssertionError(f"invalid teacher-forced vector in {field}")
        diagnostics = random_packet["actAddDirectionDiagnostics"]
        if diagnostics["mode"] != "random-matched-norm" or not math.isclose(
            diagnostics["appliedMatrixNorm"], diagnostics["semanticMatrixNorm"], rel_tol=1e-4, abs_tol=1e-6
        ):
            raise AssertionError("random floor norm mismatch")
        semantic_shift = packet["actAdd"]["topicScore"] - packet["baseline"]["topicScore"]
        random_shift = random_packet["actAdd"]["topicScore"] - random_packet["baseline"]["topicScore"]
        logit_shift = packet["steered"]["topicScore"] - packet["baseline"]["topicScore"]
        row = {
            "topic": topic,
            "prompt": prompt_id,
            "logitShift": logit_shift,
            "residualShift": semantic_shift,
            "randomResidualShift": random_shift,
            "baselineText": packet["baseline"]["text"],
            "residualText": packet["actAdd"]["text"],
            "lengths": {
                "baseline": packet["baseline"]["tokenCount"],
                "logit": packet["steered"]["tokenCount"],
                "residual": packet["actAdd"]["tokenCount"],
                "random": random_packet["actAdd"]["tokenCount"],
            },
            "repetition": {
                "baseline": repeated_trigram_fraction(packet["baseline"]["tokenIDs"]),
                "logit": repeated_trigram_fraction(packet["steered"]["tokenIDs"]),
                "residual": repeated_trigram_fraction(packet["actAdd"]["tokenIDs"]),
                "random": repeated_trigram_fraction(random_packet["actAdd"]["tokenIDs"]),
            },
            "nll": {
                "baseline": packet["baseline"]["baseModelNLL"],
                "logit": packet["steered"]["baseModelNLL"],
                "residual": packet["actAdd"]["baseModelNLL"],
                "random": random_packet["actAdd"]["baseModelNLL"],
            },
            "teacherForcedKL": {
                "logit": packet["teacherForcedLogit"]["meanNatsPerStep"],
                "residual": packet["teacherForcedActAdd"]["meanNatsPerStep"],
            },
        }
        rows.append(row)
        semantic_topic.append(semantic_shift)
        random_topic.append(random_shift)

topic_results = {}
topic_pass = True
for topic in TOPICS:
    topic_rows = [row for row in rows if row["topic"] == topic]
    semantic = [row["residualShift"] for row in topic_rows]
    random_shifts = [row["randomResidualShift"] for row in topic_rows]
    positive_count = sum(value > 0 for value in semantic)
    semantic_median = median(semantic)
    random_median = median(random_shifts)
    passed = positive_count >= 3 and semantic_median >= 0.05 and semantic_median - random_median >= 0.03
    topic_pass &= passed
    topic_results[topic] = {
        "semanticShifts": semantic,
        "semanticPositiveCount": positive_count,
        "semanticMedianShift": semantic_median,
        "randomShifts": random_shifts,
        "randomMedianShift": random_median,
        "semanticMinusRandomMedian": semantic_median - random_median,
        "passed": passed,
    }

identity = []
for prompt_id in PROMPTS:
    pair = [row for row in rows if row["prompt"] == prompt_id]
    identity.append(pair[0]["residualText"].encode() == pair[1]["residualText"].encode())
direction_pass = not any(identity)

kl_means = {}
kl_pass = True
for topic in TOPICS:
    kl_means[topic] = {}
    for method, field in (("logit", "logit"), ("residual", "residual")):
        value = statistics.mean(
            row["teacherForcedKL"][field] for row in rows if row["topic"] == topic
        )
        kl_means[topic][method] = value
        kl_pass &= TARGET <= value <= TARGET + TOLERANCE

arms = ("baseline", "logit", "residual", "random")
median_lengths = {arm: median([row["lengths"][arm] for row in rows]) for arm in arms}
length_pass = all(value >= 48 for value in median_lengths.values())
median_repetition = {arm: median([row["repetition"][arm] for row in rows]) for arm in arms}
repetition_pass = all(value <= 0.25 for value in median_repetition.values())
all_nll = [row["nll"][arm] for row in rows for arm in arms]
finite_nll = all(value is not None and math.isfinite(value) for value in all_nll)
median_nll = {arm: median([row["nll"][arm] for row in rows]) for arm in arms}
nll_pass = finite_nll and all(
    median_nll[arm] <= median_nll["baseline"] + 1 for arm in ("logit", "residual", "random")
)
valid = kl_pass and direction_pass and topic_pass and length_pass and repetition_pass and nll_pass

denominator = statistics.mean(row["residualShift"] for row in rows)
numerator = statistics.mean(row["logitShift"] for row in rows)
rng = random.Random(20_260_806)
cluster_values = {
    prompt: [row["residualShift"] for row in rows if row["prompt"] == prompt]
    for prompt in PROMPTS
}
bootstrap = []
for _ in range(100_000):
    sampled = [rng.choice(PROMPTS) for _ in PROMPTS]
    bootstrap.append(statistics.mean(value for prompt in sampled for value in cluster_values[prompt]))
denominator_interval = [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]
contains_zero = denominator_interval[0] <= 0 <= denominator_interval[1]
if not valid:
    ratio_status = "withheld-validity-gate"
    ratio = None
elif contains_zero:
    ratio_status = "undefined-denominator-interval-contains-zero"
    ratio = None
else:
    ratio_status = "defined"
    ratio = numerator / denominator

summary = {
    "status": "valid-comparison" if valid else "invalid-comparison",
    "protocol": "docs/phase6/teacher-forced-comparison/protocol.md",
    "n": {"prompts": 4, "topics": 2, "promptTopicUnits": 8},
    "targetMeanTeacherForcedKL": TARGET,
    "calibratedScalars": {
        topic: {
            "logit": calibration["calibration"][topic]["logit"]["scalar"],
            "residual": calibration["calibration"][topic]["actadd"]["scalar"],
        }
        for topic in TOPICS
    },
    "achievedMeanTeacherForcedKL": kl_means,
    "klMatchPass": kl_pass,
    "crossTopicIdentityByPrompt": identity,
    "directionDependencePass": direction_pass,
    "topics": topic_results,
    "medianLengths": median_lengths,
    "lengthPass": length_pass,
    "medianRepeatedTrigramFractions": median_repetition,
    "repetitionPass": repetition_pass,
    "medianBaseModelNLL": median_nll,
    "nllPass": nll_pass,
    "meanShifts": {"staticLogitBias": numerator, "semanticResidual": denominator},
    "denominatorPromptClusterBootstrap95": denominator_interval,
    "denominatorIntervalContainsZero": contains_zero,
    "ratioStatus": ratio_status,
    "ratio": ratio,
    "ratioPrecisionRule": "two decimal places if defined; no ratio interval",
    "rows": rows,
}
OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(
    f"teacher-forced comparison {summary['status']}: ratio={ratio_status}; "
    f"denominator CI={denominator_interval}; wrote {OUTPUT}"
)
