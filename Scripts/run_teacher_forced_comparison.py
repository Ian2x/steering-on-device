#!/usr/bin/env python3
"""Run the predeclared teacher-forced calibration and comparison without overwrites."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "DerivedData" / "Build" / "Products" / "Release" / "SteerDemo.app"
EXECUTABLE = APP / "Contents" / "MacOS" / "SteerDemo"
BASE = ROOT / "docs/phase6/teacher-forced-comparison"
CALIBRATION_DIR = BASE / "calibration-runs"
RUN_DIR = BASE / "runs"
RANDOM_DIR = BASE / "random-floor-runs"
CALIBRATION_SUMMARY = BASE / "calibration-summary.json"
TARGET = 0.43523801873284795
TOLERANCE = 0.002
ITERATIONS = 18
LAYER = 10
TOPICS = ("wedding", "ocean")
PROMPTS = {
    "bookshelf": "Describe organizing a small bookshelf in two short paragraphs.",
    "rainwalk": "Explain how to prepare for a rainy afternoon walk in two short paragraphs.",
    "library": "Describe a quiet library study session in two short paragraphs.",
    "lunch": "Explain how to pack a lunch for work in two short paragraphs.",
}
METHODS = {
    "logit": {"mode": "calibrate-logit", "high": 20.0, "field": "teacherForcedLogit"},
    "actadd": {"mode": "calibrate-actadd", "high": 40.0, "field": "teacherForcedActAdd"},
}


def build() -> None:
    subprocess.run(
        [
            "xcodebuild", "-quiet", "-project", str(ROOT / "SteerDemo.xcodeproj"),
            "-scheme", "SteerDemo", "-configuration", "Release",
            "-destination", "platform=macOS,arch=arm64",
            "-derivedDataPath", str(ROOT / "DerivedData"),
            "ONLY_ACTIVE_ARCH=YES", "CODE_SIGNING_ALLOWED=NO", "build",
        ],
        check=True,
    )


def app_pids() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", str(EXECUTABLE)], capture_output=True, text=True
    )
    return [int(value) for value in result.stdout.split()] if result.returncode == 0 else []


def stop_apps() -> None:
    for pid in app_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 5
    while app_pids() and time.time() < deadline:
        time.sleep(0.05)


def validate_packet(
    path: Path,
    mode: str,
    topic: str,
    strength: float,
    coefficient: float,
) -> dict:
    row = json.loads(path.read_text())
    assert row["status"].startswith("Complete"), row
    assert row["buildConfiguration"] == "Release", row
    assert row["stage3RunMode"] == mode, row
    assert row["lexicon"] == topic, row
    assert row["staticBiasKLCapEnabled"] is False, row
    assert row["teacherForcedTargetKL"] == TARGET, row
    assert len(row["teacherForcedContinuationTokenIDs"]) == 64, row
    if mode in ("calibrate-logit", "evaluate"):
        result = row["teacherForcedLogit"]
        assert result["appliedScalar"] == strength, row
        assert len(result["perStepNats"]) == result["continuationTokenCount"] == 64, row
        assert all(math.isfinite(value) and value >= 0 for value in result["perStepNats"]), row
    if mode in ("calibrate-actadd", "evaluate", "evaluate-random"):
        result = row["teacherForcedActAdd"]
        assert result["appliedScalar"] == coefficient, row
        assert row["actAddAppliedCoefficient"] == coefficient, row
        assert row["actAddKLCapEnabled"] is False, row
        assert len(result["perStepNats"]) == result["continuationTokenCount"] == 64, row
        assert all(math.isfinite(value) and value >= 0 for value in result["perStepNats"]), row
    return row


def run_packet(
    path: Path,
    *,
    mode: str,
    topic: str,
    prompt: str,
    strength: float,
    coefficient: float,
    direction: str = "semantic",
) -> dict:
    if path.exists():
        print(f"retaining existing packet: {path.relative_to(ROOT)}", flush=True)
        return validate_packet(path, mode, topic, strength, coefficient)
    path.parent.mkdir(parents=True, exist_ok=True)
    stop_apps()
    command = [
        "open", "-F", "-n",
        "--env", "STEERDEMO_AUTORUN=1",
        "--env", f"STEERDEMO_STAGE3_MODE={mode}",
        "--env", f"STEERDEMO_RESIDUAL_DIRECTION={direction}",
        "--env", f"STEERDEMO_LEXICON={topic}",
        "--env", f"STEERDEMO_PROMPT={prompt}",
        "--env", f"STEERDEMO_BIAS_STRENGTH={strength:.17g}",
        "--env", f"STEERDEMO_ACTADD_COEFFICIENT={coefficient:.17g}",
        "--env", f"STEERDEMO_ACTADD_LAYER={LAYER}",
        "--env", "STEERDEMO_MAX_TOKENS=64",
        "--env", f"STEERDEMO_REPORT_PATH={path}",
        str(APP),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    launch_deadline = time.time() + 5
    while not path.exists() and not app_pids() and time.time() < launch_deadline:
        time.sleep(0.1)
    if not path.exists() and not app_pids():
        raise RuntimeError(f"SteerDemo did not launch for {path}")
    deadline = time.time() + 180
    missing_process_since: float | None = None
    while not path.exists() and time.time() < deadline:
        if not app_pids():
            missing_process_since = missing_process_since or time.time()
            if time.time() - missing_process_since > 15:
                raise RuntimeError(f"SteerDemo exited before writing {path}")
        else:
            missing_process_since = None
        time.sleep(0.2)
    if not path.exists():
        raise TimeoutError(f"timed out waiting for {path}")
    row = validate_packet(path, mode, topic, strength, coefficient)
    stop_apps()
    return row


def evaluate_candidate(topic: str, method: str, label: str, scalar: float) -> tuple[float, list[dict]]:
    config = METHODS[method]
    rows = []
    for prompt_id, prompt in PROMPTS.items():
        path = CALIBRATION_DIR / f"{topic}-{method}-{label}-{prompt_id}.json"
        rows.append(
            run_packet(
                path,
                mode=config["mode"],
                topic=topic,
                prompt=prompt,
                strength=scalar if method == "logit" else 0,
                coefficient=scalar if method == "actadd" else 4,
            )
        )
    field = config["field"]
    mean = sum(row[field]["meanNatsPerStep"] for row in rows) / len(rows)
    print(f"{topic} {method} {label} scalar={scalar:.10f} meanKL={mean:.9f}", flush=True)
    return mean, rows


def calibrate(topic: str, method: str) -> dict:
    low = 0.0
    high = METHODS[method]["high"]
    high_mean, _ = evaluate_candidate(topic, method, "high", high)
    if not math.isfinite(high_mean) or high_mean < TARGET:
        raise RuntimeError(f"frozen upper bracket does not reach target: {topic}/{method}={high_mean}")
    tested = [(high, high_mean)]
    selected_mean = high_mean
    for index in range(ITERATIONS):
        midpoint = (low + high) / 2
        mean, _ = evaluate_candidate(topic, method, f"iter{index:02d}", midpoint)
        tested.append((midpoint, mean))
        if mean >= TARGET:
            high = midpoint
            selected_mean = mean
        else:
            low = midpoint
    ordered = sorted(tested)
    for (scalar_a, mean_a), (scalar_b, mean_b) in zip(ordered, ordered[1:]):
        if scalar_b > scalar_a and mean_b + 1e-5 < mean_a:
            raise RuntimeError(
                f"monotonicity reversal for {topic}/{method}: "
                f"{scalar_a}/{mean_a} -> {scalar_b}/{mean_b}"
            )
    if not (TARGET <= selected_mean <= TARGET + TOLERANCE):
        raise RuntimeError(
            f"calibration tolerance failed for {topic}/{method}: {selected_mean}"
        )
    return {
        "scalar": high,
        "meanTeacherForcedKL": selected_mean,
        "lowerBracket": low,
        "target": TARGET,
        "tolerance": TOLERANCE,
        "iterations": ITERATIONS,
        "tested": [{"scalar": scalar, "meanTeacherForcedKL": mean} for scalar, mean in tested],
    }


def main() -> None:
    build()
    try:
        calibration = {
            topic: {method: calibrate(topic, method) for method in METHODS}
            for topic in TOPICS
        }
        CALIBRATION_SUMMARY.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "targetMeanTeacherForcedKL": TARGET,
                    "promptCount": len(PROMPTS),
                    "continuationSteps": 64,
                    "calibration": calibration,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        for topic in TOPICS:
            strength = calibration[topic]["logit"]["scalar"]
            coefficient = calibration[topic]["actadd"]["scalar"]
            for prompt_id, prompt in PROMPTS.items():
                run_packet(
                    RUN_DIR / f"{topic}-{prompt_id}.json",
                    mode="evaluate",
                    topic=topic,
                    prompt=prompt,
                    strength=strength,
                    coefficient=coefficient,
                )
                run_packet(
                    RANDOM_DIR / f"{topic}-{prompt_id}.json",
                    mode="evaluate-random",
                    topic=topic,
                    prompt=prompt,
                    strength=strength,
                    coefficient=coefficient,
                    direction="random-matched-norm",
                )
        subprocess.run(
            ["python3", str(ROOT / "Scripts/summarize_teacher_forced_comparison.py")],
            check=True,
        )
    finally:
        stop_apps()


if __name__ == "__main__":
    main()
