#!/usr/bin/env python3
"""Derive a calibration golden fixture from the committed Phase 6 calibration packets.

The fixture pins `CalibrationSelector` to the frozen result: replaying the four committed
calibration curves through the Swift selector must reproduce the four scalars in
`docs/phase6/teacher-forced-comparison/results.md`.

The packets are read-only inputs. This script never writes to `docs/`.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CALIBRATION_DIR = ROOT / "docs/phase6/teacher-forced-comparison/calibration-runs"
SUMMARY = ROOT / "docs/phase6/teacher-forced-comparison/summary.json"

TARGET = 0.43523801873284795
TOLERANCE = 0.002
ITERATIONS = 18
BRACKET = {"logit": 20.0, "actadd": 40.0}
FIELD = {"logit": "teacherForcedLogit", "actadd": "teacherForcedActAdd"}
# Packet filenames say "actadd"; the frozen summary says "residual".
SUMMARY_KEY = {"logit": "logit", "actadd": "residual"}
EXPECTED_ARMS = 4
EXPECTED_CANDIDATES = ITERATIONS + 1


def build_curves() -> dict[tuple[str, str], list[dict[str, float]]]:
    """Collapse per-prompt packets into the four-prompt mean each candidate was judged on."""
    grouped: dict[tuple[str, str, str], dict[str, tuple[float, float]]] = (
        collections.defaultdict(dict)
    )
    for path in sorted(CALIBRATION_DIR.glob("*.json")):
        topic, method, label, prompt = path.stem.split("-")
        packet = json.loads(path.read_text())
        measurement = packet[FIELD[method]]
        grouped[(topic, method, label)][prompt] = (
            measurement["appliedScalar"],
            measurement["meanNatsPerStep"],
        )

    curves: dict[tuple[str, str], list[dict[str, float]]] = collections.defaultdict(list)
    for (topic, method, label), prompts in grouped.items():
        scalars = {scalar for scalar, _ in prompts.values()}
        if len(scalars) != 1:
            raise SystemExit(f"{topic}/{method}/{label} mixes scalars: {sorted(scalars)}")
        mean = sum(value for _, value in prompts.values()) / len(prompts)
        curves[(topic, method)].append(
            {
                "label": label,
                "promptCount": len(prompts),
                "scalar": scalars.pop(),
                "meanNatsPerStep": mean,
            }
        )
    return curves


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--check", type=pathlib.Path)
    arguments = parser.parse_args()
    if (arguments.output is None) == (arguments.check is None):
        parser.error("pass exactly one of --output or --check")

    curves = build_curves()
    if len(curves) != EXPECTED_ARMS:
        raise SystemExit(f"expected {EXPECTED_ARMS} arms, found {len(curves)}")

    summary = json.loads(SUMMARY.read_text())
    if summary["targetMeanTeacherForcedKL"] != TARGET:
        raise SystemExit("frozen summary target does not match this script's target")
    arms = []
    for (topic, method), candidates in sorted(curves.items()):
        if len(candidates) != EXPECTED_CANDIDATES:
            raise SystemExit(
                f"{topic}/{method} has {len(candidates)} candidates, expected {EXPECTED_CANDIDATES}"
            )
        key = SUMMARY_KEY[method]
        arms.append(
            {
                "topic": topic,
                "method": method,
                "upperBracket": BRACKET[method],
                "iterations": ITERATIONS,
                # What the frozen runner selected. The Swift selector must reproduce both.
                "expectedScalar": summary["calibratedScalars"][topic][key],
                "expectedMeanNatsPerStep": summary["achievedMeanTeacherForcedKL"][topic][key],
                "candidates": sorted(candidates, key=lambda row: row["scalar"]),
            }
        )

    fixture = {
        "source": "docs/phase6/teacher-forced-comparison/calibration-runs",
        "generator": "Scripts/generate_calibration_fixture.py",
        "target": TARGET,
        "tolerance": TOLERANCE,
        "selectionRule": "minimum absolute tested mean-KL error; ties choose lower scalar",
        "amendment": "docs/phase6/teacher-forced-comparison/amendment-1.md",
        "arms": arms,
    }
    rendered = json.dumps(fixture, indent=2, sort_keys=True) + "\n"

    if arguments.check:
        existing = arguments.check.read_text()
        if existing != rendered:
            print(f"{arguments.check} does not match the committed packets", file=sys.stderr)
            return 1
        print(f"verified {arguments.check}")
        return 0

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
