#!/usr/bin/env python3
"""Select the ActAdd layer using the predeclared, audit-independent statistic."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "docs" / "phase6" / "layer-sweep" / "runs"
OUTPUT = ROOT / "docs" / "phase6" / "layer-sweep" / "summary.json"
LAYERS = [3, 7, 11, 15, 19, 23]

rows = []
for path in sorted(RUN_DIR.glob("*.json")):
    packet = json.loads(path.read_text())
    shift = packet["actAdd"]["topicScore"] - packet["baseline"]["topicScore"]
    rows.append(
        {
            "packet": str(path.relative_to(ROOT)),
            "layer": packet["actAddLayer"],
            "lexicon": packet["lexicon"],
            "shift": shift,
            "absoluteShift": abs(shift),
            "actAddCumulativeKL": packet["actAddCumulativeKL"],
        }
    )

expected = len(LAYERS) * 4
if len(rows) != expected:
    raise SystemExit(f"Expected {expected} sweep packets, found {len(rows)}")

per_layer = []
for layer in LAYERS:
    values = [row["absoluteShift"] for row in rows if row["layer"] == layer]
    if len(values) != 4:
        raise SystemExit(f"Layer {layer} has {len(values)} packets, expected 4")
    per_layer.append(
        {
            "layer": layer,
            "medianAbsoluteTopicShift": statistics.median(values),
            "absoluteTopicShifts": values,
        }
    )

selected = sorted(
    per_layer,
    key=lambda row: (-row["medianAbsoluteTopicShift"], row["layer"]),
)[0]["layer"]
summary = {
    "protocol": "docs/phase6/layer-sweep/protocol.md",
    "selectionCriterion": "largest median absolute ActAdd-minus-baseline topic-score shift; ties choose earlier block",
    "candidateLayers": LAYERS,
    "coefficient": 12.0,
    "klBudget": 8.0,
    "maxTokens": 32,
    "runCount": len(rows),
    "selectedLayer": selected,
    "perLayer": per_layer,
    "runs": rows,
}
OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"selected layer {selected}; wrote {OUTPUT}")
