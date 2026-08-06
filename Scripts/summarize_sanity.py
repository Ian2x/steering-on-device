#!/usr/bin/env python3
"""Render the committed sanity-run JSON files as a Markdown table."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "docs" / "sanity-runs"

print("| Lexicon | Bias strength | Cumulative KL (nats) | Baseline score | Steered score | Change |")
print("|---|---:|---:|---:|---:|---:|")
rows = [json.loads(path.read_text()) for path in RUNS.glob("*.json")]
for row in sorted(rows, key=lambda item: (item["lexicon"], item["biasStrength"])):
    baseline = row["baseline"]["topicScore"]
    steered = row["steered"]["topicScore"]
    print(
        f"| {row['lexicon'].title()} | {row['biasStrength']:.0f} | "
        f"{row['cumulativeKL']:.4f} | {baseline:.4f} | {steered:.4f} | "
        f"{steered - baseline:+.4f} |"
    )
