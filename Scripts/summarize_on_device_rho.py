#!/usr/bin/env python3
"""Write the invalidated status for the preserved Phase 6 comparison packets."""

from __future__ import annotations

import json
from pathlib import Path

from analyze_invalid_phase6 import ROOT, analyze


OUTPUT = ROOT / "docs/phase6/on-device-rho/summary.json"


def main() -> None:
    invalidation = analyze()
    comparison = invalidation["comparison"]
    summary = {
        "status": "invalidated",
        "protocol": "docs/phase6/on-device-rho/protocol.md",
        "invalidationAnalysis": "docs/phase6/invalid-comparison-analysis.json",
        "rawPacketCount": comparison["runCount"],
        "promptCount": comparison["promptCount"],
        "topicCount": comparison["topicCount"],
        "ratio": None,
        "reason": invalidation["reason"],
        "supportedConclusion": invalidation["supportedConclusion"],
        "runs": [
            row["packet"] for row in comparison["firstStepRows"]
        ],
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"invalidated {summary['rawPacketCount']} packets; wrote {OUTPUT}")


if __name__ == "__main__":
    main()
