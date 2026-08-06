#!/usr/bin/env python3
"""Write the invalidated status for the preserved Phase 6 layer sweep."""

from __future__ import annotations

import json

from analyze_invalid_phase6 import ROOT, analyze


OUTPUT = ROOT / "docs/phase6/layer-sweep/summary.json"


def main() -> None:
    invalidation = analyze()
    layer = invalidation["layerSweep"]
    summary = {
        "status": "invalidated",
        "protocol": "docs/phase6/layer-sweep/protocol.md",
        "invalidationAnalysis": "docs/phase6/invalid-comparison-analysis.json",
        "rawPacketCount": layer["runCount"],
        "candidateLayers": [3, 7, 11, 15, 19, 23],
        "selectedLayer": None,
        "reason": "blocks 3 and 19 produced byte-identical text in all four matched cases",
        "supportedConclusion": "no layer selection",
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"invalidated {summary['rawPacketCount']} packets; wrote {OUTPUT}")


if __name__ == "__main__":
    main()
