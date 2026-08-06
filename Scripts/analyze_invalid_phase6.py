#!/usr/bin/env python3
"""Re-derive the Phase 6 invalidation facts from the preserved raw packets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RHO_RUNS = ROOT / "docs/phase6/on-device-rho/runs"
SWEEP_RUNS = ROOT / "docs/phase6/layer-sweep/runs"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def analyze() -> dict[str, object]:
    rho_paths = sorted(RHO_RUNS.glob("*.json"))
    if len(rho_paths) != 8:
        raise AssertionError(f"expected 8 comparison packets, found {len(rho_paths)}")

    packets = [(path, json.loads(path.read_text())) for path in rho_paths]
    by_prompt: dict[str, list[tuple[Path, dict[str, object]]]] = defaultdict(list)
    first_step_rows: list[dict[str, object]] = []
    for path, packet in packets:
        by_prompt[packet["prompt"]].append((path, packet))
        history = packet["actAddKLHistory"]
        if not history or packet["actAddCumulativeKL"] <= 0:
            raise AssertionError(f"missing positive residual-edit KL history in {path}")
        fraction = history[0]["perStep"] / packet["actAddCumulativeKL"]
        first_step_rows.append(
            {
                "packet": relative(path),
                "fraction": fraction,
                "percent": 100 * fraction,
            }
        )

    if len(by_prompt) != 4:
        raise AssertionError(f"expected 4 prompts, found {len(by_prompt)}")
    prompt_pairs: list[dict[str, object]] = []
    for prompt, rows in sorted(by_prompt.items()):
        if sorted(row[1]["lexicon"] for row in rows) != ["ocean", "wedding"]:
            raise AssertionError(f"prompt lacks ocean/wedding pair: {prompt}")
        texts = [row[1]["actAdd"]["text"] for row in rows]
        prompt_pairs.append(
            {
                "prompt": prompt,
                "packets": [relative(row[0]) for row in rows],
                "byteIdenticalAcrossLexicons": texts[0].encode() == texts[1].encode(),
                "startsWithWhenToken": all(text.startswith(" When") for text in texts),
            }
        )

    distinct_outputs = {packet["actAdd"]["text"].encode() for _, packet in packets}

    sweep_paths = sorted(SWEEP_RUNS.glob("*.json"))
    if len(sweep_paths) != 24:
        raise AssertionError(f"expected 24 sweep packets, found {len(sweep_paths)}")
    layer_pairs: dict[tuple[str, str], dict[int, tuple[Path, str]]] = defaultdict(dict)
    for path in sweep_paths:
        packet = json.loads(path.read_text())
        layer = packet["actAddLayer"]
        if layer in (3, 19):
            layer_pairs[(packet["prompt"], packet["lexicon"])][layer] = (
                path,
                packet["actAdd"]["text"],
            )
    layer_identity: list[dict[str, object]] = []
    for (prompt, lexicon), rows in sorted(layer_pairs.items()):
        if set(rows) != {3, 19}:
            raise AssertionError(f"missing block 3/19 pair for {prompt}/{lexicon}")
        layer_identity.append(
            {
                "prompt": prompt,
                "lexicon": lexicon,
                "packets": [relative(rows[3][0]), relative(rows[19][0])],
                "byteIdentical": rows[3][1].encode() == rows[19][1].encode(),
            }
        )

    return {
        "status": "invalidated",
        "reason": "residual-edit arm was not direction-dependent under the greedy cumulative-KL protocol",
        "comparison": {
            "runCount": len(packets),
            "promptCount": len(by_prompt),
            "topicCount": 2,
            "distinctResidualEditOutputs": len(distinct_outputs),
            "crossLexiconIdenticalPromptCount": sum(
                bool(row["byteIdenticalAcrossLexicons"]) for row in prompt_pairs
            ),
            "allResidualEditOutputsStartWithWhenToken": all(
                bool(row["startsWithWhenToken"]) for row in prompt_pairs
            ),
            "firstStepFractionMinimum": min(row["fraction"] for row in first_step_rows),
            "firstStepFractionMaximum": max(row["fraction"] for row in first_step_rows),
            "firstStepRows": first_step_rows,
            "promptPairs": prompt_pairs,
        },
        "layerSweep": {
            "runCount": len(sweep_paths),
            "block3Vs19CaseCount": len(layer_identity),
            "byteIdenticalCaseCount": sum(bool(row["byteIdentical"]) for row in layer_identity),
            "cases": layer_identity,
            "selectedLayer": None,
        },
        "blockingUnconstrainedControlRun": False,
        "supportedConclusion": "none about activation steering, the audit, or transfer",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    result = analyze()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        expected = args.check.read_text()
        if rendered != expected:
            raise SystemExit(f"reanalysis differs from {args.check}")
        print(f"PASS invalidation analysis matches {args.check}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
