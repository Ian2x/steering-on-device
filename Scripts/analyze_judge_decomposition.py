#!/usr/bin/env python3
"""Reproduce the topic-judge decomposition for the committed final run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    run = json.loads((root / "docs" / "final-demo-run.json").read_text())
    centroid_file = json.loads(
        (root / "Resources" / "CoreML" / "topic-centroids.json").read_text()
    )
    centroid = np.asarray(centroid_file["centroids"]["wedding"], dtype=np.float64)
    tokenizer = AutoTokenizer.from_pretrained(
        root / "Resources" / "MiniLMTokenizer", local_files_only=True
    )
    model = ct.models.MLModel(
        str(root / "Resources" / "CoreML" / "TopicEncoder.mlpackage"),
        compute_units=ct.ComputeUnit.ALL,
    )

    def score(text: str) -> float:
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np",
        )
        embedding = model.predict(
            {
                "input_ids": encoded["input_ids"].astype(np.int32),
                "attention_mask": encoded["attention_mask"].astype(np.int32),
            }
        )["embedding"].reshape(-1).astype(np.float64)
        return float(
            np.dot(embedding, centroid)
            / (np.linalg.norm(embedding) * np.linalg.norm(centroid))
        )

    baseline = run["baseline"]["text"]
    steered = run["steered"]["text"]
    prefix = "honeymoon ceremony: "
    rows = {
        "baseline": score(baseline),
        "steered": score(steered),
        "prefix_plus_baseline": score(prefix + baseline),
        "steered_prefix_removed": score(
            steered.replace(" honeymoon ceremony:", "", 1)
        ),
    }
    observed_shift = rows["steered"] - rows["baseline"]
    prefix_shift = rows["prefix_plus_baseline"] - rows["baseline"]
    payload = {
        "source_run": "docs/final-demo-run.json",
        "lexicon": "wedding",
        "removed_prefix": prefix,
        "scores": rows,
        "observed_shift": observed_shift,
        "prefix_only_shift": prefix_shift,
        "prefix_fraction_of_observed_shift": prefix_shift / observed_shift,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
