#!/usr/bin/env python3
"""Evaluate exact arbitrary-code accuracy before and after the toy LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    results = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=8,
            sampler=make_sampler(temp=0),
            verbose=False,
        ).strip()
        results.append(
            {
                "prompt": row["prompt"],
                "expected": row["completion"],
                "output": output,
                "exact": output == row["completion"],
            }
        )
    adapter_display = None
    if args.adapter:
        repository_root = args.output.resolve().parents[2]
        try:
            adapter_display = str(Path(args.adapter).resolve().relative_to(repository_root))
        except ValueError:
            adapter_display = Path(args.adapter).name

    report = {
        "label": args.label,
        "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "modelRevision": "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
        "adapter": adapter_display,
        "n": len(results),
        "exactMatches": sum(result["exact"] for result in results),
        "accuracy": sum(result["exact"] for result in results) / len(results),
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{args.label}: {report['exactMatches']}/{report['n']} exact")
    mx.clear_cache()


if __name__ == "__main__":
    main()
