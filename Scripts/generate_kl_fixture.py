#!/usr/bin/env python3
"""Generate independent high-precision KL fixtures for SteeringKit."""

from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
import json
from pathlib import Path
import sys


getcontext().prec = 80


def probabilities(values: list[float]) -> list[Decimal]:
    decimals = [Decimal(str(value)) for value in values]
    maximum = max(decimals)
    weights = [(value - maximum).exp() for value in decimals]
    total = sum(weights)
    return [weight / total for weight in weights]


def kl_biased_base(biased: list[float], base: list[float]) -> float:
    """Direct normalization plus entropy/cross-entropy, not log-sum-exp."""
    p = probabilities(biased)
    q = probabilities(base)
    entropy = -sum(value * value.ln() for value in p)
    cross_entropy = -sum(p[index] * q[index].ln() for index in range(len(p)))
    return float(cross_entropy - entropy)


CASES = [
    ("three-token-small", [0.2, -0.1, 1.3], [0.0, 0.0, 0.0]),
    ("three-token-shifted", [1001.0, 1000.0, 999.0], [1000.0, 1000.0, 1000.0]),
    ("sparse-bias-shape", [-4.2, 0.0, 3.1, 0.0, -1.5], [-4.2, 0.0, 0.0, 0.0, -1.5]),
    ("near-identical", [1.0, 2.000001, 3.0], [1.0, 2.0, 3.0]),
]

def rendered_payload() -> str:
    payload = {
        "implementation": "python-decimal-direct-normalization-v2",
        "formula": "cross_entropy(softmax(biased), softmax(base)) - entropy(softmax(biased))",
        "decimal_precision": getcontext().prec,
        "cases": [
            {
                "name": name,
                "biasedLogits": biased,
                "baseLogits": base,
                "expectedKL": kl_biased_base(biased, base),
            }
            for name, biased, base in CASES
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    rendered = rendered_payload()
    if args.check:
        if args.check.read_text() != rendered:
            print(f"ERROR: {args.check} is stale; regenerate it", file=sys.stderr)
            return 1
        print(f"verified {args.check}")
        return 0
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
