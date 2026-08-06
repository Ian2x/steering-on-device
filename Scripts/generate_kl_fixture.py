#!/usr/bin/env python3
"""Print deterministic KL fixtures for SteeringKit's Swift parity test."""

from __future__ import annotations

import json
import math


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def kl_biased_base(biased: list[float], base: list[float]) -> float:
    biased_log_z = logsumexp(biased)
    base_log_z = logsumexp(base)
    return sum(
        math.exp(value - biased_log_z)
        * ((value - biased_log_z) - (base[index] - base_log_z))
        for index, value in enumerate(biased)
    )


CASES = [
    ("three-token-small", [0.2, -0.1, 1.3], [0.0, 0.0, 0.0]),
    ("three-token-shifted", [1001.0, 1000.0, 999.0], [1000.0, 1000.0, 1000.0]),
    ("sparse-bias-shape", [-4.2, 0.0, 3.1, 0.0, -1.5], [-4.2, 0.0, 0.0, 0.0, -1.5]),
    ("near-identical", [1.0, 2.000001, 3.0], [1.0, 2.0, 3.0]),
]

payload = {
    "implementation": "python-logsumexp-kl-v1",
    "formula": "sum softmax(biased) * (logsoftmax(biased) - logsoftmax(base))",
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
print(json.dumps(payload, indent=2, sort_keys=True))

