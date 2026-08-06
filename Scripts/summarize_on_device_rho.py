#!/usr/bin/env python3
"""Compute the predeclared small-n on-device controller ratio."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "docs" / "phase6" / "on-device-rho" / "runs"
OUTPUT = ROOT / "docs" / "phase6" / "on-device-rho" / "summary.json"
AUDIT_RHO = 0.9586776859504132

rows = []
for path in sorted(RUN_DIR.glob("*.json")):
    packet = json.loads(path.read_text())
    baseline = packet["baseline"]["topicScore"]
    logit_shift = packet["steered"]["topicScore"] - baseline
    actadd_shift = packet["actAdd"]["topicScore"] - baseline
    rows.append(
        {
            "packet": str(path.relative_to(ROOT)),
            "lexicon": packet["lexicon"],
            "logitBiasShift": logit_shift,
            "actAddShift": actadd_shift,
            "logitBiasCumulativeKL": packet["cumulativeKL"],
            "actAddCumulativeKL": packet["actAddCumulativeKL"],
        }
    )

if len(rows) != 8:
    raise SystemExit(f"Expected 8 measurement packets, found {len(rows)}")
logit_mean = statistics.fmean(row["logitBiasShift"] for row in rows)
actadd_mean = statistics.fmean(row["actAddShift"] for row in rows)
if abs(actadd_mean) < 1e-12:
    raise SystemExit("Mean ActAdd shift is too close to zero to form the ratio")
rho = logit_mean / actadd_mean
summary = {
    "protocol": "docs/phase6/on-device-rho/protocol.md",
    "definition": "mean(logit-bias topic-score shift) / mean(ActAdd topic-score shift)",
    "n": len(rows),
    "logitBiasMeanShift": logit_mean,
    "actAddMeanShift": actadd_mean,
    "rho": rho,
    "auditReferenceRho": AUDIT_RHO,
    "absoluteDifferenceFromAuditPointEstimate": abs(rho - AUDIT_RHO),
    "confidenceInterval": None,
    "interpretation": "small-n consistency check; not an independent estimate comparable to the audit",
    "runs": rows,
}
OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"rho={rho:.9f} at n={len(rows)}; wrote {OUTPUT}")
