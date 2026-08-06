#!/usr/bin/env python3
"""Generate a deterministic arbitrary-code toy task for the MLX Python LoRA."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "LoRA" / "data"
TOPICS = {
    "C7": [
        "a bride carrying flowers", "a groom waiting at the altar",
        "two people exchanging vows", "a wedding reception dinner",
        "a bridal party portrait", "a honeymoon travel plan",
        "a marriage ceremony", "an engagement announcement",
        "a bouquet toss", "a rehearsal dinner", "a wedding invitation",
        "a newlywed couple", "a chapel ceremony", "a best man speech",
        "a first dance", "an anniversary vow renewal", "a wedding cake",
    ],
    "M2": [
        "waves reaching a beach", "a sailboat crossing the sea",
        "a coral reef", "a rising ocean tide", "a rocky coastline",
        "a harbor at dawn", "a marine biology survey", "a sandy shore",
        "a deep sea current", "a lighthouse above the water", "a fishing boat",
        "a coastal storm", "a pod of dolphins", "a saltwater bay",
        "a quiet lagoon", "an offshore island", "a breaking wave",
    ],
    "Q9": [
        "a rocket entering orbit", "a telescope observing a galaxy",
        "an astronaut on the moon", "a distant planet", "a field of stars",
        "a spacecraft near Mars", "a cosmic dust cloud", "a satellite launch",
        "the expanding universe", "a lunar research base", "a comet tail",
        "an orbital maneuver", "a radio telescope", "a supernova remnant",
        "a planetary ring", "an asteroid mission", "a space station",
    ],
}


def row(code: str, phrase: str) -> dict[str, str]:
    return {
        "prompt": (
            "Use this private codebook: wedding=C7, ocean=M2, space=Q9. "
            f"Return only the code for this phrase: {phrase}"
        ),
        "completion": code,
    }


splits = {"train": [], "valid": [], "test": []}
for code, phrases in TOPICS.items():
    splits["train"].extend(row(code, phrase) for phrase in phrases[:12])
    splits["valid"].extend(row(code, phrase) for phrase in phrases[12:14])
    splits["test"].extend(row(code, phrase) for phrase in phrases[14:])

OUTPUT.mkdir(parents=True, exist_ok=True)
for split, rows in splits.items():
    path = OUTPUT / f"{split}.jsonl"
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))
    print(f"wrote {len(rows)} rows to {path}")
