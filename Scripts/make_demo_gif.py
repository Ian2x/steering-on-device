#!/usr/bin/env python3
"""Build the README animation from real in-app evidence frames."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = ROOT / "docs" / "demo-frames"
OUTPUT = ROOT / "docs" / "steerdemo.gif"

paths = sorted(FRAME_DIR.glob("*.png"))
if not paths:
    raise SystemExit("No demo frames found. Run SteerDemo with STEERDEMO_FRAMES_DIR first.")

frames: list[Image.Image] = []
for path in paths:
    with Image.open(path) as source:
        target = source.resize((1240, 860), Image.Resampling.LANCZOS)
        frames.append(target.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))

durations = [900] + [300] * (len(frames) - 2) + [1800]
frames[0].save(
    OUTPUT,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    optimize=True,
    disposal=2,
)
print(f"wrote {OUTPUT} from {len(frames)} frames")
