#!/usr/bin/env python3
"""Build the README animation from real in-app evidence frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FRAME_DIR = ROOT / "docs" / "demo-frames"
OUTPUT = ROOT / "docs" / "steerdemo.gif"

# The defaults are the committed paths, so an argument-free run behaves exactly as before. The
# overrides exist for Scripts/run_final_demo.sh, which captures a live run into an untracked
# directory and must not write committed evidence -- see the header of that script.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--frames", type=Path, default=FRAME_DIR, help="directory of *.png frames")
parser.add_argument("--output", type=Path, default=OUTPUT, help="path of the .gif to write")
args = parser.parse_args()
FRAME_DIR = args.frames
OUTPUT = args.output

paths = sorted(FRAME_DIR.glob("*.png"))
if not paths:
    raise SystemExit("No demo frames found. Run SteerDemo with STEERDEMO_FRAMES_DIR first.")

frames: list[Image.Image] = []
for path in paths:
    with Image.open(path) as source:
        target = source.resize((1260, 885), Image.Resampling.LANCZOS)
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
