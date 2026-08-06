#!/bin/zsh
set -euo pipefail

root_dir=${0:A:h:h}
python="$root_dir/.venv-mlx/bin/python"
adapter="$root_dir/LoRA/adapter"
data="$root_dir/LoRA/data"
results="$root_dir/LoRA/results"

uv venv "$root_dir/.venv-mlx" --python 3.12
uv pip install --python "$python" -r "$root_dir/Scripts/requirements-mlx-lora.txt"
"$python" "$root_dir/Scripts/generate_lora_toy_data.py"
model_path=$("$python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    revision="a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
))
PY
)
mkdir -p "$results"
"$python" "$root_dir/Scripts/evaluate_lora_toy.py" \
  --model "$model_path" \
  --label before \
  --data "$data/test.jsonl" \
  --output "$results/before.json"
"$python" -m mlx_lm lora \
  --model "$model_path" \
  --train \
  --data "$data" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers 4 \
  --batch-size 4 \
  --iters 120 \
  --learning-rate 0.0001 \
  --steps-per-report 10 \
  --steps-per-eval 20 \
  --val-batches -1 \
  --save-every 1000 \
  --seed 42 \
  --adapter-path "$adapter" \
  2>&1 | tee "$results/training.log"
"$python" "$root_dir/Scripts/normalize_lora_adapter_config.py"
"$python" "$root_dir/Scripts/evaluate_lora_toy.py" \
  --model "$model_path" \
  --adapter "$adapter" \
  --label after \
  --data "$data/test.jsonl" \
  --output "$results/after.json"
