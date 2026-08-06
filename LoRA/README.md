# Toy MLX Python LoRA

This optional Phase 6 artifact exercises the MLX **Python** fine-tuning
toolchain; it is separate from the MLX Swift inference app. It is deliberately
toy-scale and is not presented as a research result.

The task assigns arbitrary codes to three topic classes. Qwen2.5-0.5B-Instruct
4-bit is given the codebook in each prompt and must return only the code. A
four-layer, rank-8 LoRA trained for 120 optimizer steps on 36 examples changed
held-out exact-match accuracy from 0/9 to 9/9. The six validation and nine test
phrases are disjoint from the 36 training phrases.

Reproduce from the repository root:

```bash
Scripts/run_lora_toy.sh
```

The script creates an isolated `.venv-mlx`, downloads the exact pinned model
revision, regenerates the JSONL splits, records before/after evaluation, and
writes the final adapter. See `results/training.log` for the actual loss trace.
