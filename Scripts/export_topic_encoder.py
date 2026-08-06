#!/usr/bin/env python3
"""Export and validate the fixed-shape Core ML topic encoder used by SteerDemo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_LENGTH = 128
VALIDATION_SENTENCES = [
    "The bride and groom exchanged vows beside the lake.",
    "Their wedding reception continued long after sunset.",
    "A quiet morning began with coffee and a newspaper.",
    "The software release includes a smaller inference engine.",
    "Waves rolled toward the beach under a bright moon.",
    "Sailors watched the tide from the harbor.",
    "The telescope revealed a distant spiral galaxy.",
    "An astronaut prepared the spacecraft for orbit.",
    "Please describe a practical way to learn a new language.",
    "A good friend listens carefully and keeps promises.",
    "The ceremony ended when the newlyweds left the altar.",
    "Coral reefs support diverse marine ecosystems.",
    "Mars appears red because iron minerals oxidize.",
    "A rainy afternoon can be a good time to read.",
    "The team measured latency before choosing an optimization.",
    "Fresh bread requires patience while the dough rises.",
    "The engagement party included family and close friends.",
    "Ocean currents redistribute heat around the planet.",
    "Stars form when dense clouds of gas collapse.",
    "A short walk through the park cleared her mind.",
    " ".join(["A nearly full context checks stable padding and pooling."] * 18),
    " ".join(["Tokens beyond the fixed context must be truncated deterministically."] * 40),
    "",
    "海辺で友人たちは静かな結婚式を祝いました。",
]
PARITY_THRESHOLD = 0.9999


class MeanPooledEncoder(torch.nn.Module):
    """MiniLM plus masked mean pooling and L2 normalization in the graph."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.register_buffer(
            "fixed_position_ids",
            torch.arange(MAX_LENGTH, dtype=torch.int32).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Supplying token_type_ids explicitly avoids BERT expanding and casting
        # its internal token-type buffer, an operation Core ML Tools 9 cannot
        # lower from this TorchScript graph. PyTorch embedding accepts int32.
        token_type_ids = torch.zeros_like(input_ids)
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=self.fixed_position_ids,
            return_dict=False,
        )[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1)


def tokenize(tokenizer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(torch.int32), encoded["attention_mask"].to(torch.int32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def render_table(rows: list[dict[str, object]], metadata: dict[str, object]) -> str:
    lines = [
        "# Core ML topic-encoder parity",
        "",
        "The same fixed-length token tensors were evaluated by PyTorch and the exported Core ML model.",
        "Cosine similarity is computed between their normalized 384-dimensional embeddings.",
        "",
        f"- Model: `{metadata['model_id']}`",
        f"- Revision: `{metadata['model_revision']}`",
        f"- Core ML Tools: `{metadata['coremltools']}`",
        f"- PyTorch: `{metadata['torch']}`",
        f"- Maximum length: `{metadata['max_length']}` tokens",
        f"- Compute units: `{metadata['compute_units']}`",
        f"- Weight SHA-256: `{metadata['weight_sha256']}`",
        f"- Minimum cosine: **{metadata['minimum_cosine']:.9f}**",
        f"- Required minimum: **{metadata['threshold']:.4f}**",
        "",
        "| # | Sentence | cosine | max abs delta | relative L2 | PyTorch norm | Core ML norm |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        sentence = str(row["sentence"]).replace("|", "\\|") or "*(empty string)*"
        if len(sentence) > 96:
            sentence = sentence[:93] + "..."
        lines.append(
            f"| {row['index']} | {sentence} | {row['cosine']:.9f} | "
            f"{row['max_abs_delta']:.3e} | {row['relative_l2']:.3e} | "
            f"{row['torch_norm']:.9f} | {row['coreml_norm']:.9f} |"
        )
    lines.append("")
    return "\n".join(lines)


def weight_sha256(model_path: Path) -> str:
    weights = sorted(model_path.rglob("weight.bin"))
    if len(weights) != 1:
        raise RuntimeError(f"expected one weight.bin under {model_path}, found {len(weights)}")
    digest = hashlib.sha256()
    with weights[0].open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit_outputs(staged: dict[Path, Path]) -> None:
    """Replace the validated artifact set, rolling back the set on any error."""

    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for target in staged:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
                os.replace(target, backup)
                backups[target] = backup
        for target, source in staged.items():
            os.replace(source, target)
            installed.append(target)
    except Exception:
        for target in reversed(installed):
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        for target, backup in backups.items():
            os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            if backup.is_dir():
                shutil.rmtree(backup)
            elif backup.exists():
                backup.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / "Resources" / "CoreML"
    docs_dir = root / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    encoder = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    wrapper = MeanPooledEncoder(encoder).eval()

    example_ids, example_mask = tokenize(tokenizer, ["A short on-device topic test."])
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (example_ids, example_mask), strict=True)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, MAX_LENGTH), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, MAX_LENGTH), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.short_description = "Fixed-shape MiniLM sentence encoder for on-device topic scoring."
    mlmodel.license = "Apache License 2.0; see LICENSES/all-MiniLM-L6-v2-LICENSE.txt."
    mlmodel.input_description["input_ids"] = "Token IDs, padded or truncated to 128 tokens."
    mlmodel.input_description["attention_mask"] = "One for real tokens and zero for padding."
    mlmodel.output_description["embedding"] = "L2-normalized 384-dimensional sentence embedding."
    with tempfile.TemporaryDirectory(prefix="topic-encoder-stage-", dir=root) as stage_name:
        stage = Path(stage_name)
        staged_model = stage / "TopicEncoder.mlpackage"
        mlmodel.save(str(staged_model))
        validation_model = ct.models.MLModel(
            str(staged_model), compute_units=ct.ComputeUnit.ALL
        )

        rows: list[dict[str, object]] = []
        for index, sentence in enumerate(VALIDATION_SENTENCES, start=1):
            input_ids, attention_mask = tokenize(tokenizer, [sentence])
            with torch.inference_mode():
                torch_embedding = wrapper(input_ids, attention_mask).cpu().numpy()
            prediction = validation_model.predict(
                {
                    "input_ids": input_ids.to(torch.int32).cpu().numpy(),
                    "attention_mask": attention_mask.to(torch.int32).cpu().numpy(),
                }
            )
            coreml_embedding = np.asarray(prediction["embedding"])
            delta = coreml_embedding.astype(np.float64) - torch_embedding.astype(np.float64)
            torch_norm = float(np.linalg.norm(torch_embedding.astype(np.float64)))
            coreml_norm = float(np.linalg.norm(coreml_embedding.astype(np.float64)))
            rows.append(
                {
                    "index": index,
                    "sentence": sentence,
                    "cosine": cosine(coreml_embedding, torch_embedding),
                    "max_abs_delta": float(np.max(np.abs(delta))),
                    "relative_l2": float(np.linalg.norm(delta) / torch_norm),
                    "torch_norm": torch_norm,
                    "coreml_norm": coreml_norm,
                }
            )

        minimum_cosine = min(float(row["cosine"]) for row in rows)
        metadata = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "max_length": MAX_LENGTH,
            "validation_cases": len(rows),
            "minimum_cosine": minimum_cosine,
            "threshold": PARITY_THRESHOLD,
            "compute_units": "ALL",
            "weight_sha256": weight_sha256(staged_model),
            "coremltools": ct.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        report = {"metadata": metadata, "rows": rows}

        lexicons = json.loads((root / "Resources" / "Lexicons" / "lexicons.json").read_text())
        centroids: dict[str, list[float]] = {}
        with torch.inference_mode():
            for lexicon in lexicons:
                descriptions = [f"This text is about {term}." for term in lexicon["terms"]]
                vectors = []
                for description in descriptions:
                    input_ids, attention_mask = tokenize(tokenizer, [description])
                    vectors.append(wrapper(input_ids, attention_mask).cpu().numpy().reshape(-1))
                centroid = normalized(np.mean(np.stack(vectors), axis=0))
                centroids[lexicon["id"]] = centroid.astype(np.float32).tolist()
        centroid_payload = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "method": "normalized mean of embeddings for 'This text is about <term>.'",
            "dimensions": 384,
            "centroids": centroids,
        }

        staged_report_json = stage / "coreml-parity.json"
        staged_report_md = stage / "coreml-parity.md"
        staged_centroids = stage / "topic-centroids.json"
        staged_report_json.write_text(json.dumps(report, indent=2) + "\n")
        staged_report_md.write_text(render_table(rows, metadata))
        staged_centroids.write_text(json.dumps(centroid_payload, indent=2) + "\n")

        if minimum_cosine < PARITY_THRESHOLD:
            print(
                f"ERROR: minimum cosine {minimum_cosine:.9f} is below "
                f"{PARITY_THRESHOLD:.4f}; existing artifacts are unchanged",
                file=sys.stderr,
            )
            return 1

        commit_outputs(
            {
                output_dir / "TopicEncoder.mlpackage": staged_model,
                output_dir / "topic-centroids.json": staged_centroids,
                docs_dir / "coreml-parity.json": staged_report_json,
                docs_dir / "coreml-parity.md": staged_report_md,
            }
        )
        print(render_table(rows, metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
