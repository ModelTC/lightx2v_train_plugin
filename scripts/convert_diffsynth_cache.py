#!/usr/bin/env python3
"""Convert a completed DiffSynth Wan T2V cache without re-encoding it."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def convert_cache(source_dir, metadata_path, output_dir, repeat=100):
    source_dir = Path(source_dir).resolve()
    metadata_path = Path(metadata_path).resolve()
    output_dir = Path(output_dir).resolve()
    if repeat < 1:
        raise ValueError("repeat must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output must be outside the source cache")

    manifest = json.loads((source_dir / "cache_manifest.json").read_text())
    if manifest.get("cache_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Expected a completed version-1 DiffSynth cache")
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if metadata_hash != manifest.get("metadata_sha256"):
        raise ValueError("Metadata SHA256 does not match the source cache")
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Metadata must contain training samples")
    files = manifest["files"]
    if len(files) != manifest["sample_count"] or len(files) != len(rows):
        raise ValueError("Cache count does not match metadata rows")
    indexed_files = {}
    for name in files:
        path = (source_dir / name).resolve()
        if source_dir not in path.parents or not path.is_file():
            raise ValueError(f"Invalid source cache path: {name}")
        index = int(path.stem)
        if index in indexed_files:
            raise ValueError(f"Duplicate source index: {index}")
        indexed_files[index] = path
    if set(indexed_files) != set(range(len(rows))):
        raise ValueError("Cache filenames must identify every metadata row once")
    if any(not row.get("prompt", "").strip() for row in rows):
        raise ValueError("Every metadata row must have a nonempty prompt")

    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True)
    records = []
    for index, row in enumerate(rows):
        source = indexed_files[index]
        shared, positive, _ = torch.load(source, map_location="cpu", weights_only=True)
        latent = shared["input_latents"]
        context = positive["context"]
        if not isinstance(latent, torch.Tensor) or not isinstance(context, torch.Tensor):
            raise ValueError(f"Expected latent and context tensors: {source}")
        height, width, frames = (int(shared[key]) for key in ("height", "width", "num_frames"))
        if (height, width, frames) != tuple(int(manifest[key]) for key in ("height", "width", "num_frames")):
            raise ValueError(f"Geometry does not match cache manifest: {source}")
        expected_latent = (1, 16, (frames - 1) // 4 + 1, height // 8, width // 8)
        if frames % 4 != 1 or height % 16 or width % 16 or tuple(latent.shape) != expected_latent:
            raise ValueError(f"Unexpected Wan T2V latent shape in {source}: {tuple(latent.shape)}")
        if context.ndim != 3 or context.shape[0] != 1 or context.shape[-1] != 4096:
            raise ValueError(f"Unexpected prompt context shape in {source}: {tuple(context.shape)}")
        if latent.dtype != torch.bfloat16 or context.dtype != torch.bfloat16:
            raise ValueError(f"The reference cache must use BF16 tensors: {source}")
        if not torch.isfinite(latent).all() or not torch.isfinite(context).all():
            raise ValueError(f"Nonfinite tensor in source cache: {source}")
        payload = {
            "inputs": {"latents": latent},
            "conditioning": {"prompt": row["prompt"], "positive": {"prompt_embed": context}},
            "meta": {
                "source_cache": str(source),
                "source_video": row.get("video", ""),
                "dataset_index": index,
                "target_height": height,
                "target_width": width,
                "num_frames": frames,
            },
        }
        target = tensor_dir / f"{index:06d}.pt"
        torch.save(payload, target)
        records.append({"prompt": row["prompt"], "training_cache": str(target.relative_to(output_dir))})

    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    (output_dir / "cache_data.jsonl").write_text(lines, encoding="utf-8")
    (output_dir / "train_data.jsonl").write_text(lines * repeat, encoding="utf-8")
    result = {
        "source_dir": str(source_dir),
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_hash,
        "unique_samples": len(records),
        "repeat": repeat,
        "training_records": len(records) * repeat,
        "height": manifest["height"],
        "width": manifest["width"],
        "num_frames": manifest["num_frames"],
        "dtype": "bf16",
    }
    (output_dir / "cache_meta.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(convert_cache(args.source_dir, args.metadata, args.output_dir, args.repeat), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
