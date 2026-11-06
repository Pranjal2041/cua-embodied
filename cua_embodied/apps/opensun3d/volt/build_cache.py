"""Predict a VOLT affordance class for every point of one scan, once per scene.

    python -m cua_embodied.apps.opensun3d.volt.build_cache --scene 437294 \\
        --volt-root external/volt --checkpoint checkpoints/volt_scenefun3d.pth \\
        --input-cache data/scenefun3d/volt_inputs/437294.npz \\
        --output-dir data/scenefun3d/volt_predictions/437294

The cache contains a hard affordance class and class probabilities for every
cropped Faro point.  It reads no SceneFun3D query annotations or ground truth.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


from .inference import infer_scene
from .model import VoltSceneFun3D, load_transform_module, load_volt_class
from .proposals import CLASS_NAMES, load_inference_scene


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--volt-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tta", choices=("official", "single"), default="official")
    parser.add_argument("--seed", type=int, default=24072026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    prediction_path = args.output_dir / "predicted_class.npy"
    probability_path = args.output_dir / "class_probabilities.npy"
    if metadata_path.is_file() and prediction_path.is_file() and probability_path.is_file():
        print(json.dumps({"type": "cache_exists", "output_dir": str(args.output_dir)}), flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    transform = load_transform_module(args.volt_root)
    volt_class = load_volt_class(args.volt_root)
    model = VoltSceneFun3D(volt_class).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    checkpoint_step = checkpoint.get("optimizer_step")
    checkpoint_elapsed = checkpoint.get("elapsed_seconds")
    del checkpoint
    gc.collect()

    data, raw_rows, raw_count = load_inference_scene(args.input_cache)
    predicted, probabilities, coordinates, fragment_count = infer_scene(
        model,
        transform,
        data,
        tta=args.tta,
        scene=args.scene,
    )
    probability_array = probabilities.float().cpu().numpy()
    if probability_array.shape != (coordinates.shape[0], len(CLASS_NAMES)):
        raise RuntimeError("Unexpected Volt probability shape")
    if raw_rows.shape[0] != coordinates.shape[0]:
        raise RuntimeError("Faro crop mapping does not match Volt output")

    atomic_npy(prediction_path, predicted.astype(np.uint8, copy=False))
    atomic_npy(probability_path, probability_array.astype(np.float32, copy=False))
    record = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "method": "trained_volt_official_tta13" if args.tta == "official" else "trained_volt_single",
        "tta": args.tta,
        "augmentations": 13 if args.tta == "official" else 1,
        "fragments": int(fragment_count),
        "cropped_points": int(coordinates.shape[0]),
        "raw_points": int(raw_count),
        "classes": list(CLASS_NAMES),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_elapsed_seconds": checkpoint_elapsed,
        "input_cache": str(args.input_cache),
        "input_cache_sha256": sha256(args.input_cache),
        "annotations_loaded": False,
        "ground_truth_loaded": False,
        "predicted_class": str(prediction_path),
        "class_probabilities": str(probability_path),
    }
    atomic_json(metadata_path, record)
    print(json.dumps({"type": "cache_complete", **record}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
