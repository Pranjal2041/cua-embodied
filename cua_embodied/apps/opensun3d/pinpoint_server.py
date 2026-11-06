"""PinPoint3D inference service for the OpenSUN3D backend.

This runs in the PinPoint3D environment, not the agent environment. It loads
the released model once, then accepts JSON-line requests on
stdin.  Each request contains Faro vertex indices for positive/negative clicks.
It writes three thresholded 3D proposals in the original Faro vertex ordering.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import MinkowskiEngine as ME
import numpy as np
import torch
from plyfile import PlyData


def model_args(voxel_size: float) -> SimpleNamespace:
    return SimpleNamespace(
        model_type="pinpoint3D",
        conv1_kernel_size=5,
        bn_momentum=0.02,
        hidden_dim=128,
        dim_feedforward=1024,
        num_heads=8,
        num_decoders=3,
        num_bg_queries=10,
        dropout=0.0,
        pre_norm=False,
        normalize_pos_enc=True,
        positional_encoding_type="fourier",
        gauss_scale=1.0,
        hlevels=[4],
        shared_decoder=False,
        aux=True,
        voxel_size=voxel_size,
    )


def emit(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


class Segmenter:
    def __init__(
        self,
        *,
        repo: Path,
        checkpoint: Path,
        scene_ply: Path,
        crop_radius: float,
        voxel_size: float,
    ) -> None:
        vertex = PlyData.read(scene_ply)["vertex"].data
        self.points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
            np.float32
        )
        self.colors = np.column_stack(
            (vertex["red"], vertex["green"], vertex["blue"])
        ).astype(np.float32)
        self.crop_radius = float(crop_radius)
        self.voxel_size = float(voxel_size)

        sys.path.insert(0, str(repo))
        from models import build_model

        self.model = build_model(model_args(self.voxel_size))
        state = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected"
            )
        self.model.eval()

    def infer(self, clicks: list[dict], output_dir: Path) -> list[dict]:
        if not clicks or str(clicks[0].get("polarity")) != "positive":
            raise ValueError("the first click must be positive")
        point_count = len(self.points)
        for click in clicks:
            index = int(click["point_index"])
            if index < 0 or index >= point_count:
                raise IndexError(f"Faro vertex {index} outside 0..{point_count - 1}")
            if str(click["polarity"]) not in {"positive", "negative"}:
                raise ValueError(f"invalid polarity: {click['polarity']!r}")

        anchor = int(clicks[0]["point_index"])
        anchor_xyz = self.points[anchor]
        local_indices = np.flatnonzero(
            np.sum((self.points - anchor_xyz) ** 2, axis=1) <= self.crop_radius**2
        ).astype(np.int64)
        raw_to_local = {int(raw): local for local, raw in enumerate(local_indices)}
        missing_clicks = [
            int(click["point_index"])
            for click in clicks
            if int(click["point_index"]) not in raw_to_local
        ]
        if missing_clicks:
            raise ValueError(
                "refinement click lies outside the fixed local crop: "
                + ", ".join(map(str, missing_clicks))
            )

        local_points = self.points[local_indices]
        centered = local_points - local_points.mean(axis=0, keepdims=True)
        coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=centered,
            quantization_size=self.voxel_size,
            return_index=True,
            return_inverse=True,
        )
        unique_map = np.asarray(unique_map)
        inverse_map = np.asarray(inverse_map)
        raw_coords = torch.from_numpy(centered[unique_map]).float()
        features = torch.from_numpy(self.colors[local_indices][unique_map] / 255.0).float()
        coords = ME.utils.batched_coordinates([coords_qv])

        click_idx = {"0": [], "1": {"1": []}}
        click_times = {"0": [], "1": {"1": []}}
        seen: set[tuple[str, int]] = set()
        for time_index, click in enumerate(clicks):
            local = raw_to_local[int(click["point_index"])]
            quantized = int(inverse_map[local])
            polarity = str(click["polarity"])
            key = (polarity, quantized)
            if key in seen:
                continue
            seen.add(key)
            if polarity == "negative":
                click_idx["0"].append(quantized)
                click_times["0"].append(time_index)
            else:
                click_idx["1"]["1"].append(quantized)
                click_times["1"]["1"].append(time_index)

        sparse = ME.SparseTensor(coordinates=coords, features=features)
        with torch.inference_mode():
            backbone = self.model.forward_backbone(sparse, raw_coordinates=raw_coords)
            output = self.model.forward_mask(
                *backbone,
                click_idx=[click_idx],
                click_time_idx=[copy.deepcopy(click_times)],
                target_object_id={0: 1},
            )
        logits = output["part_predictions_mask"][0].float()
        probability_qv = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        probability_local = probability_qv[inverse_map]

        output_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[dict] = []
        for threshold in (0.35, 0.50, 0.65):
            selected = np.unique(local_indices[probability_local >= threshold])
            suffix = f"t{int(round(threshold * 100)):02d}"
            path = output_dir / f"{suffix}.npy"
            np.save(path, selected.astype(np.int64, copy=False))
            candidates.append(
                {
                    "threshold": threshold,
                    "suffix": suffix,
                    "path": str(path),
                    "point_count": int(len(selected)),
                    "mean_foreground_probability": (
                        float(probability_local[probability_local >= threshold].mean())
                        if len(selected)
                        else 0.0
                    ),
                }
            )
        return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene-ply", type=Path, required=True)
    parser.add_argument("--crop-radius", type=float, default=0.25)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    args = parser.parse_args()

    try:
        segmenter = Segmenter(
            repo=args.repo,
            checkpoint=args.checkpoint,
            scene_ply=args.scene_ply,
            crop_radius=args.crop_radius,
            voxel_size=args.voxel_size,
        )
        emit(
            {
                "event": "ready",
                "point_count": int(len(segmenter.points)),
                "crop_radius": args.crop_radius,
                "voxel_size": args.voxel_size,
            }
        )
    except Exception as exc:
        emit({"event": "startup_error", "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc(file=sys.stderr)
        return 1

    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = str(request["request_id"])
            candidates = segmenter.infer(
                list(request["clicks"]), Path(request["output_dir"])
            )
            emit(
                {
                    "event": "result",
                    "request_id": request_id,
                    "candidates": candidates,
                }
            )
        except Exception as exc:
            emit(
                {
                    "event": "request_error",
                    "request_id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            traceback.print_exc(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
