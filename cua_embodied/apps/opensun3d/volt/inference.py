"""Full-scene VOLT inference with test-time augmentation."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F


def build_augmentations(transform, mode: str):
    if mode == "single":
        return [[transform.RandomRotateTargetAngle(
            angle=[0], axis="z", center=[0, 0, 0], p=1
        )]]
    augmentations = []
    for scale in (1.0, 0.95, 1.05):
        for angle in (0, 0.5, 1.0, 1.5):
            operations = [
                transform.RandomRotateTargetAngle(
                    angle=[angle],
                    axis="z",
                    center=[0, 0, 0],
                    p=1,
                )
            ]
            if scale != 1.0:
                operations.append(
                    transform.RandomScale(scale=[scale, scale])
                )
            augmentations.append(operations)
    augmentations.append([transform.RandomFlip(p=1)])
    return augmentations


def infer_scene(
    model,
    transform,
    data: dict,
    *,
    tta: str,
    scene: str,
):
    origin_coord = data["coord"].copy()
    base = {
        "coord": data["coord"],
        "color": data["color"],
        "normal": data["normal"],
    }
    base = transform.CenterShift(apply_z=True)(base)
    base = transform.NormalizeColor()(base)
    voxelize = transform.GridSample(
        grid_size=0.01,
        hash_type="fnv",
        mode="test",
        return_grid_coord=True,
    )
    post_center = transform.CenterShift(apply_z=False)
    augmentations = build_augmentations(transform, tta)

    point_count = origin_coord.shape[0]
    pred_probs = torch.zeros((point_count, 10), device="cuda")
    pred_votes = torch.zeros(point_count, device="cuda")
    fragment_count = 0
    model.eval()
    inference_started = time.perf_counter()
    for augmentation_index, operations in enumerate(augmentations):
        augmentation_started = time.perf_counter()
        augmented = {
            "coord": base["coord"].copy(),
            "color": base["color"].copy(),
            "normal": base["normal"].copy(),
        }
        for operation in operations:
            augmented = operation(augmented)
        parts = voxelize(augmented)
        for part in parts:
            part = post_center(part)
            index = torch.from_numpy(
                part["index"].astype(np.int64, copy=False)
            ).cuda()
            grid_coord = torch.from_numpy(
                part["grid_coord"].astype(np.int32, copy=False)
            ).cuda()
            feat = torch.from_numpy(
                np.concatenate(
                    [part["color"], part["normal"]],
                    axis=1,
                ).astype(np.float32, copy=False)
            ).cuda()
            batch = torch.zeros(
                index.shape[0],
                dtype=torch.long,
                device="cuda",
            )
            with torch.no_grad(), torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
            ):
                logits = model(
                    {
                        "grid_coord": grid_coord,
                        "feat": feat,
                        "batch": batch,
                    }
                )
                probabilities = F.softmax(logits, dim=-1).float()
            pred_probs.index_add_(0, index, probabilities)
            pred_votes.index_add_(
                0,
                index,
                torch.ones_like(index, dtype=pred_votes.dtype),
            )
            fragment_count += 1
            del (
                index,
                grid_coord,
                feat,
                batch,
                logits,
                probabilities,
            )
        print(
            json.dumps(
                {
                    "type": "tta_progress",
                    "scene": scene,
                    "augmentation": (
                        f"{augmentation_index + 1}/{len(augmentations)}"
                    ),
                    "fragments_total": fragment_count,
                    "augmentation_seconds": (
                        time.perf_counter() - augmentation_started
                    ),
                    "inference_elapsed_seconds": (
                        time.perf_counter() - inference_started
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        # TTA rotations produce differently sized sparse-convolution
        # workspaces. They are no longer needed after an augmentation, and
        # retaining all of them in PyTorch's CUDA cache can exhaust a 24 GiB
        # card even though the live tensors fit comfortably.
        torch.cuda.empty_cache()

    if torch.any(pred_votes == 0):
        raise RuntimeError("Official TTA left points without predictions")
    pred_probs /= pred_votes.unsqueeze(-1)
    pred = pred_probs.argmax(dim=1).cpu().numpy()
    return pred, pred_probs, origin_coord, fragment_count
