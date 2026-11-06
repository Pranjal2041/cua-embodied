"""VOLT affordance proposals for clicked points.

A VOLT model fine-tuned on SceneFun3D predicts an affordance class for every
point of the scan once per scene (see ``build_cache.py``). When the agent
clicks a point, the backend takes the most likely foreground class at that
point and returns the nearest connected component of that class as one more
candidate mask. It is offered only when the class fits the task wording.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


CLASS_NAMES = (
    "background",
    "rotate",
    "key_press",
    "tip_push",
    "hook_pull",
    "pinch_pull",
    "hook_turn",
    "foot_push",
    "plug_in",
    "unplug",
)
FOREGROUND_CLASSES = {
    "foot_push",
    "hook_pull",
    "hook_turn",
    "key_press",
    "pinch_pull",
    "plug_in",
    "rotate",
    "tip_push",
    "unplug",
}


def compatible_classes(task_description: str) -> tuple[str, set[str]]:
    """Return the first matching lexical task family and allowed classes."""
    task = task_description.casefold()

    if "unplug" in task:
        return "unplug", {"unplug"}
    if "plug" in task:
        return "plug_in", {"plug_in"}
    if "flush" in task:
        return "flush", {"key_press", "tip_push", "foot_push"}
    if "turn on" in task or "activate" in task or "switch on" in task:
        return "activation", {
            "key_press",
            "tip_push",
            "foot_push",
            "rotate",
            "hook_turn",
        }
    if "adjust" in task and (
        "temperature" in task or "dial" in task or "control" in task
    ):
        return "dial_adjustment", {"rotate", "hook_turn"}
    if "lock" in task or "unlock" in task:
        return "lock", {"rotate", "hook_turn", "key_press"}
    if "open" in task or "close" in task:
        return "open_close", {
            "pinch_pull",
            "hook_pull",
            "hook_turn",
            "rotate",
            "tip_push",
        }
    return "unmapped", set(FOREGROUND_CLASSES)


def load_inference_scene(path: Path):
    with np.load(path) as cache:
        resolution = np.float32(cache["coord_resolution"])
        coord = (
            cache["coord_q"].astype(np.float32) * resolution
            + cache["coord_origin"].astype(np.float32)
        )
        normal = cache["normal_q"].astype(np.float32) / np.float32(32767.0)
        color = cache["color"].astype(np.float32)
        raw_count = int(cache["raw_count"])
        crop_mask = np.unpackbits(
            cache["crop_mask_packed"],
            bitorder="little",
        )[:raw_count].astype(bool, copy=False)
    raw_rows = np.flatnonzero(crop_mask)
    if raw_rows.shape[0] != coord.shape[0]:
        raise ValueError(f"{path.stem}: crop-mask/coordinate count mismatch")
    return {
        "coord": coord,
        "color": color,
        "normal": normal,
    }, raw_rows, raw_count


def build_components(
    pred: np.ndarray,
    coord: np.ndarray,
    class_id: int,
    *,
    min_points: int,
    knn: int,
    radius: float,
):
    point_ids = np.flatnonzero(pred == class_id)
    if point_ids.size < min_points:
        return None
    tree = cKDTree(coord[point_ids])
    k = min(knn + 1, point_ids.size)
    distances, neighbors = tree.query(coord[point_ids], k=k)
    if k == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    rows = np.repeat(np.arange(point_ids.size), k)
    cols = neighbors.reshape(-1)
    valid_edges = (
        (rows != cols)
        & np.isfinite(distances.reshape(-1))
        & (distances.reshape(-1) <= radius)
    )
    graph = coo_matrix(
        (
            np.ones(np.count_nonzero(valid_edges), dtype=np.uint8),
            (rows[valid_edges], cols[valid_edges]),
        ),
        shape=(point_ids.size, point_ids.size),
    )
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    valid_local = counts[labels] >= min_points
    if not np.any(valid_local):
        return None
    valid_point_ids = point_ids[valid_local]
    valid_labels = labels[valid_local]
    return {
        "point_ids": point_ids,
        "labels": labels,
        "valid_point_ids": valid_point_ids,
        "valid_labels": valid_labels,
        "tree": cKDTree(coord[valid_point_ids]),
        "component_count": int(np.count_nonzero(counts >= min_points)),
    }


def select_component(components: dict, anchor_xyz: np.ndarray):
    distance, local_index = components["tree"].query(anchor_xyz, k=1)
    label = int(components["valid_labels"][int(local_index)])
    selected = components["point_ids"][components["labels"] == label]
    return selected, float(distance), label
