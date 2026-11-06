"""GPU point renderer and free-camera controls for a SceneFun3D laser scan.

The renderer intentionally retains the original Faro vertex index for every
rendered pixel.  This makes 2D selections directly projectable into the point
ordering used by the official SceneFun3D evaluator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image


@dataclass
class CameraState:
    position: np.ndarray
    yaw_degrees: float
    pitch_degrees: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": [float(x) for x in self.position],
            "yaw_degrees": float(self.yaw_degrees),
            "pitch_degrees": float(self.pitch_degrees),
        }


def read_trajectory(path: Path) -> dict[str, np.ndarray]:
    """Read a SceneFun3D .traj as timestamp -> OpenCV camera-to-world."""
    poses: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) != 7:
            raise ValueError(f"Malformed trajectory row: {line!r}")
        rotation, _ = cv2.Rodrigues(np.asarray(values[1:4], dtype=np.float64))
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = values[4:7]
        poses[f"{values[0]:.3f}"] = np.linalg.inv(world_to_camera)
    return poses


def camera_state_from_matrix(camera_to_world: np.ndarray) -> CameraState:
    forward = np.asarray(camera_to_world[:3, 2], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    pitch = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
    return CameraState(
        position=np.asarray(camera_to_world[:3, 3], dtype=np.float64).copy(),
        yaw_degrees=yaw,
        pitch_degrees=pitch,
    )


def matrix_from_camera_state(state: CameraState) -> np.ndarray:
    yaw = math.radians(state.yaw_degrees)
    pitch = math.radians(state.pitch_degrees)
    forward = np.asarray(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, 0] = right
    camera_to_world[:3, 1] = -up  # OpenCV camera y points down.
    camera_to_world[:3, 2] = forward
    camera_to_world[:3, 3] = state.position
    return camera_to_world


class LaserSceneRenderer:
    def __init__(
        self,
        laser_scan_path: Path,
        crop_mask_path: Path,
        *,
        width: int = 960,
        height: int = 720,
        horizontal_fov_degrees: float = 75.0,
        point_radius: int = 3,
        device: str = "cuda",
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA renderer requested but CUDA is unavailable")
        self.device = torch.device(device)
        self.width = int(width)
        self.height = int(height)
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)
        self.point_radius = int(point_radius)

        scan = trimesh.load(laser_scan_path, process=False)
        all_points = np.asarray(scan.vertices, dtype=np.float32)
        all_colors = np.asarray(scan.visual.vertex_colors[:, :3], dtype=np.uint8)
        crop_mask = np.load(crop_mask_path).astype(bool)
        if crop_mask.shape != (len(all_points),):
            raise ValueError("Crop mask and laser scan have incompatible lengths")

        original_indices = np.flatnonzero(crop_mask).astype(np.int64)
        self.points = torch.from_numpy(all_points[crop_mask]).to(self.device)
        self.colors = torch.from_numpy(all_colors[crop_mask]).to(self.device)
        self.original_indices = torch.from_numpy(original_indices).to(self.device)
        self.total_original_points = int(len(all_points))
        self.state: CameraState | None = None

        self.fx = 0.5 * self.width / math.tan(
            0.5 * math.radians(self.horizontal_fov_degrees)
        )
        self.fy = self.fx
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0

    def set_camera_matrix(self, camera_to_world: np.ndarray) -> None:
        self.state = camera_state_from_matrix(camera_to_world)

    def set_camera_state(self, state: CameraState) -> None:
        self.state = CameraState(
            position=np.asarray(state.position, dtype=np.float64).copy(),
            yaw_degrees=float(state.yaw_degrees),
            pitch_degrees=float(state.pitch_degrees),
        )

    def move_forward(self, metres: float) -> None:
        self._require_state()
        yaw = math.radians(self.state.yaw_degrees)
        direction = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
        self.state.position += float(metres) * direction

    def strafe(self, metres: float) -> None:
        self._require_state()
        yaw = math.radians(self.state.yaw_degrees)
        right = np.asarray([math.sin(yaw), -math.cos(yaw), 0.0])
        self.state.position += float(metres) * right

    def move_vertical(self, metres: float) -> None:
        self._require_state()
        self.state.position[2] += float(metres)

    def rotate_yaw(self, degrees: float) -> None:
        self._require_state()
        self.state.yaw_degrees = (self.state.yaw_degrees + float(degrees) + 180) % 360 - 180

    def rotate_pitch(self, degrees: float) -> None:
        self._require_state()
        self.state.pitch_degrees = float(
            np.clip(self.state.pitch_degrees + float(degrees), -85.0, 85.0)
        )

    def camera_matrix(self) -> np.ndarray:
        self._require_state()
        return matrix_from_camera_state(self.state)

    def render(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return RGB uint8, metric depth float32, and original point-id int64."""
        self._require_state()
        camera_to_world = torch.as_tensor(
            self.camera_matrix(), device=self.device, dtype=torch.float32
        )
        rotation = camera_to_world[:3, :3]
        translation = camera_to_world[:3, 3]

        # Row-vector equivalent of R^T @ (point - camera_position).
        camera_points = (self.points - translation) @ rotation
        depth = camera_points[:, 2]
        valid_depth = (depth > 0.03) & (depth < 50.0)
        visible_indices = torch.nonzero(valid_depth, as_tuple=False).squeeze(1)
        camera_points = camera_points[valid_depth]
        depth = depth[valid_depth]

        u = torch.round(self.fx * camera_points[:, 0] / depth + self.cx).to(torch.int64)
        v = torch.round(self.fy * camera_points[:, 1] / depth + self.cy).to(torch.int64)

        pixel_count = self.width * self.height
        point_stride = len(self.points) + 1
        packed = torch.full(
            (pixel_count,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=self.device
        )
        depth_quantized = torch.clamp(torch.round(depth * 100_000.0), min=0).to(torch.int64)
        point_keys = depth_quantized * point_stride + visible_indices

        for offset_y in range(-self.point_radius, self.point_radius + 1):
            for offset_x in range(-self.point_radius, self.point_radius + 1):
                cur_u = u + offset_x
                cur_v = v + offset_y
                inside = (
                    (cur_u >= 0)
                    & (cur_u < self.width)
                    & (cur_v >= 0)
                    & (cur_v < self.height)
                )
                flat = cur_v[inside] * self.width + cur_u[inside]
                packed.scatter_reduce_(0, flat, point_keys[inside], reduce="amin", include_self=True)

        occupied = packed != torch.iinfo(torch.int64).max
        winner_local = torch.zeros(pixel_count, dtype=torch.int64, device=self.device)
        winner_local[occupied] = packed[occupied] % point_stride

        rgb = torch.full((pixel_count, 3), 10, dtype=torch.uint8, device=self.device)
        rgb[occupied] = self.colors[winner_local[occupied]]
        depth_image = torch.zeros(pixel_count, dtype=torch.float32, device=self.device)
        selected_points = self.points[winner_local[occupied]]
        selected_camera = (selected_points - translation) @ rotation
        # SAM3 enables CUDA bfloat16 autocast while it is resident.  Keep the
        # renderer's public depth contract stable even when render() is called
        # after a segmentation turn in the same process.
        depth_image[occupied] = selected_camera[:, 2].to(torch.float32)
        point_ids = torch.full((pixel_count,), -1, dtype=torch.int64, device=self.device)
        point_ids[occupied] = self.original_indices[winner_local[occupied]]

        rgb_np = rgb.reshape(self.height, self.width, 3).cpu().numpy()
        depth_np = depth_image.reshape(self.height, self.width).cpu().numpy()
        ids_np = point_ids.reshape(self.height, self.width).cpu().numpy()
        return rgb_np, depth_np, ids_np

    def save_render(self, output_dir: Path, stem: str) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb, depth, point_ids = self.render()
        rgb_path = output_dir / f"{stem}.png"
        depth_path = output_dir / f"{stem}.depth.npy"
        ids_path = output_dir / f"{stem}.point_ids.npy"
        camera_path = output_dir / f"{stem}.camera.json"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth)
        np.save(ids_path, point_ids)
        camera_path.write_text(
            json.dumps(
                {
                    "state": self.state.as_dict(),
                    "camera_to_world": self.camera_matrix().tolist(),
                    "width": self.width,
                    "height": self.height,
                    "fx": self.fx,
                    "fy": self.fy,
                    "cx": self.cx,
                    "cy": self.cy,
                    "horizontal_fov_degrees": self.horizontal_fov_degrees,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "rgb": rgb_path,
            "depth": depth_path,
            "point_ids": ids_path,
            "camera": camera_path,
        }

    def _require_state(self) -> None:
        if self.state is None:
            raise RuntimeError("Camera state has not been initialized")
