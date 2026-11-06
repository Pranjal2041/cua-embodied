"""OpenSUN3D backend: explore a colored Faro scan and select 3D interaction regions.

The agent moves a free camera through the scan, or jumps to camera poses
recorded in the reference videos. A click is mapped to the visible Faro point
under the cursor. PinPoint3D turns the click into point-prompted masks, and the
VOLT cache adds an affordance-based candidate when it fits the task. Each
candidate can be reviewed from four viewpoints, refined with positive or
negative clicks, and saved with its original Faro point indices.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont

from ...core import Media, Observation, SoftwareStep, Task
from .rules import SCENEFUN3D_TASK_RULES
from .renderer import CameraState, LaserSceneRenderer, read_trajectory
from .volt.proposals import (
    CLASS_NAMES,
    build_components,
    compatible_classes,
    load_inference_scene,
    select_component,
)


_VOLT_SCENE_CACHE: dict[str, dict[str, Any]] = {}
_VOLT_COMPONENT_CACHE: dict[tuple[str, int], dict[str, Any] | None] = {}


@dataclass
class Proposal:
    proposal_id: str
    group_id: str
    threshold: float
    indices_path: Path
    point_count: int
    mean_probability: float
    clicks: list[dict[str, Any]]
    source: str = "pinpoint3d"
    affordance_class: str | None = None
    source_metadata: dict[str, Any] | None = None
    inspected: bool = False
    inspection_path: Path | None = None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def to_wsl(path: Path) -> str:
    absolute = PureWindowsPath(path.resolve())
    drive = absolute.drive.rstrip(":").lower()
    tail = "/".join(absolute.parts[1:])
    return f"/mnt/{drive}/{tail}"


def nearest_visible_point(point_ids: np.ndarray, x: float, y: float) -> int:
    row = int(round(y))
    column = int(round(x))
    if 0 <= row < point_ids.shape[0] and 0 <= column < point_ids.shape[1]:
        value = int(point_ids[row, column])
        if value >= 0:
            return value
    for radius in range(1, 21):
        y0, y1 = max(0, row - radius), min(point_ids.shape[0], row + radius + 1)
        x0, x1 = max(0, column - radius), min(point_ids.shape[1], column + radius + 1)
        patch = point_ids[y0:y1, x0:x1]
        valid = np.argwhere(patch >= 0)
        if len(valid):
            target = np.asarray([row - y0, column - x0])
            distances = np.sum((valid - target) ** 2, axis=1)
            selected = valid[int(np.argmin(distances))]
            return int(patch[selected[0], selected[1]])
    raise RuntimeError("No rendered Faro point lies within 20 pixels of the click")


def look_at(position: np.ndarray, target: np.ndarray) -> CameraState:
    direction = np.asarray(target - position, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    return CameraState(
        position=np.asarray(position, dtype=np.float64),
        yaw_degrees=math.degrees(math.atan2(direction[1], direction[0])),
        pitch_degrees=math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0)))),
    )


class OpenSUN3DBackend:
    """Faro navigation and 3D proposal review for one task."""

    def __init__(
        self,
        *,
        scene_id: str,
        description_id: str,
        data_root: Path,
        output_dir: Path,
        pinpoint_checkpoint: Path,
        pinpoint_repo: Path,
        pinpoint_python: str,
        wsl_distro: str | None = None,
        volt_input_cache: Path | None = None,
        volt_prediction_cache: Path | None = None,
        point_radius: int = 3,
    ) -> None:
        self.scene_id = str(scene_id)
        self.description_id = str(description_id)
        self.data_root = Path(data_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.checkpoint = Path(pinpoint_checkpoint).resolve()
        self.pinpoint_repo = Path(pinpoint_repo).resolve()
        # PinPoint3D needs MinkowskiEngine. On Windows it runs inside WSL.
        self.pinpoint_python = str(pinpoint_python)
        self.wsl_distro = wsl_distro
        self.volt_input_cache = Path(volt_input_cache).resolve() if volt_input_cache else None
        self.volt_prediction_cache = (
            Path(volt_prediction_cache).resolve() if volt_prediction_cache else None
        )
        self.point_radius = int(point_radius)

        self.frames_dir = self.output_dir / "frames"
        self.proposals_dir = self.output_dir / "proposals"
        self.inspections_dir = self.output_dir / "inspections"
        for directory in (self.frames_dir, self.proposals_dir, self.inspections_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.renderer: LaserSceneRenderer | None = None
        self.base_colors: torch.Tensor | None = None
        self.raw_points: np.ndarray | None = None
        self.crop_original_indices: np.ndarray | None = None
        self.current_rgb: np.ndarray | None = None
        self.current_point_ids: np.ndarray | None = None
        self.current_frame: Path | None = None
        self.frame_index = 0
        self.poses: dict[str, dict[str, np.ndarray]] = {}
        self.pose_times: dict[str, np.ndarray] = {}
        self.offsets: dict[str, float] = {}
        self.durations: dict[str, float] = {}
        self.groups: dict[str, list[dict[str, Any]]] = {}
        self.proposals: dict[str, Proposal] = {}
        self.events: list[dict[str, Any]] = []
        self._segmenter: subprocess.Popen[str] | None = None
        self._segmenter_stderr: Any = None
        self._request_index = 0
        self._group_index = 0
        self.objective = ""

    def reset(self, task: Task) -> Observation:
        if task.task_id != f"{self.scene_id}_{self.description_id}":
            raise ValueError("task identity does not match configured SceneFun3D query")
        self.objective = task.objective
        scene_ply = self.data_root / f"{self.scene_id}_laser_scan.ply"
        crop_mask = self.data_root / f"{self.scene_id}_crop_mask.npy"
        for path in (scene_ply, crop_mask, self.checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)

        self.renderer = LaserSceneRenderer(
            scene_ply,
            crop_mask,
            point_radius=self.point_radius,
        )
        self.base_colors = self.renderer.colors.clone()
        self.crop_original_indices = self.renderer.original_indices.cpu().numpy()
        scan = trimesh.load(scene_ply, process=False)
        self.raw_points = np.asarray(scan.vertices, dtype=np.float32)

        for trajectory in sorted(self.data_root.glob("*/hires_poses.traj")):
            video_id = trajectory.parent.name
            poses = read_trajectory(trajectory)
            times = np.asarray(sorted(float(value) for value in poses), dtype=np.float64)
            self.poses[video_id] = poses
            self.pose_times[video_id] = times
            self.offsets[video_id] = float(times[0] - 0.6)
            self.durations[video_id] = float(times[-1] - self.offsets[video_id])
        if not self.poses:
            raise RuntimeError("No registered SceneFun3D trajectories were found")

        initial_video = min(self.durations, key=self.durations.get)
        first_time = self.pose_times[initial_video][0]
        self.renderer.set_camera_matrix(self.poses[initial_video][f"{first_time:.3f}"])
        self.current_frame = self._render("initial")
        self.events.append(
            {
                "event": "reset",
                "scene_id": self.scene_id,
                "description_id": self.description_id,
                "initial_video": initial_video,
            }
        )
        return self._observation(
            "A colored Faro 3D scan is shown.\n\nSCENEFUN3D TASK RULES:\n"
            + SCENEFUN3D_TASK_RULES
        )

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> SoftwareStep:
        renderer = self._require_renderer()
        if tool_name == "seek":
            video_id = str(arguments.get("video_id", ""))
            if video_id not in self.poses:
                return SoftwareStep(
                    response={
                        "ok": False,
                        "error": "Unknown registered pose stream.",
                        "valid_video_ids": sorted(self.poses),
                    },
                    observation=self._observation("Seek was rejected; the camera did not move."),
                )
            seconds = float(arguments.get("seconds", 0.0))
            seconds = float(np.clip(seconds, 0.0, self.durations[video_id]))
            desired = self.offsets[video_id] + seconds
            nearest = float(
                self.pose_times[video_id][
                    np.argmin(np.abs(self.pose_times[video_id] - desired))
                ]
            )
            renderer.set_camera_matrix(self.poses[video_id][f"{nearest:.3f}"])
            self.current_frame = self._render(f"seek_{video_id}_{seconds:06.2f}")
            self._record(tool_name, arguments, True)
            return SoftwareStep(
                response={
                    "ok": True,
                    "video_id": video_id,
                    "requested_seconds": round(seconds, 3),
                    "resolved_seconds": round(nearest - self.offsets[video_id], 3),
                },
                observation=self._observation("Registered Faro viewpoint after seek."),
            )

        if tool_name == "move":
            direction = str(arguments.get("direction", ""))
            distance = float(np.clip(float(arguments.get("distance_m", 0.0)), 0.0, 3.0))
            movements = {
                "forward": lambda: renderer.move_forward(distance),
                "backward": lambda: renderer.move_forward(-distance),
                "right": lambda: renderer.strafe(distance),
                "left": lambda: renderer.strafe(-distance),
                "up": lambda: renderer.move_vertical(distance),
                "down": lambda: renderer.move_vertical(-distance),
            }
            if direction not in movements:
                return SoftwareStep(
                    response={"ok": False, "error": "Invalid movement direction."},
                    observation=self._observation("The camera did not move."),
                )
            movements[direction]()
            self.current_frame = self._render(f"move_{direction}_{distance:.2f}")
            self._record(tool_name, arguments, True)
            return SoftwareStep(
                response={"ok": True, "direction": direction, "distance_m": distance},
                observation=self._observation("Free-camera movement applied."),
            )

        if tool_name == "rotate":
            yaw = float(np.clip(float(arguments.get("yaw_degrees", 0.0)), -180.0, 180.0))
            pitch = float(np.clip(float(arguments.get("pitch_degrees", 0.0)), -89.0, 89.0))
            renderer.rotate_yaw(-yaw)  # Unified interface defines positive yaw as right.
            renderer.rotate_pitch(pitch)
            self.current_frame = self._render(f"rotate_y{yaw:.1f}_p{pitch:.1f}")
            self._record(tool_name, arguments, True)
            return SoftwareStep(
                response={"ok": True, "yaw_degrees": yaw, "pitch_degrees": pitch},
                observation=self._observation("Free-camera rotation applied."),
            )

        if tool_name == "click":
            return self._click(arguments)

        if tool_name == "inspect":
            proposal_id = str(arguments.get("proposal_id", ""))
            proposal = self.proposals.get(proposal_id)
            if proposal is None:
                return SoftwareStep(
                    response={
                        "ok": False,
                        "error": "Unknown proposal.",
                        "valid_proposal_ids": sorted(self.proposals),
                    },
                    observation=self._observation("No proposal was inspected."),
                )
            path = self._render_inspection(proposal)
            proposal.inspected = True
            proposal.inspection_path = path
            self._write_proposal_record(proposal)
            self._record(tool_name, arguments, True)
            return SoftwareStep(
                response={
                    "ok": True,
                    "proposal_id": proposal_id,
                    "source": proposal.source,
                    "point_count": proposal.point_count,
                    "threshold": proposal.threshold,
                    "mean_foreground_probability": proposal.mean_probability,
                    "affordance_class": proposal.affordance_class,
                    "views": 4,
                },
                observation=Observation(
                    text=(
                        "3D proposal rendered from the current camera and three nearby "
                        "camera positions. Magenta points are the proposed interaction component."
                    ),
                    media=(
                        Media(
                            mime_type="image/png",
                            path=path,
                            label=f"INSPECTED 3D PROPOSAL {proposal_id}:",
                        ),
                    ),
                    record={
                        "proposal_id": proposal_id,
                        "point_count": proposal.point_count,
                        "threshold": proposal.threshold,
                    },
                ),
            )

        if tool_name == "submit":
            proposal_ids = [str(value) for value in arguments.get("proposal_ids", [])]
            missing = [value for value in proposal_ids if value not in self.proposals]
            uninspected = [
                value
                for value in proposal_ids
                if value in self.proposals and not self.proposals[value].inspected
            ]
            selected_groups = {
                self.proposals[value].group_id
                for value in proposal_ids
                if value in self.proposals
            }
            unreviewed_siblings = sorted(
                key
                for key, proposal in self.proposals.items()
                if proposal.group_id in selected_groups and not proposal.inspected
            )
            if not proposal_ids or missing or uninspected or unreviewed_siblings:
                return SoftwareStep(
                    response={
                        "ok": False,
                        "error": (
                            "Submit requires one or more existing proposals and visual review "
                            "of every candidate returned by each selected proposal's click."
                        ),
                        "unknown": missing,
                        "uninspected": uninspected,
                        "unreviewed_siblings": unreviewed_siblings,
                    },
                    observation=self._observation("Submission was rejected."),
                )
            selected_dir = self.output_dir / "selected_proposals"
            selected_dir.mkdir(parents=True, exist_ok=True)
            arrays: list[np.ndarray] = []
            selected_records: list[dict[str, Any]] = []
            for index, proposal_id in enumerate(proposal_ids, start=1):
                proposal = self.proposals[proposal_id]
                indices = np.load(proposal.indices_path).astype(np.int64, copy=False)
                destination = selected_dir / f"{index:02d}_{proposal_id}.npy"
                np.save(destination, indices)
                arrays.append(indices)
                selected_records.append(
                    {
                        "proposal_id": proposal_id,
                        "source": proposal.source,
                        "indices_path": str(destination),
                        "point_count": int(len(indices)),
                        "threshold": proposal.threshold,
                        "affordance_class": proposal.affordance_class,
                    }
                )
            union = np.unique(np.concatenate(arrays))
            union_path = self.output_dir / "predicted_indices_union.npy"
            np.save(union_path, union)
            output = {
                "scientific_status": "prediction_frozen_before_scoring",
                "scene_id": self.scene_id,
                "description_id": self.description_id,
                "selected_proposals": selected_records,
                "union_indices_path": str(union_path),
                "union_point_count": int(len(union)),
                "reason": str(arguments.get("reason", "")),
                "events": self.events,
            }
            atomic_json(self.output_dir / "submission_output.json", output)
            self._record(tool_name, arguments, True)
            return SoftwareStep(
                response={"ok": True, "submitted": True, "proposal_count": len(arrays)},
                done=True,
                output=output,
            )

        raise ValueError(f"Unsupported SceneFun3D tool: {tool_name}")

    def close(self) -> None:
        if self._segmenter is not None:
            try:
                if self._segmenter.stdin:
                    self._segmenter.stdin.close()
                self._segmenter.terminate()
                self._segmenter.wait(timeout=10)
            except Exception:
                self._segmenter.kill()
            self._segmenter = None
        if self._segmenter_stderr is not None:
            self._segmenter_stderr.close()
            self._segmenter_stderr = None
        self.renderer = None
        self.base_colors = None
        torch.cuda.empty_cache()

    def _click(self, arguments: Mapping[str, Any]) -> SoftwareStep:
        if self.current_point_ids is None or self.current_frame is None:
            raise RuntimeError("No current Faro render")
        width = self.current_point_ids.shape[1]
        height = self.current_point_ids.shape[0]
        x_norm = float(np.clip(float(arguments.get("x", 0.0)), 0.0, 1000.0))
        y_norm = float(np.clip(float(arguments.get("y", 0.0)), 0.0, 1000.0))
        x = x_norm * (width - 1) / 1000.0
        y = y_norm * (height - 1) / 1000.0
        point_index = nearest_visible_point(self.current_point_ids, x, y)
        polarity = str(arguments.get("polarity", "positive"))
        parent_id = str(arguments.get("proposal_id", "") or "")

        if parent_id:
            parent = self.proposals.get(parent_id)
            if parent is None:
                return SoftwareStep(
                    response={"ok": False, "error": "Unknown proposal_id for refinement."},
                    observation=self._observation("Refinement click was rejected."),
                )
            clicks = [dict(value) for value in self.groups[parent.group_id]]
        else:
            if polarity != "positive":
                return SoftwareStep(
                    response={"ok": False, "error": "A new proposal must begin with a positive click."},
                    observation=self._observation("Negative click was rejected."),
                )
            clicks = []
        clicks.append(
            {
                "point_index": point_index,
                "polarity": polarity,
                "source_frame": str(self.current_frame),
                "pixel_xy": [x, y],
                "normalized_xy": [x_norm, y_norm],
            }
        )

        self._group_index += 1
        group_id = f"g{self._group_index:03d}"
        self.groups[group_id] = clicks
        seed_specs: list[tuple[str, int, tuple[int, int]]] = [
            ("center", point_index, (0, 0))
        ]
        if not parent_id:
            seen = {point_index}
            for label, dx, dy in (
                ("left", -5, 0),
                ("right", 5, 0),
                ("up", 0, -5),
                ("down", 0, 5),
            ):
                seed_point = nearest_visible_point(self.current_point_ids, x + dx, y + dy)
                if seed_point not in seen:
                    seed_specs.append((label, seed_point, (dx, dy)))
                    seen.add(seed_point)

        candidate_records: list[dict[str, Any]] = []
        center_record: dict[str, Any] | None = None
        for label, seed_point, offset in seed_specs:
            seed_clicks = [dict(value) for value in clicks]
            seed_clicks[-1]["point_index"] = seed_point
            records = [
                record
                for record in self._segment(f"{group_id}_{label}", seed_clicks)
                if abs(float(record["threshold"]) - 0.50) < 1e-9
            ]
            if not records:
                raise RuntimeError(
                    f"PinPoint3D did not return its 0.50 proposal for seed {label}"
                )
            record = records[0]
            record["suffix"] = f"pinpoint_{label}"
            record["source"] = "pinpoint3d"
            record["clicks"] = seed_clicks
            record["source_metadata"] = {
                "seed_label": label,
                "seed_offset_pixels": list(offset),
                "seed_faro_vertex": int(seed_point),
                "purpose": "local click-rasterization robustness",
            }
            candidate_records.append(record)
            if label == "center":
                center_record = record
        assert center_record is not None
        volt_record = self._volt_candidate(group_id, point_index, center_record)
        if volt_record is not None:
            candidate_records.append(volt_record)
        created: list[dict[str, Any]] = []
        for record in candidate_records:
            proposal_id = f"{group_id}_{record['suffix']}"
            path = Path(record["path"])
            proposal = Proposal(
                proposal_id=proposal_id,
                group_id=group_id,
                threshold=float(record["threshold"]),
                indices_path=path,
                point_count=int(record["point_count"]),
                mean_probability=float(record["mean_foreground_probability"]),
                clicks=[dict(value) for value in record.get("clicks", clicks)],
                source=str(record.get("source", "pinpoint3d")),
                affordance_class=record.get("affordance_class"),
                source_metadata=dict(record.get("source_metadata", {})),
            )
            self.proposals[proposal_id] = proposal
            self._write_proposal_record(proposal)
            created.append(
                {
                    "proposal_id": proposal_id,
                    "source": proposal.source,
                    "point_count": proposal.point_count,
                    "threshold": proposal.threshold,
                    "mean_foreground_probability": round(proposal.mean_probability, 4),
                    "affordance_class": proposal.affordance_class,
                }
            )
        self._record("click", arguments, True, {"point_index": point_index, "created": created})
        return SoftwareStep(
            response={
                "ok": True,
                "clicked_faro_vertex": point_index,
                "proposals": created,
                "instruction": (
                    "Inspect every returned candidate. Nearby-seed PinPoint3D variants make the "
                    "click robust to point-cloud rasterization; choose the smallest complete, "
                    "coherent actionable component. Refine one with another positive or negative "
                    "click if every candidate includes too little or too much."
                ),
            },
            observation=self._observation("3D segmentation proposals are ready for inspection."),
        )

    def _ensure_segmenter(self) -> None:
        if self._segmenter is not None:
            return
        helper = Path(__file__).with_name("pinpoint_server.py")
        scene_ply = self.data_root / f"{self.scene_id}_laser_scan.ply"
        stderr_path = self.output_dir / "pinpoint_server.stderr.log"
        self._segmenter_stderr = stderr_path.open("a", encoding="utf-8")
        path = self._service_path
        service = [
            self.pinpoint_python,
            "-u",
            path(helper),
            "--repo",
            path(self.pinpoint_repo),
            "--checkpoint",
            path(self.checkpoint),
            "--scene-ply",
            path(scene_ply),
            "--crop-radius",
            "0.25",
            "--voxel-size",
            "0.005",
        ]
        environment = ["OMP_NUM_THREADS=12", f"PYTHONPATH={path(self.pinpoint_repo)}"]
        if self.wsl_distro:
            command = ["wsl.exe", "-d", self.wsl_distro, "--cd", path(self.pinpoint_repo),
                       "--", "/usr/bin/env", *environment, *service]
        else:
            command = ["/usr/bin/env", *environment, *service]
        self._segmenter = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._segmenter_stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        while True:
            line = self._segmenter.stdout.readline() if self._segmenter.stdout else ""
            if not line:
                raise RuntimeError(
                    f"PinPoint3D service exited during startup with code {self._segmenter.poll()}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "startup_error":
                raise RuntimeError(record.get("error", "PinPoint3D startup failed"))
            if record.get("event") == "ready":
                atomic_json(self.output_dir / "pinpoint_server_ready.json", record)
                return

    def _service_path(self, value: Path) -> str:
        """A path as the PinPoint3D process sees it: inside WSL or native."""
        return to_wsl(Path(value)) if self.wsl_distro else str(Path(value).resolve())

    def _segment(self, group_id: str, clicks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._ensure_segmenter()
        assert self._segmenter is not None
        self._request_index += 1
        request_id = f"r{self._request_index:03d}"
        output_dir = self.proposals_dir / group_id
        payload = {
            "request_id": request_id,
            "clicks": [
                {"point_index": int(item["point_index"]), "polarity": item["polarity"]}
                for item in clicks
            ],
            "output_dir": self._service_path(output_dir),
        }
        assert self._segmenter.stdin is not None
        assert self._segmenter.stdout is not None
        self._segmenter.stdin.write(json.dumps(payload) + "\n")
        self._segmenter.stdin.flush()
        while True:
            line = self._segmenter.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"PinPoint3D service exited during inference with code {self._segmenter.poll()}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("request_id") != request_id:
                continue
            if record.get("event") == "request_error":
                raise RuntimeError(record.get("error", "PinPoint3D request failed"))
            if record.get("event") == "result":
                converted: list[dict[str, Any]] = []
                for candidate in record["candidates"]:
                    value = dict(candidate)
                    value["path"] = str(output_dir / Path(str(value["path"])).name)
                    converted.append(value)
                atomic_json(output_dir / "inference.json", {"request": payload, "result": converted})
                return converted

    def _load_volt_scene(self) -> dict[str, Any] | None:
        if self.volt_prediction_cache is None or self.volt_input_cache is None:
            return None
        key = str(self.volt_prediction_cache)
        cached = _VOLT_SCENE_CACHE.get(key)
        if cached is not None:
            return cached
        metadata_path = self.volt_prediction_cache / "metadata.json"
        predicted_path = self.volt_prediction_cache / "predicted_class.npy"
        probabilities_path = self.volt_prediction_cache / "class_probabilities.npy"
        for path in (metadata_path, predicted_path, probabilities_path, self.volt_input_cache):
            if not path.is_file():
                raise FileNotFoundError(path)
        data, raw_rows, raw_count = load_inference_scene(self.volt_input_cache)
        predicted = np.load(predicted_path, mmap_mode="r")
        probabilities = np.load(probabilities_path, mmap_mode="r")
        if predicted.shape != (len(raw_rows),) or probabilities.shape != (len(raw_rows), 10):
            raise RuntimeError("Volt full-scene cache shape does not match the Faro crop")
        cached = {
            "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
            "coord": data["coord"],
            "raw_rows": raw_rows,
            "raw_count": raw_count,
            "predicted": predicted,
            "probabilities": probabilities,
        }
        _VOLT_SCENE_CACHE[key] = cached
        return cached

    def _volt_candidate(
        self,
        group_id: str,
        point_index: int,
        pinpoint_record: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        scene = self._load_volt_scene()
        if scene is None:
            return None
        raw_rows = scene["raw_rows"]
        cache_index = int(np.searchsorted(raw_rows, point_index))
        if cache_index >= len(raw_rows) or int(raw_rows[cache_index]) != point_index:
            audit = {"emitted": False, "reason": "clicked_point_outside_volt_crop"}
            atomic_json(self.proposals_dir / group_id / "volt_audit.json", audit)
            return None

        probabilities = scene["probabilities"]
        predicted = scene["predicted"]
        foreground = probabilities[cache_index, 1:]
        class_id = int(np.argmax(foreground)) + 1
        selected_class = CLASS_NAMES[class_id]
        hard_class = CLASS_NAMES[int(predicted[cache_index])]
        task_family, allowed_classes = compatible_classes(self.objective)
        hard_foreground = hard_class == selected_class
        semantic_compatible = selected_class in allowed_classes
        audit: dict[str, Any] = {
            "clicked_faro_vertex": int(point_index),
            "cache_index": cache_index,
            "task_family": task_family,
            "selected_class": selected_class,
            "hard_class_at_anchor": hard_class,
            "hard_foreground": hard_foreground,
            "semantic_compatible": semantic_compatible,
            "selected_class_probability_at_anchor": float(probabilities[cache_index, class_id]),
            "background_probability_at_anchor": float(probabilities[cache_index, 0]),
        }
        if not hard_foreground or not semantic_compatible:
            audit.update({"emitted": False, "reason": "label_blind_fusion_gate"})
            atomic_json(self.proposals_dir / group_id / "volt_audit.json", audit)
            return None

        component_key = (str(self.volt_prediction_cache), class_id)
        if component_key not in _VOLT_COMPONENT_CACHE:
            _VOLT_COMPONENT_CACHE[component_key] = build_components(
                predicted,
                scene["coord"],
                class_id,
                min_points=10,
                knn=32,
                radius=0.02,
            )
        component_data = _VOLT_COMPONENT_CACHE[component_key]
        if component_data is None:
            selected = np.asarray([cache_index], dtype=np.int64)
            anchor_distance = 0.0
            component_label = -1
            fallback = "single_anchor_point"
        else:
            selected, anchor_distance, component_label = select_component(
                component_data,
                scene["coord"][cache_index],
            )
            fallback = None
        raw_indices = raw_rows[selected].astype(np.int64, copy=False)
        pinpoint_indices = np.load(str(pinpoint_record["path"])).astype(np.int64, copy=False)
        exact_duplicate = np.array_equal(np.sort(raw_indices), np.sort(pinpoint_indices))
        confidence = float(np.mean(probabilities[selected, class_id]))
        audit.update(
            {
                "component_label": int(component_label),
                "component_points": int(len(raw_indices)),
                "component_nearest_distance_metres": float(anchor_distance),
                "component_mean_probability": confidence,
                "fallback": fallback,
                "exact_duplicate": bool(exact_duplicate),
                "emitted": not exact_duplicate,
                "reason": "exact_duplicate" if exact_duplicate else "eligible_recovery",
            }
        )
        atomic_json(self.proposals_dir / group_id / "volt_audit.json", audit)
        if exact_duplicate:
            return None
        path = self.proposals_dir / group_id / "volt.npy"
        np.save(path, raw_indices)
        return {
            "suffix": "volt",
            "source": "volt_tta13",
            "threshold": 0.0,
            "path": str(path),
            "point_count": int(len(raw_indices)),
            "mean_foreground_probability": confidence,
            "affordance_class": selected_class,
            "source_metadata": audit,
        }

    def _render(self, event: str) -> Path:
        renderer = self._require_renderer()
        self.frame_index += 1
        rgb, depth, point_ids = renderer.render()
        stem = f"frame_{self.frame_index:04d}_{event}"
        path = self.frames_dir / f"{stem}.png"
        Image.fromarray(rgb).save(path)
        np.save(self.frames_dir / f"{stem}.point_ids.npy", point_ids)
        atomic_json(
            self.frames_dir / f"{stem}.camera.json",
            {"event": event, "state": renderer.state.as_dict()},
        )
        self.current_rgb = rgb
        self.current_point_ids = point_ids
        return path

    def _render_inspection(self, proposal: Proposal) -> Path:
        renderer = self._require_renderer()
        assert self.raw_points is not None
        assert self.base_colors is not None
        assert self.crop_original_indices is not None
        indices = np.load(proposal.indices_path).astype(np.int64, copy=False)
        valid_indices = indices[(indices >= 0) & (indices < len(self.raw_points))]
        if not len(valid_indices):
            raise RuntimeError(f"Proposal {proposal.proposal_id} is empty")
        points = self.raw_points[valid_indices]
        center = np.median(points, axis=0).astype(np.float64)
        extent = np.percentile(np.linalg.norm(points - center, axis=1), 95)
        saved_state = CameraState(
            position=renderer.state.position.copy(),
            yaw_degrees=renderer.state.yaw_degrees,
            pitch_degrees=renderer.state.pitch_degrees,
        )
        target_mask = np.zeros(renderer.total_original_points, dtype=bool)
        target_mask[valid_indices] = True
        selected_local = torch.from_numpy(target_mask[self.crop_original_indices]).to(
            renderer.device
        )
        overlay_colors = (self.base_colors.to(torch.float32) * 0.32).to(torch.uint8)
        overlay_colors[selected_local] = torch.tensor(
            [255, 45, 190], dtype=torch.uint8, device=renderer.device
        )
        renderer.colors = overlay_colors

        from_target = saved_state.position - center
        distance_current = float(np.linalg.norm(from_target))
        if distance_current < 1e-6:
            from_target = np.asarray([1.0, 0.0, 0.0])
        else:
            from_target /= distance_current
        close_distance = float(np.clip(max(0.38, extent * 5.0), 0.38, 1.1))

        states: list[tuple[str, CameraState]] = [("CURRENT VIEW", saved_state)]
        for label, angle in (("CLOSE VIEW", 0.0), ("LEFT ANGLE", -25.0), ("RIGHT ANGLE", 25.0)):
            radians = math.radians(angle)
            vector = np.asarray(
                [
                    math.cos(radians) * from_target[0] - math.sin(radians) * from_target[1],
                    math.sin(radians) * from_target[0] + math.cos(radians) * from_target[1],
                    from_target[2],
                ],
                dtype=np.float64,
            )
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            position = center + vector * close_distance
            states.append((label, look_at(position, center)))

        panels: list[Image.Image] = []
        font = ImageFont.load_default(size=20)
        try:
            for label, state in states:
                renderer.set_camera_state(state)
                rgb, _, _ = renderer.render()
                panel = Image.fromarray(rgb).resize((480, 360), Image.Resampling.LANCZOS)
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 480, 34), fill=(6, 12, 20))
                draw.text((12, 8), label, fill=(255, 255, 255), font=font)
                panels.append(panel)
        finally:
            renderer.colors = self.base_colors
            renderer.set_camera_state(saved_state)

        montage = Image.new("RGB", (960, 720), (6, 12, 20))
        for index, panel in enumerate(panels):
            montage.paste(panel, ((index % 2) * 480, (index // 2) * 360))
        draw = ImageDraw.Draw(montage)
        draw.rectangle((0, 680, 960, 720), fill=(6, 12, 20))
        draw.text(
            (12, 688),
            f"{proposal.proposal_id} | magenta proposal | {proposal.point_count:,} Faro points",
            fill=(255, 180, 230),
            font=font,
        )
        path = self.inspections_dir / f"{proposal.proposal_id}.png"
        montage.save(path)
        return path

    def _observation(self, text: str) -> Observation:
        if self.current_frame is None:
            raise RuntimeError("No current SceneFun3D frame")
        renderer = self._require_renderer()
        return Observation(
            text=text,
            media=(
                Media(
                    mime_type="image/png",
                    path=self.current_frame,
                    label="CURRENT FARO 3D VIEW:",
                ),
            ),
            record={
                "scene_id": self.scene_id,
                "camera": renderer.state.as_dict(),
                "registered_pose_streams": {
                    key: {"duration_seconds": round(value, 2)}
                    for key, value in sorted(self.durations.items())
                },
                "proposal_ids": sorted(self.proposals),
                "inspected_proposal_ids": sorted(
                    key for key, value in self.proposals.items() if value.inspected
                ),
            },
        )

    def _record(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        accepted: bool,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            "index": len(self.events) + 1,
            "tool": tool,
            "arguments": dict(arguments),
            "accepted": accepted,
        }
        if extra:
            record.update(extra)
        self.events.append(record)
        atomic_json(self.output_dir / "backend_events.json", self.events)

    def _write_proposal_record(self, proposal: Proposal) -> None:
        atomic_json(
            self.proposals_dir / f"{proposal.proposal_id}.json",
            {
                "proposal_id": proposal.proposal_id,
                "group_id": proposal.group_id,
                "source": proposal.source,
                "affordance_class": proposal.affordance_class,
                "threshold": proposal.threshold,
                "indices_path": str(proposal.indices_path),
                "point_count": proposal.point_count,
                "mean_foreground_probability": proposal.mean_probability,
                "clicks": proposal.clicks,
                "source_metadata": proposal.source_metadata,
                "inspected": proposal.inspected,
                "inspection_path": (
                    str(proposal.inspection_path) if proposal.inspection_path else None
                ),
            },
        )

    def _require_renderer(self) -> LaserSceneRenderer:
        if self.renderer is None:
            raise RuntimeError("SceneFun3D backend must be reset before use")
        return self.renderer
