"""VLNVerse backend: a floating camera in an Isaac Sim scene.

The agent moves the camera forward, backward, sideways or vertically and turns
or tilts it. Each operation renders a new first-person image through the Isaac
service in ``isaac_bridge.py``. Stop saves the trajectory and final pose.
"""

from __future__ import annotations

import io
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from ...core import Media, Observation, SoftwareStep, Task


CAMERA_HEIGHT_M = 1.5
MIN_CAMERA_HEIGHT_M = 0.5
MAX_CAMERA_HEIGHT_M = 2.5
TRANSLATION_CLIP_M = 3.0
ROTATION_CLIP_DEG = 180.0
COLLISION_RADIUS_M = 0.2
COLLISION_CLEARANCE_M = 0.03


def quaternion_to_agent_yaw(rotation_wxyz: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in rotation_wxyz)
    world_yaw = math.degrees(
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )
    return (90.0 - world_yaw + 180.0) % 360.0 - 180.0


def trajectory_length(points: Sequence[Sequence[float]]) -> float:
    return float(
        sum(
            np.linalg.norm(np.asarray(current) - np.asarray(previous))
            for previous, current in zip(points, points[1:])
        )
    )


class VLNVerseBackend:
    """Camera navigation for one VLNVerse episode."""

    def __init__(
        self,
        *,
        bridge_url: str,
        episode: Mapping[str, Any],
        output_dir: str | Path,
        collision: bool = False,
    ) -> None:
        self.bridge_url = bridge_url.rstrip("/")
        # With collision off the camera moves the full requested distance.
        # Camera height limits and invalid-render recovery still apply.
        self.collision = bool(collision)
        self.episode = dict(episode)
        self.output_dir = Path(output_dir).resolve()
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.position = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0
        self.pitch = 0.0
        self.step_index = 0
        self.positions: list[list[float]] = []
        self.actions: list[dict[str, Any]] = []
        self.collision_audits: list[dict[str, Any]] = []
        self.last_frame: Path | None = None

    def reset(self, task: Task) -> Observation:
        if task.task_id != str(self.episode["episode_id"]):
            raise ValueError(
                f"task {task.task_id!r} does not match episode {self.episode['episode_id']!r}"
            )
        start = self.episode["start_position"]
        self.position = np.asarray(
            [float(start[0]), float(start[1]), CAMERA_HEIGHT_M], dtype=np.float64
        )
        self.yaw = quaternion_to_agent_yaw(self.episode["start_rotation"])
        self.pitch = 0.0
        self.step_index = 0
        self.positions = [self.position.tolist()]
        self.actions = []
        self.collision_audits = []
        frame, diagnostics = self._render()
        return self._observation(
            frame,
            diagnostics,
            bumped=False,
            action_result="episode_started",
        )

    def execute(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> SoftwareStep:
        if tool_name == "stop":
            reason = str(arguments.get("reason", ""))
            self.actions.append(
                {
                    "step": self.step_index + 1,
                    "tool": "stop",
                    "arguments": dict(arguments),
                    "position_m": self.position.tolist(),
                }
            )
            print(
                f"VLN_STEP episode={self.episode['episode_id']} "
                f"step={self.step_index + 1:03d} action=stop",
                flush=True,
            )
            return SoftwareStep(
                response={
                    "stopped": True,
                    "position_m": self._rounded_position(),
                    "reason": reason,
                },
                done=True,
                output=self._terminal_output(reason),
            )

        prior_position = self.position.copy()
        prior_yaw = self.yaw
        prior_pitch = self.pitch
        bumped = False
        audit: dict[str, Any] | None = None

        if tool_name == "move":
            direction = str(arguments.get("direction", ""))
            distance = float(arguments.get("distance_m", 0.0))
            bumped, audit = self._move(direction, distance)
            action_result = (
                f"move_{direction}_applied" if not bumped else f"move_{direction}_bumped"
            )
        elif tool_name == "rotate":
            yaw_delta = float(arguments.get("yaw_degrees", 0.0))
            pitch_delta = float(arguments.get("pitch_degrees", 0.0))
            self.yaw = self._wrap_angle(
                self.yaw + float(np.clip(yaw_delta, -ROTATION_CLIP_DEG, ROTATION_CLIP_DEG))
            )
            self.pitch = float(
                np.clip(
                    self.pitch
                    + float(
                        np.clip(
                            pitch_delta,
                            -ROTATION_CLIP_DEG,
                            ROTATION_CLIP_DEG,
                        )
                    ),
                    -89.0,
                    89.0,
                )
            )
            action_result = "rotation_applied"
        else:
            raise ValueError(f"unsupported VLNVerse tool: {tool_name}")

        self.step_index += 1
        render_reverted = False
        render_error: str | None = None
        try:
            frame, diagnostics = self._render()
        except RuntimeError as exc:
            self.position = prior_position
            self.yaw = prior_yaw
            self.pitch = prior_pitch
            frame, diagnostics = self._render()
            bumped = True
            render_reverted = True
            render_error = str(exc)
            action_result = "action_reverted_after_invalid_render"

        self.positions.append(self.position.tolist())
        action_record = {
            "step": self.step_index,
            "tool": tool_name,
            "arguments": dict(arguments),
            "position_m": self.position.tolist(),
            "yaw_deg": self.yaw,
            "pitch_deg": self.pitch,
            "bumped": bumped,
            "render_reverted": render_reverted,
            "render_error": render_error,
        }
        self.actions.append(action_record)
        if audit is not None:
            self.collision_audits.append(
                {"step": self.step_index, "tool": tool_name, **audit}
            )

        print(
            f"VLN_STEP episode={self.episode['episode_id']} "
            f"step={self.step_index:03d} action={tool_name} bump={int(bumped)}",
            flush=True,
        )
        response = self._state(bumped=bumped, action_result=action_result)
        if tool_name == "move" and audit is not None:
            response.update(
                {
                    "requested_distance_m": round(
                        float(audit["requested_distance_m"]), 3
                    ),
                    "achieved_distance_m": round(
                        float(audit["achieved_distance_m"]), 3
                    ),
                }
            )
        return SoftwareStep(
            response=response,
            observation=self._observation(
                frame,
                diagnostics,
                bumped=bumped,
                action_result=action_result,
            ),
        )

    def close(self) -> None:
        return

    def _move(self, direction: str, distance: float) -> tuple[bool, dict[str, Any]]:
        if direction not in {"forward", "backward", "left", "right", "up", "down"}:
            raise ValueError(f"invalid movement direction: {direction!r}")
        distance = float(np.clip(distance, 0.0, TRANSLATION_CLIP_M))
        forward, right, up = self._basis()
        vectors = {
            "forward": forward,
            "backward": -forward,
            "left": -right,
            "right": right,
            "up": up,
            "down": -up,
        }
        target = self.position + vectors[direction] * distance
        unclamped_height = float(target[2])
        target[2] = float(
            np.clip(target[2], MIN_CAMERA_HEIGHT_M, MAX_CAMERA_HEIGHT_M)
        )
        height_clamped = not math.isclose(unclamped_height, float(target[2]), abs_tol=1e-6)
        if not self.collision:
            distance_m = float(np.linalg.norm(target - self.position))
            result = {
                "ok": True,
                "position": target.tolist(),
                "bumped": False,
                "requested_distance_m": distance_m,
                "achieved_distance_m": distance_m,
                "collision_disabled": True,
            }
        else:
            result = self._request_json(
                "/sweep",
                {
                    "x0": float(self.position[0]),
                    "y0": float(self.position[1]),
                    "z0": float(self.position[2]),
                    "x1": float(target[0]),
                    "y1": float(target[1]),
                    "z1": float(target[2]),
                    "radius": COLLISION_RADIUS_M,
                    "clearance": COLLISION_CLEARANCE_M,
                },
            )
        if not result.get("ok"):
            raise RuntimeError(f"Isaac collision query failed: {result}")
        self.position = np.asarray(result["position"], dtype=np.float64)
        audit = {
            "requested_distance_m": float(result["requested_distance_m"]),
            "achieved_distance_m": float(result["achieved_distance_m"]),
            "height_clamped": height_clamped,
            "sweep": result,
        }
        return bool(result.get("bumped")) or height_clamped, audit

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        forward = np.asarray(
            [
                math.sin(yaw) * math.cos(pitch),
                math.cos(yaw) * math.cos(pitch),
                math.sin(pitch),
            ],
            dtype=np.float64,
        )
        world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        norm = float(np.linalg.norm(right))
        right = right / norm if norm > 1e-6 else np.asarray([1.0, 0.0, 0.0])
        up = np.cross(right, forward)
        return forward, right, up

    def _render(self) -> tuple[Path, dict[str, Any]]:
        payload = {
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "z": float(self.position[2]),
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": 0.0,
        }
        query = urllib.parse.urlencode(payload)
        errors: list[str] = []
        for attempt in range(1, 6):
            try:
                with urllib.request.urlopen(
                    f"{self.bridge_url}/render?{query}", timeout=150
                ) as response:
                    data = response.read()
                diagnostics = self._image_diagnostics(data)
                diagnostics["capture_attempt"] = attempt
                if not diagnostics["valid"]:
                    raise RuntimeError(f"invalid render: {diagnostics}")
                frame_path = self.frames_dir / f"step_{self.step_index:04d}.png"
                frame_path.write_bytes(data)
                self.last_frame = frame_path
                return frame_path, diagnostics
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt < 5:
                    time.sleep(float(attempt))
        raise RuntimeError("Isaac render failed: " + " | ".join(errors))

    @staticmethod
    def _image_diagnostics(data: bytes) -> dict[str, Any]:
        image = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        mean = float(image.mean())
        black_fraction = float(np.mean(np.max(image, axis=2) <= 2))
        return {
            "valid": bool(
                image.ndim == 3
                and image.shape[2] == 3
                and mean >= 2.0
                and black_fraction < 0.995
            ),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "mean_luma_proxy": round(mean, 3),
            "std_luma_proxy": round(float(image.std()), 3),
            "black_fraction": round(black_fraction, 5),
        }

    def _observation(
        self,
        frame: Path,
        diagnostics: Mapping[str, Any],
        *,
        bumped: bool,
        action_result: str,
    ) -> Observation:
        return Observation(
            text="Current egocentric RGB observation.",
            media=(
                Media(
                    mime_type="image/png",
                    path=frame,
                    label="CURRENT VIEW:",
                ),
            ),
            record={
                **self._state(bumped=bumped, action_result=action_result),
                "render": dict(diagnostics),
            },
        )

    def _state(self, *, bumped: bool, action_result: str) -> dict[str, Any]:
        return {
            "position_m": self._rounded_position(),
            "yaw_deg": round(self.yaw, 1),
            "pitch_deg": round(self.pitch, 1),
            "bumped_wall": bool(bumped),
            "action_result": action_result,
        }

    def _terminal_output(self, reason: str) -> dict[str, Any]:
        return {
            "scientific_status": "blind_challenge_trajectory_unscored",
            "ground_truth_metrics_available": False,
            "episode_id": self.episode["episode_id"],
            "trajectory_id": self.episode["trajectory_id"],
            "scan": self.episode["scan"],
            "instruction": self.episode["instruction"],
            "termination": "model_stop",
            "stop_reason": reason,
            "start_position_annotation": self.episode["start_position"],
            "start_rotation_wxyz": self.episode["start_rotation"],
            "final_position_world": self.position.tolist(),
            "predicted_path": self.positions,
            "trajectory_length_m": round(trajectory_length(self.positions), 6),
            "actions": self.actions,
            "collision_audits_hidden_from_model": self.collision_audits,
            "final_frame": str(self.last_frame) if self.last_frame else None,
        }

    def _request_json(self, endpoint: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode(parameters)
        with urllib.request.urlopen(
            f"{self.bridge_url}{endpoint}?{query}", timeout=150
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def _rounded_position(self) -> list[float]:
        return [round(float(value), 3) for value in self.position]

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (float(value) + 180.0) % 360.0 - 180.0
