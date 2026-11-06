"""MeViSv2 backend: a video player, frame selection and SAM3 object tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ...core import Media, Observation, SoftwareStep, Task
from .sam3_interface import Sam3VideoInterface
from .video import DEFAULT_FPS, build_complete_video, load_task


class MeViSBackend:
    """Referring video segmentation for one MeViSv2 expression."""

    def __init__(
        self,
        *,
        video_id: str,
        expression_id: str,
        output_dir: Path,
        data_root: Path,
        checkpoint: Path,
        video_cache: Path | None = None,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.video_id = str(video_id)
        self.expression_id = str(expression_id)
        self.output_dir = output_dir.resolve()
        self.data_root = data_root.resolve()
        self.checkpoint = checkpoint.resolve()
        self.video_cache = (video_cache or self.output_dir / "video_cache").resolve()
        self.fps = float(fps)
        self.interface: Sam3VideoInterface | None = None
        self.frame_dir: Path | None = None
        self.frame_names: list[str] = []
        self.complete_video: Path | None = None
        self.prediction_dir = self.output_dir / "prediction"
        self.events: list[dict[str, Any]] = []

    def reset(self, task: Task) -> Observation:
        if task.task_id != f"{self.video_id}_expr{self.expression_id}":
            raise ValueError(f"Task ID {task.task_id!r} does not match backend task")
        expression, self.frame_names = load_task(
            self.data_root, self.video_id, self.expression_id
        )
        if task.objective.strip() != expression.strip():
            raise ValueError("Task objective does not match public expression metadata")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir = self.data_root / "JPEGImages" / self.video_id
        self.video_cache.mkdir(parents=True, exist_ok=True)
        self.complete_video = build_complete_video(
            self.frame_dir,
            self.frame_names,
            self.video_cache / f"{self.video_id}_all_frames_{self.fps:g}fps.mp4",
            self.fps,
        )
        self.interface = Sam3VideoInterface(
            frame_dir=self.frame_dir,
            frame_names=self.frame_names,
            output_dir=self.output_dir / "sam3",
            checkpoint=self.checkpoint,
            fps=self.fps,
        )
        self._write_events()
        return Observation(
            text=(
                "The attached media is the complete original video. Inspect the video and "
                "select every object matching the objective."
            ),
            media=(
                Media(
                    mime_type="video/mp4",
                    path=self.complete_video,
                    label="COMPLETE ORIGINAL VIDEO:",
                ),
            ),
            record={
                "video_id": self.video_id,
                "frame_count": len(self.frame_names),
                "fps": self.fps,
                "coordinate_system": "native full frame, normalized 0..1000",
                "active_object_ids": [],
            },
        )

    def execute(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> SoftwareStep:
        interface = self._require_interface()
        try:
            if tool_name == "seek":
                frame_index = int(arguments["frame_index"])
                path = interface.frame_path(frame_index)
                response = {
                    "ok": True,
                    "frame_index": frame_index,
                    "frame_count": len(self.frame_names),
                    "coordinate_system": "this full frame, normalized 0..1000",
                }
                step = SoftwareStep(
                    response=response,
                    observation=Observation(
                        text=f"Full-resolution official frame {frame_index}.",
                        media=(Media(mime_type="image/jpeg", path=path, label="SELECTED FRAME:"),),
                        record=self._state(),
                    ),
                )
            elif tool_name == "click":
                frame_index = int(arguments["frame_index"])
                x = float(arguments["x"])
                y = float(arguments["y"])
                mode = str(arguments["mode"])
                if mode == "new_object":
                    record, image = interface.select_object(
                        frame_index=frame_index, click_x=x, click_y=y
                    )
                else:
                    object_id = self._object_id(arguments.get("object_id"))
                    point = {"x": x, "y": y}
                    record, image = interface.refine_object(
                        object_id=object_id,
                        frame_index=frame_index,
                        positive_points=[point] if mode == "refine_positive" else [],
                        negative_points=[point] if mode == "refine_negative" else [],
                    )
                response = {
                    "ok": True,
                    **record,
                    "active_object_ids": [str(value) for value in sorted(interface.object_ids)],
                }
                step = SoftwareStep(
                    response=response,
                    observation=Observation(
                        text="SAM3 mask at the selected frame.",
                        media=(Media(mime_type="image/png", path=image, label="CURRENT SELECTION:"),),
                        record=self._state(),
                    ),
                )
            elif tool_name == "remove":
                record = interface.remove_object(
                    object_id=self._object_id(arguments.get("object_id"))
                )
                step = SoftwareStep(
                    response={"ok": True, **record},
                    observation=Observation(text="Object removed.", record=self._state()),
                )
            elif tool_name == "inspect":
                summary, video = interface.preview_tracking()
                step = SoftwareStep(
                    response={
                        "ok": True,
                        "preview_generation": summary["preview_generation"],
                        "object_ids": [str(value) for value in summary["object_ids"]],
                        "frame_count": summary["frame_count"],
                        "nonempty_frames": summary["nonempty_frames"],
                    },
                    observation=Observation(
                        text=(
                            "The attached media is the complete tracked-mask overlay. Review "
                            "the complete video before submitting or revising."
                        ),
                        media=(
                            Media(
                                mime_type="video/mp4",
                                path=video,
                                label="COMPLETE TRACKED-MASK OVERLAY:",
                            ),
                        ),
                        record=self._state(),
                    ),
                )
            elif tool_name == "submit":
                requested = {self._object_id(value) for value in arguments["object_ids"]}
                active = set(interface.object_ids)
                if requested != active:
                    raise ValueError(
                        f"submit object_ids must equal active selections {sorted(active)}; "
                        "remove unwanted selections or add missing IDs, inspect, then submit"
                    )
                output = interface.materialize_submission(
                    self.prediction_dir, no_target=False
                )
                output.update(
                    {
                        "video_id": self.video_id,
                        "expression_id": self.expression_id,
                        "reason": str(arguments.get("reason", "")),
                    }
                )
                step = SoftwareStep(
                    response={"ok": True, "submitted": True, **output},
                    done=True,
                    output=output,
                )
            elif tool_name == "submit_no_target":
                output = interface.materialize_submission(
                    self.prediction_dir, no_target=True
                )
                output.update(
                    {
                        "video_id": self.video_id,
                        "expression_id": self.expression_id,
                        "reason": str(arguments.get("reason", "")),
                    }
                )
                step = SoftwareStep(
                    response={"ok": True, "submitted": True, **output},
                    done=True,
                    output=output,
                )
            else:
                raise ValueError(f"Unknown tool {tool_name!r}")
        except Exception as exc:
            step = SoftwareStep(
                response={
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "active_object_ids": [
                        str(value) for value in sorted(interface.object_ids)
                    ],
                },
                observation=Observation(
                    text="The action was rejected. Correct the arguments or selection state.",
                    record=self._state(),
                ),
            )
        self.events.append(
            {
                "index": len(self.events) + 1,
                "tool": tool_name,
                "arguments": dict(arguments),
                "response": dict(step.response),
                "done": step.done,
            }
        )
        self._write_events()
        return step

    def close(self) -> None:
        if self.interface is not None:
            self.interface.close()
            self.interface = None

    @staticmethod
    def _object_id(value: Any) -> int:
        if value is None or str(value).strip() == "":
            raise ValueError("object_id is required for this operation")
        return int(str(value))

    def _require_interface(self) -> Sam3VideoInterface:
        if self.interface is None:
            raise RuntimeError("Backend must be reset before use")
        return self.interface

    def _state(self) -> dict[str, Any]:
        interface = self._require_interface()
        return {
            "video_id": self.video_id,
            "frame_count": len(self.frame_names),
            "active_object_ids": [str(value) for value in sorted(interface.object_ids)],
            "preview_generation": interface.preview_generation,
            "preview_current": interface.preview_is_current,
        }

    def _write_events(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "backend_events.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.events, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
