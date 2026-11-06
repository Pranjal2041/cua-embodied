"""Small stateful SAM3 interface for MeViSv2 text-guided video segmentation.

The interface deliberately exposes only the operations needed by the CUA:

* add an object with a positive click on any official frame;
* refine an existing object with positive/negative clicks;
* remove an object;
* propagate the current selections over the complete video; and
* materialize benchmark-ready union masks and a complete overlay video.

Semantic interpretation remains Gemini's job. SAM3 never receives the referring
expression or an object-name text prompt.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    from sam3.model_builder import build_sam3_predictor
except ImportError:  # Compatibility with pre-SAM3.1 installations.
    from sam3.model_builder import build_sam3_video_predictor

    def build_sam3_predictor(**kwargs: Any) -> Any:
        version = kwargs.pop("version", "sam3")
        if version != "sam3":
            raise RuntimeError(
                "This SAM checkout does not provide SAM3.1 support"
            )
        return build_sam3_video_predictor(**kwargs)


COLORS = (
    (45, 232, 210),
    (255, 103, 126),
    (255, 198, 74),
    (149, 117, 255),
    (90, 170, 255),
    (125, 224, 116),
)


@dataclass(frozen=True)
class Click:
    x: float
    y: float
    label: int


def _font(size: int = 30) -> ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/consolab.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _configure_connected_components_backend() -> None:
    """Avoid SAM3's unreliable Triton connected-components fallback.

    SAM3 prefers ``cc_torch`` when it is installed, otherwise it falls back to
    Triton on CUDA tensors.  The latter raises ``CUDA_ERROR_INVALID_VALUE`` on
    some otherwise-supported videos/GPUs.  SAM3 also ships an official CPU
    implementation backed by scikit-image; use that deterministic backend by
    default for this small post-processing operation.
    """

    backend = os.environ.get(
        "MEVIS_SAM3_CONNECTED_COMPONENTS_BACKEND", "cpu"
    ).strip().lower()
    if backend == "auto":
        return
    if backend != "cpu":
        raise ValueError(
            "MEVIS_SAM3_CONNECTED_COMPONENTS_BACKEND must be 'cpu' or 'auto'"
        )

    connected_components_module = importlib.import_module(
        "sam3.perflib.connected_components"
    )
    try:
        importlib.import_module("skimage.measure")
    except ImportError as error:
        raise RuntimeError(
            "The CPU connected-components backend requires scikit-image"
        ) from error
    connected_components_module.connected_components = (
        connected_components_module.connected_components_cpu
    )


class Sam3VideoInterface:
    """Stateful wrapper around the official SAM3 video predictor."""

    def __init__(
        self,
        *,
        frame_dir: Path,
        frame_names: list[str],
        output_dir: Path,
        checkpoint: Path,
        fps: float,
        sam3_version: str = "sam3",
    ) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not frame_names:
            raise ValueError("The video has no frames")
        if sam3_version not in {"sam3", "sam3.1"}:
            raise ValueError(
                f"Unsupported SAM version: {sam3_version!r}"
            )

        self.frame_dir = frame_dir
        self.frame_names = frame_names
        self.output_dir = output_dir
        self.checkpoint = checkpoint
        self.sam3_version = sam3_version
        self.fps = float(fps)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preview_root = self.output_dir / "tracking_previews"
        self.preview_root.mkdir(exist_ok=True)

        first = self.frame_path(0)
        with Image.open(first) as image:
            self.width, self.height = image.size

        _configure_connected_components_backend()
        predictor_kwargs: dict[str, Any] = {
            "checkpoint_path": str(checkpoint),
            "version": sam3_version,
            "async_loading_frames": False,
        }
        if sam3_version == "sam3":
            predictor_kwargs["gpus_to_use"] = [0]
        else:
            # Daytona's H100 image does not provide the optional
            # flash_attn_interface module. SAM3.1 officially supports the
            # equivalent PyTorch attention backend through this flag.
            predictor_kwargs["use_fa3"] = False
        self.predictor = build_sam3_predictor(**predictor_kwargs)
        if sam3_version == "sam3.1":
            # SAM3.1's current public base predictor forwards the legacy
            # offload_state_to_cpu argument to a multiplex init_state method
            # that does not accept it. Keep the public request API and discard
            # only that unused compatibility argument.
            multiplex_init_state = self.predictor.model.init_state

            def compatible_init_state(
                *args: Any,
                offload_state_to_cpu: bool = False,
                video_loader_type: str | None = None,
                **kwargs: Any,
            ) -> Any:
                del offload_state_to_cpu
                if video_loader_type == "torchcodec":
                    kwargs["use_torchcodec"] = True
                elif video_loader_type == "cv2":
                    kwargs["use_cv2"] = True
                return multiplex_init_state(*args, **kwargs)

            self.predictor.model.init_state = compatible_init_state
            multiplex_model = self.predictor.model
            # Point-selected benchmark objects should not be suppressed while
            # waiting for an open-vocabulary detector confirmation. Offload
            # retained frame outputs so long validation videos fit on H100.
            multiplex_model.masklet_confirmation_enable = False
            multiplex_model.hotstart_delay = 0
            multiplex_model.offload_output_to_cpu_for_eval = True
            if hasattr(multiplex_model.detector, "offload_outputs_to_cpu_for_eval"):
                multiplex_model.detector.offload_outputs_to_cpu_for_eval = True
            tracker_model = getattr(multiplex_model.tracker, "model", None)
            if tracker_model is not None:
                tracker_model.offload_output_to_cpu_for_eval = False
            self._sam31_tracker_model = tracker_model
        else:
            self._sam31_tracker_model = None
        response = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(frame_dir),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": False,
            }
        )
        self.session_id = str(response["session_id"])
        if sam3_version == "sam3.1":
            # The public multiplex interactivity path merges refined masks
            # into cached_frame_outputs. Text-first demos populate every
            # frame through detection; this point-only interface must create
            # the corresponding empty entries itself.
            inference_state = self.predictor._get_session(
                self.session_id
            )["state"]
            cached_outputs = inference_state["cached_frame_outputs"]
            for frame_index in range(len(frame_names)):
                cached_outputs.setdefault(frame_index, {})
        self.object_ids: set[int] = set()
        self.object_prompt_frames: dict[int, set[int]] = {}
        self.next_object_id = 1
        self.prompt_history: list[dict[str, Any]] = []
        self.preview_generation = 0
        self.preview_is_current = False
        self.latest_masks_dir: Path | None = None
        self.latest_overlay_video: Path | None = None
        self.closed = False

    def frame_path(self, frame_index: int) -> Path:
        self._validate_frame(frame_index)
        base = self.frame_dir / self.frame_names[frame_index]
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = base.with_suffix(suffix)
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No image found for frame {frame_index}: {base}")

    def _validate_frame(self, frame_index: int) -> None:
        if frame_index < 0 or frame_index >= len(self.frame_names):
            raise ValueError(
                f"frame_index must be between 0 and {len(self.frame_names) - 1}"
            )

    @staticmethod
    def _validate_coordinate(value: float, name: str) -> float:
        value = float(value)
        if value < 0 or value > 1000:
            raise ValueError(f"{name} must be in normalized range 0..1000")
        return value / 1000.0

    @staticmethod
    def _outputs(outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        object_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64)
        masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        return object_ids, masks

    def _render_prompt_result(
        self,
        *,
        frame_index: int,
        outputs: dict[str, Any],
        clicks: Iterable[Click],
        label: str,
    ) -> Path:
        object_ids, masks = self._outputs(outputs)
        image = Image.open(self.frame_path(frame_index)).convert("RGB")
        base = np.asarray(image, dtype=np.float32)
        blended = base.copy()
        for object_id, mask in zip(object_ids.tolist(), masks):
            color = np.asarray(COLORS[(int(object_id) - 1) % len(COLORS)])
            blended[mask] = 0.35 * base[mask] + 0.65 * color
        result = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(result)
        font = _font()
        small = _font(22)
        draw.rectangle((0, 0, min(self.width, 980), 58), fill=(4, 10, 18))
        draw.text(
            (18, 12),
            f"{label}  ·  FRAME {frame_index:05d}",
            fill=(235, 242, 249),
            font=font,
        )
        for click in clicks:
            x = click.x / 1000.0 * self.width
            y = click.y / 1000.0 * self.height
            color = (52, 235, 209) if click.label == 1 else (255, 86, 112)
            radius = 12
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(255, 255, 255),
                width=3,
            )
            draw.text(
                (x + 16, y - 12),
                "+" if click.label == 1 else "−",
                fill=(255, 255, 255),
                font=small,
            )
        destination = (
            self.output_dir
            / "prompt_frames"
            / f"prompt_{len(self.prompt_history):03d}_frame_{frame_index:05d}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.save(destination)
        return destination

    def select_object(
        self, *, frame_index: int, click_x: float, click_y: float
    ) -> tuple[dict[str, Any], Path]:
        self._validate_frame(frame_index)
        x = self._validate_coordinate(click_x, "click_x")
        y = self._validate_coordinate(click_y, "click_y")
        object_id = self.next_object_id
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": frame_index,
                "points": [[x, y]],
                "point_labels": [1],
                "obj_id": object_id,
                "rel_coordinates": True,
                "clear_old_points": True,
            }
        )
        outputs = response["outputs"]
        returned_ids, returned_masks = self._outputs(outputs)
        if object_id not in set(returned_ids.tolist()):
            raise RuntimeError(
                f"SAM3 did not return newly selected object {object_id}; "
                f"returned {returned_ids.tolist()}"
            )
        self.object_ids.add(object_id)
        self.object_prompt_frames[object_id] = {frame_index}
        self.next_object_id += 1
        self.preview_is_current = False
        event = {
            "action": "select_object",
            "object_id": object_id,
            "frame_index": frame_index,
            "click_x": float(click_x),
            "click_y": float(click_y),
            "returned_object_ids": returned_ids.tolist(),
            "returned_mask_pixels": [
                int(mask.sum()) for mask in returned_masks
            ],
        }
        self.prompt_history.append(event)
        preview = self._render_prompt_result(
            frame_index=frame_index,
            outputs=outputs,
            clicks=[Click(float(click_x), float(click_y), 1)],
            label=f"ADDED OBJECT {object_id}",
        )
        return event, preview

    def refine_object(
        self,
        *,
        object_id: int,
        frame_index: int,
        positive_points: list[dict[str, float]],
        negative_points: list[dict[str, float]],
    ) -> tuple[dict[str, Any], Path]:
        if int(object_id) not in self.object_ids:
            raise ValueError(f"Unknown object_id {object_id}")
        self._validate_frame(frame_index)
        clicks: list[Click] = []
        for point in positive_points:
            clicks.append(Click(float(point["x"]), float(point["y"]), 1))
        for point in negative_points:
            clicks.append(Click(float(point["x"]), float(point["y"]), 0))
        if not clicks:
            raise ValueError("refine_object requires at least one click")
        points = [
            [
                self._validate_coordinate(click.x, "point.x"),
                self._validate_coordinate(click.y, "point.y"),
            ]
            for click in clicks
        ]
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": frame_index,
                "points": points,
                "point_labels": [click.label for click in clicks],
                "obj_id": int(object_id),
                "rel_coordinates": True,
                "clear_old_points": False,
            }
        )
        outputs = response["outputs"]
        returned_ids, returned_masks = self._outputs(outputs)
        self.object_prompt_frames[int(object_id)].add(frame_index)
        self.preview_is_current = False
        event = {
            "action": "refine_object",
            "object_id": int(object_id),
            "frame_index": frame_index,
            "positive_points": positive_points,
            "negative_points": negative_points,
            "returned_object_ids": returned_ids.tolist(),
            "returned_mask_pixels": [
                int(mask.sum()) for mask in returned_masks
            ],
        }
        self.prompt_history.append(event)
        preview = self._render_prompt_result(
            frame_index=frame_index,
            outputs=outputs,
            clicks=clicks,
            label=f"REFINED OBJECT {object_id}",
        )
        return event, preview

    def remove_object(self, *, object_id: int) -> dict[str, Any]:
        object_id = int(object_id)
        if object_id not in self.object_ids:
            raise ValueError(f"Unknown object_id {object_id}")
        self.predictor.handle_request(
            {
                "type": "remove_object",
                "session_id": self.session_id,
                "obj_id": object_id,
            }
        )
        self.object_ids.remove(object_id)
        self.object_prompt_frames.pop(object_id)
        self.preview_is_current = False
        event = {
            "action": "remove_object",
            "object_id": object_id,
            "remaining_object_ids": sorted(self.object_ids),
        }
        self.prompt_history.append(event)
        return event

    def _render_overlay_video(self, masks_dir: Path, destination: Path) -> None:
        writer = imageio.get_writer(
            destination,
            fps=self.fps,
            codec="libx264",
            quality=9,
            macro_block_size=None,
            ffmpeg_log_level="error",
        )
        font = _font(28)
        try:
            for frame_index in range(len(self.frame_names)):
                image = Image.open(self.frame_path(frame_index)).convert("RGB")
                base = np.asarray(image, dtype=np.float32)
                mask = (
                    np.asarray(
                        Image.open(masks_dir / f"{self.frame_names[frame_index]}.png")
                    )
                    > 0
                )
                blended = base.copy()
                blended[mask] = (
                    0.34 * base[mask]
                    + 0.66 * np.asarray((40, 235, 205), dtype=np.float32)
                )
                rendered = Image.fromarray(
                    np.clip(blended, 0, 255).astype(np.uint8)
                )
                draw = ImageDraw.Draw(rendered)
                draw.rectangle((0, 0, 650, 54), fill=(4, 10, 18))
                draw.text(
                    (17, 10),
                    (
                        f"TRACKED TARGET  ·  FRAME {frame_index:05d}/"
                        f"{len(self.frame_names) - 1:05d}"
                    ),
                    fill=(62, 239, 212),
                    font=font,
                )
                frame = np.asarray(rendered)
                pad_height = frame.shape[0] % 2
                pad_width = frame.shape[1] % 2
                if pad_height or pad_width:
                    frame = np.pad(
                        frame,
                        ((0, pad_height), (0, pad_width), (0, 0)),
                        mode="edge",
                    )
                writer.append_data(frame)
        finally:
            writer.close()

    def preview_tracking(self) -> tuple[dict[str, Any], Path]:
        if not self.object_ids:
            raise RuntimeError("Select at least one object before tracking")
        self.preview_generation += 1
        root = self.preview_root / f"preview_{self.preview_generation:02d}"
        masks_dir = root / "masks"
        masks_dir.mkdir(parents=True, exist_ok=False)
        frame_summaries: dict[int, dict[str, Any]] = {}
        active_prompt_frames = sorted(
            {
                frame_index
                for object_id in self.object_ids
                for frame_index in self.object_prompt_frames[object_id]
            }
        )
        if not active_prompt_frames:
            raise RuntimeError("Active objects have no recorded prompt frames")
        propagation_start_frame = active_prompt_frames[0]

        for response in self.predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": "both",
                # The official predictor executes "both" as a forward pass followed
                # by a backward pass. The forward pass consumes its transient
                # previous_stages_out state, so leaving this unset can make the
                # backward pass incorrectly report that there was no prompt.
                # Supplying the real active seed frame is the supported way to make
                # both passes independent of that transient state.
                "start_frame_index": propagation_start_frame,
                "output_prob_thresh": 0.5,
            }
        ):
            frame_index = int(response["frame_index"])
            object_ids, masks = self._outputs(response["outputs"])
            union = (
                np.any(masks, axis=0)
                if masks.size
                else np.zeros((self.height, self.width), dtype=bool)
            )
            Image.fromarray(union.astype(np.uint8) * 255).save(
                masks_dir / f"{self.frame_names[frame_index]}.png"
            )
            frame_summaries[frame_index] = {
                "object_ids": object_ids.tolist(),
                "object_mask_pixels": [int(mask.sum()) for mask in masks],
                "union_pixels": int(union.sum()),
            }

        for frame_index, frame_name in enumerate(self.frame_names):
            destination = masks_dir / f"{frame_name}.png"
            if not destination.is_file():
                Image.fromarray(
                    np.zeros((self.height, self.width), dtype=np.uint8)
                ).save(destination)
                frame_summaries[frame_index] = {
                    "object_ids": [],
                    "object_mask_pixels": [],
                    "union_pixels": 0,
                }

        overlay = root / "complete_tracked_overlay.mp4"
        self._render_overlay_video(masks_dir, overlay)
        summary = {
            "preview_generation": self.preview_generation,
            "object_ids": sorted(self.object_ids),
            "active_prompt_frames": active_prompt_frames,
            "propagation_start_frame": propagation_start_frame,
            "frame_count": len(self.frame_names),
            "nonempty_frames": sum(
                1
                for value in frame_summaries.values()
                if value["union_pixels"] > 0
            ),
            "masks_dir": str(masks_dir.resolve()),
            "overlay_video": str(overlay.resolve()),
            "per_frame": {
                str(index): frame_summaries[index]
                for index in sorted(frame_summaries)
            },
        }
        _json(root / "summary.json", summary)
        self.latest_masks_dir = masks_dir
        self.latest_overlay_video = overlay
        self.preview_is_current = True
        self.prompt_history.append(
            {
                "action": "preview_tracking",
                "preview_generation": self.preview_generation,
                "object_ids": sorted(self.object_ids),
            }
        )
        return summary, overlay

    def materialize_submission(
        self, destination: Path, *, no_target: bool = False
    ) -> dict[str, Any]:
        # Check every precondition before creating the destination. Otherwise a
        # rejected early submit strands an empty directory and every later,
        # valid submit deterministically fails with FileExistsError.
        if (
            not no_target
            and (not self.preview_is_current or self.latest_masks_dir is None)
        ):
            raise RuntimeError(
                "Run preview_tracking after the most recent edit before submitting"
            )
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True)
        try:
            if no_target:
                for frame_name in self.frame_names:
                    Image.fromarray(
                        np.zeros((self.height, self.width), dtype=np.uint8)
                    ).save(destination / f"{frame_name}.png")
            else:
                for frame_name in self.frame_names:
                    shutil.copy2(
                        self.latest_masks_dir / f"{frame_name}.png",
                        destination / f"{frame_name}.png",
                    )
            record = {
                "no_target": bool(no_target),
                "frame_count": len(self.frame_names),
                "object_ids": [] if no_target else sorted(self.object_ids),
                "source_preview_generation": (
                    None if no_target else self.preview_generation
                ),
                "destination": str(destination.resolve()),
            }
            _json(destination / "prediction_manifest.json", record)
            return record
        except Exception:
            # This invocation created the directory, so a partial write must
            # not poison a later retry.
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": self.session_id,
                    "run_gc_collect": True,
                }
            )
        finally:
            self.predictor.shutdown()
            self.closed = True

    def __enter__(self) -> "Sam3VideoInterface":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
