"""MeViSv2 task loading and complete-video encoding."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


DEFAULT_FPS = 5.0


def load_task(
    data_root: Path, video_id: str, expression_id: str
) -> tuple[str, list[str]]:
    metadata = json.loads(
        (data_root / "meta_expressions_v2.json").read_text(encoding="utf-8")
    )
    try:
        video = metadata["videos"][video_id]
    except KeyError as exc:
        raise ValueError(f"Unknown video_id {video_id}") from exc
    try:
        expression = video["expressions"][str(expression_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown expression_id {expression_id} for {video_id}"
        ) from exc
    # Intentionally return only public inference inputs. Annotation IDs and
    # object IDs in public validation metadata are not carried into the agent.
    return str(expression["exp"]), [str(value) for value in video["frames"]]


def source_frame_path(frame_dir: Path, frame_name: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = (frame_dir / frame_name).with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(frame_dir / frame_name)


def build_complete_video(
    frame_dir: Path, frame_names: list[str], destination: Path, fps: float
) -> Path:
    """Encode every official frame, in order, without cropping any content.

    H.264's broadly compatible yuv420p format requires even frame dimensions.
    Some official MeViS videos have an odd width or height, so edge-pad only
    that final row/column when necessary. The benchmark images and masks remain
    untouched.
    """
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.mp4")
    writer = imageio.get_writer(
        temporary,
        fps=fps,
        codec="libx264",
        quality=10,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for frame_name in frame_names:
            source = source_frame_path(frame_dir, frame_name)
            frame = np.asarray(Image.open(source).convert("RGB"))
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
    temporary.replace(destination)
    return destination
