"""MeViSv2 text track: segment the objects described by a motion expression."""
from __future__ import annotations

from pathlib import Path

from ...core import Media, Task
from ..base import App
from .api import MeViSAPI


RULES = ("Select every and only object matching the expression. The expression may "
    "refer to one object, multiple objects or no object. Select each whole target object. "
    "Track the selections and review the complete tracked-mask overlay after the latest "
    "selection edit. Save only after the tracks match the expression throughout the video.")


class MeViS(App):
    name = "mevis"
    title = "MeViSv2 Text"
    rules = RULES
    actions = frozenset({"seek", "click", "remove", "inspect", "submit", "submit_no_target"})
    observation_fields = frozenset({"video_id", "frame_count", "fps", "coordinate_system",
                                    "active_object_ids", "preview_generation", "preview_current"})
    response_fields = frozenset({"ok", "error", "action", "frame_index", "frame_count",
        "coordinate_system", "object_id", "click_x", "click_y", "positive_points",
        "negative_points", "returned_object_ids", "returned_mask_pixels", "active_object_ids",
        "remaining_object_ids", "preview_generation", "object_ids", "nonempty_frames",
        "submitted", "no_target", "source_preview_generation", "reason", "video_id",
        "expression_id"})
    software = MeViSAPI

    def open(self, config, output_dir):
        from .video import load_task
        from .backend import MeViSBackend
        data_root = Path(config["data_root"])
        expression, _ = load_task(data_root, config["video_id"], config["expression_id"])
        backend = MeViSBackend(
            video_id=config["video_id"], expression_id=config["expression_id"],
            output_dir=Path(output_dir), data_root=data_root,
            checkpoint=Path(config["sam3_checkpoint"]),
            video_cache=Path(config["video_cache"]) if config.get("video_cache") else None)
        return Task(task_id=f'{config["video_id"]}_expr{config["expression_id"]}',
                    objective=expression), backend

    def api_initial_media(self, backend):
        # The GUI opens on frame zero. The API arm starts from the same frame.
        return (Media(mime_type="image/jpeg", path=backend.interface.frame_path(0),
                      label="SELECTED FRAME 0:"),)

    def prepare_gui(self, session):
        session.observe(session.backend.execute("seek", {"frame_index": 0}).observation)
        session.state["frame_count"] = len(session.backend.frame_names)
        session.state["fps"] = session.backend.fps

    def refresh(self, session):
        interface = session.backend.interface
        session.state["objects"] = sorted(interface.object_ids)
        session.state["preview_current"] = interface.preview_is_current
        session.state["preview_generation"] = interface.preview_generation

    def before_action(self, session, action, arguments):
        if action == "submit" and not session.backend.interface.preview_is_current:
            raise ValueError("Track and review the current selections before saving.")

    def after_action(self, session, action, arguments, response):
        if action in ("seek", "click"):
            session.state["frame_index"] = int(arguments["frame_index"])
        if action == "inspect":
            session.state["display"] = "overlay"


APP = MeViS()
