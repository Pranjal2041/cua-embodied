"""OpenSUN3D: segment the 3D interaction region named by a SceneFun3D task."""
from __future__ import annotations

from pathlib import Path

from ...core import Media, Task
from ..base import App
from .api import OpenSUN3DAPI
from .rules import SCENEFUN3D_TASK_RULES


def _optional_path(value):
    return Path(value) if value else None


class OpenSUN3D(App):
    name = "opensun3d"
    title = "OpenSUN3D"
    rules = SCENEFUN3D_TASK_RULES
    actions = frozenset({"seek", "move", "rotate", "click", "inspect", "submit"})
    observation_fields = frozenset({"scene_id", "camera", "registered_pose_streams",
        "proposal_ids", "inspected_proposal_ids", "proposal_id", "point_count", "threshold"})
    response_fields = frozenset({"ok", "error", "valid_video_ids", "video_id",
        "requested_seconds", "resolved_seconds", "direction", "distance_m", "yaw_degrees",
        "pitch_degrees", "proposal_id", "source", "point_count", "threshold",
        "mean_foreground_probability", "affordance_class", "views", "valid_proposal_ids",
        "clicked_faro_vertex", "proposals", "instruction", "unknown", "uninspected",
        "unreviewed_siblings", "submitted", "proposal_count"})
    software = OpenSUN3DAPI

    def open(self, config, output_dir):
        from .backend import OpenSUN3DBackend
        pinpoint = config["pinpoint"]
        backend = OpenSUN3DBackend(
            scene_id=config["scene_id"], description_id=config["description_id"],
            data_root=Path(config["data_root"]), output_dir=Path(output_dir),
            pinpoint_checkpoint=Path(pinpoint["checkpoint"]), pinpoint_repo=Path(pinpoint["repo"]),
            pinpoint_python=pinpoint["python"], wsl_distro=pinpoint.get("wsl_distro"),
            volt_input_cache=_optional_path(config.get("volt_input_cache")),
            volt_prediction_cache=_optional_path(config.get("volt_prediction_cache")),
            point_radius=config.get("point_radius", 3))
        task = Task(task_id=f'{config["scene_id"]}_{config["description_id"]}',
                    objective=config["objective"])
        return task, backend

    def _reference_paths(self, backend, config):
        directory = Path(config["reference_video_dir"])
        return {video_id: directory / f"{video_id}.mp4" for video_id in sorted(backend.durations)}

    def context_videos(self, backend, config):
        videos = []
        for video_id, path in self._reference_paths(backend, config).items():
            if not path.is_file():
                raise FileNotFoundError(path)
            videos.append(Media(mime_type="video/mp4", path=path,
                                label=f"COMPLETE REFERENCE VIDEO {video_id}:"))
        return tuple(videos)

    def prepare_gui(self, session):
        paths = self._reference_paths(session.backend, session.config)
        session.state["references"] = [
            dict(id=video_id, seconds=session.backend.durations[video_id],
                 url=session.media.file(path))
            for video_id, path in paths.items()]

    def camera(self, fields):
        return fields.get("camera")

    def refresh(self, session):
        session.state["proposals"] = [
            dict(id=p.proposal_id, source=p.source, count=p.point_count, inspected=p.inspected,
                 group=p.group_id, threshold=p.threshold,
                 mean_foreground_probability=p.mean_probability,
                 affordance_class=p.affordance_class)
            for p in session.backend.proposals.values()]

    def after_action(self, session, action, arguments, response):
        session.state["reviewing"] = arguments.get("proposal_id") if action == "inspect" else None


APP = OpenSUN3D()
