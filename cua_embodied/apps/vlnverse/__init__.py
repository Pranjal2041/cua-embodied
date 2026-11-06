"""VLNVerse: navigate to a destination described in language."""
from __future__ import annotations

import json
from pathlib import Path

from ...core import Task
from ..base import App
from .api import VLNVerseAPI


RULES = ("Navigate to the object described in the objective and stop beside it. "
    "This task requires navigation only, even if the instruction describes manipulating "
    "an object. Explore using the images and maintain spatial orientation. Get physically "
    "close to the target before stopping. Camera collision is disabled.")


class VLNVerse(App):
    name = "vlnverse"
    title = "VLNVerse"
    rules = RULES
    actions = frozenset({"move", "rotate", "stop"})
    observation_fields = frozenset({"position_m", "yaw_deg", "pitch_deg", "bumped_wall",
                                    "action_result", "render"})
    response_fields = observation_fields | {"requested_distance_m", "achieved_distance_m",
                                            "stopped", "reason", "error"}
    software = VLNVerseAPI

    def open(self, config, output_dir):
        from .backend import VLNVerseBackend
        manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
        episode = manifest["episodes"][config["episode_index"]]
        backend = VLNVerseBackend(bridge_url=config["bridge_url"], episode=episode,
                                  output_dir=output_dir, collision=config.get("collision", False))
        return Task(task_id=str(episode["episode_id"]), objective=episode["instruction"]), backend

    def prepare_gui(self, session):
        session.state["collision"] = bool(session.config.get("collision", False))

    def camera(self, fields):
        return {k: fields[k] for k in ("position_m", "yaw_deg", "pitch_deg") if k in fields}


APP = VLNVerse()
