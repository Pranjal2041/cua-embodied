"""DialNav: reach a destination through dialogue with a Guide."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ...core import Task
from ..base import App
from .api import DialNavAPI


RULES = ("Navigate to the destination region specified by the Guide. The panorama's "
    "left and right edges join. Numbered circles indicate legal moves. The Guide cannot "
    "see your panorama. Describe visible rooms, objects, layout and recent motion in "
    "ordinary language when asking for directions. Do not transmit local move labels, "
    "coordinates or hidden identifiers. Stop when you reach the destination.")


class DialNav(App):
    name = "dialnav"
    title = "DialNav"
    rules = RULES
    actions = frozenset({"move", "type", "stop"})
    observation_fields = frozenset({"event", "navigation_step", "valid_markers"})
    response_fields = frozenset({"moved", "message", "error", "valid_markers", "accepted",
                                 "answer", "stopped", "reason"})
    software = DialNavAPI

    def open(self, config, output_dir):
        from .backend import DialNavBackend
        index = config["episode_index"]
        episode = json.loads(Path(config["split"]).read_text(encoding="utf-8"))[index]
        backend = DialNavBackend(
            episode=episode, dataset_index=index, output_dir=output_dir,
            guide_model=config["guide_model"], guide_seed=config.get("guide_seed", 0),
            connectivity_dir=config["connectivity_dir"], rgb_dir=config["rgb_dir"])
        return Task(task_id=str(episode["instr_id"]), objective=str(episode["target"])), backend

    def refresh(self, session):
        backend = session.backend
        # Pixel positions of the numbered exit circles drawn on the panorama.
        with Image.open(backend.current_frame) as image:
            width, height = image.size
        session.state["markers"] = [
            dict(id=c.action, x=((c.yaw_deg + 180) % 360) / 360,
                 y=(58 + (0.5 - c.pitch_deg / 180) * (height - 108)) / height)
            for c in backend.current_candidates]
        session.state["moves"] = len(backend.env.path) - 1

    def after_action(self, session, action, arguments, response):
        if action == "type":
            session.state["dialog"].append(dict(role="Navigator", text=arguments["text"]))
            session.state["dialog"].append(dict(role="Guide", text=response.get("answer", "")))


APP = DialNav()
