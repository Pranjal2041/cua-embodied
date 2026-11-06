"""One open task in the GUI: the backend plus the state the web page renders.

Only allowlisted fields leave this module. Evaluation output, ground truth and
DialNav Guide internals never reach the page.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

from ..apps import App
from ..core import Media, Observation


class Session:
    def __init__(self, app: App, config: Mapping[str, Any], directory: Path, media, *,
                 evaluation: bool = False):
        self.app, self.config, self.directory, self.media = app, config, directory, media
        self.backend = None
        self.task = None
        self.version = 0
        self.initial_observation: Observation | None = None
        self._pending_videos: list[Media] = []
        self.state = dict(app=app.name, status="idle", busy=False, done=False, version=0,
                          objective="", task_id="", image=None, original_video=None,
                          overlay_video=None, objects=[], proposals=[], dialog=[],
                          references=[], markers=[], log=[], frame_index=0, frame_count=0,
                          evaluation=evaluation)

    def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.task, self.backend = self.app.open(self.config, self.directory)
        self.state.update(objective=self.task.objective, task_id=self.task.task_id,
                          instructions=self.app.rules)
        initial = self.backend.reset(self.task)
        videos = self.app.context_videos(self.backend, self.config)
        self._pending_videos.extend(videos)
        # The text-API arm starts from this observation. The GUI arm sees the
        # same videos and a screenshot of the same starting view.
        self.initial_observation = replace(
            initial, media=videos + initial.media + self.app.api_initial_media(self.backend))
        self.observe(initial)
        self.app.prepare_gui(self)
        self.refresh()
        self.log("Task loaded")

    def take_new_videos(self) -> tuple[Media, ...]:
        """Videos produced since the last call. The agent receives them in full."""
        videos, self._pending_videos = tuple(self._pending_videos), []
        return videos

    def observe(self, observation: Observation | None):
        if observation is None:
            return
        self._pending_videos.extend(m for m in observation.media if m.mime_type.startswith("video/"))
        visible = {k: v for k, v in observation.record.items() if k in self.app.observation_fields}
        self.state["observation"] = dict(text=observation.text, fields=visible)
        camera = self.app.camera(visible)
        if camera:
            self.state["camera"] = camera
        for item in observation.media:
            if item.path:
                url = self.media.file(item.path)
            elif item.data:
                url = self.media.blob(item.data, item.mime_type)
            else:
                continue
            if item.mime_type.startswith("image/"):
                self.state["image"] = url
                if "INSPECTED" not in (item.label or ""):
                    self.state["scene_image"] = url
                self.state["display"] = "image"
            elif item.mime_type.startswith("video/"):
                key = "overlay_video" if self.state["original_video"] else "original_video"
                self.state[key] = url
        self.state["message"] = observation.text.split("\n")[0][:250]

    def refresh(self):
        self.app.refresh(self)

    def execute(self, action: str, arguments: Mapping[str, Any]):
        if action not in self.app.actions:
            raise ValueError("Unavailable control")
        if self.state["done"]:
            raise ValueError("This task is finished. Start a new session to make another prediction.")
        self.app.before_action(self, action, arguments)
        result = self.backend.execute(action, arguments)
        self.observe(result.observation)
        response = dict(result.response)
        self.state["feedback"] = {k: v for k, v in response.items() if k in self.app.response_fields}
        if response.get("error"):
            raise ValueError(response["error"])
        self.app.after_action(self, action, arguments, response)
        if result.done:
            self.state.update(done=True, message="Prediction saved locally.")
            (self.directory / "gui_output.json").write_text(json.dumps(result.output, indent=2, default=str))
        self.refresh()
        label = {"move": "Moved", "rotate": "View rotated", "click": "Selection updated",
                 "seek": "View changed", "inspect": "Review ready", "remove": "Object removed",
                 "type": "Guide replied", "stop": "Trajectory saved", "submit": "Prediction saved",
                 "submit_no_target": "No-target prediction saved"}[action]
        self.log(label)
        with (self.directory / "gui_actions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(time=time.time(), action=action, arguments=arguments,
                                    done=result.done)) + "\n")

    def log(self, text):
        self.state["log"].append(dict(time=time.strftime("%H:%M:%S"), text=text))
        self.state["log"] = self.state["log"][-12:]

    def close(self):
        if self.backend:
            self.backend.close()
