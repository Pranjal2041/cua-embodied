"""The agent's only interface to the task software: a mouse tool and a keyboard tool.

A headless Chrome page shows the task GUI at 1440 by 1000 pixels. Each tool call
runs up to 16 inputs in order, waits for every visible operation to finish, and
returns one screenshot. Videos produced by the software, such as reference videos
or tracked-mask overlays, are attached to that observation in full.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
from typing import Callable, Sequence
import urllib.request

from ..core import BaseSoftware, Media, Observation, SoftwareStep, ToolDefinition, object_schema


MAX_INPUTS = 16
REPEAT = {"type": "integer", "minimum": 1, "maximum": MAX_INPUTS,
          "description": "Repeat this input sequentially, waiting for the screen after each repetition."}
COORDINATE = {"type": "number", "minimum": 0, "maximum": 1000,
              "description": "Normalized screenshot coordinate: 0 is the left/top edge and 1000 is the right/bottom edge."}
MOUSE_FIELDS = {
    "operation": {"type": "string", "enum": ["move", "click", "double_click", "drag", "scroll"]},
    "x": COORDINATE, "y": COORDINATE,
    "button": {"type": "string", "enum": ["left", "right", "middle"]},
    "to_x": COORDINATE, "to_y": COORDINATE,
    "delta_x": {"type": "number", "description": "Horizontal scroll distance in pixels."},
    "delta_y": {"type": "number", "description": "Vertical scroll distance in pixels."}, "repeat": REPEAT,
}
KEYBOARD_FIELDS = {
    "text": {"type": "string"}, "keys": {"type": "array", "items": {"type": "string"}},
    "repeat": REPEAT,
}


def input_schema(fields, required=()):
    return object_schema({**fields, "sequence": {
        "type": "array", "minItems": 1, "maxItems": MAX_INPUTS,
        "description": "Use sequence alone for successive inputs. At most 16 inputs including repeats.",
        "items": object_schema(fields, required=required),
    }})


class PixelComputer(BaseSoftware):
    name = "computer"
    description = "Operate the task software using only keyboard and mouse. Screenshots are 1440 by 1000 pixels. All mouse positions use normalized coordinates from 0 to 1000 on each axis: (0,0) is the top-left, (500,500) the center, and (1000,1000) the bottom-right. This includes drag endpoints and scroll anchors. Scroll distances are in pixels. Each tool accepts one input, repeat, or a sequence of up to 16 inputs. Inputs execute sequentially, waiting for each visible operation to finish. A fresh screenshot is returned after the batch; intermediate screenshots are not supplied. Click a text field before entering text. Use keys for shortcuts; keys means one simultaneous chord. Use sequence for successive key presses. Observation details and instructions are available through visible buttons. Complete task videos may also be supplied directly in context. Save the final prediction through the visible GUI."
    tool_definitions = (
        ToolDefinition("mouse", "Move, click, drag or scroll. All positions (x/y/to_x/to_y) use 0 to 1000 normalized screenshot coordinates. Scroll deltas are pixel distances. Supply operation/x/y for one input, optional repeat, or sequence alone for successive mouse inputs.", input_schema(MOUSE_FIELDS, ("operation", "x", "y"))),
        ToolDefinition("keyboard", "Insert text into the focused field or press a keys chord, e.g. [\"Control\",\"a\"]. Click the field before entering text. Text entry does not trigger shortcuts; use keys for navigation and shortcuts. Optional repeat repeats that input. Use sequence alone for successive inputs, e.g. [{\"keys\":[\"w\"],\"repeat\":3},{\"keys\":[\"ArrowRight\"]}].", input_schema(KEYBOARD_FIELDS)),
    )


class BrowserBackend:
    """Drives one task GUI in headless Chrome through the Node driver.

    ``video_source`` returns the videos the software produced since the last
    call. They are attached after the screenshot so the agent sees them in full.
    """

    def __init__(self, app: str, base_url: str = "http://127.0.0.1:8808", *,
                 screenshot_dir: Path | None = None,
                 video_source: Callable[[], Sequence[Media]] | None = None):
        self.app, self.base_url = app, base_url.rstrip("/")
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        self.video_source = video_source
        self.process = None
        self.index = 0
        if self.screenshot_dir:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, task):
        driver = Path(__file__).with_name("driver") / "browser_driver.cjs"
        self.process = subprocess.Popen(["node", str(driver), f"{self.base_url}/?app={self.app}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        return self._observation(self._read())

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("GUI browser stopped")
        return json.loads(line)

    def _observation(self, data):
        image = base64.b64decode(data["image"])
        if self.screenshot_dir:
            path = self.screenshot_dir / f"screen_{self.index:04d}.png"
            path.write_bytes(image)
            screenshot = Media(mime_type="image/png", path=path)
        else:
            screenshot = Media(mime_type="image/png", data=image)
        self.index += 1
        videos = tuple(self.video_source()) if self.video_source else ()
        return Observation(text=data.get("error", "Current screen. Use keyboard and mouse."),
                           media=(screenshot, *videos))

    def execute(self, name, arguments):
        if name not in {"mouse", "keyboard"}:
            raise ValueError("Unsupported input")
        # The driver validates the whole batch before any input. Invalid calls
        # return an error and a fresh screenshot so the agent can correct them.
        self.process.stdin.write(json.dumps({"name": name, "arguments": arguments}) + "\n")
        self.process.stdin.flush()
        data = self._read()
        # The terminal flag only tells the runner to stop. No server state enters context.
        with urllib.request.urlopen(f"{self.base_url}/api/state/{self.app}", timeout=5) as r:
            done = bool(json.load(r).get("done"))
        response = {"ok": data.get("ok", False)}
        for key in ("executed_inputs", "requested_inputs"):
            if key in data:
                response[key] = data[key]
        if data.get("error"):
            response["error"] = data["error"]
        return SoftwareStep(response=response, observation=self._observation(data), done=done,
                            output={"prediction_saved": True} if done else None)

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.stdin.write('{"name":"close"}\n')
            self.process.stdin.flush()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
