"""Run the computer-use agent on one task.

    python -m cua_embodied.run --app opensun3d --config configs/tasks.json --output outputs/runs/opensun3d

By default the agent operates the task GUI with mouse and keyboard only. Use
``--interface api`` for the text-API arm of the GUI versus API comparison,
where the same backend operations are exposed as typed tools instead.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

from .agent import GeminiAgent
from .apps import NAMES, load_app
from .computer import BrowserBackend, PixelComputer
from .config import load_config, load_dotenv
from .gui import server as gui_server
from .gui.session import Session
from .runner import Runner


class ApiBackend:
    """Exposes a started session's backend operations directly as tools."""

    def __init__(self, session: Session):
        self.session = session

    def reset(self, task):
        if task.task_id != self.session.task.task_id:
            raise ValueError("Task mismatch")
        return self.session.initial_observation

    def execute(self, name, arguments):
        return self.session.backend.execute(name, arguments)

    def close(self):
        self.session.close()


def run(args) -> dict:
    load_dotenv()
    config = load_config(args.config)[args.app]
    if args.app == "dialnav":
        # The Guide uses the agent's model and seed unless the config sets them.
        config = {"guide_model": args.model, "guide_seed": args.seed, **config}
    app = load_app(args.app)
    directory = args.output.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    agent = GeminiAgent(model=args.model, temperature=args.temperature, seed=args.seed,
                        thinking_level="high", image_context_mode="sliding",
                        image_context_window=20, image_context_drop=10)
    httpd = application = session = software = None
    try:
        if args.interface == "api":
            session = Session(app, config, directory / "backend", gui_server.MediaStore(),
                              evaluation=True)
            session.start()
            software = app.software(ApiBackend(session))
        else:
            application = gui_server.Application({args.app: config}, directory / "sessions",
                                                 evaluation=True)
            application.dispatch(args.app, "start", {})
            while application.state(args.app).get("busy"):
                time.sleep(0.25)
            state = application.state(args.app)
            if state.get("error"):
                raise RuntimeError(state["error"])
            session = application.sessions[args.app]
            httpd = ThreadingHTTPServer(("127.0.0.1", args.port), gui_server.handler(application))
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            software = PixelComputer(BrowserBackend(
                args.app, f"http://127.0.0.1:{args.port}",
                screenshot_dir=directory / "screenshots",
                video_source=lambda: application.sessions[args.app].take_new_videos()))
        # The task instructions travel with the objective in the agent's first message.
        task = replace(session.task, inputs={"task_rules": app.rules})
        result = Runner(agent, software, max_turns=args.max_turns, output_dir=directory).run(task)
        output = result.output
        if args.interface == "gui" and result.status == "completed":
            output = json.loads((session.directory / "gui_output.json").read_text())
        summary = dict(app=args.app, interface=args.interface, model=args.model, seed=args.seed,
                       task_id=task.task_id, status=result.status, turns=len(result.turns),
                       session_directory=str(session.directory))
        if result.status == "completed":
            (directory / "prediction.json").write_text(json.dumps(output, indent=2, default=str))
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        if httpd:
            httpd.shutdown()
            httpd.server_close()
        if application:
            for executor in application.executors.values():
                executor.shutdown(wait=True)
            for item in application.sessions.values():
                item.close()
        elif session and software is None:
            session.close()  # Otherwise the runner already closed it.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--app", choices=NAMES, required=True)
    parser.add_argument("--interface", choices=["gui", "api"], default="gui")
    parser.add_argument("--config", type=Path, default=Path("configs/tasks.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=192)
    parser.add_argument("--port", type=int, default=8810, help="Port for the GUI during the run")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
