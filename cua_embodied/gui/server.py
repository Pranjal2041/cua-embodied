"""Local web server for the four task GUIs.

Its HTTP endpoints are internal to the software. The agent never calls them.
It only sees screenshots of the page and sends mouse and keyboard input
through ``cua_embodied.computer``.

    python -m cua_embodied.gui.server --config configs/tasks.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse
import uuid

from ..apps import load_app
from ..config import load_config, load_dotenv
from .session import Session


class MediaStore:
    def __init__(self):
        self.entries, self.paths = {}, {}

    def file(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            return None
        key = str(path)
        if key not in self.paths:
            token = uuid.uuid4().hex
            self.paths[key] = token
            self.entries[token] = (path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        return "/media/"+self.paths[key]

    def blob(self, data, mime):
        token = uuid.uuid4().hex
        self.entries[token] = (data, mime)
        return "/media/"+token


class Application:
    def __init__(self, config, output, *, evaluation=False):
        self.config, self.output = config, output
        # In evaluation runs the page hides app switching and New session,
        # because the text-API arm has no such operations.
        self.evaluation = evaluation
        self.media = MediaStore()
        self.sessions, self.executors = {}, {}
        self.lock = threading.RLock()

    def _session(self, app, directory):
        return Session(load_app(app), self.config[app], directory, self.media,
                       evaluation=self.evaluation)

    def state(self, app):
        with self.lock:
            session = self.sessions.get(app)
            if not session:
                return dict(app=app, status="idle", busy=False, version=0)
            return copy.deepcopy(session.state)

    def dispatch(self, app, action, arguments):
        if app not in self.config:
            raise ValueError("Unknown application")
        with self.lock:
            if app not in self.sessions:
                if action not in {"start", "restart"}:
                    raise ValueError("Open a task first")
                directory = self.output / app / (time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])
                self.sessions[app] = self._session(app, directory)
                self.executors[app] = ThreadPoolExecutor(max_workers=1)
            session = self.sessions[app]
            if session.state["busy"]:
                raise ValueError("An operation is already running")
            if action == "start" and session.backend is not None:
                raise ValueError("This session is already open")
            session.state.update(busy=True, status="loading" if action in {"start","restart"} else "working", error=None,
                message="Loading task data…" if action in {"start","restart"} else "Processing…")

        def work():
            nonlocal session
            try:
                if action == "restart":
                    session.close()
                    previous_version=session.version
                    directory=self.output/app/(time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])
                    session=self._session(app,directory)
                    session.version=previous_version
                    session.state.update(busy=True,status="loading")
                    with self.lock: self.sessions[app]=session
                    session.start()
                elif action == "start":
                    session.start()
                else:
                    session.execute(action, arguments)
                session.state["status"] = "finished" if session.state["done"] else "ready"
            except Exception as e:
                import traceback
                traceback.print_exc()
                session.state.update(status="error", error=str(e), message=str(e))
            finally:
                with self.lock:
                    session.version += 1
                    session.state.update(busy=False, version=session.version)
        self.executors[app].submit(work)
        return self.state(app)


def handler(application):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if self.command != "GET" or self.path.startswith("/media/") is False and self.path.startswith("/api/") is False:
                print(fmt % args, flush=True)

        def json(self, data, status=200):
            content = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlparse(self.path).path
            if path.startswith("/api/state/"):
                return self.json(application.state(path.rsplit("/", 1)[-1]))
            if path.startswith("/media/"):
                entry = application.media.entries.get(path.rsplit("/",1)[-1])
                if entry is None:
                    return self.json({"error":"Unknown media"}, 404)
                data, mime = entry
                size = data.stat().st_size if isinstance(data, Path) else len(data)
                start, end, partial = 0, size-1, False
                value = self.headers.get("Range")
                if value:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
                    if not match:
                        return self.json({"error":"Unsupported range"},416)
                    a,b = match.groups()
                    if a:
                        start, end = int(a), min(int(b) if b else end, end)
                    elif b:
                        start = max(0,size-int(b))
                    else:
                        return self.json({"error":"Empty range"},416)
                    if start > end or start >= size:
                        return self.json({"error":"Range outside media"},416)
                    partial=True
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", mime)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length",str(end-start+1))
                if partial:
                    self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
                self.end_headers()
                try:
                    if isinstance(data, Path):
                        with data.open("rb") as f:
                            f.seek(start)
                            remaining=end-start+1
                            while remaining:
                                chunk=f.read(min(256*1024,remaining))
                                if not chunk: break
                                self.wfile.write(chunk)
                                remaining-=len(chunk)
                    else:
                        self.wfile.write(data[start:end+1])
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                return
            files = {"/":"index.html", "/app.js":"app.js", "/style.css":"style.css"}
            if path not in files:
                return self.json({"error":"Not found"},404)
            file = Path(__file__).parent/"static"/files[path]
            content=file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0])
            self.send_header("Content-Length",str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            # Reject cross-origin mutation; this service is local only.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.json({"error":"Origin rejected"},403)
            try:
                length=int(self.headers.get("Content-Length",0))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request length")
                if self.path != "/api/control":
                    return self.json({"error":"Not found"},404)
                data=json.loads(self.rfile.read(length))
                return self.json(application.dispatch(data["app"],data["action"],data.get("arguments",{})))
            except (KeyError,ValueError,TypeError) as e:
                return self.json({"error":str(e)},400)
    return Handler


def main():
    load_dotenv()
    p = argparse.ArgumentParser(description="Serve the task GUIs for a person to use.")
    p.add_argument("--port", type=int, default=8808)
    p.add_argument("--config", type=Path, default=Path("configs/tasks.json"))
    p.add_argument("--output", type=Path, default=Path("outputs/gui_sessions"))
    args = p.parse_args()
    app = Application(load_config(args.config), args.output.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(app))
    print(f"Task GUIs: http://127.0.0.1:{args.port}/?app=vlnverse", flush=True)
    try:
        server.serve_forever()
    finally:
        for executor in app.executors.values():
            executor.shutdown(wait=True)
        for session in app.sessions.values():
            session.close()


if __name__ == "__main__":
    main()
