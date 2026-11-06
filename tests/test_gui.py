"""GUI contract and HTTP tests without model calls or GPUs."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from cua_embodied.apps import load_app
from cua_embodied.computer import PixelComputer
from cua_embodied.core import Observation, SoftwareStep
from cua_embodied.gui.server import Application, MediaStore, handler
from cua_embodied.gui.session import Session


def session(app, root, config=None):
    return Session(load_app(app), config or {}, Path(root), MediaStore())


class TestContract(unittest.TestCase):
    def test_exactly_two_tools(self):
        self.assertEqual([t.name for t in PixelComputer.tool_definitions], ["mouse", "keyboard"])
        for t in PixelComputer.tool_definitions:
            self.assertFalse(t.parameters["additionalProperties"])

    def test_gui_controls_match_api_tools(self):
        inventories = {"vlnverse": {"move", "rotate", "stop"},
                       "dialnav": {"move", "type", "stop"},
                       "opensun3d": {"seek", "move", "rotate", "click", "inspect", "submit"},
                       "mevis": {"seek", "click", "remove", "inspect", "submit", "submit_no_target"}}
        for name, expected in inventories.items():
            app = load_app(name)
            self.assertEqual({t.name for t in app.software.tool_definitions}, expected)
            self.assertEqual(set(app.actions), expected)

    def test_no_hidden_records_in_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = session("dialnav", tmp)
            s.observe(Observation(text="Visible observation",
                                  record={"goal": "secret", "atlas": "hidden", "nav_trajectory": [1, 2]}))
            encoded = json.dumps(s.state)
            for forbidden in ("secret", "hidden", "nav_trajectory", "atlas"):
                self.assertNotIn(forbidden, encoded)

    def test_stale_tracking_rejects_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = session("mevis", tmp)
            s.backend = SimpleNamespace(interface=SimpleNamespace(preview_is_current=False))
            with self.assertRaisesRegex(ValueError, "Track and review"):
                s.execute("submit", {"object_ids": [1]})

    def test_stopped_session_rejects_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = session("vlnverse", tmp)
            s.state["done"] = True
            with self.assertRaisesRegex(ValueError, "finished"):
                s.execute("move", {})

    def test_backend_output_is_saved_without_exposing_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = session("vlnverse", tmp)
            s.backend = SimpleNamespace(execute=lambda *a: SoftwareStep(
                response={"stopped": True}, done=True, output={"evaluator_secret": 123}))
            s.execute("stop", {"reason": "QA"})
            self.assertTrue(s.state["done"])
            self.assertNotIn("evaluator_secret", json.dumps(s.state))
            self.assertEqual(json.loads((Path(tmp) / "gui_output.json").read_text())["evaluator_secret"], 123)


class TestHTTP(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = Application({}, Path(self.tmp.name))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def test_media_byte_range(self):
        media = self.app.media.blob(b"0123456789", "video/mp4")
        request = urllib.request.Request(self.url + media, headers={"Range": "bytes=3-5"})
        with urllib.request.urlopen(request) as r:
            self.assertEqual(r.status, 206)
            self.assertEqual(r.read(), b"345")
            self.assertEqual(r.headers["Content-Range"], "bytes 3-5/10")

    def test_no_filesystem_endpoint(self):
        for path in ["/media/../../.env", "/api/config", "/.env"]:
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(self.url + path)
            self.assertEqual(e.exception.code, 404)

    def test_cross_origin_write_rejected(self):
        r = urllib.request.Request(self.url + "/api/control", data=b"{}",
                                   headers={"Origin": "https://example.com"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(r)
        self.assertEqual(e.exception.code, 403)

    def test_pages_are_served(self):
        for path in ["/", "/app.js", "/style.css"]:
            with urllib.request.urlopen(self.url + path) as r:
                self.assertEqual(r.status, 200)


if __name__ == "__main__":
    unittest.main()
