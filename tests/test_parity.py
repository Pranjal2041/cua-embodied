"""GUI and text-API parity checks without model calls or GPUs.

Both interfaces must show the agent the same task information: every public
field, the same complete videos, and the same starting view.
"""
import base64
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from cua_embodied.apps import NAMES, load_app
from cua_embodied.computer import BrowserBackend, PixelComputer
from cua_embodied.core import Media, Observation, SoftwareStep, Task
from cua_embodied.gui.server import MediaStore
from cua_embodied.gui.session import Session


def session(app, root, config=None):
    return Session(load_app(app), config or {}, Path(root), MediaStore())


class TestVisibleFeedback(unittest.TestCase):
    def test_all_public_observation_fields_survive(self):
        with tempfile.TemporaryDirectory() as root:
            for name in NAMES:
                with self.subTest(app=name):
                    s = session(name, root)
                    fields = s.app.observation_fields
                    record = {field: {} if field == "camera" else 1 for field in fields}
                    record["evaluation_secret"] = "hidden"
                    s.observe(Observation(text="Complete feedback\nSecond line", record=record))
                    self.assertEqual(s.state["observation"]["fields"], {k: record[k] for k in fields})
                    self.assertEqual(s.state["observation"]["text"], "Complete feedback\nSecond line")
                    self.assertNotIn("evaluation_secret", str(s.state))

    def test_all_public_response_fields_survive(self):
        with tempfile.TemporaryDirectory() as root:
            for name in NAMES:
                with self.subTest(app=name):
                    s = session(name, root)
                    fields = s.app.response_fields
                    response = {field: 1 for field in fields if field != "error"}
                    response["evaluation_secret"] = "hidden"
                    s.backend = SimpleNamespace(execute=lambda *args: SoftwareStep(response=response))
                    with patch.object(s, "refresh"), patch.object(s.app, "after_action"):
                        s.execute("stop" if name in {"dialnav", "vlnverse"} else "inspect", {})
                    self.assertEqual(set(s.state["feedback"]), fields - {"error"})
                    self.assertNotIn("evaluation_secret", str(s.state))

    def test_rejected_guide_message_does_not_enter_dialog(self):
        with tempfile.TemporaryDirectory() as root:
            s = session("dialnav", root)
            s.backend = SimpleNamespace(execute=lambda *args: SoftwareStep(
                response={"accepted": False, "error": "Describe the scene in ordinary language."}))
            with self.assertRaisesRegex(ValueError, "ordinary language"):
                s.execute("type", {"text": "marker 1"})
            self.assertEqual(s.state["dialog"], [])
            self.assertFalse(s.state["feedback"]["accepted"])


class FakeBackend:
    def __init__(self, observation, **attributes):
        self.observation = observation
        self.__dict__.update(attributes)

    def reset(self, task):
        return self.observation

    def execute(self, name, arguments):
        return SoftwareStep(response={"ok": True},
                            observation=Observation(text="Frame", media=(Media(mime_type="image/jpeg", data=b"f"),)))

    def close(self):
        pass


def started(name, root, backend, config=None):
    s = session(name, root, config)
    with patch.object(s.app, "open", return_value=(Task("t", "Objective"), backend)), \
         patch.object(s.app, "refresh"):
        s.start()
    return s


class TestVideoContext(unittest.TestCase):
    def test_original_and_each_overlay_delivered_once(self):
        with tempfile.TemporaryDirectory() as root:
            s = session("mevis", root)
            original = Media(mime_type="video/mp4", data=b"original", label="COMPLETE ORIGINAL VIDEO:")
            overlays = [Media(mime_type="video/mp4", data=f"overlay{i}".encode(),
                              label="COMPLETE TRACKED-MASK OVERLAY:") for i in (1, 2)]
            browser = BrowserBackend("mevis", "http://127.0.0.1:8813",
                                     screenshot_dir=Path(root) / "screenshots",
                                     video_source=s.take_new_videos)
            screenshot = {"image": base64.b64encode(b"pixels").decode(), "hidden_goal": "secret"}
            for video in [original, *overlays]:
                s.observe(Observation(text="Complete video", media=(video,)))
                delivered = browser._observation(screenshot)
                self.assertEqual(delivered.media[1:], (video,))
                self.assertEqual(delivered.record, {})
                self.assertNotIn("secret", delivered.text)
                self.assertEqual(len(browser._observation(screenshot).media), 1)

    def test_opensun3d_reference_videos_match(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in ("1", "2", "3"):
                (root / f"{name}.mp4").write_bytes(name.encode())
            backend = FakeBackend(Observation(text="Faro view", media=(Media(mime_type="image/png", data=b"rgb"),)),
                                  durations={"3": 3, "1": 1, "2": 2})
            s = started("opensun3d", root, backend, {"reference_video_dir": str(root)})
            api = tuple(m for m in s.initial_observation.media if m.mime_type.startswith("video/"))
            gui = s.take_new_videos()
            self.assertEqual(api, gui)
            self.assertEqual([m.path.name for m in gui], ["1.mp4", "2.mp4", "3.mp4"])
            self.assertEqual(s.take_new_videos(), ())
            self.assertEqual([r["id"] for r in s.state["references"]], ["1", "2", "3"])

    def test_mevis_initial_frame_and_video_match(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            video = Media(mime_type="video/mp4", path=root / "full.mp4", label="COMPLETE ORIGINAL VIDEO:")
            frame = root / "frame.jpg"
            backend = FakeBackend(Observation(text="Full video", media=(video,)),
                                  interface=SimpleNamespace(frame_path=lambda i: frame),
                                  frame_names=["a", "b"], fps=5.0)
            s = started("mevis", root, backend)
            self.assertEqual(s.initial_observation.media[0], video)
            self.assertEqual(s.initial_observation.media[1].path, frame)
            self.assertEqual(s.take_new_videos(), (video,))
            self.assertEqual(s.state["frame_count"], 2)


class TestComputer(unittest.TestCase):
    def test_invalid_keyboard_call_returns_error_and_allows_correction(self):
        image = base64.b64encode(b"pixels").decode()
        message = "Provide exactly one of text or keys. No keyboard input was executed."
        replies = [dict(ok=False, error=message, image=image), dict(ok=True, image=image)]
        backend = BrowserBackend("mevis")
        backend.process = SimpleNamespace(stdin=io.StringIO(),
            stdout=io.StringIO("".join(__import__("json").dumps(r) + "\n" for r in replies)))
        with patch("cua_embodied.computer.pixel_computer.urllib.request.urlopen",
                   side_effect=[io.BytesIO(b'{"done": false}'), io.BytesIO(b'{"done": false}')]):
            failed = backend.execute("keyboard", {"text": "66", "keys": ["Backspace"]})
            corrected = backend.execute("keyboard", {"text": "66"})
        self.assertEqual(failed.response, {"ok": False, "error": message})
        self.assertEqual(failed.observation.text, message)
        self.assertEqual(failed.observation.media[0].data, b"pixels")
        self.assertFalse(failed.done)
        self.assertEqual(corrected.response, {"ok": True})

    def test_gui_observation_contains_only_screenshot(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = BrowserBackend("dialnav", screenshot_dir=Path(directory))
            obs = backend._observation({"image": base64.b64encode(b"pixels").decode(),
                                        "hidden_goal": "secret", "dom": "secret"})
            self.assertEqual(obs.record, {})
            self.assertNotIn("secret", obs.text)
            self.assertEqual(obs.media[0].path.read_bytes(), b"pixels")
            self.assertEqual({t.name for t in PixelComputer(backend).tools}, {"keyboard", "mouse"})


class TestGuideArrival(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cua_embodied.apps.dialnav import env, guide
        cls.guide, cls.env_module = guide, env

    def test_terminal_preview_has_no_required_exit(self):
        raw = io.BytesIO()
        Image.new("RGB", (1280, 640), (10, 20, 30)).save(raw, format="PNG")
        with patch.object(self.env_module, "panorama_bytes", return_value=raw.getvalue()) as pano, \
             patch.object(self.env_module, "render_action_view") as exits:
            image = self.env_module.render_route_preview(None, ["destination"])
        pano.assert_called_once()
        exits.assert_not_called()
        self.assertEqual(Image.open(io.BytesIO(image)).size, (1280, 400))

    def test_arrival_uses_inferred_location_only(self):
        class Env:
            target = "plant"
            end_panos = ("destination",)
            rgb_dir = Path("unused")
            dialog = []
            graph = SimpleNamespace(shortest_path=lambda start, ends:
                ["destination"] if start == "destination" else ["corridor", "destination"])

            @property
            def current(self):
                raise AssertionError("The agent's true location must not be read here")

        for inferred, arrived in [("destination", True), ("corridor", False)]:
            with self.subTest(inferred=inferred), tempfile.TemporaryDirectory() as root:
                calls = []
                guide = self.guide.Guide(client=None, model="test-model")

                def generate(**kwargs):
                    calls.append(kwargs)
                    return SimpleNamespace()
                with patch.object(self.guide, "render_route_preview", return_value=b"preview"), \
                     patch.object(self.guide, "panorama_bytes", return_value=b"arrival"), \
                     patch.object(guide, "_generate", side_effect=generate), \
                     patch.object(self.guide, "persist_response"), \
                     patch.object(self.guide, "response_text", return_value="Stop here."), \
                     patch.object(self.guide, "response_thought", return_value=""), \
                     patch.object(self.guide, "usage_dict", return_value={}):
                    _, _, record = guide.answer(env=Env(), localized_viewpoint=inferred,
                                                question="Where next?", raw_dir=Path(root),
                                                guide_dir=Path(root), ask_index=1)
                prompt = calls[0]["contents"][0].parts[0].text
                self.assertEqual("Zero moves remain" in prompt, arrived)
                self.assertEqual(record["localized_in_target_region"], arrived)


if __name__ == "__main__":
    unittest.main()
