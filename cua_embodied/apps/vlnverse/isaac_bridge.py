"""Isaac Sim renderer and collision-query service for VLNVerse scenes.

Run this with Isaac Sim's own Python (python.bat or python.sh). It loads one
VLNVerse USD scene, renders a floating egocentric camera and answers PhysX
sphere sweeps against the scene's collision geometry. The task backend talks to
it over a localhost HTTP API and receives only RGB frames and movement results.

    <isaac-sim>/python.bat cua_embodied/apps/vlnverse/isaac_bridge.py --scene <scene.usd> --port 8793
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=79.0)
    parser.add_argument("--warmup-frames", type=int, default=24)
    args, _unknown = parser.parse_known_args()
    return args


ARGS = parse_args()
SCENE = Path(ARGS.scene).resolve()
if not SCENE.is_file():
    raise SystemExit(f"VLNVerse USD scene does not exist: {SCENE}")
if not (1.0 < ARGS.hfov < 179.0):
    raise SystemExit("--hfov must be between 1 and 179 degrees")

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp  # noqa: E402 - must precede Kit imports


simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": ARGS.width,
        "height": ARGS.height,
    }
)

import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.prims import define_prim  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402


def camera_quaternion(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return a wxyz quaternion for a USD camera.

    Agent convention: yaw 0 faces +Y and +90 faces +X; pitch is positive
    upward.  A USD camera looks down its local -Z axis with local +Y upward.
    """

    yaw_r, pitch_r, roll_r = np.radians([yaw, pitch, roll])
    forward = np.array(
        [
            np.sin(yaw_r) * np.cos(pitch_r),
            np.cos(yaw_r) * np.cos(pitch_r),
            np.sin(pitch_r),
        ],
        dtype=np.float64,
    )
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    norm = np.linalg.norm(right)
    right = right / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)
    if abs(roll) > 1e-8:
        right, up = (
            np.cos(roll_r) * right + np.sin(roll_r) * up,
            -np.sin(roll_r) * right + np.cos(roll_r) * up,
        )
    matrix = np.column_stack((right, up, -forward))
    quat_xyzw = R.from_matrix(matrix).as_quat()
    return np.array([quat_xyzw[3], *quat_xyzw[:3]], dtype=np.float64)


class Renderer:
    def __init__(self) -> None:
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        scene_root = define_prim("/World/Scene", "Xform")
        scene_root.GetReferences().AddReference(SCENE.as_posix(), "/Root")

        self.stage = simulation_app.context.get_stage()
        self.camera_path = "/World/FloatingCamera"
        camera = UsdGeom.Camera.Define(self.stage, self.camera_path)
        horizontal_aperture_mm = 20.0
        focal_length_mm = horizontal_aperture_mm / (
            2.0 * math.tan(math.radians(ARGS.hfov) / 2.0)
        )
        camera.CreateFocalLengthAttr(focal_length_mm)
        camera.CreateClippingRangeAttr().Set((0.01, 1000.0))
        camera.CreateHorizontalApertureAttr(horizontal_aperture_mm)
        camera.CreateVerticalApertureAttr(
            horizontal_aperture_mm * ARGS.height / ARGS.width
        )

        dome = self.stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)

        self.world.reset()
        self.render_product = rep.create.render_product(
            self.camera_path, (ARGS.width, ARGS.height)
        )
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.render_product])
        for _ in range(max(1, ARGS.warmup_frames)):
            self.world.step(render=True)
        self.scene_query = omni.physx.get_physx_scene_query_interface()
        print(
            "READY "
            + json.dumps(
                {
                    "scene": str(SCENE),
                    "resolution": [ARGS.width, ARGS.height],
                    "hfov_deg": ARGS.hfov,
                    "renderer": "RayTracedLighting",
                    "collision_query": "PhysX sphere sweep against scene-authored collision",
                }
            ),
            flush=True,
        )

    def set_camera(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        pitch: float,
        roll: float,
    ) -> None:
        orientation = camera_quaternion(yaw, pitch, roll)
        prim = self.stage.GetPrimAtPath(self.camera_path)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xformable.AddOrientOp().Set(
            Gf.Quatf(
                float(orientation[0]),
                Gf.Vec3f(
                    float(orientation[1]),
                    float(orientation[2]),
                    float(orientation[3]),
                ),
            )
        )

    def capture(self, pose: dict[str, float]) -> bytes:
        self.set_camera(**pose)
        self.world.step(render=True)
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
        self.world.step(render=True)
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
        rgba = np.asarray(self.rgb.get_data())
        if rgba.ndim != 3 or rgba.shape[:2] != (ARGS.height, ARGS.width):
            raise RuntimeError(f"invalid RGB capture shape: {rgba.shape}")
        rgb = np.ascontiguousarray(rgba[:, :, :3].astype(np.uint8, copy=False))
        output = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(output, format="PNG", compress_level=3)
        return output.getvalue()

    def sweep(self, payload: dict[str, float]) -> dict[str, object]:
        start = np.array(
            [payload["x0"], payload["y0"], payload["z0"]], dtype=np.float64
        )
        target = np.array(
            [payload["x1"], payload["y1"], payload["z1"]], dtype=np.float64
        )
        radius = float(payload["radius"])
        clearance = float(payload["clearance"])
        delta = target - start
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            return {
                "ok": True,
                "position": start.tolist(),
                "requested_distance_m": 0.0,
                "achieved_distance_m": 0.0,
                "bumped": False,
                "hit": None,
            }
        direction = delta / distance
        hit = self.scene_query.sweep_sphere_closest(
            radius,
            tuple(float(v) for v in start),
            tuple(float(v) for v in direction),
            distance,
            True,
        )
        if hit.get("hit"):
            hit_distance = max(0.0, float(hit.get("distance", 0.0)))
            achieved = max(0.0, min(distance, hit_distance - clearance))
            reached = start + direction * achieved
            hit_payload = {
                "distance_m": hit_distance,
                "position": [float(v) for v in hit.get("position", ())],
                "normal": [float(v) for v in hit.get("normal", ())],
                "collision": str(hit.get("collision", "")),
                "rigid_body": str(hit.get("rigidBody", "")),
            }
            bumped = achieved < distance - 1e-5
        else:
            achieved = distance
            reached = target
            hit_payload = None
            bumped = False
        return {
            "ok": True,
            "position": [float(v) for v in reached],
            "requested_distance_m": distance,
            "achieved_distance_m": achieved,
            "bumped": bumped,
            "hit": hit_payload,
        }


class WorkItem:
    def __init__(self, kind: str, payload: dict[str, float] | None = None) -> None:
        self.kind = kind
        self.payload = payload or {}
        self.done = threading.Event()
        self.result: bytes | dict[str, object] | None = None
        self.error: str | None = None


WORK: queue.Queue[WorkItem] = queue.Queue()
STOP = threading.Event()


class Handler(BaseHTTPRequestHandler):
    server_version = "VLNVerseBlindTestBridge/1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _queue(self, item: WorkItem, timeout: float = 120.0) -> bool:
        WORK.put(item)
        if not item.done.wait(timeout=timeout):
            self._json(504, {"error": "Isaac worker timed out"})
            return False
        if item.error:
            self._json(500, {"error": item.error})
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "scene": str(SCENE),
                    "resolution": [ARGS.width, ARGS.height],
                    "hfov_deg": ARGS.hfov,
                    "collision_query": True,
                },
            )
            return
        query = parse_qs(parsed.query)
        if parsed.path == "/render":
            keys = ("x", "y", "z", "yaw", "pitch", "roll")
            try:
                pose = {key: float(query[key][0]) for key in keys}
            except (KeyError, TypeError, ValueError) as exc:
                self._json(400, {"error": f"invalid pose: {exc}"})
                return
            item = WorkItem("render", pose)
            if not self._queue(item):
                return
            assert isinstance(item.result, bytes)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(item.result)))
            self.end_headers()
            self.wfile.write(item.result)
            return
        if parsed.path == "/sweep":
            keys = (
                "x0",
                "y0",
                "z0",
                "x1",
                "y1",
                "z1",
                "radius",
                "clearance",
            )
            try:
                payload = {key: float(query[key][0]) for key in keys}
            except (KeyError, TypeError, ValueError) as exc:
                self._json(400, {"error": f"invalid sweep: {exc}"})
                return
            if not (0.01 <= payload["radius"] <= 2.0):
                self._json(400, {"error": "radius must be in [0.01, 2.0] metres"})
                return
            if not (0.0 <= payload["clearance"] <= 1.0):
                self._json(400, {"error": "clearance must be in [0, 1.0] metres"})
                return
            item = WorkItem("sweep", payload)
            if not self._queue(item):
                return
            assert isinstance(item.result, dict)
            self._json(200, item.result)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/shutdown":
            self._json(404, {"error": "not found"})
            return
        WORK.put(WorkItem("shutdown"))
        self._json(200, {"ok": True})


def main() -> None:
    renderer = Renderer()
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"HTTP http://{ARGS.host}:{ARGS.port}", flush=True)
    try:
        while not STOP.is_set() and simulation_app.is_running():
            try:
                item = WORK.get(timeout=0.02)
            except queue.Empty:
                simulation_app.update()
                continue
            if item.kind == "shutdown":
                STOP.set()
                item.done.set()
                continue
            try:
                if item.kind == "render":
                    item.result = renderer.capture(item.payload)
                elif item.kind == "sweep":
                    item.result = renderer.sweep(item.payload)
                else:
                    raise ValueError(f"unknown work item: {item.kind}")
            except Exception as exc:
                item.error = f"{type(exc).__name__}: {exc}"
                print(f"WORK_ERROR kind={item.kind} error={item.error}", flush=True)
            finally:
                item.done.set()
    finally:
        server.shutdown()
        server.server_close()
        simulation_app.close()


if __name__ == "__main__":
    main()
