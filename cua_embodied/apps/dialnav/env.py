"""Graph environment and panorama renderer for DialNav.

The official benchmark operates on the Matterport3D navigation graph.  The
Navigator sees a panoramic observation and may move only to an adjacent graph
node, ask the Guide a natural-language question, or stop.  Ground-truth goal
nodes are retained inside :class:`DialNavEpisodeEnv` solely for scoring and for
the Guide, which the challenge explicitly permits to know the target.
"""

from __future__ import annotations

import heapq
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Data locations. The backend passes the configured paths explicitly. These
# environment variables are only fallbacks for standalone use.
DEFAULT_CONNECTIVITY_DIR = Path(os.environ.get("DIALNAV_CONNECTIVITY_DIR", "data/dialnav/connectivity"))
DEFAULT_RGB_DIR = Path(os.environ.get("DIALNAV_RGB_DIR", "data/dialnav/rgb"))


FONT_CANDIDATES = {
    False: ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/Library/Fonts/Arial.ttf"),
    True: ("C:/Windows/Fonts/consolab.ttf", "C:/Windows/Fonts/arialbd.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/Library/Fonts/Arial Bold.ttf"),
}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES[bold]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default(size=size)


FONT_SMALL = _font(18)
FONT_LABEL = _font(24, bold=True)
FONT_HEADER = _font(27, bold=True)


@dataclass(frozen=True)
class GraphNode:
    viewpoint_id: str
    position: np.ndarray
    rotation: np.ndarray
    neighbors: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    action: int
    viewpoint_id: str
    yaw_deg: float
    pitch_deg: float
    distance_m: float


class MatterportGraph:
    """One scan's official connectivity graph plus panorama geometry."""

    def __init__(self, scan: str, connectivity_dir: Path = DEFAULT_CONNECTIVITY_DIR):
        self.scan = scan
        path = connectivity_dir / f"{scan}_connectivity.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        included = {row["image_id"] for row in rows if row["included"]}
        self.nodes: dict[str, GraphNode] = {}
        for index, row in enumerate(rows):
            if not row["included"]:
                continue
            pose = np.asarray(row["pose"], dtype=np.float64).reshape(4, 4)
            neighbors = tuple(
                rows[j]["image_id"]
                for j, connected in enumerate(row["unobstructed"])
                if connected
                and j != index
                and rows[j]["image_id"] in included
            )
            self.nodes[row["image_id"]] = GraphNode(
                viewpoint_id=row["image_id"],
                position=pose[:3, 3].copy(),
                rotation=pose[:3, :3].copy(),
                neighbors=neighbors,
            )

    def candidates(self, viewpoint_id: str) -> list[Candidate]:
        """Return legal moves ordered left-to-right in the equirectangular view.

        After the Matterport cubemap model rotation is accounted for, the
        processed panorama's yaw 0 faces pose-local -X and +90 faces local +Z.
        Matterport's camera-pose convention has local +Y pointing down, so
        positive panorama pitch is local -Y.  The connectivity pose maps those
        local camera axes into the scan's world frame.  The horizontal 180
        degree offset is independently checked by multi-view feature
        triangulation in ``audit_marker_geometry.py``.
        """

        current = self.nodes[viewpoint_id]
        projected: list[tuple[float, float, float, str]] = []
        for neighbor_id in current.neighbors:
            neighbor = self.nodes[neighbor_id]
            world_delta = neighbor.position - current.position
            # Connectivity poses are local-camera to world transforms.
            local_delta = np.linalg.solve(current.rotation, world_delta)
            horizontal = math.hypot(float(local_delta[0]), float(local_delta[2]))
            yaw = math.degrees(math.atan2(float(local_delta[2]), -float(local_delta[0])))
            pitch = -math.degrees(math.atan2(float(local_delta[1]), horizontal))
            projected.append((yaw, pitch, float(np.linalg.norm(world_delta)), neighbor_id))
        projected.sort(key=lambda item: (item[0], item[1], item[3]))
        return [
            Candidate(
                action=index + 1,
                viewpoint_id=neighbor_id,
                yaw_deg=yaw,
                pitch_deg=pitch,
                distance_m=distance,
            )
            for index, (yaw, pitch, distance, neighbor_id) in enumerate(projected)
        ]

    def shortest_path(self, start: str, goals: Iterable[str]) -> list[str]:
        goals = set(goals)
        if start in goals:
            return [start]
        queue: list[tuple[float, str]] = [(0.0, start)]
        distance = {start: 0.0}
        parent: dict[str, str] = {}
        reached: str | None = None
        while queue:
            cost, node_id = heapq.heappop(queue)
            if cost != distance.get(node_id):
                continue
            if node_id in goals:
                reached = node_id
                break
            node = self.nodes[node_id]
            for neighbor_id in node.neighbors:
                edge = float(
                    np.linalg.norm(self.nodes[neighbor_id].position - node.position)
                )
                new_cost = cost + edge
                if new_cost < distance.get(neighbor_id, math.inf):
                    distance[neighbor_id] = new_cost
                    parent[neighbor_id] = node_id
                    heapq.heappush(queue, (new_cost, neighbor_id))
        if reached is None:
            raise ValueError(f"no path from {start} to any goal in scan {self.scan}")
        path = [reached]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    def shortest_distance(self, start: str, goals: Iterable[str]) -> float:
        path = self.shortest_path(start, goals)
        return sum(
            float(
                np.linalg.norm(
                    self.nodes[right].position - self.nodes[left].position
                )
            )
            for left, right in zip(path[:-1], path[1:])
        )


def panorama_path(viewpoint_id: str, rgb_dir: Path = DEFAULT_RGB_DIR) -> Path:
    return rgb_dir / f"{viewpoint_id}.png"


def panorama_bytes(viewpoint_id: str, rgb_dir: Path = DEFAULT_RGB_DIR) -> bytes:
    return panorama_path(viewpoint_id, rgb_dir).read_bytes()


def render_labeled_panorama(
    viewpoint_id: str,
    label: str,
    *,
    rgb_dir: Path = DEFAULT_RGB_DIR,
) -> bytes:
    """Guide-only full-resolution panorama with an internal atlas label."""

    pano = Image.open(panorama_path(viewpoint_id, rgb_dir)).convert("RGB")
    header_h = 54
    canvas = Image.new("RGB", (pano.width, pano.height + header_h), (7, 15, 23))
    canvas.paste(pano, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 10),
        f"GUIDE-ONLY LOCALIZATION CANDIDATE: {label}",
        font=FONT_HEADER,
        fill=(88, 231, 216),
    )
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _label_circle(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] = (8, 18, 28),
) -> None:
    radius = 23
    width = image.width
    for x_copy in (x - width, x, x + width):
        if x_copy + radius < 0 or x_copy - radius >= width:
            continue
        draw.ellipse(
            (x_copy - radius, y - radius, x_copy + radius, y + radius),
            fill=fill,
            outline=outline,
            width=4,
        )
        bbox = draw.textbbox((0, 0), text, font=FONT_LABEL)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x_copy - tw / 2, y - th / 2 - 2),
            text,
            font=FONT_LABEL,
            fill=(3, 12, 20),
        )


def render_action_view(
    graph: MatterportGraph,
    viewpoint_id: str,
    *,
    target: str | None = None,
    step: int | None = None,
    highlight_neighbor: str | None = None,
    rgb_dir: Path = DEFAULT_RGB_DIR,
) -> tuple[bytes, list[Candidate]]:
    """Render a full panorama with numbered legal-transition markers."""

    pano = Image.open(panorama_path(viewpoint_id, rgb_dir)).convert("RGB")
    candidates = graph.candidates(viewpoint_id)
    draw = ImageDraw.Draw(pano, "RGBA")
    for candidate in candidates:
        x = ((candidate.yaw_deg + 180.0) % 360.0) / 360.0 * pano.width
        y = (0.5 - candidate.pitch_deg / 180.0) * pano.height
        highlighted = candidate.viewpoint_id == highlight_neighbor
        fill = (90, 238, 128, 245) if highlighted else (61, 226, 214, 235)
        draw.line((x, y + 23, x, min(pano.height - 1, y + 68)), fill=fill, width=5)
        _label_circle(pano, draw, x, y, str(candidate.action), fill)

    header_h, footer_h = 58, 50
    canvas = Image.new("RGB", (pano.width, pano.height + header_h + footer_h), (8, 16, 25))
    canvas.paste(pano, (0, header_h))
    cdraw = ImageDraw.Draw(canvas)
    left = "DIALNAV 360 OBSERVATION"
    if step is not None:
        left += f"  /  STEP {step:02d}"
    cdraw.text((20, 14), left, font=FONT_HEADER, fill=(233, 241, 248))
    if target:
        target_text = f"TARGET CLUE: {target}"
        bbox = cdraw.textbbox((0, 0), target_text, font=FONT_HEADER)
        cdraw.text(
            (canvas.width - (bbox[2] - bbox[0]) - 20, 14),
            target_text,
            font=FONT_HEADER,
            fill=(255, 204, 102),
        )
    footer = (
        "NUMBERED CIRCLES = LEGAL MOVES   |   IMAGE EDGES JOIN   |   "
        "CENTER = LOCAL PANORAMA FRONT"
    )
    cdraw.text((18, pano.height + header_h + 13), footer, font=FONT_SMALL, fill=(157, 181, 199))
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue(), candidates


def render_scene_atlas(
    graph: MatterportGraph,
    *,
    rgb_dir: Path = DEFAULT_RGB_DIR,
    columns: int = 4,
    rows: int = 4,
) -> tuple[list[bytes], dict[str, str]]:
    """Build labeled panorama sheets for Guide-only visual localization."""

    node_ids = sorted(graph.nodes)
    code_to_viewpoint = {f"L{index:03d}": node for index, node in enumerate(node_ids)}
    items = list(code_to_viewpoint.items())
    thumb_w, thumb_h, label_h = 256, 128, 30
    per_sheet = columns * rows
    sheets: list[bytes] = []
    for offset in range(0, len(items), per_sheet):
        page = items[offset : offset + per_sheet]
        sheet = Image.new(
            "RGB",
            (columns * thumb_w, rows * (thumb_h + label_h) + 44),
            (7, 15, 23),
        )
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (14, 8),
            f"GUIDE-ONLY SCENE ATLAS  {graph.scan}  /  {offset // per_sheet + 1}",
            font=FONT_LABEL,
            fill=(226, 238, 246),
        )
        for local_index, (code, viewpoint_id) in enumerate(page):
            row, col = divmod(local_index, columns)
            x, y = col * thumb_w, 44 + row * (thumb_h + label_h)
            pano = Image.open(panorama_path(viewpoint_id, rgb_dir)).convert("RGB")
            pano.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_w, thumb_h), (0, 0, 0))
            tile.paste(pano, ((thumb_w - pano.width) // 2, (thumb_h - pano.height) // 2))
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill=(15, 29, 42))
            draw.text((x + 9, y + thumb_h + 4), code, font=FONT_SMALL, fill=(86, 230, 216))
        output = io.BytesIO()
        sheet.save(output, format="PNG", optimize=True)
        sheets.append(output.getvalue())
    return sheets, code_to_viewpoint


def render_route_preview(
    graph: MatterportGraph,
    route: Sequence[str],
    *,
    max_hops: int = 6,
    rgb_dir: Path = DEFAULT_RGB_DIR,
) -> bytes:
    """Guide-only montage highlighting each correct outgoing direction."""

    if not route or max_hops < 1:
        raise ValueError("A route and a positive preview hop limit are required")
    at_destination = len(route) == 1
    hops = min(max_hops, len(route) - 1)
    if at_destination:
        frames = [panorama_bytes(route[0], rgb_dir)]
    else:
        frames = [
            render_action_view(
                graph,
                route[index],
                step=index + 1,
                highlight_neighbor=route[index + 1],
                rgb_dir=rgb_dir,
            )[0]
            for index in range(hops)
        ]
    images = [Image.open(io.BytesIO(frame)).convert("RGB") for frame in frames]
    tile_w, tile_h = 640, 400
    columns = 2
    rows = math.ceil(len(images) / columns)
    montage = Image.new("RGB", (columns * tile_w, rows * tile_h), (5, 12, 20))
    draw = ImageDraw.Draw(montage)
    for index, image in enumerate(images):
        image.thumbnail((tile_w, tile_h - 34), Image.Resampling.LANCZOS)
        row, col = divmod(index, columns)
        x = col * tile_w + (tile_w - image.width) // 2
        y = row * tile_h + 34
        montage.paste(image, (x, y))
        draw.text(
            (col * tile_w + 12, row * tile_h + 5),
            "DESTINATION REACHED: STOP HERE" if at_destination else
            f"ROUTE STEP {index + 1}: GREEN MARK = REQUIRED EXIT",
            font=FONT_SMALL,
            fill=(104, 239, 143),
        )
    output = io.BytesIO()
    montage.save(output, format="PNG", optimize=True)
    return output.getvalue()


class DialNavEpisodeEnv:
    def __init__(
        self,
        episode: dict,
        *,
        connectivity_dir: Path = DEFAULT_CONNECTIVITY_DIR,
        rgb_dir: Path = DEFAULT_RGB_DIR,
    ) -> None:
        self.episode = episode
        self.scan = str(episode["scan"])
        self.target = str(episode["target"])
        self.start_pano = str(episode["start_pano"])
        self.end_panos = tuple(str(x) for x in episode["end_panos"])
        self.graph = MatterportGraph(self.scan, connectivity_dir)
        self.rgb_dir = rgb_dir
        self.current = self.start_pano
        self.path = [self.current]
        self.dialog: list[dict] = []
        self.stopped = False

    def observe(self, step: int) -> tuple[bytes, list[Candidate]]:
        return render_action_view(
            self.graph,
            self.current,
            target=self.target,
            step=step,
            rgb_dir=self.rgb_dir,
        )

    def move(self, action: int) -> str:
        candidates = self.graph.candidates(self.current)
        match = next((item for item in candidates if item.action == action), None)
        if match is None:
            raise ValueError(
                f"action {action} is not legal at the current node; "
                f"valid actions are {[item.action for item in candidates]}"
            )
        self.current = match.viewpoint_id
        self.path.append(self.current)
        return self.current

    def stop(self) -> None:
        self.stopped = True

    def is_success(self) -> bool:
        """Match the official submission evaluator's endpoint semantics."""

        return self.current in self.end_panos

    def model_stop_success(self) -> bool:
        return self.stopped and self.current in self.end_panos

    def oracle_success(self) -> bool:
        return any(viewpoint in self.end_panos for viewpoint in self.path)

    def navigation_error(self) -> float:
        return self.graph.shortest_distance(self.current, self.end_panos)

    def oracle_route(self, start: str | None = None) -> list[str]:
        return self.graph.shortest_path(start or self.current, self.end_panos)
