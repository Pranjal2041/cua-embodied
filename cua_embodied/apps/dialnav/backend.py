"""DialNav backend: a 360-degree panorama, numbered exits and a text channel to the Guide."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from google import genai
from google.genai import types

from ...core import Media, Observation, SoftwareStep, Task
from .env import DEFAULT_CONNECTIVITY_DIR, DEFAULT_RGB_DIR, DialNavEpisodeEnv, render_scene_atlas
from .guide import Guide


FORBIDDEN_GUIDE_SIGNAL = re.compile(
    r"\bL\d{3}\b|\bviewpoint\b|\bcoordinates?\b|\bmarkers?\b|\baction\s*\d+\b",
    re.I,
)


class DialNavBackend:
    """Unified-software adapter for one official DialNav episode."""

    def __init__(
        self,
        *,
        episode: Mapping[str, Any],
        dataset_index: int,
        output_dir: str | Path,
        guide_model: str = "gemini-3.5-flash",
        guide_seed: int = 0,
        connectivity_dir: str | Path = DEFAULT_CONNECTIVITY_DIR,
        rgb_dir: str | Path = DEFAULT_RGB_DIR,
    ) -> None:
        self.episode = dict(episode)
        self.dataset_index = int(dataset_index)
        self.output_dir = Path(output_dir).resolve()
        self.frames_dir = self.output_dir / "frames"
        self.guide_dir = self.output_dir / "guide_internal"
        self.guide_raw_dir = self.output_dir / "guide_raw_responses"
        for directory in (self.frames_dir, self.guide_dir, self.guide_raw_dir):
            directory.mkdir(parents=True, exist_ok=True)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required by the DialNav Guide")
        self.guide_model = guide_model
        self.connectivity_dir = Path(connectivity_dir)
        self.rgb_dir = Path(rgb_dir)
        self.guide = Guide(
            genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=180_000)),
            model=guide_model,
            seed=guide_seed,
        )
        self.env: DialNavEpisodeEnv | None = None
        self.atlas_pages: list[bytes] = []
        self.code_to_viewpoint: dict[str, str] = {}
        self.current_frame: Path | None = None
        self.current_candidates: list[Any] = []
        self.ask_index = 0
        self.interaction_step = 0
        self.events: list[dict[str, Any]] = []
        self.guide_records: list[dict[str, Any]] = []

    def reset(self, task: Task) -> Observation:
        expected_task_id = str(self.episode["instr_id"])
        if task.task_id != expected_task_id:
            raise ValueError(
                f"task {task.task_id!r} does not match DialNav instr_id {expected_task_id!r}"
            )
        self.env = DialNavEpisodeEnv(
            self.episode, connectivity_dir=self.connectivity_dir, rgb_dir=self.rgb_dir
        )
        self.atlas_pages, self.code_to_viewpoint = render_scene_atlas(
            self.env.graph,
            rgb_dir=self.env.rgb_dir,
        )
        for page_index, page in enumerate(self.atlas_pages, start=1):
            (self.guide_dir / f"scene_atlas_{page_index:02d}.png").write_bytes(page)
        self.ask_index = 0
        self.interaction_step = 0
        self.events = []
        self.guide_records = []
        self.current_frame, self.current_candidates = self._render_current()
        return self._observation("episode_started")

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> SoftwareStep:
        env = self._require_env()
        self.interaction_step += 1

        if tool_name == "move":
            marker = int(arguments.get("marker", -1))
            legal = {candidate.action: candidate for candidate in self.current_candidates}
            if marker not in legal:
                response = {
                    "moved": False,
                    "error": "That marker is absent from the current panorama.",
                    "valid_markers": sorted(legal),
                }
                self.events.append(
                    {
                        "interaction_step": self.interaction_step,
                        "tool": "move",
                        "arguments": dict(arguments),
                        "accepted": False,
                        "current_viewpoint_hidden_from_agent": env.current,
                    }
                )
                self._print_step("move_invalid")
                return SoftwareStep(
                    response=response,
                    observation=self._observation("move_rejected"),
                )

            before = env.current
            selected = legal[marker]
            env.move(marker)
            self.current_frame, self.current_candidates = self._render_current()
            self.events.append(
                {
                    "interaction_step": self.interaction_step,
                    "tool": "move",
                    "arguments": dict(arguments),
                    "accepted": True,
                    "from_viewpoint_hidden_from_agent": before,
                    "to_viewpoint_hidden_from_agent": selected.viewpoint_id,
                    "yaw_deg_hidden_from_agent_text": selected.yaw_deg,
                    "distance_m_hidden_from_agent_text": selected.distance_m,
                }
            )
            self._print_step("move")
            return SoftwareStep(
                response={
                    "moved": True,
                    "message": "Fresh 360-degree observation follows.",
                },
                observation=self._observation("move_applied"),
            )

        if tool_name == "type":
            question = str(arguments.get("text", "")).strip()
            if not question:
                question = "Please help me determine where I am and where I should go next."
            if FORBIDDEN_GUIDE_SIGNAL.search(question):
                self.events.append(
                    {
                        "interaction_step": self.interaction_step,
                        "tool": "type",
                        "arguments": dict(arguments),
                        "accepted": False,
                        "current_viewpoint_hidden_from_agent": env.current,
                    }
                )
                self._print_step("type_rejected")
                return SoftwareStep(
                    response={
                        "accepted": False,
                        "error": (
                            "The Guide cannot see local action labels or hidden identifiers. "
                            "Describe rooms, objects, layout, and recent motion in ordinary language."
                        ),
                    },
                    observation=self._observation("guide_message_rejected"),
                )

            self.ask_index += 1
            answer, localized, guide_record = self.guide.turn(
                env=env,
                question=question,
                atlas_pages=self.atlas_pages,
                code_to_viewpoint=self.code_to_viewpoint,
                raw_dir=self.guide_raw_dir,
                guide_dir=self.guide_dir,
                ask_index=self.ask_index,
            )
            dialog_item = {
                "nav_idx": len(env.path) - 1,
                "question": question,
                "answer": answer,
                "viewpoint": env.current,
                "localized_viewpoint": localized,
            }
            env.dialog.append(dialog_item)
            self.guide_records.append(guide_record)
            self.events.append(
                {
                    "interaction_step": self.interaction_step,
                    "tool": "type",
                    "arguments": dict(arguments),
                    "accepted": True,
                    "guide_answer": answer,
                    "current_viewpoint_hidden_from_agent": env.current,
                    "localized_viewpoint_hidden_from_agent": localized,
                    "guide_internal": guide_record,
                }
            )
            self._print_step("type")
            return SoftwareStep(
                response={"accepted": True, "answer": answer},
                observation=self._observation("guide_answer_received"),
            )

        if tool_name == "stop":
            env.stop()
            reason = str(arguments.get("reason", ""))
            self.events.append(
                {
                    "interaction_step": self.interaction_step,
                    "tool": "stop",
                    "arguments": dict(arguments),
                    "current_viewpoint_hidden_from_agent": env.current,
                }
            )
            self._print_step("stop")
            return SoftwareStep(
                response={"stopped": True, "reason": reason},
                done=True,
                output=self.evaluation_output("model_stop"),
            )

        raise ValueError(f"unsupported DialNav tool: {tool_name}")

    def evaluation_output(self, termination: str) -> dict[str, Any]:
        env = self._require_env()
        gt_dialog_count = len(self.episode["dialog"])
        gt_navigation_nodes = len(self.episode["nav_trajectory"])
        denominator = gt_navigation_nodes - gt_dialog_count
        dialog_efficiency = (
            0.0
            if denominator <= 0
            else 1.0
            - min(
                max(len(env.dialog) - gt_dialog_count, 0) / denominator,
                1.0,
            )
        )
        endpoint_success = int(env.is_success())
        return {
            "scientific_status": "public_val_unseen_locally_scored",
            "dataset_index": self.dataset_index,
            "instr_id": self.episode["instr_id"],
            "scan": env.scan,
            "target": env.target,
            "termination": termination,
            "guide_model": self.guide_model,
            "metrics": {
                "official_endpoint_success": endpoint_success,
                "model_stop_success": int(env.model_stop_success()),
                "oracle_success": int(env.oracle_success()),
                "navigation_error_m": env.navigation_error(),
                "dialog_efficiency": dialog_efficiency,
                "score_x_success": dialog_efficiency * endpoint_success,
                "navigation_steps": len(env.path) - 1,
                "dialog_turns": len(env.dialog),
                "gt_navigation_nodes": gt_navigation_nodes,
                "gt_dialog_turns": gt_dialog_count,
            },
            "submission_item": {
                "instr_id": self.episode["instr_id"],
                "scan": env.scan,
                "path": env.path,
                "dialog": env.dialog,
            },
            "events": self.events,
            "guide_internal_records": self.guide_records,
            "final_frame": str(self.current_frame) if self.current_frame else None,
        }

    def close(self) -> None:
        close = getattr(self.guide.client, "close", None)
        if callable(close):
            close()

    def _render_current(self) -> tuple[Path, list[Any]]:
        env = self._require_env()
        navigation_step = len(env.path) - 1
        frame, candidates = env.observe(step=navigation_step)
        path = self.frames_dir / f"nav_{navigation_step:03d}.png"
        path.write_bytes(frame)
        return path, candidates

    def _observation(self, event: str) -> Observation:
        if self.current_frame is None:
            raise RuntimeError("DialNav observation requested before rendering")
        return Observation(
            text=(
                "Complete 360-degree panorama. The image edges join. Numbered circles "
                "are the currently legal moves. The Guide cannot see this image; describe "
                "visible rooms, objects, layout, and recent motion when sending a message."
            ),
            media=(
                Media(
                    mime_type="image/png",
                    path=self.current_frame,
                    label="CURRENT 360-DEGREE OBSERVATION:",
                ),
            ),
            record={
                "event": event,
                "navigation_step": len(self._require_env().path) - 1,
                "valid_markers": sorted(
                    candidate.action for candidate in self.current_candidates
                ),
            },
        )

    def _require_env(self) -> DialNavEpisodeEnv:
        if self.env is None:
            raise RuntimeError("DialNav backend must be reset before use")
        return self.env

    def _print_step(self, action: str) -> None:
        env = self._require_env()
        print(
            f"DIALNAV_STEP dataset_index={self.dataset_index:03d} "
            f"turn={self.interaction_step:03d} action={action} "
            f"nav_steps={len(env.path) - 1} asks={len(env.dialog)}",
            flush=True,
        )


def write_evaluation(path: Path, output: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
