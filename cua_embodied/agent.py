"""The computer-use agent: one Gemini model that acts only through registered tools.

The agent keeps the full interaction text. When the context holds twenty images,
the oldest ten are replaced by a short placeholder. Videos are never dropped.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from google import genai
from google.genai import types

from .core import Media, Observation, Task, ToolDefinition


SYSTEM_INSTRUCTION = """You are a multimodal agent operating software to complete an objective.
Use the registered tools to interact with the software. Inspect each returned observation before
choosing the next action. Call exactly one tool per turn. Use a software completion tool only when
the objective is satisfied."""

IMAGE_CONTEXT_MODES = frozenset({"all", "sliding"})
EARLIER_IMAGE_PLACEHOLDER = "[earlier view dropped to bound context]"


class AgentProtocolError(RuntimeError):
    """The model response did not follow the one-action interaction contract."""


@dataclass(frozen=True)
class AgentAction:
    name: str
    arguments: Mapping[str, Any]
    thought: str = ""
    text: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw_response_json: str | None = None


class GeminiAgent:
    """Generic Gemini agent whose available actions are registered at runtime."""

    def __init__(
        self,
        *,
        model: str | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        system_instruction: str = SYSTEM_INSTRUCTION,
        temperature: float = 0.2,
        seed: int = 0,
        thinking_level: str = "high",
        max_output_tokens: int = 4096,
        no_tool_retries: int = 3,
        request_timeout_seconds: int = 180,
        request_attempts: int = 5,
        media_processing_timeout_seconds: int = 600,
        image_context_mode: str = "sliding",
        image_context_window: int = 20,
        image_context_drop: int = 10,
    ) -> None:
        if image_context_mode not in IMAGE_CONTEXT_MODES:
            raise ValueError(
                "image_context_mode must be one of "
                f"{sorted(IMAGE_CONTEXT_MODES)}, got {image_context_mode!r}"
            )
        if image_context_window < 1:
            raise ValueError("image_context_window must be positive")
        if not 0 < image_context_drop < image_context_window:
            raise ValueError(
                "image_context_drop must be positive and below image_context_window"
            )
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.seed = seed
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.no_tool_retries = max(1, no_tool_retries)
        self.media_processing_timeout_seconds = max(1, media_processing_timeout_seconds)
        self.image_context_mode = image_context_mode
        self.image_context_window = image_context_window
        self.image_context_drop = image_context_drop

        if client is None:
            resolved_key = api_key or os.getenv("GEMINI_API_KEY")
            if not resolved_key:
                raise RuntimeError("GEMINI_API_KEY is required when no client is supplied")
            client = genai.Client(
                api_key=resolved_key,
                http_options=types.HttpOptions(
                    timeout=max(1, request_timeout_seconds) * 1000,
                    retry_options=types.HttpRetryOptions(
                        attempts=max(1, request_attempts),
                        initial_delay=1.0,
                        max_delay=8.0,
                        exp_base=2.0,
                        jitter=0.2,
                        http_status_codes=[408, 429, *range(500, 600)],
                    ),
                ),
            )
        self.client = client
        self.reset()

    def reset(self) -> None:
        self._tool_definitions: dict[str, ToolDefinition] = {}
        self._contents: list[types.Content] = []
        self._started = False
        self._uploaded_media: dict[Path, tuple[str, str]] = {}

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tool_definitions.values())

    @property
    def contents(self) -> tuple[types.Content, ...]:
        return tuple(self._contents)

    def register_tool(self, definition: ToolDefinition) -> None:
        if self._started:
            raise RuntimeError("tools must be registered before the task begins")
        if definition.name in self._tool_definitions:
            raise ValueError(f"tool {definition.name!r} is already registered")
        self._tool_definitions[definition.name] = definition

    def register_tools(self, definitions: Iterable[ToolDefinition]) -> None:
        for definition in definitions:
            self.register_tool(definition)

    def begin_task(
        self,
        task: Task,
        *,
        software_description: str,
        observation: Observation,
    ) -> None:
        if self._started:
            raise RuntimeError("reset the agent before beginning another task")
        if not self._tool_definitions:
            raise RuntimeError("at least one tool must be registered")

        task_text = [
            f"OBJECTIVE:\n{task.objective}",
            f"SOFTWARE:\n{software_description}",
        ]
        if task.inputs:
            task_text.append(
                "TASK CONTEXT:\n"
                + json.dumps(task.inputs, ensure_ascii=False, sort_keys=True, default=str)
            )
        parts: list[types.Part] = [types.Part(text="\n\n".join(task_text))]
        parts.extend(self._observation_parts(observation))
        self._contents.append(types.Content(role="user", parts=parts))
        self._started = True

    def act(self) -> AgentAction:
        if not self._started:
            raise RuntimeError("begin_task() must be called before act()")

        accumulated_thoughts: list[str] = []
        accumulated_text: list[str] = []
        last_raw: str | None = None
        last_usage: Mapping[str, Any] = {}

        for _ in range(self.no_tool_retries):
            self._apply_image_context_policy()
            response = self.client.models.generate_content(
                model=self.model,
                contents=self._contents,
                config=self._generation_config(),
            )
            last_raw = self._serialize_response(response)
            last_usage = self._usage(response)
            candidate = self._candidate(response)
            parts = list(candidate.content.parts or [])
            accumulated_thoughts.extend(
                str(part.text)
                for part in parts
                if getattr(part, "thought", False) and getattr(part, "text", None)
            )
            accumulated_text.extend(
                str(part.text)
                for part in parts
                if getattr(part, "text", None) and not getattr(part, "thought", False)
            )
            calls = [
                part.function_call
                for part in parts
                if getattr(part, "function_call", None) is not None
            ]
            self._contents.append(candidate.content)

            if len(calls) == 1:
                call = calls[0]
                name = str(call.name)
                if name not in self._tool_definitions:
                    raise AgentProtocolError(f"model called unregistered tool {name!r}")
                return AgentAction(
                    name=name,
                    arguments=dict(call.args or {}),
                    thought="\n".join(accumulated_thoughts).strip(),
                    text="\n".join(accumulated_text).strip(),
                    usage=last_usage,
                    raw_response_json=last_raw,
                )
            if len(calls) > 1:
                raise AgentProtocolError(
                    f"model called {len(calls)} tools in one turn; exactly one is allowed"
                )
            self._contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text="Call exactly one registered tool.")],
                )
            )

        raise AgentProtocolError(
            "model did not call a tool after "
            f"{self.no_tool_retries} response attempts; text={accumulated_text!r}"
        )

    def observe(self, action: AgentAction, step_response: Mapping[str, Any], observation: Observation | None) -> None:
        if not self._started:
            raise RuntimeError("begin_task() must be called before observe()")
        response = dict(step_response)
        parts: list[types.Part] = [
            types.Part.from_function_response(name=action.name, response=response)
        ]
        if observation is not None:
            parts.extend(self._observation_parts(observation))
        self._contents.append(types.Content(role="user", parts=parts))

    def _generation_config(self) -> types.GenerateContentConfig:
        declarations = [
            types.FunctionDeclaration(
                name=definition.name,
                description=definition.description,
                parameters_json_schema=dict(definition.parameters),
            )
            for definition in self._tool_definitions.values()
        ]
        return types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=[types.Tool(function_declarations=declarations)],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=list(self._tool_definitions),
                )
            ),
            temperature=self.temperature,
            seed=self.seed,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=self.thinking_level,
                include_thoughts=True,
            ),
        )

    def _observation_parts(self, observation: Observation) -> list[types.Part]:
        parts: list[types.Part] = []
        if observation.text:
            parts.append(types.Part(text=observation.text))
        if observation.record:
            parts.append(
                types.Part(
                    text="SOFTWARE STATE:\n"
                    + json.dumps(
                        observation.record,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
            )
        for media in observation.media:
            if media.label:
                parts.append(types.Part(text=media.label))
            parts.append(self._media_part(media))
        return parts

    def _media_part(self, media: Media) -> types.Part:
        if media.data is not None:
            return types.Part.from_bytes(data=media.data, mime_type=media.mime_type)
        if media.uri is not None:
            return types.Part.from_uri(file_uri=media.uri, mime_type=media.mime_type)

        assert media.path is not None
        path = media.path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if media.mime_type.startswith("image/"):
            return types.Part.from_bytes(data=path.read_bytes(), mime_type=media.mime_type)

        uri, mime_type = self._upload_media(path, media.mime_type)
        return types.Part.from_uri(file_uri=uri, mime_type=mime_type)

    def _apply_image_context_policy(self) -> None:
        if self.image_context_mode == "all":
            return
        image_parts = [
            (content_index, part_index)
            for content_index, content in enumerate(self._contents)
            for part_index, part in enumerate(content.parts or [])
            if self._is_image_part(part)
        ]
        if len(image_parts) < self.image_context_window:
            return
        for content_index, part_index in image_parts[: self.image_context_drop]:
            self._contents[content_index].parts[part_index] = types.Part(
                text=EARLIER_IMAGE_PLACEHOLDER
            )

    @staticmethod
    def _is_image_part(part: types.Part) -> bool:
        for attribute in ("inline_data", "file_data"):
            data = getattr(part, attribute, None)
            mime_type = str(getattr(data, "mime_type", "") or "")
            if mime_type.startswith("image/"):
                return True
        return False

    def _upload_media(self, path: Path, mime_type: str) -> tuple[str, str]:
        cached = self._uploaded_media.get(path)
        if cached is not None:
            return cached
        uploaded = self.client.files.upload(
            file=path,
            config=types.UploadFileConfig(
                mime_type=mime_type,
                display_name=path.name,
            ),
        )
        deadline = time.monotonic() + self.media_processing_timeout_seconds
        while str(getattr(uploaded, "state", "")) not in {
            "FileState.ACTIVE",
            "ACTIVE",
        }:
            state = str(getattr(uploaded, "state", ""))
            if state in {"FileState.FAILED", "FAILED"}:
                raise RuntimeError(f"Gemini media processing failed for {path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Gemini media processing timed out for {path}")
            time.sleep(2)
            uploaded = self.client.files.get(name=uploaded.name)
        if not uploaded.uri:
            raise RuntimeError(f"Gemini upload returned no URI for {path}")
        value = (str(uploaded.uri), str(uploaded.mime_type or mime_type))
        self._uploaded_media[path] = value
        return value

    @staticmethod
    def _candidate(response: Any) -> Any:
        candidates = list(getattr(response, "candidates", None) or [])
        if not candidates or getattr(candidates[0], "content", None) is None:
            raise AgentProtocolError("Gemini returned no candidate content")
        return candidates[0]

    @staticmethod
    def _serialize_response(response: Any) -> str | None:
        dump = getattr(response, "model_dump_json", None)
        if callable(dump):
            return str(dump(exclude_none=True))
        return None

    @staticmethod
    def _usage(response: Any) -> Mapping[str, Any]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        dump = getattr(usage, "model_dump", None)
        if callable(dump):
            return dict(dump(exclude_none=True))
        if isinstance(usage, Mapping):
            return dict(usage)
        return {"value": str(usage)}
