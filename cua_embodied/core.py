"""Core data types shared by the agent, the runner and the task software."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


JSON = Any


@dataclass(frozen=True)
class Task:
    """One competition task presented to the multimodal agent."""

    task_id: str
    objective: str
    inputs: Mapping[str, JSON] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, JSON]) -> "Task":
        objective = str(value.get("objective", "")).strip()
        if not objective:
            raise ValueError("task objective is required")
        return cls(
            task_id=str(value.get("task_id", value.get("id", "task"))),
            objective=objective,
            inputs=dict(value.get("inputs", {})),
        )


@dataclass(frozen=True)
class Media:
    """A model-visible image, video, or other media attachment."""

    mime_type: str
    data: bytes | None = None
    path: Path | None = None
    uri: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        sources = sum(value is not None for value in (self.data, self.path, self.uri))
        if sources != 1:
            raise ValueError("Media requires exactly one of data, path, or uri")
        if not self.mime_type:
            raise ValueError("Media mime_type is required")


@dataclass(frozen=True)
class Observation:
    """Information returned by software after reset or an action."""

    text: str = ""
    media: tuple[Media, ...] = ()
    record: Mapping[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDefinition:
    """Model-visible definition for one software action."""

    name: str
    description: str
    parameters: Mapping[str, JSON]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name is required")
        if not self.description:
            raise ValueError(f"tool {self.name!r} requires a description")
        if self.parameters.get("type") != "object":
            raise ValueError(f"tool {self.name!r} parameters must be an object schema")


@dataclass(frozen=True)
class SoftwareStep:
    """Public feedback, new observations, and optional terminal output."""

    response: Mapping[str, JSON]
    observation: Observation | None = None
    done: bool = False
    output: JSON = None


class SoftwareBackend(Protocol):
    """Backend that implements one task's software operations."""

    def reset(self, task: Task) -> Observation: ...

    def execute(self, tool_name: str, arguments: Mapping[str, JSON]) -> SoftwareStep: ...

    def close(self) -> None: ...


class BaseSoftware(ABC):
    """Software exposed to the agent through a fixed set of tools."""

    name = "base"
    description = ""
    tool_definitions: Sequence[ToolDefinition] = ()

    def __init__(self, backend: SoftwareBackend):
        self.backend = backend
        self.task: Task | None = None
        self._tools = {tool.name: tool for tool in self.tool_definitions}
        if len(self._tools) != len(self.tool_definitions):
            raise ValueError(f"{type(self).__name__} declares duplicate tool names")

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        return tuple(self.tool_definitions)

    def reset(self, task: Task) -> Observation:
        self.task = task
        observation = self.backend.reset(task)
        if not isinstance(observation, Observation):
            raise TypeError("software backend reset() must return Observation")
        return observation

    def execute(self, tool_name: str, arguments: Mapping[str, JSON]) -> SoftwareStep:
        if self.task is None:
            raise RuntimeError("software must be reset before execute")
        if tool_name not in self._tools:
            raise ValueError(f"{self.name} does not expose tool {tool_name!r}")
        step = self.backend.execute(tool_name, dict(arguments))
        if not isinstance(step, SoftwareStep):
            raise TypeError("software backend execute() must return SoftwareStep")
        return step

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


def object_schema(
    properties: Mapping[str, JSON] | None = None,
    *,
    required: Sequence[str] = (),
    additional_properties: bool = False,
) -> dict[str, JSON]:
    schema: dict[str, JSON] = {
        "type": "object",
        "properties": dict(properties or {}),
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = list(required)
    return schema


def number(description: str, *, minimum: float | None = None, maximum: float | None = None) -> dict[str, JSON]:
    schema: dict[str, JSON] = {"type": "number", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def integer(description: str, *, minimum: int | None = None, maximum: int | None = None) -> dict[str, JSON]:
    schema: dict[str, JSON] = {"type": "integer", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def string(description: str, *, enum: Sequence[str] | None = None) -> dict[str, JSON]:
    schema: dict[str, JSON] = {"type": "string", "description": description}
    if enum is not None:
        schema["enum"] = list(enum)
    return schema


def string_array(description: str, *, min_items: int = 0) -> dict[str, JSON]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": description,
        "minItems": min_items,
    }

