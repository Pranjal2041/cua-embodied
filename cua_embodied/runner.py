"""The agent loop: act, execute in the software, observe, until the software reports done."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from .agent import AgentAction
from .core import BaseSoftware, Media, Observation, Task


class Agent(Protocol):
    model: str

    def reset(self) -> None: ...

    def register_tools(self, definitions: Any) -> None: ...

    def begin_task(
        self,
        task: Task,
        *,
        software_description: str,
        observation: Observation,
    ) -> None: ...

    def act(self) -> AgentAction: ...

    def observe(
        self,
        action: AgentAction,
        step_response: Mapping[str, Any],
        observation: Observation | None,
    ) -> None: ...


@dataclass(frozen=True)
class RunTurn:
    index: int
    action: AgentAction
    response: Mapping[str, Any]
    observation: Mapping[str, Any] | None
    done: bool
    output: Any = None


@dataclass(frozen=True)
class RunResult:
    task_id: str
    software: str
    model: str
    status: str
    turns: tuple[RunTurn, ...] = ()
    output: Any = None
    error: str | None = None


@dataclass
class Runner:
    agent: Agent
    software: BaseSoftware
    max_turns: int = 128
    output_dir: Path | None = None
    _turns: list[RunTurn] = field(default_factory=list, init=False)

    def run(self, task: Task) -> RunResult:
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        self._turns = []
        if self.output_dir is not None:
            self.output_dir = self.output_dir.resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.agent.reset()
            self.agent.register_tools(self.software.tools)
            initial = self.software.reset(task)
            self.agent.begin_task(
                task,
                software_description=self.software.description,
                observation=initial,
            )
            self._write_initial(task, initial)

            for index in range(1, self.max_turns + 1):
                action = self.agent.act()
                self._write_raw_response(index, action.raw_response_json)
                step = self.software.execute(action.name, action.arguments)
                feedback = dict(step.response)
                feedback.setdefault("done", step.done)
                self.agent.observe(action, feedback, step.observation)
                turn = RunTurn(
                    index=index,
                    action=action,
                    response=feedback,
                    observation=_observation_record(step.observation),
                    done=step.done,
                    output=step.output,
                )
                self._turns.append(turn)
                self._write_trace(task, status="running")
                if step.done:
                    result = self._result(task, status="completed", output=step.output)
                    self._write_result(result)
                    return result

            result = self._result(task, status="max_turns")
            self._write_result(result)
            return result
        except Exception as exc:
            result = self._result(
                task,
                status="error",
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
            self._write_result(result)
            raise
        finally:
            self.software.close()

    def _result(
        self,
        task: Task,
        *,
        status: str,
        output: Any = None,
        error: str | None = None,
    ) -> RunResult:
        return RunResult(
            task_id=task.task_id,
            software=self.software.name,
            model=str(getattr(self.agent, "model", type(self.agent).__name__)),
            status=status,
            turns=tuple(self._turns),
            output=output,
            error=error,
        )

    def _write_initial(self, task: Task, observation: Observation) -> None:
        if self.output_dir is None:
            return
        _atomic_json(
            self.output_dir / "task.json",
            {
                "task": asdict(task),
                "software": self.software.name,
                "software_description": self.software.description,
                "tools": [asdict(tool) for tool in self.software.tools],
                "initial_observation": _observation_record(observation),
            },
        )

    def _write_raw_response(self, index: int, raw: str | None) -> None:
        if self.output_dir is None or raw is None:
            return
        path = self.output_dir / "raw_model_responses" / f"turn_{index:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw, encoding="utf-8")

    def _write_trace(self, task: Task, *, status: str) -> None:
        if self.output_dir is None:
            return
        result = self._result(task, status=status)
        _atomic_json(self.output_dir / "trace.json", _result_record(result))

    def _write_result(self, result: RunResult) -> None:
        if self.output_dir is None:
            return
        record = _result_record(result)
        _atomic_json(self.output_dir / "trace.json", record)
        _atomic_json(self.output_dir / "result.json", record)


def run_task(
    agent: Agent,
    software: BaseSoftware,
    task: Task,
    *,
    max_turns: int = 128,
    output_dir: str | Path | None = None,
) -> RunResult:
    return Runner(
        agent=agent,
        software=software,
        max_turns=max_turns,
        output_dir=Path(output_dir) if output_dir is not None else None,
    ).run(task)


def _observation_record(observation: Observation | None) -> Mapping[str, Any] | None:
    if observation is None:
        return None
    return {
        "text": observation.text,
        "record": _jsonable(observation.record),
        "media": [_media_record(media) for media in observation.media],
    }


def _media_record(media: Media) -> Mapping[str, Any]:
    source: str
    if media.path is not None:
        source = str(media.path)
    elif media.uri is not None:
        source = media.uri
    else:
        source = f"inline:{len(media.data or b'')} bytes"
    return {
        "label": media.label,
        "mime_type": media.mime_type,
        "source": source,
    }


def _action_record(action: AgentAction) -> Mapping[str, Any]:
    return {
        "name": action.name,
        "arguments": _jsonable(action.arguments),
        "thought": action.thought,
        "text": action.text,
        "usage": _jsonable(action.usage),
        "raw_response_path": None,
    }


def _result_record(result: RunResult) -> Mapping[str, Any]:
    return {
        "task_id": result.task_id,
        "software": result.software,
        "model": result.model,
        "status": result.status,
        "turns": [
            {
                "index": turn.index,
                "action": _action_record(turn.action),
                "response": _jsonable(turn.response),
                "observation": _jsonable(turn.observation),
                "done": turn.done,
                "output": _jsonable(turn.output),
            }
            for turn in result.turns
        ],
        "output": _jsonable(result.output),
        "error": result.error,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
