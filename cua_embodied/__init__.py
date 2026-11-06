"""Computer-use agents as embodied generalists.

One computer-use agent, with a fixed design, operates task software through
screenshots and mouse and keyboard input. Only the software changes between
tasks. See README.md for setup and usage.
"""
from .agent import AgentAction, AgentProtocolError, GeminiAgent
from .core import BaseSoftware, Media, Observation, SoftwareStep, Task, ToolDefinition
from .runner import RunResult, RunTurn, Runner, run_task

__all__ = [
    "AgentAction",
    "AgentProtocolError",
    "BaseSoftware",
    "GeminiAgent",
    "Media",
    "Observation",
    "RunResult",
    "RunTurn",
    "Runner",
    "SoftwareStep",
    "Task",
    "ToolDefinition",
    "run_task",
]
