"""What each task app provides to the GUI server and to the text-API arm.

An app wraps one competition's software backend. The backend implements the
operations (move, segment, track, ask the Guide, and so on). The GUI turns
those operations into buttons, shortcuts and clicks. The agent never calls the
backend directly in the GUI setting. It only sees screenshots and sends mouse
and keyboard input.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..core import BaseSoftware, Media, SoftwareBackend, Task

if TYPE_CHECKING:
    from ..gui.session import Session


class App:
    name = ""
    title = ""
    # Task instructions. The GUI shows them under Instructions and the agent
    # receives them in its first message.
    rules = ""
    # Backend operations that GUI controls may call.
    actions: frozenset[str] = frozenset()
    # Backend fields the GUI may display. Evaluation output, ground truth and
    # Guide internals are never listed here.
    observation_fields: frozenset[str] = frozenset()
    response_fields: frozenset[str] = frozenset()
    # Text-API tool definitions, used only for the GUI versus API comparison.
    software: type[BaseSoftware] = BaseSoftware

    def open(self, config: Mapping[str, Any], output_dir: Path) -> tuple[Task, SoftwareBackend]:
        """Load the configured task and build its backend."""
        raise NotImplementedError

    def context_videos(self, backend: Any, config: Mapping[str, Any]) -> tuple[Media, ...]:
        """Complete videos given to the agent at the start, in both interfaces."""
        return ()

    def api_initial_media(self, backend: Any) -> tuple[Media, ...]:
        """Extra media for the text-API arm so it starts from the same view as the GUI."""
        return ()

    def prepare_gui(self, session: "Session") -> None:
        """Set up GUI-only state after the backend is reset."""

    def camera(self, fields: Mapping[str, Any]) -> dict[str, Any] | None:
        """Camera pose shown in the GUI, taken from the visible observation fields."""
        return None

    def refresh(self, session: "Session") -> None:
        """Copy app state that the GUI needs after every operation."""

    def before_action(self, session: "Session", action: str, arguments: Mapping[str, Any]) -> None:
        """Reject an operation the GUI should not allow in the current state."""

    def after_action(self, session: "Session", action: str, arguments: Mapping[str, Any],
                     response: Mapping[str, Any]) -> None:
        """Update GUI state that depends on the operation just executed."""
