"""Text-API tools for VLNVerse, used only in the GUI versus API comparison."""

from __future__ import annotations

from ...core import BaseSoftware, ToolDefinition, number, object_schema, string


class VLNVerseAPI(BaseSoftware):
    name = "vlnverse"
    description = (
        "An egocentric 3D home. The software returns a fresh RGB view and movement "
        "feedback after every action. The objective is complete when the camera reaches "
        "the described target location."
    )
    tool_definitions = (
        ToolDefinition(
            name="move",
            description="Move relative to the current camera orientation.",
            parameters=object_schema(
                {
                    "direction": string(
                        "Movement direction.",
                        enum=("forward", "backward", "left", "right", "up", "down"),
                    ),
                    "distance_m": number("Distance in metres.", minimum=0.0, maximum=3.0),
                },
                required=("direction", "distance_m"),
            ),
        ),
        ToolDefinition(
            name="rotate",
            description="Rotate the current camera in place.",
            parameters=object_schema(
                {
                    "yaw_degrees": number("Horizontal rotation; positive turns right.", minimum=-180, maximum=180),
                    "pitch_degrees": number("Vertical rotation; positive looks up.", minimum=-89, maximum=89),
                }
            ),
        ),
        ToolDefinition(
            name="stop",
            description="Finish after reaching the location described by the objective.",
            parameters=object_schema(
                {"reason": string("Brief reason the objective is complete.")},
                required=("reason",),
            ),
        ),
    )

