"""Text-API tools for DialNav, used only in the GUI versus API comparison."""

from __future__ import annotations

from ...core import BaseSoftware, ToolDefinition, integer, object_schema, string


class DialNavAPI(BaseSoftware):
    name = "dialnav"
    description = (
        "A 360-degree navigation environment with legal movement markers and a text "
        "channel to the benchmark Guide. The Guide is a capability of this software."
    )
    tool_definitions = (
        ToolDefinition(
            name="move",
            description="Traverse one legal transition shown in the current panorama.",
            parameters=object_schema(
                {
                    "marker": integer("Number printed on the selected legal move marker.", minimum=0),
                    "reason": string("Brief visual or Guide-based reason for the move."),
                },
                required=("marker", "reason"),
            ),
        ),
        ToolDefinition(
            name="type",
            description="Send an ordinary natural-language message to the benchmark Guide.",
            parameters=object_schema(
                {"text": string("Visible-scene description and navigation question.")},
                required=("text",),
            ),
        ),
        ToolDefinition(
            name="stop",
            description="Finish after reaching the destination region.",
            parameters=object_schema(
                {"reason": string("Brief reason the destination has been reached.")},
                required=("reason",),
            ),
        ),
    )

