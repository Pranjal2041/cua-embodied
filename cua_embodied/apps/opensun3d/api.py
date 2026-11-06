"""Text-API tools for OpenSUN3D, used only in the GUI versus API comparison."""

from __future__ import annotations

from ...core import (
    BaseSoftware,
    ToolDefinition,
    number,
    object_schema,
    string,
    string_array,
)


class OpenSUN3DAPI(BaseSoftware):
    name = "opensun3d"
    description = (
        "An interactive colored 3D scan with registered video viewpoints and a 3D "
        "segmentation service. Clicking a target produces inspectable 3D proposals."
    )
    tool_definitions = (
        ToolDefinition(
            name="seek",
            description="Move the 3D camera to a registered source-video timestamp.",
            parameters=object_schema(
                {
                    "video_id": string("Source video identifier."),
                    "seconds": number("Time from the start of the source video.", minimum=0.0),
                },
                required=("video_id", "seconds"),
            ),
        ),
        ToolDefinition(
            name="move",
            description="Move the free camera relative to its current orientation.",
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
            description="Rotate the free camera in place.",
            parameters=object_schema(
                {
                    "yaw_degrees": number("Horizontal rotation; positive turns right.", minimum=-180, maximum=180),
                    "pitch_degrees": number("Vertical rotation; positive looks up.", minimum=-89, maximum=89),
                }
            ),
        ),
        ToolDefinition(
            name="click",
            description=(
                "Click the exact physical interaction region in the current 3D render. "
                "The software returns one or more 3D segmentation proposals."
            ),
            parameters=object_schema(
                {
                    "x": number("Horizontal image coordinate normalized to 0..1000.", minimum=0, maximum=1000),
                    "y": number("Vertical image coordinate normalized to 0..1000.", minimum=0, maximum=1000),
                    "polarity": string("Positive adds evidence; negative removes it.", enum=("positive", "negative")),
                    "proposal_id": string("Optional proposal to refine with this click."),
                },
                required=("x", "y", "polarity"),
            ),
        ),
        ToolDefinition(
            name="inspect",
            description="Render one 3D segmentation proposal as an overlay for visual review.",
            parameters=object_schema(
                {"proposal_id": string("Proposal identifier returned by the software.")},
                required=("proposal_id",),
            ),
        ),
        ToolDefinition(
            name="submit",
            description="Submit the reviewed 3D proposal or proposals that satisfy the objective.",
            parameters=object_schema(
                {
                    "proposal_ids": string_array("One or more accepted 3D proposal identifiers.", min_items=1),
                    "reason": string("Brief reason these proposals satisfy the objective."),
                },
                required=("proposal_ids", "reason"),
            ),
        ),
    )

