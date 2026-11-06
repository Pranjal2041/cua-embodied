"""Text-API tools for MeViSv2, used only in the GUI versus API comparison."""

from __future__ import annotations

from ...core import BaseSoftware, ToolDefinition, integer, number, object_schema, string, string_array


VIDEO_SELECTION_TOOLS = (
    ToolDefinition(
        name="seek",
        description="Inspect one full-resolution frame from the complete video.",
        parameters=object_schema(
            {"frame_index": integer("Zero-based frame index.", minimum=0)},
            required=("frame_index",),
        ),
    ),
    ToolDefinition(
        name="click",
        description="Add or refine an object selection on a chosen video frame.",
        parameters=object_schema(
            {
                "frame_index": integer("Zero-based frame index.", minimum=0),
                "x": number("Horizontal coordinate normalized to 0..1000.", minimum=0, maximum=1000),
                "y": number("Vertical coordinate normalized to 0..1000.", minimum=0, maximum=1000),
                "mode": string(
                    "Selection operation.",
                    enum=("new_object", "refine_positive", "refine_negative"),
                ),
                "object_id": string("Object identifier for a refinement click."),
            },
            required=("frame_index", "x", "y", "mode"),
        ),
    ),
    ToolDefinition(
        name="remove",
        description="Remove one selected object from the current answer.",
        parameters=object_schema(
            {"object_id": string("Selected object identifier.")},
            required=("object_id",),
        ),
    ),
    ToolDefinition(
        name="inspect",
        description="Track the current selections and return the complete overlay video.",
        parameters=object_schema(),
    ),
    ToolDefinition(
        name="submit",
        description="Submit the reviewed object tracks as the answer.",
        parameters=object_schema(
            {
                "object_ids": string_array("Selected object identifiers to submit.", min_items=1),
                "reason": string("Brief reason the tracks satisfy the objective."),
            },
            required=("object_ids", "reason"),
        ),
    ),
    ToolDefinition(
        name="submit_no_target",
        description="Submit an empty answer when the video contains no matching target.",
        parameters=object_schema(
            {"reason": string("Brief reason no target is present.")},
            required=("reason",),
        ),
    ),
)


class MeViSAPI(BaseSoftware):
    name = "mevis"
    description = (
        "A complete video and text motion description with interactive object selection, "
        "SAM3 tracking, correction, and full-video overlay review."
    )
    tool_definitions = VIDEO_SELECTION_TOOLS

