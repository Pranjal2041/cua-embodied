"""The DialNav Guide: a separate model context that knows the scene and the target.

The Guide never sees the agent's screen. For each message it receives the
agent's text, a Guide-only atlas of every panorama in the scene and the
conversation so far. It first infers the agent's location from the text, then
looks at route previews from that location to the destination and writes
directions in ordinary language. Only that text goes back to the agent.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Iterable

from google import genai
from google.genai import types

from .env import (
    DialNavEpisodeEnv,
    panorama_bytes,
    render_labeled_panorama,
    render_route_preview,
)


TEMPERATURE = 0.6
THINKING_LEVEL = "high"
MAX_OUTPUT_TOKENS = 4096
SEED = 0
GUIDE_ROUTE_HOPS = 6
GUIDE_PROTOCOL = "conversation_and_arrival_v2"


GUIDE_LOCALIZER_PROMPT = """You are the remote Guide's localization component in DialNav. The Navigator cannot send images or hidden identifiers; you receive only its ordinary natural-language question. You do have the complete Guide-only panorama atlas of this Matterport scene. Each atlas location has an internal L### label that is never shown or repeated to the Navigator.

Match the Navigator's described room, objects, geometry, levels, doors, and recent motion against the atlas. Return the three most plausible atlas labels in descending order. Do not infer from the target clue alone: localize from the Navigator's question. Call rank_locations exactly once."""


GUIDE_VERIFY_PROMPT = """You are the remote Guide's final visual localizer. Compare the Navigator's natural-language question against the full-resolution candidate panoramas. Select the single candidate label that best matches the described visible scene. This internal label is never communicated to the Navigator. Call select_location exactly once."""


GUIDE_BASE_ANSWER_PROMPT = """You are the remote Guide in DialNav. You know the full environment and the target. A separate localization stage selected the Navigator's probable location, and the Guide-only route preview shows successive panoramas along a shortest route to the target. In each preview panel, the green mark indicates the exit that the Navigator should take from that panel.

Write a concise, ordinary natural-language route instruction for the Navigator. Translate the highlighted exits into visible landmarks, relative turns, room transitions, doors, corridors, and stairs. Never mention colors or numbers of overlay markers, route-step labels, atlas labels, viewpoint IDs, coordinates, graph nodes, shortest paths, internal localization, or any machine-readable code. Do not claim to see the Navigator's live image. Describe at most the shown route segment; if it does not reach the destination, ask the Navigator to report what they see after completing the described segment. If the selected location is already in the target region, clearly tell the Navigator to stop there."""

GUIDE_TEMPORAL_RULE = (
    "\n\nKeep the current location separate from positions the Navigator will reach "
    "after following your directions. A destination panorama at the end of a "
    "route with remaining moves shows a future position. Give the remaining movement "
    "instructions and describe the stopping condition using future or conditional "
    "language, such as 'Once you reach ..., stop there.' Say that the Navigator "
    "has already arrived only when its inferred current location is inside the "
    "target region and zero moves remain."
)

GUIDE_DESTINATION_RULE = (
    "\n\nWhen an arrival panorama is provided, use it to describe the precise stopping "
    "position relative to the surrounding landmarks. Compare it with the final "
    "departure panorama and explain the remaining movement between them. Identify "
    "which part or side of a landmark the Navigator should stand beside, and the "
    "arrangement of nearby landmarks at arrival. Choose visible details that "
    "distinguish that position from places along the approach."
)

GUIDE_ANSWER_PROMPT = GUIDE_BASE_ANSWER_PROMPT + GUIDE_TEMPORAL_RULE + GUIDE_DESTINATION_RULE


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def schema_string(description: str) -> types.Schema:
    return types.Schema(type=types.Type.STRING, description=description)


RANK_TOOL = types.FunctionDeclaration(
    name="rank_locations",
    description="Return the three most plausible Guide-atlas location labels in descending order.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "location_codes": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="Exactly three L### atlas labels, most likely first.",
            ),
            "evidence": schema_string("Brief visual-language evidence for the ranking."),
        },
        required=["location_codes", "evidence"],
    ),
)


SELECT_TOOL = types.FunctionDeclaration(
    name="select_location",
    description="Select one final Guide-atlas location label.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "location_code": schema_string("One of the supplied L### candidate labels."),
            "evidence": schema_string("Brief evidence for the final match."),
        },
        required=["location_code", "evidence"],
    ),
)


def usage_dict(response: Any) -> dict[str, int]:
    usage = response.usage_metadata
    return {
        "prompt": int(usage.prompt_token_count or 0),
        "output": int(usage.candidates_token_count or 0),
        "thinking": int(getattr(usage, "thoughts_token_count", 0) or 0),
        "cached": int(getattr(usage, "cached_content_token_count", 0) or 0),
    }


def response_parts(response: Any) -> list[Any]:
    if response.candidates and response.candidates[0].content:
        return list(response.candidates[0].content.parts or [])
    return []


def response_thought(response: Any) -> str:
    return " ".join(
        part.text
        for part in response_parts(response)
        if getattr(part, "thought", False) and getattr(part, "text", None)
    )


def response_text(response: Any) -> str:
    return " ".join(
        part.text
        for part in response_parts(response)
        if getattr(part, "text", None) and not getattr(part, "thought", False)
    ).strip()


def response_call(response: Any) -> Any | None:
    return next(
        (
            part.function_call
            for part in response_parts(response)
            if getattr(part, "function_call", None)
        ),
        None,
    )


def response_truncated(response: Any) -> bool:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    return getattr(reason, "value", reason) == "MAX_TOKENS"


def guide_dialog_context(dialog: Iterable[dict[str, Any]]) -> str:
    """Expose only the natural-language channel, excluding evaluation metadata."""
    turns = [
        f"Navigator: {item.get('question', '')}\nGuide: {item.get('answer', '')}"
        for item in dialog
    ]
    if not turns:
        return ""
    return (
        "\n\nPrevious conversation, oldest first:\n"
        + "\n\n".join(turns)
        + "\nEnd previous conversation.\n"
        "Use the conversation to interpret references and recent movements. "
        "Earlier Guide directions may have been mistaken; check the current "
        "description against the visual evidence before choosing a location."
    )


def persist_response(raw_dir: Path, name: str, response: Any) -> None:
    try:
        (raw_dir / f"{name}.json").write_text(
            response.model_dump_json(exclude_none=True), encoding="utf-8"
        )
    except Exception:
        (raw_dir / f"{name}.txt").write_text(repr(response), encoding="utf-8")


def sum_usage(records: Iterable[dict[str, int]]) -> dict[str, int]:
    records = list(records)
    return {
        key: sum(record.get(key, 0) for record in records)
        for key in ("prompt", "output", "thinking", "cached")
    }


def normalize_code(value: Any) -> str:
    match = re.search(r"L\d{3}", str(value).upper())
    return match.group(0) if match else ""


class Guide:
    def __init__(self, client: genai.Client, *, model: str, seed: int = SEED,
                 temperature: float = TEMPERATURE) -> None:
        self.client = client
        self.model = model
        self.seed = seed
        self.temperature = temperature

    def _config(self, tools: list[types.FunctionDeclaration] | None = None) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "seed": self.seed,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking_config": types.ThinkingConfig(
                thinking_level=THINKING_LEVEL,
                include_thoughts=True,
            ),
        }
        if tools:
            kwargs["tools"] = [types.Tool(function_declarations=tools)]
            kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[tool.name for tool in tools],
                )
            )
        return types.GenerateContentConfig(**kwargs)

    def _generate(
        self,
        *,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except Exception as error:
                last_error = error
                if attempt == 4:
                    break
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Gemini call failed after retries: {last_error}")

    def _generate_required(
        self,
        *,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        required_function_name: str,
        raw_dir: Path,
        raw_name: str,
    ) -> tuple[Any, Any, int]:
        """Retry a Guide response that violates its already-forced tool schema.

        Guide rank/verify calls use FunctionCallingConfigMode.ANY with one allowed
        function.  A text-only response is therefore a backend/schema violation,
        not a valid policy action.  Preserve every invalid response and retry the
        request up to five times. Token-truncated responses receive a larger output
        budget on retry; semantic argument validation remains unchanged and fail-closed.
        """

        observed: list[str] = []
        config = config.model_copy()
        for schema_attempt in range(5):
            response = self._generate(contents=contents, config=config)
            suffix = "" if schema_attempt == 0 else f"_schema_retry{schema_attempt}"
            persist_response(raw_dir, raw_name + suffix, response)
            if response_truncated(response):
                observed.append("MAX_TOKENS")
                config.max_output_tokens = min(16384, (config.max_output_tokens or MAX_OUTPUT_TOKENS) * 2)
                continue
            call = response_call(response)
            if call is not None and call.name == required_function_name:
                return response, call, schema_attempt
            observed.append(call.name if call is not None else "<no_function_call>")
        raise RuntimeError(
            f"required function {required_function_name!r} absent after schema retries; "
            f"observed {observed}"
        )

    def localize(
        self,
        *,
        question: str,
        target: str,
        atlas_pages: list[bytes],
        code_to_viewpoint: dict[str, str],
        rgb_dir: Path,
        raw_dir: Path,
        guide_dir: Path,
        ask_index: int,
        dialog: Iterable[dict[str, Any]] = (),
    ) -> tuple[str, dict[str, Any]]:
        context = guide_dialog_context(dialog)
        rank_parts = [
            types.Part(
                text=(
                    GUIDE_LOCALIZER_PROMPT
                    + f"\n\nTarget clue known to the Guide: {target}"
                    + context
                    + f"\nNavigator question: {question}"
                )
            )
        ] + [types.Part.from_bytes(data=page, mime_type="image/png") for page in atlas_pages]
        rank_response, rank_call, rank_schema_retries = self._generate_required(
            contents=[types.Content(role="user", parts=rank_parts)],
            config=self._config([RANK_TOOL]),
            required_function_name="rank_locations",
            raw_dir=raw_dir,
            raw_name=f"guide_{ask_index:02d}_rank",
        )
        rank_args = dict(rank_call.args or {})
        ranked: list[str] = []
        for value in rank_args.get("location_codes", []):
            code = normalize_code(value)
            if code in code_to_viewpoint and code not in ranked:
                ranked.append(code)
        if not ranked:
            raise RuntimeError(f"Guide localizer returned no valid atlas labels: {rank_args}")
        ranked = ranked[:3]

        candidate_parts = [
            types.Part(
                text=(
                    GUIDE_VERIFY_PROMPT
                    + f"\n\nTarget clue known to the Guide: {target}"
                    + context
                    + f"\nNavigator question: {question}"
                    + f"\nAllowed final candidates: {', '.join(ranked)}"
                )
            )
        ]
        for code in ranked:
            image = render_labeled_panorama(code_to_viewpoint[code], code, rgb_dir=rgb_dir)
            (guide_dir / f"ask{ask_index:02d}_candidate_{code}.png").write_bytes(image)
            candidate_parts.append(types.Part.from_bytes(data=image, mime_type="image/png"))
        verify_response, verify_call, verify_schema_retries = self._generate_required(
            contents=[types.Content(role="user", parts=candidate_parts)],
            config=self._config([SELECT_TOOL]),
            required_function_name="select_location",
            raw_dir=raw_dir,
            raw_name=f"guide_{ask_index:02d}_verify",
        )
        verify_args = dict(verify_call.args or {})
        selected = normalize_code(verify_args.get("location_code", ""))
        if selected not in ranked:
            raise RuntimeError(
                f"Guide verifier selected {selected!r}, outside ranked candidates {ranked}"
            )
        return code_to_viewpoint[selected], {
            "ranked_codes": ranked,
            "selected_code": selected,
            "rank_evidence": rank_args.get("evidence", ""),
            "verify_evidence": verify_args.get("evidence", ""),
            "rank_thought": response_thought(rank_response),
            "verify_thought": response_thought(verify_response),
            "rank_schema_retries": rank_schema_retries,
            "verify_schema_retries": verify_schema_retries,
            "conversation_context": context,
            "usage": sum_usage([usage_dict(rank_response), usage_dict(verify_response)]),
        }

    def answer(
        self,
        *,
        env: DialNavEpisodeEnv,
        localized_viewpoint: str,
        question: str,
        raw_dir: Path,
        guide_dir: Path,
        ask_index: int,
    ) -> tuple[str, list[str], dict[str, Any]]:
        route = env.graph.shortest_path(localized_viewpoint, env.end_panos)
        preview = render_route_preview(
            env.graph,
            route,
            max_hops=GUIDE_ROUTE_HOPS,
            rgb_dir=env.rgb_dir,
        )
        (guide_dir / f"ask{ask_index:02d}_route_preview.png").write_bytes(preview)
        reaches_goal = len(route) - 1 <= GUIDE_ROUTE_HOPS
        localized_in_target = localized_viewpoint in env.end_panos
        arrival = None
        if reaches_goal and not localized_in_target:
            # The move montage contains departure views. The answerer also needs
            # the arrival view to describe the precise stopping location.
            arrival = panorama_bytes(route[-1], env.rgb_dir)
            (guide_dir / f"ask{ask_index:02d}_arrival.png").write_bytes(arrival)
        route_status = (
            "Your inferred location is already in the target region. Zero moves remain. "
            "Tell the Navigator to stop at its current location."
            if localized_in_target else
            f"Your inferred current location requires further movement. {len(route) - 1} "
            "moves remain on the complete route from that inferred location. "
            f"The shown segment {'reaches' if reaches_goal else 'does not reach'} the target region."
        )
        base_prompt = (
            GUIDE_ANSWER_PROMPT
            + f"\n\nTarget clue: {env.target}"
            + guide_dialog_context(env.dialog)
            + f"\nNavigator question: {question}"
            + "\n" + route_status
        )
        responses: list[Any] = []
        violations: list[str] = []
        answer = ""
        output_budget = MAX_OUTPUT_TOKENS
        for answer_attempt in range(1, 4):
            correction = ""
            if violations:
                correction = (
                    "\n\nYour previous draft could not be sent: "
                    f"{violations[-1]}. Write a complete, concise instruction "
                    "using ordinary visual route language."
                )
            parts = [types.Part(text=base_prompt + correction),
                     types.Part.from_bytes(data=preview, mime_type="image/png")]
            if arrival is not None:
                parts.extend([
                    types.Part(text=(
                        "The following panorama shows future arrival at the destination after "
                        "the last highlighted move. Use its visible landmarks to describe "
                        "where the Navigator should stop."
                    )),
                    types.Part.from_bytes(data=arrival, mime_type="image/png"),
                ])
            config = self._config()
            config.max_output_tokens = output_budget
            response = self._generate(
                contents=[
                    types.Content(
                        role="user",
                        parts=parts,
                    )
                ],
                config=config,
            )
            responses.append(response)
            suffix = "answer" if answer_attempt == 1 else f"answer_repair{answer_attempt - 1}"
            persist_response(raw_dir, f"guide_{ask_index:02d}_{suffix}", response)
            if response_truncated(response):
                violations.append("response truncated at MAX_TOKENS")
                output_budget = min(16384, output_budget * 2)
                continue
            answer = response_text(response)
            forbidden = re.search(
                r"\b(?:L\d{3}|viewpoint|graph nodes?|green (?:mark|marker|circle|line)|"
                r"markers?\s*\d+|route steps?|shortest path|coordinates?|overlay|highlighted)\b",
                answer,
                flags=re.IGNORECASE,
            )
            if answer and forbidden is None:
                break
            violations.append(repr(forbidden.group(0)) if forbidden else "empty answer")
        else:
            raise RuntimeError(
                "Guide answer could not be converted to a legal natural-language message: "
                + ", ".join(violations)
            )
        return answer, route, {
            "thought": response_thought(responses[-1]),
            "usage": sum_usage(usage_dict(response) for response in responses),
            "answer_attempts": len(responses),
            "rejected_draft_violations": violations,
            "preview_sha256": bytes_sha256(preview),
            "arrival_image_sha256": bytes_sha256(arrival) if arrival is not None else None,
            "dialog_turns_in_context": len(env.dialog),
            "preview_hops": min(GUIDE_ROUTE_HOPS, max(0, len(route) - 1)),
            "remaining_route_hops": max(0, len(route) - 1),
            "segment_reaches_goal": reaches_goal,
            "localized_in_target_region": localized_in_target,
        }

    def turn(
        self,
        *,
        env: DialNavEpisodeEnv,
        question: str,
        atlas_pages: list[bytes],
        code_to_viewpoint: dict[str, str],
        raw_dir: Path,
        guide_dir: Path,
        ask_index: int,
    ) -> tuple[str, str, dict[str, Any]]:
        context = guide_dialog_context(env.dialog)
        (guide_dir / f"ask{ask_index:02d}_conversation.txt").write_text(context, encoding="utf-8")
        localized, localization = self.localize(
            question=question,
            target=env.target,
            atlas_pages=atlas_pages,
            code_to_viewpoint=code_to_viewpoint,
            rgb_dir=env.rgb_dir,
            raw_dir=raw_dir,
            guide_dir=guide_dir,
            ask_index=ask_index,
            dialog=env.dialog,
        )
        answer, route, answering = self.answer(
            env=env,
            localized_viewpoint=localized,
            question=question,
            raw_dir=raw_dir,
            guide_dir=guide_dir,
            ask_index=ask_index,
        )
        return answer, localized, {
            "guide_protocol": GUIDE_PROTOCOL,
            "localization": localization,
            "answering": answering,
            "internal_route": route,
            "localization_error_m": env.graph.shortest_distance(
                env.current, [localized]
            ),
        }
