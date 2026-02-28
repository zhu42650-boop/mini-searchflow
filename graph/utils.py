# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import logging
import json
from typing import Any
from graph.types import State
ASSISTANT_SPEAKER_NAMES = {
    "coordinator",
    "planner",
    "researcher",
    "coder",
    "reporter",
    "background_investigator",
}

logger = logging.getLogger(__name__)

def get_message_content(message: Any) -> str:
    """Extract message content from dict or LangChain message."""
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def is_user_message(message: Any) -> bool:
    """Return True if the message originated from the end user."""
    if isinstance(message, dict):
        role = (message.get("role") or "").lower()
        if role in {"user", "human"}:
            return True
        if role in {"assistant", "system"}:
            return False
        name = (message.get("name") or "").lower()
        if name and name in ASSISTANT_SPEAKER_NAMES:
            return False
        return role == "" and name not in ASSISTANT_SPEAKER_NAMES

    message_type = (getattr(message, "type", "") or "").lower()
    name = (getattr(message, "name", "") or "").lower()
    if message_type == "human":
        return not (name and name in ASSISTANT_SPEAKER_NAMES)

    role_attr = getattr(message, "role", None)
    if isinstance(role_attr, str) and role_attr.lower() in {"user", "human"}:
        return True

    additional_role = getattr(message, "additional_kwargs", {}).get("role")
    if isinstance(additional_role, str) and additional_role.lower() in {
        "user",
        "human",
    }:
        return True

    return False


def get_latest_user_message(messages: list[Any]) -> tuple[Any, str]:
    """Return the latest user-authored message and its content."""
    for message in reversed(messages or []):
        if is_user_message(message):
            content = get_message_content(message)
            if content:
                return message, content
    return None, ""


def build_clarified_topic_from_history(
    clarification_history: list[str],
) -> tuple[str, list[str]]:
    """Construct clarified topic string from an ordered clarification history."""
    sequence = [item for item in clarification_history if item]
    if not sequence:
        return "", []
    if len(sequence) == 1:
        return sequence[0], sequence
    head, *tail = sequence
    clarified_string = f"{head} - {', '.join(tail)}"
    return clarified_string, sequence


def reconstruct_clarification_history(
    messages: list[Any],
    fallback_history: list[str] | None = None,
    base_topic: str = "",
) -> list[str]:
    """Rebuild clarification history from user-authored messages, with fallback.

    Args:
        messages: Conversation messages in chronological order.
        fallback_history: Optional existing history to use if no user messages found.
        base_topic: Optional topic to use when no user messages are available.

    Returns:
        A cleaned clarification history containing unique consecutive user contents.
    """
    sequence: list[str] = []
    for message in messages or []:
        if not is_user_message(message):
            continue
        content = get_message_content(message)
        if not content:
            continue
        if sequence and sequence[-1] == content:
            continue
        sequence.append(content)

    if sequence:
        return sequence

    fallback = [item for item in (fallback_history or []) if item]
    if fallback:
        return fallback

    base_topic = (base_topic or "").strip()
    return [base_topic] if base_topic else []


def preserve_state_meta_fields(state: State) -> dict:
    """
    Extract meta/config fields that should be preserved across state transitions.
    
    These fields are critical for workflow continuity and should be explicitly
    included in all Command.update dicts to prevent them from reverting to defaults.
    
    Args:
        state: Current state object
        
    Returns:
        Dict of meta fields to preserve
    """
    return {
        "locale": state.get("locale", "en-US"),
        "research_topic": state.get("research_topic", ""),
        "clarified_research_topic": state.get("clarified_research_topic", ""),
        "clarification_history": state.get("clarification_history", []),
        "enable_clarification": state.get("enable_clarification", False),
        "max_clarification_rounds": state.get("max_clarification_rounds", 3),
        "clarification_rounds": state.get("clarification_rounds", 0),
        "resources": state.get("resources", []),
    }


def extract_subquestions_content(subquestions_data: str | dict | Any) -> str:
    """
    Safely extract sub-questions content from different types.

    Args:
        subquestions_data: The decomposition output which can be a string, an
            object with a `content` attribute, or a dict.

    Returns:
        str: The sub-questions content as a string (JSON string for dict inputs,
        or extracted/original string for other types).
    """
    if isinstance(subquestions_data, str):
        return subquestions_data

    # If already a DecompositionResult-like object, return as JSON string
    if hasattr(subquestions_data, "model_dump_json"):
        try:
            return subquestions_data.model_dump_json()
        except Exception:
            return str(subquestions_data)

    if hasattr(subquestions_data, "content") and isinstance(
        subquestions_data.content, str
    ):
        logger.debug(
            "Extracting sub-questions content from message object: %s",
            type(subquestions_data).__name__,
        )
        return subquestions_data.content

    if isinstance(subquestions_data, dict):
        if "content" in subquestions_data:
            content = subquestions_data["content"]
            if isinstance(content, str):
                logger.debug("Extracting sub-questions content from dict content field")
                return content
            if isinstance(content, dict):
                return json.dumps(content, ensure_ascii=False)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
                raise ValueError(
                    "No valid text content found in multimodal list"
                )
            logger.debug(
                "Unexpected type for 'content' field (%s), converting to string",
                type(content).__name__,
            )
            return str(content)

        logger.debug("Converting sub-questions dictionary to JSON string")
        return json.dumps(subquestions_data, ensure_ascii=False)

    logger.warning(
        "Unexpected sub-questions data type %s, converting to string",
        type(subquestions_data).__name__,
    )
    return str(subquestions_data)


def validate_and_fix_subquestions(subquestions: dict, enforce_web_search: bool = False, enable_web_search: bool = True) -> dict:
    """
    Validate and fix a plan to ensure it meets requirements.

    Args:
        plan: The plan dict to validate
        enforce_web_search: If True, ensure at least one step has need_search=true
        enable_web_search: If False, skip web search enforcement (takes precedence)

    Returns:
        The validated/fixed plan dict
    """
    if not isinstance(subquestions, dict):
        return subquestions

    # Ensure required top-level fields exist
    if "has_enough_context" not in subquestions:
        subquestions["has_enough_context"] = False
    if not subquestions.get("title"):
        subquestions["title"] = subquestions.get("thought", "") or "Decomposition Result"

    questions = subquestions.get("questions", [])

    # ============================================================
    # SECTION 1: Repair missing step_type fields (Issue #650 fix)
    # ============================================================
    for idx, subquestion in enumerate(questions):
        if not isinstance(subquestion, dict):
            continue
        
        # Check if step_type is missing or empty
        if "step_type" not in subquestion or not subquestion.get("step_type"):
            # Infer step_type based on need_search value
            # Default to "analysis" for non-search steps (Issue #677: not all processing needs code)
            inferred_type = "research" if subquestion.get("need_search", False) else "analysis"
            subquestion["step_type"] = inferred_type
            logger.info(
                "Repaired missing step_type for question %s (%s): inferred as '%s' based on need_search=%s",
                idx,
                subquestion.get("question", "Untitled"),
                inferred_type,
                subquestion.get("need_search", False),
            )

    # ============================================================
    # SECTION 2: Enforce web search requirements
    # Skip enforcement if web search is disabled (enable_web_search=False takes precedence)
    # ============================================================
    if enforce_web_search and enable_web_search:
        # Check if any step has need_search=true (only check dict steps)
        has_search_step = any(
            subquestion.get("need_search", False)
            for subquestion in questions
            if isinstance(subquestion, dict)
        )

        if not has_search_step and questions:
            # Ensure first research step has web search enabled
            for idx, subquestion in enumerate(questions):
                if isinstance(subquestion, dict) and subquestion.get("step_type") == "research":
                    subquestion["need_search"] = True
                    logger.info(f"Enforced web search on research step at index {idx}")
                    break
            else:
                # Fallback: If no research step exists, convert the first step to a research step with web search enabled.
                # This ensures that at least one step will perform a web search as required.
                if isinstance(questions[0], dict):
                    questions[0]["step_type"] = "research"
                    questions[0]["need_search"] = True
                    logger.info(
                        "Converted first step to research with web search enforcement"
                    )
        elif not has_search_step and not questions:
            # Add a default research step if no steps exist
            logger.warning("subquestions has no question. Adding default research step.")
            subquestions["questions"] = [
                {
                    "need_search": True,
                    "question": "Initial Research",
                    "description": "Gather information about the topic",
                    "step_type": "research",
                }
            ]

    return subquestions
