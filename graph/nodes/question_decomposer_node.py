import logging
import json
from typing import Any, Literal

from graph.types import State
from config.configuration import Configuration
from prompt.template import apply_prompt_template
from llms.llm import get_llm_by_type
from config.agents import AGENT_LLM_MAP
from utils.json_utils import repair_json_output
from graph.utils import (
    get_message_content,
    preserve_state_meta_fields,
    extract_subquestions_content,
    validate_and_fix_subquestions,
)
from prompt.decomposer_model import DecompositionResult


from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)



def question_decomposer_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter", "__end__"]]:
    """decomposer node that generate the son_questions"""
    logger.info("Decomposer generating son_questions with locale: %s",state.get("locale", "en-US"))
    configurable = Configuration.from_runnable_config(config)
    decompose_interations = (
        state["decompose_iterations"] if state.get("decompose_iterations", 0) else 0
    )

    # If all sub-questions already have execution results, skip re-decomposition
    existing = state.get("son_questions")
    if existing and getattr(existing, "questions", None):
        if all(q.execution_res for q in existing.questions):
            return Command(
                update={
                    "son_questions": existing,
                    **preserve_state_meta_fields(state),
                },
                goto="reporter",
            )

    if state.get("enable_clarification", False) and state.get("clarified_research_topic"):
        logger.info("Clarification enabled for decomposer; using clarified topic.")
        messages = apply_prompt_template(
            "question_decomposer", state, configurable, state.get("locale", "en-US")
        )
        pass
        #to be continue
    
    #no clarification
    else:
        messages = apply_prompt_template("question_decomposer", state, configurable, state.get("locale", "en-US"))

    if state.get("enable_background_investigation") and state.get("background_investigation_results"):
        messages +=[
            {
                "role": "user",
                "content": (
                    "background investigation results of user query:\n"
                    + state["background_investigation_results"]
                    +"\n"
                )
            }
        ]
    
    if configurable.enable_deep_thinking:
        llm = get_llm_by_type("reasoning")
    elif AGENT_LLM_MAP["decomposer"] == "basic":
        llm = get_llm_by_type("basic")
    else:
        llm = get_llm_by_type(AGENT_LLM_MAP["decomposer"])
    
    if decompose_interations >= configurable.max_decompose_interations:
        return Command(
            update=preserve_state_meta_fields(state),
            goto = "reporter"
        )

    full_response = ""
    if AGENT_LLM_MAP["decomposer"] == "basic" and not configurable.enable_deep_thinking:
        response = llm.invoke(messages)
        if hasattr(response, "model_dump_json"):
            full_response = response.model_dump_json(indent=4, exclude_none=True)
        else:
            full_response = get_message_content(response) or ""
    
    else:
        response = llm.stream(messages)
        for chunk in response:
            full_response +=chunk.content
    
    logger.debug("Current state messages: %s", state.get("messages", []))
    logger.info(f"Decomposer response:{full_response}")

    cleaned_response = repair_json_output(full_response)

    if not cleaned_response.strip().startswith('{') and not cleaned_response.strip().startswith('['):
        logger.warning("Decomposer response does not appear to be valid JSON after cleanup")
        if decompose_interations > 0:
            return Command(
                update= preserve_state_meta_fields(state),
                goto ="reporter"
            )
        else:
            return Command(
                update =preserve_state_meta_fields(state),
                goto = "__end__"
            )
    
    try:
        curr_questions = json.loads(cleaned_response)
        curr_questions_content = extract_subquestions_content(curr_questions)
        curr_questions = json.loads(repair_json_output(curr_questions_content))
    except json.JSONDecodeError:
        logger.warning("Decomposer response is not a valid JSON")
        if decompose_interations > 0:
            return Command(
                update= preserve_state_meta_fields(state),
                goto = "reporter"
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto = "__end__"
            )

    if isinstance(curr_questions, dict):
        curr_questions = validate_and_fix_subquestions(
            curr_questions,
            configurable.enforce_web_search,
            configurable.enable_web_search,
        )

    if isinstance(curr_questions, dict) and curr_questions.get("has_enough_context"):
        logger.info("Decomposer response has enough context.")
        new_subquestions = DecompositionResult.model_validate(curr_questions)
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="decomposer")],
                "son_questions": new_subquestions,
                **preserve_state_meta_fields(state),
            },
            goto="reporter",
        )

    new_subquestions = DecompositionResult.model_validate(curr_questions)
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="decomposer")],
            "son_questions": new_subquestions,
            **preserve_state_meta_fields(state),
        },
        goto="human_feedback",
    )
    

