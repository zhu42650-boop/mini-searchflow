import logging

from typing import Annotated, Literal

from config.configuration import Configuration
from graph.types import State
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command

from config.agents import AGENT_LLM_MAP
from llms.llm import get_llm_by_type
from prompt.template import apply_prompt_template

logger = logging.getLogger(__name__)

@tool
def direct_response(
    message: Annotated[str,"The response message to send directly to user"],
    locale: Annotated[str, "The user's detected language locale(e.g., en-US, zh-CN)."],
):
    """Respond directly to user for greetings or simple requests."""
    return 

@tool
def handoff_to_question_decomposer(
    research_topic: Annotated[str, "The topic of the research task to be handed off"],
    locale: Annotated[str, "The user's detected language locale(e.g., en-US, zh-CN)."],
):
    """Handoff to question decomposer for research tasks."""
    return 

@tool
def handoff_after_clarification(
    locale: Annotated[str, "The user's detected language locale(e.g., en-US, zh-CN)."],
    research_topic: Annotated[
        str, "The clarified research topic based on all clarification rounds."
    ],
):
    """Handoff after clarification with clarified research topic."""
    return 


def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["decomposer", "background_investigator", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)

    enable_clarification = state.get("enable_clarification", False)
    initial_topic = state.get("research_topic", "")
    clarified_topic = initial_topic



    if enable_clarification:
        # TODO: implement clarification flow (multi-turn)
        # Placeholder for future clarification logic
        pass

    messages = apply_prompt_template(
        "coordinator", state, locale=state.get("locale", "en-US")
    )
    messages.append(
        {
            "role": "system",
            "content": (
                "Clarification is DISABLED. For research questions, use handoff_to_question_decomposer. For greetings or small talk，use direct_response. Do NOT ask clarifying questions.",
            ),
        }
    )

    tools = [handoff_to_question_decomposer, direct_response]
    # Normalize message content to avoid invalid list-of-strings payloads
    normalized_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                if all(isinstance(item, str) for item in content):
                    msg = {**msg, "content": "\n".join(content)}
            elif content is not None and not isinstance(content, str):
                msg = {**msg, "content": str(content)}
        normalized_messages.append(msg)

    response = (
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools(tools)
        .invoke(normalized_messages)
    )

    goto = "__end__"
    locale = state.get("locale", "en-US")
    research_topic = state.get("research_topic", "")

    # Process tool calls
    direct_reply = None
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})

            if tool_name == "handoff_to_question_decomposer":
                logger.info("handing off to decomposer")
                goto = "decomposer"
                if tool_args.get("research_topic"):
                    research_topic = tool_args.get("research_topic")
                break

            if tool_name == "direct_response":
                goto = "__end__"
                if tool_args.get("message"):
                    direct_reply = tool_args.get("message")
                break


    # Append coordinator response content to messages
    messages = list(state.get("messages", []) or [])
    if response.content:
        messages.append(HumanMessage(content=response.content, name="coordinator"))
    if direct_reply:
        messages.append(AIMessage(content=direct_reply, name="coordinator"))

    # Route to background investigation if enabled
    if goto == "decomposer" and state.get("enable_background_investigation"):
        goto = "background_investigator"

    clarified_research_topic_value = clarified_topic or research_topic

    return Command(
        update={
            "messages": messages,
            "locale": locale,
            "research_topic": research_topic,
            "clarified_research_topic": clarified_research_topic_value,
            "resources": configurable.resources,
            "goto": goto,
            "citations": state.get("citations", []),
        },
        goto=goto,
    )



