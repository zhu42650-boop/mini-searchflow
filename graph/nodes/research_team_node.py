import logging
import os
from functools import partial
from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from citations import extract_citations_from_messages, merge_citations
from prompt.template import get_system_prompt_template
from config.configuration import Configuration
from llms.llm import get_llm_token_limit_by_type,get_llm_by_type
from config import AGENT_LLM_MAP
from agents.agents import create_agent
from utils.context_manager import ContextManager
from utils.json_utils import sanitize_tool_response
from graph.utils import preserve_state_meta_fields
from utils.context_manager import validate_message_content
from graph.types import State
from tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
)


logger = logging.getLogger(__name__)

def research_team_node(
    state: State,
) -> Command[Literal["researcher", "analyst", "coder", "question_decomposer"]]:
    """Route to the next agent based on unfinished sub-questions."""
    logger.info("Research team is collaborating on tasks")
    logger.debug("Entering research_team_node - coordinating agents")

    son_questions = state.get("son_questions")
    if not son_questions or not getattr(son_questions, "questions", None):
        return Command(update=preserve_state_meta_fields(state), goto="reporter")

    for question in son_questions.questions:
        if not question.execution_res:
            if question.step_type == "research":
                return Command(update=preserve_state_meta_fields(state), goto="researcher")
            if question.step_type == "analysis":
                return Command(update=preserve_state_meta_fields(state), goto="analyst")
            if question.step_type == "processing":
                return Command(update=preserve_state_meta_fields(state), goto="coder")

    return Command(update=preserve_state_meta_fields(state), goto="reporter")

def validate_web_search_usage(messages: list, agent_name: str = "agent") -> bool:
    """
    Validate if the agent has used the web search tool during execution.
    
    Args:
        messages: List of messages from the agent execution
        agent_name: Name of the agent (for logging purposes)
        
    Returns:
        bool: True if web search tool was used, False otherwise
    """
    web_search_used = False
    
    for message in messages:
        # Check for ToolMessage instances indicating web search was used
        if isinstance(message, ToolMessage) and message.name == "web_search":
            web_search_used = True
            logger.info(f"[VALIDATION] {agent_name} received ToolMessage from web_search tool")
            break
            
        # Check for AIMessage content that mentions tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get('name') == "web_search":
                    web_search_used = True
                    logger.info(f"[VALIDATION] {agent_name} called web_search tool")
                    break
            # break outer loop if web search was used
            if web_search_used:
                break
                    
        # Check for message name attribute
        if hasattr(message, 'name') and message.name == "web_search":
            web_search_used = True
            logger.info(f"[VALIDATION] {agent_name} used web_search tool")
            break
    
    if not web_search_used:
        logger.warning(f"[VALIDATION] {agent_name} did not use web_search tool")
        
    return web_search_used

async def _handle_recursion_limit_fallback(
    messages: list,
    agent_name: str,
    current_step,
    state: State,
) -> list:
    """Handle GraphRecursionError with graceful fallback using LLM summary.

    When the agent hits the recursion limit, this function generates a final output
    using only the observations already gathered, without calling any tools.

    Args:
        messages: Messages accumulated during agent execution before hitting limit
        agent_name: Name of the agent that hit the limit
        current_step: The current step being executed
        state: Current workflow state

    Returns:
        list: Messages including the accumulated messages plus the fallback summary

    Raises:
        Exception: If the fallback LLM call fails
    """
    logger.warning(
        f"Recursion limit reached for {agent_name} agent. "
        f"Attempting graceful fallback with {len(messages)} accumulated messages."
    )

    if len(messages) == 0:
        return messages

    cleared_messages = messages.copy()
    while len(cleared_messages) > 0 and cleared_messages[-1].type == "system":
        cleared_messages = cleared_messages[:-1]

    # Prepare state for prompt template
    fallback_state = {
        "locale": state.get("locale", "en-US"),
    }

    # Apply the recursion_fallback prompt template
    system_prompt = get_system_prompt_template(agent_name, fallback_state, None, fallback_state.get("locale", "en-US"))
    limit_prompt = get_system_prompt_template("recursion_fallback", fallback_state, None, fallback_state.get("locale", "en-US"))
    fallback_messages = cleared_messages + [
        SystemMessage(content=system_prompt),
        SystemMessage(content=limit_prompt)
    ]

    # Get the LLM without tools (strip all tools from binding)
    fallback_llm = get_llm_by_type(AGENT_LLM_MAP[agent_name])

    # Call the LLM with the updated messages
    fallback_response = fallback_llm.invoke(fallback_messages)
    fallback_content = fallback_response.content

    logger.info(
        f"Graceful fallback succeeded for {agent_name} agent. "
        f"Generated summary of {len(fallback_content)} characters."
    )

    # Sanitize response
    fallback_content = sanitize_tool_response(str(fallback_content))

    # Update the step with the fallback result
    current_step.execution_res = fallback_content

    # Return the accumulated messages plus the fallback response
    result_messages = list(cleared_messages)
    result_messages.append(AIMessage(content=fallback_content, name=agent_name))

    return result_messages


async def _execute_agent_step(
        state: State, agent, agent_name: str, config: RunnableConfig = None
) -> Command[Literal["research_team"]]:
    logger.debug(f"[_execute_agent_step] Starting execution for agent {agent_name}")
    
    son_questions = state.get("son_questions")
    if not son_questions or not getattr(son_questions, "questions", None):
        logger.warning("[_execute_agent_step] No sub-questions available")
        return Command(
            update=preserve_state_meta_fields(state),
            goto="research_team",
        )

    title = son_questions.title
    answers = state.get("answers", [])
    logger.debug(
        "[_execute_agent_step] Plan title: %s, answers count: %s",
        title,
        len(answers),
    )

    #find the first unexecuted step
    current_question = None
    completed_questions = []
    for idx, question in enumerate(son_questions.questions):
        if not question.execution_res:
            current_question = question
            logger.debug(f"[_execute_agent_step] Found unexecuted step at index {idx}: {question.question}")
            break
        else:
            completed_questions.append(question)
    
    if not current_question:
        logger.warning(f"[_execute_agent_step] No unexecuted step found in {len(son_questions.questions)} total steps")
        return Command(
            update=preserve_state_meta_fields(state),
            goto="research_team"
        )

    logger.info(
        "[_execute_agent_step] Executing step: %s, agent: %s",
        current_question.question,
        agent_name,
    )
    logger.debug(f"[_execute_agent_step] Completed steps so far: {len(completed_questions)}")

    # Format completed steps information
    completed_questions_info = ""
    if completed_questions:
        completed_questions_info = "# Completed Research Steps\n\n"
        for i, step in enumerate(completed_questions):
            completed_questions_info += f"## Completed Step {i + 1}: {step.question}\n\n"
            completed_questions_info += f"<finding>\n{step.execution_res}\n</finding>\n\n"

    #prepare the input of the agent 
    agent_input = {
        "messages": [
            HumanMessage(
                content=(
                    f"# Research Topic\n\n{title}\n\n{completed_questions_info}"
                    f"# Current Step\n\n## Question\n\n{current_question.question}\n\n"
                    f"## Description\n\n{current_question.description}\n\n"
                    f"## Locale\n\n{state.get('locale', 'en-US')}"
                )
            )
        ]
    }

    # Add citation reminder for researcher agent
    if agent_name == "researcher":
        if state.get("resources"):
            resources_info = "**The user mentioned the following resource files:**\n\n"
            for resource in state.get("resources"):
                resources_info += f"- {resource.title} ({resource.description})\n"

            agent_input["messages"].append(
                HumanMessage(
                    content=resources_info
                    + "\n\n"
                    + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )

        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )
    
    # Invoke the agent
    default_recursion_limit = 25
    try:
        env_value_str = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
        parsed_limit = int(env_value_str)

        if parsed_limit > 0:
            recursion_limit = parsed_limit
            logger.info(f"Recursion limit set to: {recursion_limit}")
        else:
            logger.warning(
                f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit
    except ValueError:
        raw_env_value = os.getenv("AGENT_RECURSION_LIMIT")
        logger.warning(
            f"Invalid AGENT_RECURSION_LIMIT value: '{raw_env_value}'. "
            f"Using default value {default_recursion_limit}."
        )
        recursion_limit = default_recursion_limit

    logger.info(f"Agent input: {agent_input}")
    
    # Validate message content before invoking agent
    try:
        validated_messages = validate_message_content(agent_input["messages"])
        agent_input["messages"] = validated_messages
    except Exception as validation_error:
        logger.error(f"Error validating agent input messages: {validation_error}")
    

    # Apply context compression to prevent token overflow (Issue #721)
    llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP[agent_name])
    if llm_token_limit:
        token_count_before = sum(
            len(str(msg.content).split()) for msg in agent_input.get("messages", []) if hasattr(msg, "content")
        )
        compressed_state = ContextManager(llm_token_limit, preserve_prefix_message_count=3).compress_messages(
            {"messages": agent_input["messages"]}
        )
        agent_input["messages"] = compressed_state.get("messages", [])
        token_count_after = sum(
            len(str(msg.content).split()) for msg in agent_input.get("messages", []) if hasattr(msg, "content")
        )
        logger.info(
            f"Context compression for {agent_name}: {len(compressed_state.get('messages', []))} messages, "
            f"estimated tokens before: ~{token_count_before}, after: ~{token_count_after}"
        )

    try:
        # Use astream (async) from the start to capture messages in real-time
        # This allows us to retrieve accumulated messages even if recursion limit is hit
        # NOTE: astream is required for MCP tools which only support async invocation
        accumulated_messages = []
        async for chunk in agent.astream(
            input=agent_input,
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            if isinstance(chunk, dict) and "messages" in chunk:
                accumulated_messages = chunk["messages"]

        # If we get here, execution completed successfully
        result = {"messages": accumulated_messages}
    except GraphRecursionError:
        # Check if recursion fallback is enabled
        configurable = Configuration.from_runnable_config(config) if config else Configuration()

        if configurable.enable_recursion_fallback:
            try:
                # Call fallback with accumulated messages (function returns list of messages)
                response_messages = await _handle_recursion_limit_fallback(
                    messages=accumulated_messages,
                    agent_name=agent_name,
                    current_step=current_question,
                    state=state,
                )

                # Create result dict so the code can continue normally from line 1178
                result = {"messages": response_messages}
            except Exception as fallback_error:
                # If fallback fails, log and fall through to standard error handling
                logger.error(
                    f"Recursion fallback failed for {agent_name} agent: {fallback_error}. "
                    "Falling back to standard error handling."
                )
                raise
        else:
            # Fallback disabled, let error propagate to standard handler
            logger.info(
                f"Recursion limit reached but graceful fallback is disabled. "
                "Using standard error handling."
            )
            raise
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = (
            f"Error executing {agent_name} agent for step '{current_question.question}': {str(e)}"
        )
        logger.exception(error_message)
        logger.error(f"Full traceback:\n{error_traceback}")
        
        # Enhanced error diagnostics for content-related errors
        if "Field required" in str(e) and "content" in str(e):
            logger.error(f"Message content validation error detected")
            for i, msg in enumerate(agent_input.get('messages', [])):
                logger.error(f"Message {i}: type={type(msg).__name__}, "
                            f"has_content={hasattr(msg, 'content')}, "
                            f"content_type={type(msg.content).__name__ if hasattr(msg, 'content') else 'N/A'}, "
                            f"content_len={len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0}")

        detailed_error = (
            f"[ERROR] {agent_name.capitalize()} Agent Error\n\n"
            f"Step: {current_question.question}\n\nError Details:\n{str(e)}\n\n"
            "Please check the logs for more information."
        )
        current_question.execution_res = detailed_error

        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=detailed_error,
                        name=agent_name,
                    )
                ],
                "answers": answers
                + [
                    {
                        "question": current_question.question,
                        "answer": detailed_error,
                        "agent": agent_name,
                    }
                ],
                **preserve_state_meta_fields(state),
            },
            goto="research_team",
        )
    
    response_messages = result["messages"]

    response_content = response_messages[-1].content

    response_content = sanitize_tool_response(str(response_content))

    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")

    # Validate web search usage for researcher agent if enforcement is enabled
    web_search_validated = True
    should_validate = agent_name == "researcher"
    validation_info = ""

    if should_validate:
        # Check if enforcement is enabled in configuration
        configurable = Configuration.from_runnable_config(config) if config else Configuration()
        # Skip validation if web search is disabled (user intentionally disabled it)
        if configurable.enforce_researcher_search and configurable.enable_web_search:
            web_search_validated = validate_web_search_usage(result["messages"], agent_name)
            
            # If web search was not used, add a warning to the response
            if not web_search_validated:
                logger.warning(f"[VALIDATION] Researcher did not use web_search tool. Adding reminder to response.")
                # Add validation information to observations
                validation_info = (
                    "\n\n[WARNING] This research was completed without using the web_search tool. "
                    "Please verify that the information provided is accurate and up-to-date."
                    "\n\n[VALIDATION WARNING] Researcher did not use the web_search tool as recommended."
                )

    # Update the step with the execution result
    current_question.execution_res = response_content
    logger.info(
        "Step '%s' execution completed by %s",
        current_question.question,
        agent_name,
    )

    # Include all messages from agent result to preserve intermediate tool calls/results
    # This ensures multiple web_search calls all appear in the stream, not just the final result
    agent_messages = result.get("messages", [])
    logger.debug(
        f"{agent_name.capitalize()} returned {len(agent_messages)} messages. "
        f"Message types: {[type(msg).__name__ for msg in agent_messages]}"
    )
    
    # Count tool messages for logging
    tool_message_count = sum(1 for msg in agent_messages if isinstance(msg, ToolMessage))
    if tool_message_count > 0:
        logger.info(
            f"{agent_name.capitalize()} agent made {tool_message_count} tool calls. "
            f"All tool results will be preserved and streamed to frontend."
        )

    # Extract citations from tool call results (web_search, crawl)
    existing_citations = state.get("citations", [])
    new_citations = extract_citations_from_messages(agent_messages)
    merged_citations = merge_citations(existing_citations, new_citations)
    
    if new_citations:
        logger.info(
            f"Extracted {len(new_citations)} new citations from {agent_name} agent. "
            f"Total citations: {len(merged_citations)}"
        )

    return Command(
        update={
            **preserve_state_meta_fields(state),
            "messages": agent_messages,
            "answers": answers
            + [
                {
                    "question": current_question.question,
                    "answer": response_content + validation_info,
                    "agent": agent_name,
                }
            ],
            "citations": merged_citations,  # Store merged citations based on existing state and new tool results
        },
        goto="research_team",
    )


async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_type: str,
    default_tools: list,
) -> Command[Literal["research_team"]]:
    """Helper function to set up an agent with appropriate tools and execute a step.

    This function handles the common logic for both researcher_node and coder_node:
    1. Configures MCP servers and tools based on agent type
    2. Creates an agent with the appropriate tools or uses the default agent
    3. Executes the agent on the current step

    Args:
        state: The current state
        config: The runnable config
        agent_type: The type of agent ("researcher" or "coder")
        default_tools: The default tools to add to the agent

    Returns:
        Command to update state and go to research_team
    """
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = {}
    enabled_tools = {}
    loaded_tools = default_tools[:]
    
    # Get locale from workflow state to pass to agent creation
    # This fixes issue #743 where locale was not correctly retrieved in agent prompt
    locale = state.get("locale", "en-US")


    llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP[agent_type])
    pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)
    agent = create_agent(
        agent_type,
        agent_type,
        loaded_tools,
        agent_type,
        pre_model_hook,
        interrupt_before_tools=configurable.interrupt_before_tools,
        locale=locale,
    )
    return await _execute_agent_step(state, agent, agent_type, config)

async def researcher_node(
        state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    "Researcher node that do researching"
    logger.info("researcher node is researching.")
    logger.debug(f"[research_node] Starting researcher agent")

    configurable = Configuration.from_runnable_config(config)
    logger.debug(f"[research_node] Max search results: {configurable.max_search_results}")

    tools = []

    if configurable.enable_web_search:
        tools.extend([get_web_search_tool(configurable.max_search_results), crawl_tool])
    else:
        logger.info("[researcher_node] web search is disabled, using only local rag")
    
    retriever_tool = get_retriever_tool(state.get("resources", []))
    if retriever_tool:
        logger.debug(f"[research_node] Adding retriver tool to tools list")
        tools.insert(0, retriever_tool)
    
    # Warn if no tools are available
    if not tools:
        logger.warning("[researcher_node] No tools available (web search disabled, no resources). "
                       "Researcher will operate in pure reasoning mode.")
    
    logger.info(f"[researcher_node] Researcher tools count: {len(tools)}")
    logger.debug(f"[researcher_node] Researcher tools: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]}")
    logger.info(f"[researcher_node] enforce_researcher_search={configurable.enforce_researcher_search}, "
                f"enable_web_search={configurable.enable_web_search}")
    
    return await _setup_and_execute_agent_step(
        state,
        config,
        "researcher",
        tools,
    )
    

async def coder_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Coder node that do code analysis."""
    logger.info("Coder node is coding.")
    logger.debug(f"[coder_node] Starting coder agent with python_repl_tool")
    
    return await _setup_and_execute_agent_step(
        state,
        config,
        "coder",
        [python_repl_tool],
    )

async def analyst_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Analyst node that performs reasoning and analysis without code execution.
    
    This node handles tasks like:
    - Cross-validating information from multiple sources
    - Synthesizing research findings
    - Comparative analysis
    - Pattern recognition and trend analysis
    - General reasoning tasks that don't require code
    """
    logger.info("Analyst node is analyzing.")
    logger.debug(f"[analyst_node] Starting analyst agent for reasoning/analysis tasks")
    
    # Analyst uses no tools - pure LLM reasoning
    return await _setup_and_execute_agent_step(
        state,
        config,
        "analyst",
        [],  # No tools - pure reasoning
    )
