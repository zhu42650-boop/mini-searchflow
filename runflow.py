import logging
import uuid

from langgraph.types import Command

from graph import build_graph, build_graph_with_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

async def run_agent_workflow_async(
    user_input: str,
    max_decompose_interations: int = 1,
    max_son_questions: int = 5,
    enable_background_investigation: bool = True,
    enable_human_feedback: bool = False,
    enable_clarification: bool | None = None,
    max_clarification_rounds: int | None = None,
    locale: str | None = "zh-CN",
):
    if not user_input:
        raise ValueError("Input could not be empty")

    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "research_topic": user_input,
        "clarified_research_topic": user_input,
        "enable_background_investigation": enable_background_investigation,
        "auto_accepted_questions": not enable_human_feedback,
        "locale": locale or "zh-CN",
    }

    if enable_clarification is not None:
        initial_state["enable_clarification"] = enable_clarification
    if max_clarification_rounds is not None:
        initial_state["max_clarification_rounds"] = max_clarification_rounds

    graph = build_graph_with_memory() if enable_human_feedback else build_graph()

    config = {
        "configurable": {
            "max_decompose_interations": max_decompose_interations,
            "max_son_questions": max_son_questions,
        },
        "recursion_limit": 100,
    }
    if enable_human_feedback:
        config["configurable"]["thread_id"] = str(uuid.uuid4())

    final_state = None
    current_input = initial_state

    while True:
        interrupted = False
        async for s in graph.astream(input=current_input, config=config, stream_mode="values"):
            final_state = s
            if isinstance(s, dict) and "__interrupt__" in s:
                interrupt_info = s["__interrupt__"][0]
                user_feedback = input(f"{interrupt_info.value}\n").strip()
                current_input = Command(resume=user_feedback)
                interrupted = True
                break

            if isinstance(s, dict) and "messages" in s and s["messages"]:
                message = s["messages"][-1]
                if hasattr(message, "pretty_print"):
                    message.pretty_print()
                else:
                    logger.info(message)
        if not interrupted:
            break

    return final_state


if __name__ == "__main__":
    import asyncio

    user_query = input("Enter your query: ").strip()
    asyncio.run(run_agent_workflow_async(user_query, enable_human_feedback=True))
