import logging
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from config.configuration import Configuration
from config.agents import AGENT_LLM_MAP
from graph.types import State
from llms.llm import get_llm_by_type
from prompt.template import apply_prompt_template

logger = logging.getLogger(__name__)


def _write_report_to_file(content: str) -> str:
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("report_%Y%m%d_%H%M%S.md")
    path = output_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def reporter_node(state: State, config: RunnableConfig) -> Command:
    """Reporter node that generates the final report."""
    logger.info("Reporter is generating final report.")
    configurable = Configuration.from_runnable_config(config)

    # Avoid duplicate report generation
    if state.get("final_report"):
        logger.info("Final report already exists. Skipping reporter generation.")
        return Command(update={}, goto="__end__")

    messages = apply_prompt_template(
        "reporter", state, configurable, state.get("locale", "zh-CN")
    )

    llm = get_llm_by_type(AGENT_LLM_MAP["reporter"])
    response = llm.invoke(messages)
    content = getattr(response, "content", "") or ""

    report_path = _write_report_to_file(content)

    return Command(
        update={
            "messages": [AIMessage(content=content, name="reporter")],
            "final_report": content,
            "report_path": report_path,
        },
        goto="__end__",
    )
