import logging
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from config.configuration import Configuration
from config.agents import AGENT_LLM_MAP
from graph.types import State
from llms.llm import get_llm_by_type, get_llm_token_limit_by_type
from prompt.template import apply_prompt_template
from utils.context_manager import ContextManager, validate_message_content

logger = logging.getLogger(__name__)


def _is_table_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("```"):
        return False
    return stripped.count("|") >= 2


def _normalize_markdown(content: str) -> str:
    """Normalize common LLM markdown issues for stable rendering."""
    lines = content.replace("\r\n", "\n").split("\n")
    normalized: list[str] = []

    for idx, line in enumerate(lines):
        prev_line = lines[idx - 1] if idx > 0 else ""
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        # Remove empty lines inside markdown table blocks.
        if not line.strip() and _is_table_like_line(prev_line) and _is_table_like_line(next_line):
            continue
        normalized.append(line)

    fixed = "\n".join(normalized).strip()
    # Normalize accidental escaped heading markers like "\#\# title".
    fixed = re.sub(r"(?m)^\\(#+\s)", r"\1", fixed)
    return fixed + "\n"


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

    # Validate and compress context to avoid token overflow
    try:
        messages = validate_message_content(messages)
    except Exception as exc:
        logger.warning("Reporter message validation failed: %s", exc)

    llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP["reporter"])
    if llm_token_limit:
        compressed_state = ContextManager(llm_token_limit, preserve_prefix_message_count=3).compress_messages(
            {"messages": messages}
        )
        messages = compressed_state.get("messages", messages)

    llm = get_llm_by_type(AGENT_LLM_MAP["reporter"])
    response = llm.invoke(messages)
    content = _normalize_markdown(getattr(response, "content", "") or "")

    report_path = _write_report_to_file(content)

    return Command(
        update={
            "messages": [AIMessage(content=content, name="reporter")],
            "final_report": content,
            "report_path": report_path,
        },
        goto="__end__",
    )
