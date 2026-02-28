from dataclasses import dataclass, field, fields
import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
import enum

from rag.retriever import Resource

class ReportStyle(enum.Enum):
    ACADEMIC = "academic"
    POPULAR_SCIENCE = "popular_science"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    STRATEGIC_INVESTMENT = "strategic_investment"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the workflow."""

    resources: list[Resource] = field(
        default_factory=list
    )

    # Workflow limits
    max_decompose_interations: int =1
    max_son_questions: int = 5
    max_new_question_round: int = 3

    max_search_results: int =3


    # Tooling / behavior toggles
    enforce_web_search: bool = (
        True
    )
    enable_web_search: bool = (
        True
    )
    enforce_researcher_search: bool = (
        True  # Enforce that researcher must use web search tool at least once
    )
    enable_recursion_fallback: bool = (
        True
    )
    enable_deep_thinking: bool = False

    # Output preferences
    report_style: str = ReportStyle.ACADEMIC.value

    # Optional control hooks
    interrupt_before_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig-like dict."""
        configurable = (
            config.get("configurable", {}) if isinstance(config, dict) else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v is not None})


__all__ = ["Configuration"]
