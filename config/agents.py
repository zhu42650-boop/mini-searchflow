
from typing import Literal

# LLM类型
LLMType = Literal["basic", "reasoning", "vision", "code"]

AGENT_LLM_MAP: dict[str, LLMType] = {
    "coordinator": "basic",
    "decomposer": "basic",
    "researcher": "basic",
    "coder": "basic",
    "analyst": "basic",
    "reporter": "basic",
}

__all__ = ["LLMType", "AGENT_LLM_MAP"]
