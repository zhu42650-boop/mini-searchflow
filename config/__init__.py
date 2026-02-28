from dotenv import load_dotenv

from .loader import load_yaml_config
from .questions import BUILT_IN_QUESTIONS, BUILT_IN_QUESTIONS_ZH_CN
from .tools import SearchEngine, SELECTED_SEARCH_ENGINE
from .agents import AGENT_LLM_MAP
load_dotenv()

# Team configuration
TEAM_MEMBER_CONFIGURATIONS = {
    "researcher": {
        "name": "researcher",
        "desc": (
            "Responsible for searching and collecting relevant information, understanding user needs and conducting research analysis"
        ),
        "desc_for_llm": (
            "Uses search engines and web crawlers to gather information from the internet. "
            "Outputs a Markdown report summarizing findings. Researcher can not do math or programming."
        ),
        "is_optional": False,
    },
    "coder": {
        "name": "coder",
        "desc": (
            "Responsible for code implementation, debugging and optimization, handling technical programming tasks"
        ),
        "desc_for_llm": (
            "Executes Python or Bash commands, performs mathematical calculations, and outputs a Markdown report. "
            "Must be used for all mathematical computations."
        ),
        "is_optional": True,
    },
}
TEAM_MEMBERS = list(TEAM_MEMBER_CONFIGURATIONS.keys())

__all__ = [
    load_yaml_config,
    "SearchEngine",
    "SELECTED_SEARCH_ENGINE",
    "BUILT_IN_QUESTIONS",
    "BUILT_IN_QUESTIONS_ZH_CN",
    "TEAM_MEMBER_CONFIGURATIONS",
    "TEAM_MEMBERS",
    "AGENT_LLM_MAP",

]
