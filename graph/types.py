from dataclasses import field
from typing import Any

from langgraph.graph import MessagesState
from prompt.decomposer_model import DecompositionResult
from rag.retriever import Resource
class State(MessagesState):
    
    # 主题和question
    locale: str = "en-US"
    research_topic: str = ""
    clarified_research_topic : str = (
        ""
    )

    resources: list[Resource] = []
    # 拆分子问题
    decompose_iterations: int = 0
    son_questions: DecompositionResult | str = None
    auto_accepted_questions: bool = False

    # 子问题的answer和evidence
    # Each item can be {"question": str, "answer": str, "agent": str, "confidence": float}
    answers: list[dict[str, Any]] = field(default_factory=list)
    # Each item can be {"question": str, "evidence": list[dict], "source": str}
    evidence: list[dict[str, Any]] = field(default_factory=list)

    #Clarification state tracking
    enable_clarification: bool = (
        False )
    clarification_rounds: int = 0
    clarification_history: list[str] = field(default_factory= list)
    is_clarification_complete: bool = False
    max_clarification_rounds: int = (
        3 )
    
    #config about background investigation
    enable_background_investigation: bool = True
    background_investigation_results: str = None
    
    # new question 的 loop
    need_more: bool = False
    new_questions: list[str] = field(default_factory=list)
    new_question_round: int = 3

    # Workflow control
    goto: str = "human_feedback"

    # Final output
    final_report: str = ""
