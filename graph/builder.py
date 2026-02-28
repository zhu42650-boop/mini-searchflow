from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.types import State
from graph.nodes.coordinator_node import coordinator_node
from graph.nodes.background_investigator_node import background_investigation_node
from graph.nodes.question_decomposer_node import question_decomposer_node
from graph.nodes.human_feedback_node import human_feedback_node
from graph.nodes.research_team_node import (
    research_team_node,
    researcher_node,
    analyst_node,
    coder_node,
)
from graph.nodes.reporter_node import reporter_node


def continue_to_running_research_team(state: State) -> str:
    son_questions = state.get("son_questions")
    if not son_questions or not getattr(son_questions, "questions", None):
        return "reporter"

    for question in son_questions.questions:
        if not question.execution_res:
            if question.step_type == "research":
                return "researcher"
            if question.step_type == "analysis":
                return "analyst"
            if question.step_type == "processing":
                return "coder"
            return "researcher"

    return "reporter"


def _build_base_graph() -> StateGraph:
    builder = StateGraph(State)

    builder.add_edge(START, "coordinator")

    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("question_decomposer", question_decomposer_node)
    builder.add_node("decomposer", question_decomposer_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reporter", reporter_node)

    builder.add_edge("background_investigator", "decomposer")
    builder.add_edge("question_decomposer", "human_feedback")
    builder.add_edge("decomposer", "human_feedback")
    builder.add_edge("human_feedback", "research_team")

    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["researcher", "analyst", "coder", "reporter"],
    )

    builder.add_edge("researcher", "research_team")
    builder.add_edge("analyst", "research_team")
    builder.add_edge("coder", "research_team")

    builder.add_edge("reporter", END)

    return builder


def build_graph_with_memory():
    memory = MemorySaver()
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()
