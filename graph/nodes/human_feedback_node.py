import logging
import json
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage
from prompt.decomposer_model import DecompositionResult
from graph.utils import preserve_state_meta_fields,extract_subquestions_content, validate_and_fix_subquestions
from graph.types import State
from utils.json_utils import repair_json_output
from config.configuration import Configuration

logger = logging.getLogger(__name__)


async def human_feedback_node(
        state: State, config: RunnableConfig
) -> Command[Literal["decomposer", "research_team", "reporter", "__end__"]]:
    son_questions = state.get("son_questions","")

    #check wheather or not accepted
    auto_accepted_questions=state.get("auto_accepted_questions",False)
    #ask feedback
    if not auto_accepted_questions:
        try:
            feedback = interrupt("Please Review the subquestions")
        except RuntimeError as e:
            logger.warning(
                "Interrupt not available in current context, auto-accepting subquestions. Error: %s",
                e,
            )
            feedback = "[ACCEPTED]"
        #no feedback
        if not feedback:
            logger.warning(f"Received empty or None feedback: {feedback}. Returning to decomposer for new subquestion.")
            return Command(
                update=preserve_state_meta_fields(state),
                goto = "decomposer"
            )

        feedback_normalized = str(feedback).strip().upper()

        if feedback_normalized.startswith("[EDIT_PLAN]"):
            logger.info(f"Plan edit requested by user: {feedback}")
            return Command(
                update={
                    "messages":[
                        HumanMessage(content = feedback, name = "feedback"),
                    ],
                    **preserve_state_meta_fields(state),
                },
                goto = "decomposer"
            )
    
    #subquestion are accepted
    decompose_iterations = state["decompose_iterations"] if state.get("decompose_iterations",0) else 0
    goto = "research_team"
    try:
        # copy questions
        original_son_questions = son_questions
        # extract JSON string safely
        son_questions_content = extract_subquestions_content(son_questions)
        # repair and parse JSON
        son_questions = json.loads(repair_json_output(son_questions_content))

        decompose_iterations +=1
        #parse the questions
        new_questions = json.loads(repair_json_output(son_questions_content))
        #fix the questions to ensure web search open
        configurable = Configuration.from_runnable_config(config)
        new_questions = validate_and_fix_subquestions(new_questions, configurable.enforce_web_search, configurable.enable_web_search)
    
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to parse son_questions: {str(e)}. son_question type: {type(son_questions).__name__}")
        if isinstance(original_son_questions, dict) and "content" in original_son_questions:
            logger.warning(f"son_questions appear to be an AIMessage object with content field")
        if decompose_iterations > 1:
            return Command(
                update=preserve_state_meta_fields(state),
                goto = "reporter"
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto = "__end__"
            )
    
    update_dict = {
        "son_questions": DecompositionResult.model_validate(new_questions),
        "decompose_iterations": decompose_iterations,
        **preserve_state_meta_fields(state),
    }

    if new_questions.get("locale"):
        update_dict["locale"] = new_questions["locale"]
    
    return Command(
        update = update_dict,
        goto = goto,
    )

