from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    PROCESSING = "processing"



class SubQuestion(BaseModel):
    question: str 
    description: str = Field(
        ..., description="What the answer should cover and why it matters"
    )
    step_type: TaskType = Field(..., description="Nature of the sub-question")
    need_search: bool = Field(
        ..., description="Whether web/RAG search is required"
    )
    execution_res: Optional[str] = Field(
        default= None, description= "The Step execution result"
    )



class DecompositionResult(BaseModel):
    locale: str = Field(..., description="e.g. 'en-US' or 'zh-CN'")
    has_enough_context: bool
    thought: str = Field(default="", description="Brief rationale for the split")
    title: str
    questions: List[SubQuestion] = Field(
        default_factory=list, description="List of sub-questions"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {   
                    "has_enough_context": False,
                    "locale": "en-US",
                    "thought": (
                        "Break down the topic into evidence-gathering and analysis steps."
                    ),
                    "questions": [
                        {
                            "question": "What are the main use cases of RAG in industry?",
                            "description": "Collect concrete applications across sectors.",
                            "step_type": "research",
                            "need_search": True,
                        },
                        {
                            "question": "What are the tradeoffs between retrieval precision and recall?",
                            "description": "Explain typical tradeoffs and implications.",
                            "step_type": "analysis",
                            "need_search": False,
                        },
                    ],
                }
            ]
        }
