import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from graph import build_graph_with_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
ASSISTANT_NAMES = {
    "coordinator",
    "decomposer",
    "researcher",
    "analyst",
    "coder",
    "reporter",
    "background_investigator",
}

app = FastAPI(title="Mini SearchFlow API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent / "web"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

# Shared graph with checkpointer for interrupt/resume workflows.
GRAPH = build_graph_with_memory()


class ChatRequest(BaseModel):
    message: Optional[str] = Field(default=None)
    thread_id: str = Field(default="__default__")
    locale: str = Field(default="zh-CN")
    max_decompose_interations: int = Field(default=1)
    max_son_questions: int = Field(default=5)
    enable_background_investigation: bool = Field(default=True)
    enable_human_feedback: bool = Field(default=False)
    recursion_limit: int = Field(default=60)
    interrupt_feedback: Optional[str] = Field(default=None)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _message_to_payload(message: Any, idx: int) -> dict[str, Any]:
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        elif not isinstance(content, str):
            content = str(content)
        role = message.get("role", "assistant")
        name = message.get("name")
        if role in {"user", "human"} and isinstance(name, str) and name in ASSISTANT_NAMES:
            role = "assistant"

        return {
            "id": message.get("id", f"msg-{idx}"),
            "role": role,
            "name": name,
            "content": content,
        }

    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False)
    elif not isinstance(content, str):
        content = str(content)

    role = getattr(message, "type", "assistant")
    name = getattr(message, "name", None)
    if role == "ai":
        role = "assistant"
    elif role == "human":
        role = "user"
    if role == "user" and isinstance(name, str) and name in ASSISTANT_NAMES:
        role = "assistant"

    return {
        "id": getattr(message, "id", f"msg-{idx}"),
        "role": role,
        "name": name,
        "content": content,
    }


def _message_signature(payload: dict[str, Any]) -> str:
    """Build a stable signature to deduplicate repeated state snapshots."""
    return "|".join(
        [
            str(payload.get("role", "")),
            str(payload.get("name", "")),
            str(payload.get("content", "")),
        ]
    )


async def _chat_event_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    thread_id = request.thread_id if request.thread_id != "__default__" else str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
            "max_decompose_interations": request.max_decompose_interations,
            "max_son_questions": request.max_son_questions,
        },
        "recursion_limit": max(20, min(request.recursion_limit, 120)),
    }

    if request.interrupt_feedback:
        current_input: Any = Command(resume=request.interrupt_feedback)
    else:
        if not request.message:
            yield _sse("error", {"error": "message is required for a new request"})
            return
        current_input = {
            "messages": [{"role": "user", "content": request.message}],
            "research_topic": request.message,
            "clarified_research_topic": request.message,
            "enable_background_investigation": request.enable_background_investigation,
            "auto_accepted_questions": not request.enable_human_feedback,
            "locale": request.locale,
        }

    yield _sse("thread", {"thread_id": thread_id})

    sent_message_keys: set[str] = set()
    last_report = ""
    last_report_path = ""
    research_mode = False

    try:
        async for state in GRAPH.astream(input=current_input, config=config, stream_mode="values"):
            if not isinstance(state, dict):
                continue

            if "__interrupt__" in state and state["__interrupt__"]:
                interrupt_obj = state["__interrupt__"][0]
                value = getattr(interrupt_obj, "value", "Need your feedback")
                yield _sse(
                    "interrupt",
                    {
                        "thread_id": thread_id,
                        "content": value,
                        "options": [
                            {"text": "接受并继续", "value": "[ACCEPTED]"},
                            {"text": "编辑子问题", "value": "[EDIT_PLAN] 请重新拆分并更具体"},
                        ],
                    },
                )

            messages = state.get("messages") or []
            for i, m in enumerate(messages):
                payload = _message_to_payload(m, i)
                key = _message_signature(payload)
                if key in sent_message_keys:
                    continue
                sent_message_keys.add(key)
                yield _sse("message", payload)

            if state.get("son_questions"):
                if not research_mode:
                    research_mode = True
                    yield _sse("mode", {"kind": "research"})
                sq = state.get("son_questions")
                questions = []
                if hasattr(sq, "questions"):
                    questions = [
                        {
                            "question": q.question,
                            "description": q.description,
                            "step_type": str(q.step_type),
                            "done": bool(q.execution_res),
                        }
                        for q in sq.questions
                    ]
                if questions:
                    yield _sse("subquestions", {"items": questions})

            report = state.get("final_report") or ""
            report_path = state.get("report_path") or ""
            if report and report != last_report:
                if not research_mode:
                    research_mode = True
                    yield _sse("mode", {"kind": "research"})
                last_report = report
                last_report_path = report_path
                yield _sse(
                    "final_report",
                    {
                        "content": report,
                        "report_path": report_path,
                        "report_name": Path(report_path).name if report_path else "report.md",
                    },
                )

        # Defensive fallback: read final state from checkpointer to avoid missing report events.
        if not last_report:
            try:
                final_state = await GRAPH.aget_state(
                    {"configurable": {"thread_id": thread_id}}
                )
                values = getattr(final_state, "values", {}) or {}
                final_report = values.get("final_report") or ""
                final_report_path = values.get("report_path") or ""
                final_messages = values.get("messages") or []

                # Guarantee direct-response output reaches frontend left panel.
                if final_messages:
                    for i, message in enumerate(final_messages):
                        payload = _message_to_payload(message, i)
                        key = _message_signature(payload)
                        if key not in sent_message_keys:
                            sent_message_keys.add(key)
                            yield _sse("message", payload)

                if final_report:
                    last_report = final_report
                    last_report_path = final_report_path
                    if not research_mode:
                        research_mode = True
                        yield _sse("mode", {"kind": "research"})
                    yield _sse(
                        "final_report",
                        {
                            "content": final_report,
                            "report_path": final_report_path,
                            "report_name": Path(final_report_path).name if final_report_path else "report.md",
                        },
                    )
            except Exception:
                logger.exception("failed to fetch final state from checkpointer")

        yield _sse("done", {"thread_id": thread_id, "report_path": last_report_path})
    except Exception as exc:
        logger.exception("chat stream failed")
        yield _sse("error", {"error": str(exc), "thread_id": thread_id})


@app.get("/")
async def root() -> FileResponse:
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return FileResponse(Path(__file__).resolve())


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _chat_event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/report/{thread_id}")
async def get_report(thread_id: str) -> dict[str, Any]:
    try:
        graph_state = await GRAPH.aget_state({"configurable": {"thread_id": thread_id}})
        values = getattr(graph_state, "values", {}) or {}
        report = values.get("final_report") or ""
        report_path = values.get("report_path") or ""
        if not report:
            raise HTTPException(status_code=404, detail="report not found")
        return {
            "thread_id": thread_id,
            "content": report,
            "report_path": report_path,
            "report_name": Path(report_path).name if report_path else "report.md",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _safe_report_path(path: str) -> Path:
    raw = Path(path).expanduser()
    candidate = raw if raw.is_absolute() else (OUTPUT_DIR / raw)
    resolved = candidate.resolve()
    output_root = OUTPUT_DIR.resolve()
    if output_root not in resolved.parents and resolved != output_root:
        raise HTTPException(status_code=400, detail="invalid report path")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="report file not found")
    return resolved


@app.get("/api/report/file")
async def get_report_by_file(path: str) -> dict[str, Any]:
    report_file = _safe_report_path(path)
    content = report_file.read_text(encoding="utf-8")
    return {
        "thread_id": "",
        "content": content,
        "report_path": str(report_file),
        "report_name": report_file.name,
    }
