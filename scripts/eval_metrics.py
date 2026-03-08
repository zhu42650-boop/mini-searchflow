import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph.builder import build_graph
from llms.llm import get_llm_by_type
from config import AGENT_LLM_MAP
from utils.json_utils import repair_json_output


DEFAULT_QUERIES = [
    "詹姆斯是不是历史上最伟大的篮球运动员？",
    "2026年世界杯潜在冠军分析",
    "中国新能源汽车市场的竞争格局与未来趋势",
    "生成式AI在教育行业的落地场景和风险",
    "LoRA 和 QLoRA 微调方式在显存占用和效果上有何区别？",
    "2024 年以来多智能体系统（MAS）有哪些代表性工作和研究趋势？",
    "当前主流的 LLM 幻觉缓解方法有哪些，各自的核心思路是什么？",
    "目前开源的 Embedding 模型中，哪些在中文科研语料上表现最好？",
    "为一个医疗问诊场景设计多智能体工作流，需要考虑哪些关键模块？",
    "如何设计一个支持实时更新的企业级知识库问答系统？",
    "Transformer 和 Mamba 架构在长序列处理上的性能差异是什么？",
    "RAG 系统的评估指标体系目前有哪些主流方案？",
    "如何构建一个能处理百页 PDF 的长文档问答流程？",
    "multi-agent system领域在近近几年有哪些重要进展？",
    "今年葡萄牙队能在世界杯夺冠吗？概率有多大",
    "AI对未来科技发展的利弊有哪些？",
]

AGENT_BY_STEP = {
    "research": "researcher",
    "analysis": "analyst",
    "processing": "coder",
}


@dataclass
class EvalResult:
    query: str
    elapsed_sec: float
    subquestion_count: int
    routing_accuracy: float
    report_length_chars: int
    unique_sources: int
    judge_multi_score: Optional[float]
    judge_baseline_score: Optional[float]


async def run_workflow(
    query: str,
    max_decompose_iterations: int,
    max_son_questions: int,
    enable_background: bool,
    max_search_results: int,
    recursion_limit: int,
) -> Dict:
    graph = build_graph()
    initial_state = {
        "messages": [{"role": "user", "content": query}],
        "research_topic": query,
        "clarified_research_topic": query,
        "enable_background_investigation": enable_background,
        "auto_accepted_questions": True,
        "locale": "zh-CN",
    }
    config = {
        "configurable": {
            "max_decompose_interations": max_decompose_iterations,
            "max_son_questions": max_son_questions,
            "max_search_results": max_search_results,
        },
        "recursion_limit": recursion_limit,
    }

    final_state = None
    async for s in graph.astream(input=initial_state, config=config, stream_mode="values"):
        final_state = s
    return final_state or {}


def compute_routing_accuracy(state: Dict) -> float:
    son_questions = state.get("son_questions")
    answers = state.get("answers", [])
    if not son_questions or not getattr(son_questions, "questions", None):
        return 0.0

    answer_map = {item.get("question"): item.get("agent") for item in answers}
    total = 0
    correct = 0
    for question in son_questions.questions:
        expected_agent = AGENT_BY_STEP.get(str(question.step_type))
        if not expected_agent:
            continue
        total += 1
        actual_agent = answer_map.get(question.question)
        if actual_agent == expected_agent:
            correct += 1
    return (correct / total) if total else 0.0


def count_unique_sources(state: Dict) -> int:
    citations = state.get("citations", [])
    urls = set()
    for item in citations:
        if hasattr(item, "url"):
            urls.add(item.url)
        elif isinstance(item, dict):
            metadata = item.get("metadata", {})
            url = metadata.get("url") or item.get("url")
            if url:
                urls.add(url)
    return len(urls)


def run_baseline_answer(query: str) -> str:
    llm = get_llm_by_type("basic")
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer in Chinese."),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    return getattr(response, "content", str(response))


def judge_coverage(query: str, multi_answer: str, baseline_answer: str) -> Tuple[float, float]:
    judge = get_llm_by_type("basic")
    prompt = (
        "You are a strict evaluator. Compare two answers to the question. "
        "Score each answer from 0 to 100 for coverage of key information. "
        "Return only JSON: {\"multi\": number, \"baseline\": number, \"reason\": string}.\n\n"
        f"Question: {query}\n\n"
        f"Answer A (multi-agent):\n{multi_answer}\n\n"
        f"Answer B (baseline):\n{baseline_answer}\n"
    )
    response = judge.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", str(response))
    cleaned = repair_json_output(content)
    data = json.loads(cleaned)
    return float(data.get("multi", 0)), float(data.get("baseline", 0))


def summarize_results(results: List[EvalResult]) -> Dict:
    if not results:
        return {}
    avg = lambda xs: sum(xs) / len(xs)
    summary = {
        "avg_latency_sec": avg([r.elapsed_sec for r in results]),
        "avg_subquestions": avg([r.subquestion_count for r in results]),
        "avg_routing_accuracy": avg([r.routing_accuracy for r in results]),
        "avg_report_length_chars": avg([r.report_length_chars for r in results]),
        "avg_unique_sources": avg([r.unique_sources for r in results]),
    }
    judge_scores = [r for r in results if r.judge_multi_score is not None]
    if judge_scores:
        summary["avg_multi_score"] = avg([r.judge_multi_score for r in judge_scores])
        summary["avg_baseline_score"] = avg([r.judge_baseline_score for r in judge_scores])
        summary["avg_score_gain"] = summary["avg_multi_score"] - summary["avg_baseline_score"]
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to a txt file of queries (one per line)")
    parser.add_argument("--enable-judge", action="store_true", help="Enable LLM-as-judge coverage evaluation")
    parser.add_argument("--output", type=str, default="outputs/eval_metrics.json", help="Output JSON path")
    parser.add_argument("--sample", type=int, default=0, help="Only run the first N queries")
    parser.add_argument("--max-questions", type=int, default=4, help="Max sub-questions per query")
    parser.add_argument("--max-iterations", type=int, default=1, help="Max decomposition iterations")
    parser.add_argument("--recursion-limit", type=int, default=60, help="LangGraph recursion limit")
    parser.add_argument("--disable-background", action="store_true", help="Disable background investigation node")
    parser.add_argument("--max-search-results", type=int, default=3, help="Max web search results per query")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = DEFAULT_QUERIES

    if args.sample and args.sample > 0:
        queries = queries[: args.sample]

    os.environ["AGENT_RECURSION_LIMIT"] = str(args.recursion_limit)

    results: List[EvalResult] = []
    for query in queries:
        start = time.perf_counter()
        state = await run_workflow(
            query,
            max_decompose_iterations=args.max_iterations,
            max_son_questions=args.max_questions,
            enable_background=not args.disable_background,
            max_search_results=args.max_search_results,
            recursion_limit=args.recursion_limit,
        )
        elapsed = time.perf_counter() - start

        son_questions = state.get("son_questions")
        subquestion_count = len(son_questions.questions) if son_questions and getattr(son_questions, "questions", None) else 0
        routing_accuracy = compute_routing_accuracy(state)
        report = state.get("final_report", "") or ""
        report_length = len(report)
        unique_sources = count_unique_sources(state)

        judge_multi = None
        judge_baseline = None
        if args.enable_judge:
            baseline = run_baseline_answer(query)
            judge_multi, judge_baseline = judge_coverage(query, report, baseline)

        results.append(
            EvalResult(
                query=query,
                elapsed_sec=elapsed,
                subquestion_count=subquestion_count,
                routing_accuracy=routing_accuracy,
                report_length_chars=report_length,
                unique_sources=unique_sources,
                judge_multi_score=judge_multi,
                judge_baseline_score=judge_baseline,
            )
        )

    summary = summarize_results(results)
    payload = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
