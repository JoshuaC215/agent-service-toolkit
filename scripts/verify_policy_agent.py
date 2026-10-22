"""Verify real DeepSeek tool calls against local policy evidence, without business writes."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.messages.tool import ToolCall  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agents.order_agent import agent as graph_builder  # noqa: E402
from agents.policy_tools import get_policy_rag  # noqa: E402
from core.settings import settings  # noqa: E402
from schema.models import DeepseekModelName  # noqa: E402

POLICY_TOOL = "search_after_sales_policy"
QUESTION = (
    "商品出现质量问题时，申请退款需要提供哪些材料？退款是否需要人工审批？"
    "我只是咨询演示售后政策，不要查询具体订单，也不要创建工单。请引用政策文件名和标题。"
)


def policy_calls_to_resume(messages: list) -> list[ToolCall]:
    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        raise RuntimeError("Expected a model tool request before resuming")
    if any(call["name"] != POLICY_TOOL for call in last.tool_calls):
        raise RuntimeError("Blocked a non-policy tool; no business operation was executed")
    return last.tool_calls


def inspect_trace(messages: list) -> tuple[dict[str, bool], list[dict], str]:
    requested = {
        call["id"]: call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    results = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call = requested.get(message.tool_call_id)
        if call is None or call["name"] != POLICY_TOOL:
            raise RuntimeError("Tool result does not match an allowed model request")
        if not isinstance(message.content, str):
            raise RuntimeError("Expected JSON text from the policy tool")
        results.append(json.loads(message.content))
    final = messages[-1]
    answer = (
        final.content if isinstance(final, AIMessage) and isinstance(final.content, str) else ""
    )
    evidence = [item for result in results for item in result.get("evidence", [])]
    refund_evidence = [item for item in evidence if item["source"] == "refund_policy.md"]
    checks = {
        "DeepSeek requested the policy tool": bool(requested),
        "Only policy tools were requested": bool(requested)
        and all(call["name"] == POLICY_TOOL for call in requested.values()),
        "Every tool request has a result": bool(requested) and len(results) == len(requested),
        "Retrieval returned refund evidence": any(result.get("found") for result in results)
        and bool(refund_evidence),
        "A final answer was generated": bool(answer)
        and isinstance(final, AIMessage)
        and not final.tool_calls,
        "Answer cites the retrieved refund filename and title": any(
            item["source"] in answer and item["title"] in answer for item in refund_evidence
        ),
        "Answer labels the policy as a demo": "演示" in answer,
        "Answer mentions human review": "人工" in answer
        and any(word in answer for word in ("审批", "审核")),
        "Answer includes application materials": "订单号" in answer
        and any(word in answer for word in ("照片", "视频")),
    }
    return checks, evidence, answer


async def run_verification(model_name: DeepseekModelName) -> bool:
    graph = graph_builder.compile(checkpointer=InMemorySaver(), interrupt_before=["tools"])
    config = RunnableConfig(
        configurable={"thread_id": uuid4().hex, "model": model_name}, recursion_limit=32
    )
    await graph.ainvoke({"messages": [HumanMessage(content=QUESTION)]}, config)
    for _ in range(3):
        state = graph.get_state(config)
        if not state.next:
            break
        if state.next != ("tools",):
            raise RuntimeError("Unexpected graph interruption")
        for call in policy_calls_to_resume(state.values["messages"]):
            print(
                f"Tool request: {call['name']} {json.dumps(call['args'], ensure_ascii=False)}",
                flush=True,
            )
        await graph.ainvoke(None, config)
    state = graph.get_state(config)
    if state.next:
        raise RuntimeError("Policy tool round limit reached")
    checks, evidence, answer = inspect_trace(state.values["messages"])
    if settings.ORDER_AGENT_TRIAGE_ENABLED:
        checks["Triage completed without fallback"] = state.values.get("triage_status") == "ok"
    if settings.ORDER_AGENT_REVIEW_ENABLED:
        checks["Independent review approved the answer"] = (
            state.values.get("review_passed") is True
            and state.values.get("review_status") == "passed"
        )
    print("\nWorkflow diagnostics:", flush=True)
    print(
        json.dumps(
            {
                key: state.values.get(key)
                for key in (
                    "triage_status",
                    "route",
                    "review_status",
                    "review_attempts",
                    "review_feedback",
                )
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not state.values.get("review_passed"):
        print("Rejected draft:\n" + state.values.get("draft", ""), flush=True)
    print("\nRetrieved evidence:", flush=True)
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
    print("\nDeepSeek final answer:\n" + answer, flush=True)
    print("\nStructural checks:", flush=True)
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {label}", flush=True)
    print(
        "\nReview the answer against the displayed evidence; structural checks cannot prove every claim."
    )
    return all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout", type=float, default=90, help="timeout for the full Agent run in seconds"
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    if settings.POLICY_EMBEDDING_PROVIDER != "local":
        raise RuntimeError("This verification requires POLICY_EMBEDDING_PROVIDER=local")
    model_name = (
        settings.DEFAULT_MODEL
        if isinstance(settings.DEFAULT_MODEL, DeepseekModelName)
        else DeepseekModelName.DEEPSEEK_V4_FLASH
    )
    print(
        f"Model: {model_name}; local embeddings only. This run uses DeepSeek API credits.",
        flush=True,
    )
    print(f"Local policy index ready: {get_policy_rag().build_index()} chunks.", flush=True)
    return 0 if asyncio.run(asyncio.wait_for(run_verification(model_name), args.timeout)) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"Verification could not complete: {type(exc).__name__}. No API key is printed.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
