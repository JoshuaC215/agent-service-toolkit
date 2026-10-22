"""Trace validation and the pre-execution tool guard need no network or model calls."""

import json
import runpy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.policy_rag import PROJECT_ROOT

verification_script = runpy.run_path(str(PROJECT_ROOT / "scripts/verify_policy_agent.py"))
inspect_trace = verification_script["inspect_trace"]
policy_calls_to_resume = verification_script["policy_calls_to_resume"]


def sample_trace(answer=None):
    evidence = {
        "source": "refund_policy.md",
        "title": "退款政策（演示）",
        "content": "需要订单号、照片或视频；所有退款必须人工审批。",
        "score": 0.7,
    }
    return [
        HumanMessage(content="质量问题退款需要什么材料？"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "policy-1",
                    "name": "search_after_sales_policy",
                    "args": {"question": "退款材料"},
                }
            ],
        ),
        ToolMessage(
            content=json.dumps({"found": True, "evidence": [evidence]}), tool_call_id="policy-1"
        ),
        AIMessage(
            content=answer
            if answer is not None
            else (
                "根据退款政策（演示）（refund_policy.md），请提供订单号、照片或视频。退款需要人工审批。"
            )
        ),
    ]


def test_complete_trace_passes_structural_checks():
    checks, evidence, answer = inspect_trace(sample_trace())
    assert all(checks.values()), checks
    assert evidence[0]["source"] == "refund_policy.md"
    assert "人工审批" in answer


def test_answer_without_retrieval_cannot_pass():
    checks, _, _ = inspect_trace([HumanMessage(content="退款"), AIMessage(content="需要人工审批")])
    assert not checks["DeepSeek requested the policy tool"]
    assert not checks["Retrieval returned refund evidence"]


def test_answer_without_citation_fails():
    checks, _, _ = inspect_trace(sample_trace("演示政策要求订单号和照片，并需要人工审批。"))
    assert not checks["Answer cites the retrieved refund filename and title"]


def test_wrong_tool_result_id_is_rejected():
    messages = sample_trace()
    messages[2].tool_call_id = "unrequested"
    with pytest.raises(RuntimeError, match="does not match"):
        inspect_trace(messages)


def test_policy_call_can_resume():
    assert policy_calls_to_resume(sample_trace()[:2])[0]["name"] == "search_after_sales_policy"


@pytest.mark.parametrize(
    "tool_name", ["create_ticket", "get_order", "get_ticket", "list_tickets_by_order"]
)
def test_business_tools_are_blocked_before_execution(tool_name):
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "blocked",
                "name": tool_name,
                "args": {},
            }
        ],
    )
    with pytest.raises(RuntimeError, match="Blocked a non-policy tool"):
        policy_calls_to_resume([message])
