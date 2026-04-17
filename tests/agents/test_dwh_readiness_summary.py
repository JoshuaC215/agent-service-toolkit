import json

import pytest
from langchain_community.chat_models import FakeListChatModel

import agents.dwh_readiness_summary as dwh_module  # type: ignore[import,import-not-found]


@pytest.fixture(autouse=True)
def patch_get_model(monkeypatch):
    """Return deterministic FakeListChatModel so the graph has reproducible output."""
    fake_response = json.dumps(
        {
            "summary": "Dies ist eine Test-Summary.",
            "recommendations": ["Empfehlung 1", "Empfehlung 2", "Empfehlung 3"],
        }
    )

    monkeypatch.setattr(
        dwh_module,
        "get_model",
        lambda *args, **kwargs: FakeListChatModel(responses=[fake_response]),
    )


def test_dwh_readiness_summary_graph():
    """Graph should return llm_output with summary & recommendations."""
    graph = dwh_module.dwh_readiness_summary
    readiness_result = {
        "company_size": 200,
        "tech_affinity": 4,
        "challenge_result": "Hoher Handlungsbedarf",
    }
    final_state = graph.invoke({"readiness_result": readiness_result})  # type: ignore[attr-defined]
    assert "llm_output" in final_state
    llm_out = final_state["llm_output"]
    assert llm_out["summary"].startswith("Dies ist eine Test-Summary")
    assert len(llm_out["recommendations"]) == 3
