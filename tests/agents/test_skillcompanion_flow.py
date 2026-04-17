from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agents.skillcompanion import (
    after_process_answer,
    assign_category,
    finish,
    interrupt_process,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "idx,expected",
    [
        (0, "interrupt_process"),
        (1, "interrupt_process"),
        (8, "interrupt_process"),
        (9, "categorize"),
        (10, "categorize"),
    ],
)
async def test_after_process_answer_routing(idx, expected):
    state = {"testid": idx}
    nxt = await after_process_answer(state)
    assert nxt == expected


@pytest.mark.asyncio
async def test_interrupt_process_appends_user_message(monkeypatch):
    # Monkeypatch the LangGraph interrupt primitive used inside helpers.interrupt_and_append
    import agents.helpers as helpers

    monkeypatch.setattr(helpers, "interrupt", lambda prompt: "Stub user clarification")

    state = {"messages": []}
    updated = await interrupt_process(state)
    assert updated is state  # in-place update
    assert len(state["messages"]) == 1
    msg = state["messages"][0]
    assert isinstance(msg, HumanMessage)
    assert msg.content == "Stub user clarification"


@pytest.mark.asyncio
async def test_assign_category_with_mock_model(monkeypatch):
    # Return a fake model whose ainvoke returns a simple object with .content
    class FakeModel:
        async def ainvoke(self, messages, config=None):
            return SimpleNamespace(content="Advanced")

    import agents.skillcompanion as sc

    monkeypatch.setattr(sc, "_get_llm_model", lambda: FakeModel())

    state = {"answers": [{"question": "Q1", "answer": "A1"}]}
    out = await assign_category(state)
    assert out["category"] == "Advanced"


@pytest.mark.asyncio
async def test_finish_sets_finished_and_contains_link_default_bento(monkeypatch):
    # Ensure default base URL (env var not set) to avoid coupling to environment
    monkeypatch.delenv("HUBSPOT_URL", raising=False)
    monkeypatch.delenv("BENTO_URL", raising=False)

    state = {"category": "Beginner"}
    config = {"configurable": {}}
    out = await finish(state, config)
    assert out["finished"] is True
    assert out["messages"], "Should include AIMessage with content"
    content = out["messages"][0].content
    assert "https://bento.roosi.ai/?kiskill=Beginner" in content
