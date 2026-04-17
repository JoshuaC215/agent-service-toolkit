from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import agents.skillcompanion as sc


@pytest.mark.asyncio
async def test_ask_question_increments_and_emits_message(monkeypatch):
    # Fake model returning a deterministic question text
    class FakeModel:
        async def ainvoke(self, messages, config=None):
            return SimpleNamespace(content="Was ist KI?")

    # Ensure skillcompanion uses our fake model
    monkeypatch.setattr(sc, "_get_llm_model", lambda: FakeModel())

    state = {"messages": [], "answers": [], "testid": 0}
    out = await sc.ask_question(state)

    assert out["testid"] == 1  # incremented
    assert isinstance(out["messages"][0], AIMessage)
    assert out["messages"][0].content == "Was ist KI?"
    assert out["question"] == "Was ist KI?"
