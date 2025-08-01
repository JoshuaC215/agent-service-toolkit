import os
import pytest
from langchain_core.messages import AIMessage
from agents.skillcompanion_interrupted import finish_skill_check, AgentState

@pytest.mark.asyncio
async def test_finish_skill_check_without_url_parameters(monkeypatch):
    # Standardfall: Kein hubspot_id, Link zu Bento mit kiskill
    state = AgentState(category="Advanced")
    config = {"configurable": {}}
    result = await finish_skill_check(state, config)
    assert isinstance(result, dict)
    assert "messages" in result
    msg = result["messages"][0].content
    assert "https://bento.roosi.ai/?kiskill=Advanced" in msg
    assert "hubspot_id" not in msg

@pytest.mark.asyncio
async def test_finish_skill_check_with_hubspot_id(monkeypatch):
    # Mit hubspot_id: Link zu Hubspot-Basis-URL mit hubspot_id-Query
    state = AgentState(category="Expert")
    config = {"configurable": {"url_parameters": {"hubspot_id": "12345"}}}
    monkeypatch.setenv("HUBSPOT_URL", "https://hubspot.example.com")
    result = await finish_skill_check(state, config)
    assert isinstance(result, dict)
    assert "messages" in result
    msg = result["messages"][0].content
    assert "https://hubspot.example.com/?hubspot_id=12345" in msg
    assert "kiskill" not in msg