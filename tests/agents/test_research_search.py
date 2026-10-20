"""Exercise search selection and MCP through the research graph with a scripted model."""

import asyncio
import importlib
import json
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from agents import get_agent
from agents.safeguard import SafeguardOutput, SafetyAssessment
from agents.web_search import PARALLEL_MCP_URL, PROJECT_USER_AGENT, get_search_tools
from core import settings
from core.settings import Settings

research = importlib.import_module("agents.research_assistant")


class MCPServer:
    def __init__(self):
        self.requests = []
        self.calls = []
        self.error = False
        self.delay = False
        self.transport_error = False
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def respond(self, request):
        self.requests.append(request)
        assert str(request.url) == PARALLEL_MCP_URL
        assert request.headers["User-Agent"] == PROJECT_USER_AGENT
        assert "authorization" not in request.headers
        if self.transport_error:
            raise httpx.ConnectError("offline", request=request)
        if request.method != "POST":
            return httpx.Response(405)
        body = json.loads(request.content)
        method = body["method"]
        if method.startswith("notifications/"):
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-search", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": name,
                        "description": name,
                        "inputSchema": {"type": "object", "additionalProperties": True},
                    }
                    for name in ("web_search", "web_fetch")
                ]
            }
        elif method == "tools/call":
            self.calls.append(body["params"])
            self.started.set()
            if self.delay:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": "quota exceeded"
                        if self.error
                        else "Source https://example.com/topic",
                    }
                ],
                "isError": self.error,
            }
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


@pytest.fixture
def server(monkeypatch):
    server = MCPServer()

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(server.respond), **kwargs)

    monkeypatch.setattr("langchain_mcp_adapters.sessions.create_mcp_http_client", client_factory)
    return server


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_PROVIDER", "parallel")
    safe = SafeguardOutput(safety_assessment=SafetyAssessment.SAFE)
    monkeypatch.setattr(research, "Safeguard", lambda: Mock(ainvoke=AsyncMock(return_value=safe)))

    def respond(messages):
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        results = [m for m in messages[last_human:] if isinstance(m, ToolMessage)]
        if results and results[-1].status == "error":
            return AIMessage(content="Search unavailable: " + str(results[-1].content))
        if len(results) == 2:
            return AIMessage(content="Answer from " + str(results[-1].content))
        name = "WebFetch" if results else "WebSearch"
        args = (
            {"urls": ["https://example.com/topic"]}
            if results
            else {"objective": messages[last_human].content, "search_queries": ["example topic"]}
        )
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])

    model = Mock()
    model.bind_tools.side_effect = lambda tools: RunnableLambda(respond)
    monkeypatch.setattr(research, "get_model", lambda _: model)
    graph = get_agent("research-assistant")
    monkeypatch.setattr(graph, "checkpointer", InMemorySaver())
    return graph


def config(thread, **overrides):
    return {"configurable": {"thread_id": thread, **overrides}}


async def ask(graph, thread, **overrides):
    return await graph.ainvoke(
        {"messages": [HumanMessage(content="Find a source")]}, config(thread, **overrides)
    )


@pytest.mark.asyncio
async def test_search_fetch_followup_and_new_chat(graph, server):
    first = await ask(graph, "one")
    followup = await ask(graph, "one")
    new = await ask(graph, "new-chat")
    assert "https://example.com/topic" in first["messages"][-1].content
    assert first["search_session_id"] == followup["search_session_id"]
    assert UUID(first["search_session_id"]) != UUID(new["search_session_id"])
    assert [c["name"] for c in server.calls] == ["web_search", "web_fetch"] * 3
    assert [c["arguments"]["session_id"] for c in server.calls] == [
        first["search_session_id"]
    ] * 4 + [new["search_session_id"]] * 2
    assert server.calls[0]["arguments"]["search_queries"] == ["example topic"]
    assert server.calls[1]["arguments"]["urls"] == ["https://example.com/topic"]
    assert any(json.loads(r.content).get("method") == "tools/list" for r in server.requests)


@pytest.mark.asyncio
async def test_interleaved_conversations(graph, server):
    a, b = await asyncio.gather(ask(graph, "a"), ask(graph, "b"))
    assert a["search_session_id"] != b["search_session_id"]
    for state in (a, b):
        calls = [
            c for c in server.calls if c["arguments"]["session_id"] == state["search_session_id"]
        ]
        assert [c["name"] for c in calls] == ["web_search", "web_fetch"]
        assert "https://example.com/topic" in state["messages"][-1].content


@pytest.mark.asyncio
async def test_per_request_parallel_override(graph, server, monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_PROVIDER", "duckduckgo")
    state = await ask(graph, "override", search_provider="parallel")
    assert [c["name"] for c in server.calls] == ["web_search", "web_fetch"]
    assert "https://example.com/topic" in state["messages"][-1].content


@pytest.mark.asyncio
async def test_transport_error_reaches_model(graph, server):
    server.transport_error = True
    state = await ask(graph, "offline")
    assert "Parallel search failed" in state["messages"][-1].content
    assert state["messages"][-2].status == "error"
    assert server.calls == []


@pytest.mark.asyncio
async def test_quota_and_retry_keep_conversation_id(graph, server):
    server.error = True
    first = await ask(graph, "retry")
    assert "quota exceeded" in first["messages"][-1].content
    assert first["messages"][-2].status == "error"
    assert len(server.calls) == 1
    server.error = False
    retry = await ask(graph, "retry")
    assert retry["search_session_id"] == first["search_session_id"]
    assert {c["arguments"]["session_id"] for c in server.calls} == {first["search_session_id"]}


@pytest.mark.asyncio
async def test_timeout_reaches_model(graph, server, monkeypatch):
    server.delay = True
    monkeypatch.setattr(settings, "SEARCH_TIMEOUT_SECONDS", 0.15)
    state = await ask(graph, "timeout")
    assert "timed out" in state["messages"][-1].content
    assert state["messages"][-2].status == "error"
    assert server.cancelled.is_set()


@pytest.mark.asyncio
async def test_cancellation_propagates(graph, server):
    server.delay = True
    task = asyncio.create_task(ask(graph, "cancel"))
    await asyncio.wait_for(server.started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(server.cancelled.wait(), 5)


def test_default_saved_setting_and_request_override(monkeypatch):
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    fresh = Settings(_env_file=None, USE_FAKE_MODEL=True)
    assert fresh.SEARCH_PROVIDER == "duckduckgo"
    monkeypatch.setattr(settings, "SEARCH_PROVIDER", fresh.SEARCH_PROVIDER)
    assert len(get_search_tools({})) == 1
    assert get_search_tools({})[0].__class__.__name__ == "DuckDuckGoSearchResults"
    assert len(get_search_tools(config("override", search_provider="parallel"))) == 2
    monkeypatch.setenv("SEARCH_PROVIDER", "parallel")
    saved = Settings(_env_file=None, USE_FAKE_MODEL=True)
    monkeypatch.setattr(settings, "SEARCH_PROVIDER", saved.SEARCH_PROVIDER)
    assert len(get_search_tools({})) == 2
    assert get_search_tools(config("off", search_provider="none")) == []
    assert get_search_tools(config("explicit", search_provider="duckduckgo"))[0].name == "WebSearch"
    with pytest.raises(ValueError, match="Unknown search provider"):
        get_search_tools(config("invalid", search_provider="invalid"))


@pytest.mark.asyncio
async def test_disabled_search_cannot_be_reenabled(graph, server, monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_PROVIDER", "none")
    bindings = []

    def bind(tools):
        bindings.extend(t.name for t in tools)
        return RunnableLambda(lambda _: AIMessage(content="No search tools available"))

    monkeypatch.setattr(research, "get_model", lambda _: Mock(bind_tools=bind))
    result = await ask(graph, "disabled", search_provider="parallel")
    assert "WebSearch" not in bindings and "WebFetch" not in bindings
    assert "Calculator" in bindings
    assert server.requests == []
    assert result["messages"][-1].content == "No search tools available"


@pytest.mark.asyncio
async def test_safeguard_blocks_search(graph, server, monkeypatch):
    unsafe = SafeguardOutput(
        safety_assessment=SafetyAssessment.UNSAFE, unsafe_categories=["test violation"]
    )
    monkeypatch.setattr(research, "Safeguard", lambda: Mock(ainvoke=AsyncMock(return_value=unsafe)))
    result = await ask(graph, "unsafe")
    assert "flagged for unsafe content" in result["messages"][-1].content
    assert server.requests == []


def test_session_id_is_not_model_visible():
    for tool in get_search_tools(config("schema", search_provider="parallel")):
        assert "session_id" not in tool.tool_call_schema.model_json_schema()["properties"]
