"""Integration tests for /invoke and /stream against real compiled graphs.

The unit tests in test_service.py drive an AsyncMock agent, so every tuple shape and
event ordering they assert is hand-built rather than produced by LangGraph. These run the
same endpoints against real graphs on a real checkpointer, so a change in how LangGraph
reports interrupts, orders stream events, or surfaces pending tasks shows up here.
"""

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter, interrupt

from agents.utils import CustomData
from service import app


async def greet(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content="let me check")]}


async def ask_color(state: MessagesState) -> MessagesState:
    answer = interrupt("What is your favorite color?")
    return {"messages": [AIMessage(content=f"Your favorite color is {answer}")]}


def build_interrupt_agent(checkpointer):
    """Emits a message before interrupting, so the interrupt arrives mid-stream."""
    graph = StateGraph(MessagesState)
    graph.add_node("greet", greet)
    graph.add_node("ask", ask_color)
    graph.set_entry_point("greet")
    graph.add_edge("greet", "ask")
    graph.add_edge("ask", END)
    return graph.compile(checkpointer=checkpointer)


async def count_messages(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content=f"heard {len(state['messages'])} messages")]}


def build_counting_agent(checkpointer):
    graph = StateGraph(MessagesState)
    graph.add_node("count", count_messages)
    graph.set_entry_point("count")
    graph.add_edge("count", END)
    return graph.compile(checkpointer=checkpointer)


async def working(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content="working on it")]}


async def finished(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content="all done")]}


def build_two_step_agent(checkpointer):
    graph = StateGraph(MessagesState)
    graph.add_node("working", working)
    graph.add_node("finished", finished)
    graph.set_entry_point("working")
    graph.add_edge("working", "finished")
    graph.add_edge("finished", END)
    return graph.compile(checkpointer=checkpointer)


async def report_progress(state: MessagesState, writer: StreamWriter) -> MessagesState:
    CustomData(data={"status": "running"}).dispatch(writer)
    return {"messages": [AIMessage(content="all done")]}


def build_custom_data_agent(checkpointer):
    """The bg-task-agent shape: a node writing custom data alongside its messages."""
    graph = StateGraph(MessagesState)
    graph.add_node("report", report_progress)
    graph.set_entry_point("report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def checkpointer(request, tmp_path):
    """Pending interrupts and accumulated messages both round-trip through the
    checkpointer, so the tests that depend on them run against a real DB as well."""
    if request.param == "memory":
        yield MemorySaver()
    else:
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as saver:
            yield saver


@asynccontextmanager
async def client_for(agents: dict[str, Any]) -> AsyncGenerator[httpx.AsyncClient, None]:
    transport = httpx.ASGITransport(app=app)
    with patch("service.service.get_agent", side_effect=lambda agent_id: agents[agent_id]):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def stream_events(client: httpx.AsyncClient, path: str, body: dict) -> list[dict[str, Any]]:
    events = []
    async with client.stream("POST", path, json=body) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            events.append(json.loads(line.removeprefix("data: ")))
    return events


def streamed_messages(events: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    return [
        (e["content"]["type"], e["content"]["content"]) for e in events if e["type"] == "message"
    ]


async def history_of(client: httpx.AsyncClient, path: str, thread_id: str) -> list[tuple[str, str]]:
    response = await client.post(path, json={"thread_id": thread_id})
    assert response.status_code == 200
    return [(m["type"], m["content"]) for m in response.json()["messages"]]


@pytest.mark.asyncio
async def test_invoke_resumes_an_interrupted_thread(checkpointer) -> None:
    """The second call must resume the pending interrupt, not start a fresh turn."""
    agents = {"interrupt-graph": build_interrupt_agent(checkpointer)}
    async with client_for(agents) as client:
        first = await client.post(
            "/interrupt-graph/invoke", json={"message": "hi", "thread_id": "t"}
        )
        assert first.status_code == 200
        assert (first.json()["type"], first.json()["content"]) == (
            "ai",
            "What is your favorite color?",
        )

        second = await client.post(
            "/interrupt-graph/invoke", json={"message": "blue", "thread_id": "t"}
        )
        assert second.status_code == 200
        assert (second.json()["type"], second.json()["content"]) == (
            "ai",
            "Your favorite color is blue",
        )

        assert await history_of(client, "/interrupt-graph/history", "t") == [
            ("human", "hi"),
            ("ai", "let me check"),
            ("ai", "Your favorite color is blue"),
        ]


@pytest.mark.asyncio
async def test_stream_resumes_an_interrupted_thread(checkpointer) -> None:
    agents = {"interrupt-graph": build_interrupt_agent(checkpointer)}
    async with client_for(agents) as client:
        first = await stream_events(
            client, "/interrupt-graph/stream", {"message": "hi", "thread_id": "t"}
        )
        assert streamed_messages(first) == [
            ("ai", "let me check"),
            ("ai", "What is your favorite color?"),
        ]

        second = await stream_events(
            client, "/interrupt-graph/stream", {"message": "blue", "thread_id": "t"}
        )
        assert streamed_messages(second) == [("ai", "Your favorite color is blue")]

        assert await history_of(client, "/interrupt-graph/history", "t") == [
            ("human", "hi"),
            ("ai", "let me check"),
            ("ai", "Your favorite color is blue"),
        ]


@pytest.mark.asyncio
async def test_invoke_accumulates_state_across_turns(checkpointer) -> None:
    agents = {"counting-agent": build_counting_agent(checkpointer)}
    async with client_for(agents) as client:
        contents = []
        for message in ["first", "second"]:
            response = await client.post(
                "/counting-agent/invoke", json={"message": message, "thread_id": "t"}
            )
            assert response.status_code == 200
            contents.append(response.json()["content"])

        assert contents == ["heard 1 messages", "heard 3 messages"]
        assert await history_of(client, "/counting-agent/history", "t") == [
            ("human", "first"),
            ("ai", "heard 1 messages"),
            ("human", "second"),
            ("ai", "heard 3 messages"),
        ]


@pytest.mark.asyncio
async def test_invoke_returns_only_the_final_message() -> None:
    """Pins the documented limitation: intermediate AIMessages are dropped by /invoke."""
    agents = {"two-step-agent": build_two_step_agent(MemorySaver())}
    async with client_for(agents) as client:
        response = await client.post(
            "/two-step-agent/invoke", json={"message": "hi", "thread_id": "t"}
        )
        assert response.status_code == 200
        assert (response.json()["type"], response.json()["content"]) == ("ai", "all done")

        assert await history_of(client, "/two-step-agent/history", "t") == [
            ("human", "hi"),
            ("ai", "working on it"),
            ("ai", "all done"),
        ]


@pytest.mark.asyncio
async def test_stream_forwards_custom_data() -> None:
    agents = {"custom-agent": build_custom_data_agent(MemorySaver())}
    async with client_for(agents) as client:
        events = await stream_events(
            client, "/custom-agent/stream", {"message": "hi", "thread_id": "t"}
        )

        messages = [e["content"] for e in events if e["type"] == "message"]
        assert [m["type"] for m in messages] == ["custom", "ai"]
        assert messages[0]["custom_data"] == {"status": "running"}
        assert messages[1]["content"] == "all done"
