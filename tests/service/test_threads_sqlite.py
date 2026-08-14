"""Integration tests for /threads against a real SQLite checkpointer.

The unit tests in test_service.py drive a fake checkpointer, so they can't catch a
metadata filter that the database rejects or a head-checkpoint assumption that LangGraph
doesn't actually hold. These run real graphs through a real checkpointer instead.
"""

from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.func import entrypoint
from langgraph.graph import END, MessagesState, StateGraph

from service import app


async def echo(state: MessagesState) -> MessagesState:
    return {"messages": [AIMessage(content=f"echo: {state['messages'][-1].content}")]}


def build_graph_agent(checkpointer):
    graph = StateGraph(MessagesState)
    graph.add_node("echo", echo)
    graph.set_entry_point("echo")
    graph.add_edge("echo", END)
    return graph.compile(checkpointer=checkpointer)


def build_subgraph_agent(checkpointer):
    """A graph that calls a checkpointed subgraph, like the supervisor agents do."""
    inner = StateGraph(MessagesState)
    inner.add_node("echo", echo)
    inner.set_entry_point("echo")
    inner.add_edge("echo", END)

    outer = StateGraph(MessagesState)
    outer.add_node("worker", inner.compile())
    outer.set_entry_point("worker")
    outer.add_edge("worker", END)
    return outer.compile(checkpointer=checkpointer)


def build_functional_agent(checkpointer):
    @entrypoint(checkpointer=checkpointer)
    async def functional(inputs: dict, *, previous: dict | None = None) -> dict:
        messages = inputs["messages"]
        if previous:
            messages = previous["messages"] + messages
        response = AIMessage(content=f"echo: {messages[-1].content}")
        return entrypoint.final(
            value={"messages": [response]}, save={"messages": messages + [response]}
        )

    return functional


async def run_turns(agent, thread_id: str, user_id: str, agent_id: str, messages: list[str]):
    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        metadata={"user_id": user_id, "agent_id": agent_id},
    )
    for message in messages:
        await agent.ainvoke({"messages": [HumanMessage(content=message)]}, config=config)


@pytest_asyncio.fixture
async def seeded(tmp_path):
    """Seed two agents x two users, with both single-turn and multi-turn threads."""
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db")) as checkpointer:
        agents = {
            "graph-agent": build_graph_agent(checkpointer),
            "functional-agent": build_functional_agent(checkpointer),
            "subgraph-agent": build_subgraph_agent(checkpointer),
        }
        threads = {
            ("graph-agent", "alice", "g-alice-single"): ["only turn"],
            ("graph-agent", "alice", "g-alice-multi"): ["first turn", "second", "third"],
            ("graph-agent", "bob", "g-bob-single"): ["bob only turn"],
            ("functional-agent", "alice", "f-alice-single"): ["fn only turn"],
            ("functional-agent", "alice", "f-alice-multi"): ["fn first turn", "fn second"],
            ("subgraph-agent", "alice", "s-alice-single"): ["sub only turn"],
            ("subgraph-agent", "alice", "s-alice-multi"): ["sub first turn", "sub second"],
        }
        for (agent_id, user_id, thread_id), messages in threads.items():
            await run_turns(agents[agent_id], thread_id, user_id, agent_id, messages)

        transport = httpx.ASGITransport(app=app)
        lookup = {"side_effect": lambda agent_id: agents[agent_id]}
        with (
            patch("service.service.get_agent", **lookup),
            patch("service.agui.get_agent", **lookup),
        ):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client


@pytest.mark.asyncio
async def test_threads_lists_single_and_multi_turn_threads(seeded) -> None:
    """Single-turn threads never advance past their head checkpoint - they must still list."""
    response = await seeded.get("/graph-agent/threads", params={"user_id": "alice"})

    assert response.status_code == 200
    threads = response.json()["threads"]
    assert [t["thread_id"] for t in threads] == ["g-alice-multi", "g-alice-single"]
    assert [t["title"] for t in threads] == ["first turn", "only turn"]
    assert all(t["updated_at"] for t in threads)


@pytest.mark.asyncio
async def test_threads_isolates_users_and_agents(seeded) -> None:
    async def thread_ids(agent_id: str, user_id: str) -> list[str]:
        response = await seeded.get(f"/{agent_id}/threads", params={"user_id": user_id})
        assert response.status_code == 200
        return sorted(t["thread_id"] for t in response.json()["threads"])

    assert await thread_ids("graph-agent", "alice") == ["g-alice-multi", "g-alice-single"]
    assert await thread_ids("graph-agent", "bob") == ["g-bob-single"]
    assert await thread_ids("functional-agent", "alice") == ["f-alice-multi", "f-alice-single"]
    assert await thread_ids("functional-agent", "bob") == []
    assert await thread_ids("graph-agent", "nobody") == []


@pytest.mark.asyncio
async def test_threads_titles_functional_api_agent(seeded) -> None:
    """Functional-API agents keep messages in `__previous__`, not the `messages` channel."""
    response = await seeded.get("/functional-agent/threads", params={"user_id": "alice"})

    assert response.status_code == 200
    threads = response.json()["threads"]
    assert [t["thread_id"] for t in threads] == ["f-alice-multi", "f-alice-single"]
    assert [t["title"] for t in threads] == ["fn first turn", "fn only turn"]


@pytest.mark.asyncio
async def test_threads_lists_subgraph_threads_once(seeded) -> None:
    """Subgraph runs write their own head checkpoint under a nested namespace.

    Those inherit the parent run's user_id/agent_id metadata, so they match the same
    query and would otherwise be listed as extra copies of the thread, each costing
    its own tip lookup.
    """
    tip_lookups: list[str] = []
    original = AsyncSqliteSaver.aget_tuple

    async def counting_aget_tuple(self, config):
        tip_lookups.append(config["configurable"]["thread_id"])
        return await original(self, config)

    with patch.object(AsyncSqliteSaver, "aget_tuple", counting_aget_tuple):
        response = await seeded.get("/subgraph-agent/threads", params={"user_id": "alice"})

    assert response.status_code == 200
    threads = response.json()["threads"]
    assert [t["thread_id"] for t in threads] == ["s-alice-multi", "s-alice-single"]
    assert [t["title"] for t in threads] == ["sub first turn", "sub only turn"]
    assert sorted(tip_lookups) == ["s-alice-multi", "s-alice-single"]


@pytest.mark.asyncio
async def test_threads_orders_by_most_recent_update(seeded) -> None:
    """A reply to the oldest thread should move it to the top of the list."""
    before = await seeded.get("/graph-agent/threads", params={"user_id": "alice"})
    assert before.json()["threads"][-1]["thread_id"] == "g-alice-single"

    response = await seeded.post(
        "/graph-agent/invoke",
        json={"message": "a later reply", "thread_id": "g-alice-single", "user_id": "alice"},
    )
    assert response.status_code == 200

    after = await seeded.get("/graph-agent/threads", params={"user_id": "alice"})
    assert [t["thread_id"] for t in after.json()["threads"]] == [
        "g-alice-single",
        "g-alice-multi",
    ]


@pytest.mark.asyncio
async def test_threads_respects_limit(seeded) -> None:
    response = await seeded.get("/graph-agent/threads", params={"user_id": "alice", "limit": 1})

    assert response.status_code == 200
    assert [t["thread_id"] for t in response.json()["threads"]] == ["g-alice-multi"]


@pytest.mark.asyncio
async def test_agui_runs_are_listed_by_threads(seeded) -> None:
    """An AG-UI run records the same metadata, so its thread lists like any other."""
    response = await seeded.post(
        "/agui/graph-agent/run",
        json={
            "threadId": "agui-thread",
            "runId": "agui-run",
            "messages": [{"id": "m1", "role": "user", "content": "from ag-ui"}],
            "tools": [],
            "context": [],
            "state": {},
            "forwardedProps": {"configurable": {"user_id": "alice"}},
        },
    )
    assert response.status_code == 200

    threads = (await seeded.get("/graph-agent/threads", params={"user_id": "alice"})).json()
    listed = {t["thread_id"]: t for t in threads["threads"]}
    assert "agui-thread" in listed
    assert listed["agui-thread"]["title"] == "from ag-ui"
    assert listed["agui-thread"]["agent_id"] == "graph-agent"


@pytest.mark.asyncio
async def test_history_returns_full_conversation(seeded) -> None:
    for agent_id, thread_id, first, turns in [
        ("graph-agent", "g-alice-multi", "first turn", 3),
        ("functional-agent", "f-alice-multi", "fn first turn", 2),
    ]:
        response = await seeded.post(f"/{agent_id}/history", json={"thread_id": thread_id})
        assert response.status_code == 200
        messages = response.json()["messages"]
        assert [m["type"] for m in messages] == ["human", "ai"] * turns
        assert messages[0]["content"] == first
