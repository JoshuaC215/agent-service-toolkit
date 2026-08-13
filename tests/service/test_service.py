import json
from unittest.mock import AsyncMock, patch

import langsmith
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Interrupt, StateSnapshot

from agents.agents import Agent
from schema import ChatHistory, ChatMessage, ServiceMetadata
from schema.models import AnthropicModelName, OpenAIModelName


def test_invoke(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent(test_client, mock_agent) -> None:
    """Test that /invoke works with a custom agent_id path parameter."""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    CUSTOM_ANSWER = "The weather in Tokyo is sunny."
    DEFAULT_ANSWER = "This is from the default agent."

    # Create a separate mock for the default agent
    default_mock = AsyncMock()
    default_mock.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=DEFAULT_ANSWER)]})
    ]

    # Configure our custom mock agent
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=CUSTOM_ANSWER)]})]

    # Patch get_agent to return the correct agent based on the provided agent_id
    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return mock_agent
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(f"/{CUSTOM_AGENT}/invoke", json={"message": QUESTION})
        assert response.status_code == 200

        # Verify custom agent was called and default wasn't
        mock_agent.ainvoke.assert_awaited_once()
        default_mock.ainvoke.assert_not_awaited()

        input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
        assert input_message.content == QUESTION

        output = ChatMessage.model_validate(response.json())
        assert output.type == "ai"
        assert output.content == CUSTOM_ANSWER  # Verify we got the custom agent's response


def test_invoke_model_param(test_client, mock_agent) -> None:
    """Test that the model parameter is correctly passed to the agent if specified."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_MODEL = OpenAIModelName.GPT_5_MINI
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION, "model": CUSTOM_MODEL})
    assert response.status_code == 200

    # Verify the model was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["model"] == CUSTOM_MODEL

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify a valid enum outside the configured allowlist returns a 400.
    unavailable_model = AnthropicModelName.SONNET_45
    response = test_client.post("/invoke", json={"message": QUESTION, "model": unavailable_model})
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]

    # Verify a malformed model string still fails request validation.
    INVALID_MODEL = "gpt-7-notreal"
    response = test_client.post("/invoke", json={"message": QUESTION, "model": INVALID_MODEL})
    assert response.status_code == 422


def test_invoke_no_model_param_uses_none_default(test_client, mock_agent) -> None:
    """Test that when no model is specified, UserInput defaults to None and isn't passed to the runnable config (not hardcoded gpt-5-nano)."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    # Don't specify model in the request
    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert "model" not in config["configurable"]  # Should not be present when None

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent_config(test_client, mock_agent) -> None:
    """Test that the agent_config parameter is correctly passed to the agent."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_CONFIG = {"spicy_level": 0.1, "additional_param": "value_foo"}

    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": CUSTOM_CONFIG}
    )
    assert response.status_code == 200

    # Verify the agent_config was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["spicy_level"] == 0.1
    assert config["configurable"]["additional_param"] == "value_foo"

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify a reserved key in agent_config throws a validation error
    INVALID_CONFIG = {"model": "gpt-5-nano"}
    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": INVALID_CONFIG}
    )
    assert response.status_code == 422


def test_invoke_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    INTERRUPT = "Confirm weather check"
    mock_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=ANSWER)]}),
        ("updates", {"__interrupt__": [Interrupt(value=INTERRUPT)]}),
    ]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == INTERRUPT


@patch("service.service.LangsmithClient")
def test_feedback(mock_client: langsmith.Client, test_client) -> None:
    ls_instance = mock_client.return_value
    ls_instance.create_feedback.return_value = None
    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    ls_instance.create_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="human-feedback-stars",
        score=0.8,
    )


def test_history(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    user_question = HumanMessage(content=QUESTION)
    agent_response = AIMessage(content=ANSWER)
    mock_agent.aget_state.return_value = StateSnapshot(
        values={"messages": [user_question, agent_response]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    response = test_client.post(
        "/history", json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"}
    )
    assert response.status_code == 200

    output = ChatHistory.model_validate(response.json())
    assert output.messages[0].type == "human"
    assert output.messages[0].content == QUESTION
    assert output.messages[1].type == "ai"
    assert output.messages[1].content == ANSWER


class FakeCheckpointTuple:
    def __init__(self, thread_id: str, checkpoint_id: str, checkpoint: dict, metadata: dict):
        self.config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        self.checkpoint = checkpoint
        self.metadata = metadata


class FakeCheckpointer:
    """A checkpointer with the semantics /threads relies on.

    Checkpoints are globally ordered by checkpoint_id, `alist` applies the metadata
    filter as exact matches, and each thread's first checkpoint is written at step -1
    the way LangGraph writes an input checkpoint.
    """

    def __init__(self):
        self.rows: list[FakeCheckpointTuple] = []
        self.alist_filters: list[dict | None] = []
        self._next_id = 0

    def _checkpoint_id(self) -> str:
        self._next_id += 1
        return f"cp-{self._next_id:06d}"

    def add_thread(
        self,
        thread_id: str,
        user_id: str = "user-123",
        agent_id: str = "research-assistant",
        turns: int = 1,
        title: str = "Hello",
        functional_api: bool = False,
        tip_ts: str | None = "2024-07-31T20:14:19.804150+00:00",
    ) -> None:
        def add(step: int, channel_values: dict, ts: str | None) -> None:
            self.rows.append(
                FakeCheckpointTuple(
                    thread_id,
                    self._checkpoint_id(),
                    {"ts": ts, "channel_values": channel_values},
                    {"step": step, "user_id": user_id, "agent_id": agent_id},
                )
            )

        add(-1, {"__start__": {"messages": [HumanMessage(content=title)]}}, "2024-01-01T00:00:00Z")
        messages: list = []
        for turn in range(turns):
            messages = messages + [
                HumanMessage(content=title if turn == 0 else f"{title} {turn}"),
                AIMessage(content="reply"),
            ]
            channels = (
                {"__end__": {"messages": messages[-1:]}, "__previous__": {"messages": messages}}
                if functional_api
                else {"messages": messages}
            )
            add(turn * 2, channels, tip_ts if turn == turns - 1 else "2024-01-01T00:00:01Z")

    async def alist(self, config, *, filter=None, before=None, limit=None):
        self.alist_filters.append(filter)
        rows = sorted(self.rows, key=lambda r: r.config["configurable"]["checkpoint_id"])
        rows.reverse()
        yielded = 0
        for row in rows:
            if filter and any(row.metadata.get(key) != value for key, value in filter.items()):
                continue
            if (
                before
                and row.config["configurable"]["checkpoint_id"]
                >= (before["configurable"]["checkpoint_id"])
            ):
                continue
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    async def aget_tuple(self, config):
        thread_id = config["configurable"]["thread_id"]
        rows = [r for r in self.rows if r.config["configurable"]["thread_id"] == thread_id]
        if not rows:
            return None
        return max(rows, key=lambda r: r.config["configurable"]["checkpoint_id"])


def test_threads_without_checkpointer_returns_empty(test_client, mock_agent) -> None:
    """Test that /threads returns an empty list when the agent has no checkpointer configured."""
    mock_agent.checkpointer = None

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    assert response.json() == {"threads": []}


def test_threads_filters_and_orders_from_checkpointer(test_client, mock_agent) -> None:
    """Test that /threads only returns matching summaries, most recently updated first."""
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-b", title="Second", tip_ts="2024-07-31T20:14:19.804150+00:00")
    checkpointer.add_thread("thread-c", user_id="other-user", title="Ignored")
    checkpointer.add_thread("thread-a", title="First", tip_ts="2024-07-31T20:15:19.804150+00:00")
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert [thread["thread_id"] for thread in payload["threads"]] == ["thread-a", "thread-b"]
    assert [thread["title"] for thread in payload["threads"]] == ["First", "Second"]
    assert [thread["agent_id"] for thread in payload["threads"]] == [
        "research-assistant",
        "research-assistant",
    ]
    assert [thread["updated_at"].replace("Z", "+00:00") for thread in payload["threads"]] == [
        "2024-07-31T20:15:19.804150+00:00",
        "2024-07-31T20:14:19.804150+00:00",
    ]


def test_threads_enumerates_by_head_checkpoint(test_client, mock_agent) -> None:
    """Test that /threads enumerates threads by head checkpoint instead of scanning them all."""
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-single", title="One turn", turns=1)
    checkpointer.add_thread("thread-many", title="Many turns", turns=30)
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    # A single-turn thread never advances past its head, so it must still be listed.
    assert sorted(thread["thread_id"] for thread in payload["threads"]) == [
        "thread-many",
        "thread-single",
    ]
    assert checkpointer.alist_filters == [
        {"user_id": "user-123", "agent_id": "research-assistant", "step": -1}
    ]


def test_threads_filters_by_agent_id(test_client) -> None:
    """Test that /threads scopes the checkpointer query to the requested agent."""
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-mine", agent_id="custom-agent", title="Mine")
    checkpointer.add_thread("thread-theirs", agent_id="other-agent", title="Theirs")

    custom_agent = AsyncMock()
    custom_agent.checkpointer = checkpointer
    default_agent = AsyncMock()
    default_agent.checkpointer = None

    agent_calls = {"default": 0, "custom": 0}

    def agent_lookup(agent_id):
        if agent_id == "custom-agent":
            agent_calls["custom"] += 1
            return custom_agent
        agent_calls["default"] += 1
        return default_agent

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.get("/custom-agent/threads", params={"user_id": "user-123"})

    assert response.status_code == 200
    payload = response.json()
    assert [thread["thread_id"] for thread in payload["threads"]] == ["thread-mine"]
    assert checkpointer.alist_filters == [
        {"user_id": "user-123", "agent_id": "custom-agent", "step": -1}
    ]
    assert agent_calls == {"custom": 1, "default": 0}


def test_threads_truncates_to_limit(test_client, mock_agent) -> None:
    """Test that /threads returns the most recently updated threads up to the limit."""
    checkpointer = FakeCheckpointer()
    for index in range(5):
        checkpointer.add_thread(f"thread-{index}", title=f"Thread {index}")
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 2})

    assert response.status_code == 200
    payload = response.json()
    assert [thread["thread_id"] for thread in payload["threads"]] == ["thread-4", "thread-3"]


@pytest.mark.parametrize("limit", [0, -1, 101, 999999])
def test_threads_rejects_out_of_range_limit(test_client, mock_agent, limit: int) -> None:
    """Test that /threads bounds limit so a client can't ask for an unbounded scan."""
    mock_agent.checkpointer = FakeCheckpointer()

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": limit})

    assert response.status_code == 422


def test_threads_skips_checkpoints_with_mismatched_metadata(test_client, mock_agent) -> None:
    """Test that /threads drops threads the checkpointer filter should not have returned."""

    class LeakyCheckpointer(FakeCheckpointer):
        async def alist(self, config, *, filter=None, before=None, limit=None):
            async for row in super().alist(config, filter=None, before=before, limit=limit):
                if row.metadata.get("step") == -1:
                    yield row

    checkpointer = LeakyCheckpointer()
    checkpointer.add_thread("thread-ok", title="Mine")
    checkpointer.add_thread("thread-other-user", user_id="other-user", title="Theirs")
    checkpointer.add_thread("thread-other-agent", agent_id="other-agent", title="Elsewhere")
    checkpointer.rows.append(
        FakeCheckpointTuple(
            "thread-no-metadata",
            "cp-999999",
            {"ts": None, "channel_values": {}},
            {"step": -1},
        )
    )
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    assert [thread["thread_id"] for thread in response.json()["threads"]] == ["thread-ok"]


def test_threads_deduplicates_threads(test_client, mock_agent) -> None:
    """Test that a checkpointer returning several rows per thread still yields one summary."""

    class UnfilteredCheckpointer(FakeCheckpointer):
        async def alist(self, config, *, filter=None, before=None, limit=None):
            async for row in super().alist(config, filter=None, before=before, limit=limit):
                yield row

    checkpointer = UnfilteredCheckpointer()
    checkpointer.add_thread("thread-a", title="Repeated", turns=4)
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    assert [thread["thread_id"] for thread in response.json()["threads"]] == ["thread-a"]


def test_threads_titles_functional_api_threads(test_client, mock_agent) -> None:
    """Test that /threads titles threads whose messages live in the `__previous__` channel."""
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-fn", title="Functional title", turns=3, functional_api=True)
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    assert response.json()["threads"][0]["title"] == "Functional title"


def test_threads_tolerates_missing_timestamp(test_client, mock_agent) -> None:
    """Test that /threads returns a thread whose tip checkpoint has no timestamp."""
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-no-ts", title="No timestamp", tip_ts=None)
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["threads"][0]["thread_id"] == "thread-no-ts"
    assert payload["threads"][0]["updated_at"] is None


def test_threads_checkpointer_error_returns_500(test_client, mock_agent) -> None:
    async def broken_alist(*args, **kwargs):
        raise RuntimeError("db unavailable")
        yield  # pragma: no cover

    mock_agent.checkpointer = type("Checkpointer", (), {})()
    mock_agent.checkpointer.alist = broken_alist

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})
    assert response.status_code == 500


def test_history_reads_functional_api_previous_channel(test_client, mock_agent) -> None:
    """Test that /history returns the full conversation for a functional-API agent.

    Their state lives in `__previous__`; aget_state only returns the entrypoint's
    final value, which is a single stray AI message.
    """
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-fn", title="Hi", turns=2, functional_api=True)
    mock_agent.checkpointer = checkpointer
    mock_agent.aget_state.return_value = StateSnapshot(
        values={"messages": [AIMessage(content="stray")]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    response = test_client.post("/history", json={"thread_id": "thread-fn"})

    assert response.status_code == 200
    output = ChatHistory.model_validate(response.json())
    assert [message.type for message in output.messages] == ["human", "ai", "human", "ai"]
    assert output.messages[0].content == "Hi"


def test_history_custom_agent(test_client) -> None:
    """Test that /{agent_id}/history reads the thread through the requested agent's graph."""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."

    custom_snapshot = StateSnapshot(
        values={"messages": [HumanMessage(content=QUESTION), AIMessage(content=ANSWER)]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )
    # The default agent's graph doesn't know about this thread, so it returns no messages.
    default_snapshot = StateSnapshot(
        values={"messages": []},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    custom_mock = AsyncMock()
    custom_mock.aget_state.return_value = custom_snapshot
    custom_mock.checkpointer = None
    default_mock = AsyncMock()
    default_mock.aget_state.return_value = default_snapshot
    default_mock.checkpointer = None

    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return custom_mock
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(
            f"/{CUSTOM_AGENT}/history",
            json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"},
        )
        assert response.status_code == 200

        # The custom agent's graph was used, not the default one.
        custom_mock.aget_state.assert_awaited_once()
        default_mock.aget_state.assert_not_awaited()

        output = ChatHistory.model_validate(response.json())
        assert output.messages[0].type == "human"
        assert output.messages[0].content == QUESTION
        assert output.messages[1].type == "ai"
        assert output.messages[1].content == ANSWER


@pytest.mark.asyncio
async def test_stream(test_client, mock_agent) -> None:
    """Test streaming tokens and messages."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": True}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify streamed tokens
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == len(TOKENS)
        for i, msg in enumerate(token_messages):
            assert msg["content"] == TOKENS[i]

        # Verify final message
        final_messages = [msg for msg in messages if msg["type"] == "message"]
        assert len(final_messages) == 1
        assert final_messages[0]["content"]["content"] == FINAL_ANSWER
        assert final_messages[0]["content"]["type"] == "ai"


@pytest.mark.asyncio
async def test_stream_no_tokens(test_client, mock_agent) -> None:
    """Test streaming without tokens."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify no token messages
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == 0

        # Verify final message
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["content"]["content"] == FINAL_ANSWER
        assert messages[0]["content"]["type"] == "ai"


def test_stream_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    INTERRUPT = "Confirm weather check"
    # Configure mock to use our async iterator function
    events = [
        (
            "updates",
            {"__interrupt__": [Interrupt(value=INTERRUPT)]},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify interrupt message
        assert len(messages) == 1
        assert messages[0]["content"]["content"] == INTERRUPT
        assert messages[0]["content"]["type"] == "ai"


def test_info(test_client, mock_settings) -> None:
    """Test that /info returns the correct service metadata."""

    base_agent = Agent(description="A base agent.", graph_like=None)
    mock_settings.AUTH_SECRET = None
    mock_settings.DEFAULT_MODEL = OpenAIModelName.GPT_5_NANO
    mock_settings.AVAILABLE_MODELS = {OpenAIModelName.GPT_5_NANO, OpenAIModelName.GPT_5_MINI}
    with patch.dict("agents.agents.agents", {"base-agent": base_agent}, clear=True):
        response = test_client.get("/info")
        assert response.status_code == 200
        output = ServiceMetadata.model_validate(response.json())

    assert output.default_agent == "research-assistant"
    assert len(output.agents) == 1
    assert output.agents[0].key == "base-agent"
    assert output.agents[0].description == "A base agent."

    assert output.default_model == OpenAIModelName.GPT_5_NANO
    assert output.models == [OpenAIModelName.GPT_5_MINI, OpenAIModelName.GPT_5_NANO]
