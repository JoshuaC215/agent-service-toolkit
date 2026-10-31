import asyncio
from types import SimpleNamespace

import pytest

from service.service import _SSE_PING, _agent_stream_with_keepalive


@pytest.mark.asyncio
async def test_keepalive_emits_ping_during_silent_agent() -> None:
    event = ("messages", ("answer", {"tags": []}))

    async def astream(**kwargs):
        await asyncio.sleep(0.02)
        yield event

    agent = SimpleNamespace(astream=astream)
    stream = _agent_stream_with_keepalive(agent, {}, interval=0.005)

    assert await anext(stream) == _SSE_PING
    for _ in range(10):
        next_event = await asyncio.wait_for(anext(stream), timeout=1)
        if next_event == event:
            break
    else:
        pytest.fail("agent event was not yielded")
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_keepalive_cancels_agent_when_client_disconnects() -> None:
    cancelled = asyncio.Event()

    async def astream(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        if False:
            yield None

    agent = SimpleNamespace(astream=astream)
    stream = _agent_stream_with_keepalive(agent, {}, interval=0.005)

    assert await anext(stream) == _SSE_PING
    await stream.aclose()

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_keepalive_propagates_agent_errors() -> None:
    async def astream(**kwargs):
        raise RuntimeError("agent failed")
        if False:
            yield None

    agent = SimpleNamespace(astream=astream)
    stream = _agent_stream_with_keepalive(agent, {}, interval=0.005)

    with pytest.raises(RuntimeError, match="agent failed"):
        await anext(stream)
