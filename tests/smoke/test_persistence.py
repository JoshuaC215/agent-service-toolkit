import os

import pytest

from client import AgentClient

# Shared with scripts/smoke_test.sh via the environment so the script can verify
# this exact thread's checkpoints landed in the intended backend. Falls back to a
# fixed id when the test is run on its own.
THREAD_ID = os.environ.get("SMOKE_THREAD_ID", "smoke-test-persistence-thread")


@pytest.mark.docker
def test_checkpointer_persists_history():
    """Confirm the configured checkpointer persists conversation state across turns.

    Backend-agnostic: exercises whichever DATABASE_TYPE the service was started
    with. scripts/smoke_test.sh runs this against both postgres and mongo, then
    separately verifies the data actually landed in that backend (this test alone
    can't tell the backends apart, since any working checkpointer would pass).
    Requires a running service (USE_FAKE_MODEL=true) backed by a live database.

    Uses the default agent for both invoke and get_history. Since /history is
    agent-aware and AgentClient scopes it to the client's selected agent, the same
    graph that created the thread also reads it back — which is what makes the
    round-trip work now that each agent is a distinct graph with its own state.
    """
    client = AgentClient("http://localhost:8080")

    client.invoke("Tell me a joke?", thread_id=THREAD_ID, model="fake")
    client.invoke("Tell me another?", thread_id=THREAD_ID, model="fake")

    history = client.get_history(thread_id=THREAD_ID)
    human_messages = [m for m in history.messages if m.type == "human"]
    assert len(human_messages) == 2
    assert human_messages[0].content == "Tell me a joke?"
    assert human_messages[1].content == "Tell me another?"
