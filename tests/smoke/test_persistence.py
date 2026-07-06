import pytest

from client import AgentClient


@pytest.mark.docker
def test_checkpointer_persists_history():
    """Confirm the configured checkpointer persists conversation state across turns.

    Backend-agnostic: exercises whichever DATABASE_TYPE the service was started
    with. scripts/smoke_test.sh runs this against both postgres and mongo.
    Requires a running service (USE_FAKE_MODEL=true) backed by a live database.

    Uses the default agent (rather than pinning one) because /history always reads
    state through DEFAULT_AGENT regardless of which agent a thread was invoked with.
    """
    client = AgentClient("http://localhost:8080")
    thread_id = "smoke-test-persistence-thread"

    client.invoke("Tell me a joke?", thread_id=thread_id, model="fake")
    client.invoke("Tell me another?", thread_id=thread_id, model="fake")

    history = client.get_history(thread_id=thread_id)
    human_messages = [m for m in history.messages if m.type == "human"]
    assert len(human_messages) == 2
    assert human_messages[0].content == "Tell me a joke?"
    assert human_messages[1].content == "Tell me another?"
