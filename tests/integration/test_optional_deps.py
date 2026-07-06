import pytest

from client import AgentClient


@pytest.mark.docker
def test_mongo_checkpointer_persists_history():
    """Confirm the Mongo-backed checkpointer actually persists conversation state.

    Requires the service container running with DATABASE_TYPE=mongo against a live
    MongoDB (see docker/compose.optional.yaml) and USE_FAKE_MODEL=true.
    """
    client = AgentClient("http://0.0.0.0", agent="chatbot")
    thread_id = "smoke-test-mongo-thread"

    client.invoke("Tell me a joke?", thread_id=thread_id, model="fake")
    client.invoke("Tell me another?", thread_id=thread_id, model="fake")

    history = client.get_history(thread_id=thread_id)
    human_messages = [m for m in history.messages if m.type == "human"]
    assert len(human_messages) == 2
    assert human_messages[0].content == "Tell me a joke?"
    assert human_messages[1].content == "Tell me another?"
