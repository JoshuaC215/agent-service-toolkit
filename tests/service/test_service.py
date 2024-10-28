from unittest.mock import AsyncMock, Mock, patch

import langsmith
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.pregel.types import StateSnapshot

from agents import DEFAULT_AGENT
from schema import ChatHistory, ChatMessage
from service import app

test_client = TestClient(app)


def test_invoke() -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    agent_response = {"messages": [AIMessage(content=ANSWER)]}
    agent_mock = AsyncMock()
    agent_mock.ainvoke = AsyncMock(return_value=agent_response)

    with patch.dict("service.service.agents", {DEFAULT_AGENT: agent_mock}):
        with test_client as c:
            response = c.post("/invoke", json={"message": QUESTION})
            assert response.status_code == 200

    agent_mock.ainvoke.assert_awaited_once()
    input_message = agent_mock.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


@patch("service.service.LangsmithClient")
def test_feedback(mock_client: langsmith.Client) -> None:
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


def test_history() -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    user_question = HumanMessage(content=QUESTION)
    agent_response = AIMessage(content=ANSWER)
    agent_mock = AsyncMock()
    agent_mock.get_state = Mock(
        return_value=StateSnapshot(
            values={"messages": [user_question, agent_response]},
            next=(),
            config={},
            metadata=None,
            created_at=None,
            parent_config=None,
            tasks=(),
        )
    )

    with patch.dict("service.service.agents", {DEFAULT_AGENT: agent_mock}):
        with test_client as c:
            response = c.post(
                "/history", json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"}
            )
            assert response.status_code == 200

    output = ChatHistory.model_validate(response.json())
    assert output.messages[0].type == "human"
    assert output.messages[0].content == QUESTION
    assert output.messages[1].type == "ai"
    assert output.messages[1].content == ANSWER
