from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from service import app
from schema import ChatMessage

client = TestClient(app)

@patch("service.service.research_assistant")
def test_invoke(mock_agent):
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    agent_response = {"messages": [AIMessage(content=ANSWER)]}
    mock_agent.ainvoke = AsyncMock(return_value=agent_response)
    
    with client as c:
        response = c.post("/invoke", json={"message": QUESTION})
        assert response.status_code == 200
    
    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION
    
    output = ChatMessage.parse_obj(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

@patch("service.service.LangsmithClient")
def test_feedback(mock_client):
    ls_instance = mock_client.return_value
    ls_instance.create_feedback.return_value = None
    body = {"run_id": "847c6285-8fc9-4560-a83f-4e6285809254", "key": "human-feedback-stars", "score": 0.8}
    response = client.post("/feedback", json=body)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    ls_instance.create_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="human-feedback-stars",
        score=0.8,
    )
