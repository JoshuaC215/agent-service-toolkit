import json
from unittest.mock import patch

import pytest
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langgraph.checkpoint.memory import MemorySaver

from schema import ChatMessage, StreamInput


class FakeToolModel(FakeMessagesListChatModel):
    """A fake model that supports tool calls."""

    def __init__(self, responses: list[BaseMessage]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        return self


@pytest.mark.asyncio
async def test_three_layer_supervisor_hierarchy_agent_with_fake_model():
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="transfer_to_sub-agent-research_expert", args={}, id="call-1")
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[ToolCall(name="transfer_to_sub-agent-math_expert", args={}, id="call-2")],
        ),
        AIMessage(
            content="", tool_calls=[ToolCall(name="add", args={"a": 2, "b": 3}, id="call-3")]
        ),
        AIMessage(content="2+3 is 5"),  # This is the response from the math expert,
        AIMessage(
            content="The Maths Expert says the answer is 5."
        ),  # This is the response from the research expert
        AIMessage(content="The result is 5."),
    ]

    from agents.langgraph_supervisor_hierarchy_agent import workflow

    agent = workflow(FakeToolModel(responses)).compile(checkpointer=MemorySaver())

    with patch("service.service.get_agent", return_value=agent):
        from service.service import message_generator

        messages = []
        async for chunk in message_generator(
            StreamInput(message="Add 2 and 3"), agent_id="langgraph-supervisor-hierarchy-agent"
        ):
            if chunk and chunk.strip() != "data: [DONE]":  # Skip [DONE] message
                chat_message = json.loads(chunk.lstrip("data: "))["content"]
                messages.append(ChatMessage.model_validate(chat_message))

        for msg in messages:
            print(msg)

        assert messages[0].tool_calls[0]["name"] == "transfer_to_sub-agent-research_expert"
        assert messages[1].content == "Successfully transferred to sub-agent-research_expert"
        assert messages[2].tool_calls[0]["name"] == "transfer_to_sub-agent-math_expert"
        assert messages[3].content == "Successfully transferred to sub-agent-math_expert"
        assert messages[4].tool_calls[0]["name"] == "add"
        assert messages[5].content == "5.0"
        assert messages[6].content == "2+3 is 5"
        assert messages[7].tool_calls[0]["name"] == "transfer_back_to_supervisor-research_expert"
        assert messages[8].content == "Successfully transferred back to supervisor-research_expert"
        assert messages[9].content == "The Maths Expert says the answer is 5."
        assert messages[10].tool_calls[0]["name"] == "transfer_back_to_supervisor"
        assert messages[11].content == "Successfully transferred back to supervisor"
        assert messages[12].content == "The result is 5."
