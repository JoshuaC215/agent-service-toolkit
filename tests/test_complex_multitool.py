import pytest
import httpx
import json
from unittest.mock import patch, MagicMock

# Define the instructions to test
instructions = [
    "Find the prime minister of UK, then add 40 to their birth year",
    "Calculate the square root of the current year and add to first mission to the moon",
    "Find the current king of the UK and add 10 to his age",
]

# Define the endpoint URL
endpoint_url = "http://localhost:8080/complex-multitool-agent/stream"

# Mock response data for each instruction
mock_responses = {
    instructions[0]: [
        {
            "type": "message",
            "content": {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "WebSearch",
                        "args": {"query": "current Prime Minister of the UK February 2025"},
                        "id": "call_NouNqWR8AI4YaKxtRsXFc4HC",
                        "type": "tool_call",
                    }
                ],
            },
        },
        {
            "type": "message",
            "content": {
                "type": "tool",
                "content": "snippet: What to know about Keir Starmer, UK's new prime minister ...",
                "tool_calls": [],
            },
        },
        {
            "type": "message",
            "content": {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "Calculator",
                        "args": {"expression": "1962 + 40"},
                        "id": "call_EUhdPfoeI8vp69kNSoJXfPgC",
                        "type": "tool_call",
                    }
                ],
            },
        },
        {
            "type": "message",
            "content": {"type": "tool", "content": "2002", "tool_calls": []},
        },
        {
            "type": "token",
            "content": "The current Prime Minister of the UK is Sir Keir Starmer, who was born in 1962. Adding 40 to his birth year gives us 2002.",
        },
        {"type": "token", "content": "[DONE]"},
    ],
    instructions[1]: [
        {
            "type": "message",
            "content": {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "Calculator",
                        "args": {"expression": "sqrt(2023) + 1969"},
                        "id": "call_Calculator",
                        "type": "tool_call",
                    }
                ],
            },
        },
        {
            "type": "message",
            "content": {"type": "tool", "content": "2014.5", "tool_calls": []},
        },
        {
            "type": "token",
            "content": "The result of the calculation is 2014.5.",
        },
        {"type": "token", "content": "[DONE]"},
    ],
    instructions[2]: [
        {
            "type": "message",
            "content": {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "WebSearch",
                        "args": {"query": "current King of the UK and his age"},
                        "id": "call_WebSearch",
                        "type": "tool_call",
                    }
                ],
            },
        },
        {
            "type": "message",
            "content": {
                "type": "tool",
                "content": "snippet: King Charles III is the current king of the UK, born in 1948.",
                "tool_calls": [],
            },
        },
        {
            "type": "message",
            "content": {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "Calculator",
                        "args": {"expression": "1948 + 10"},
                        "id": "call_Calculator",
                        "type": "tool_call",
                    }
                ],
            },
        },
        {
            "type": "message",
            "content": {"type": "tool", "content": "1958", "tool_calls": []},
        },
        {
            "type": "token",
            "content": "The current King of the UK is King Charles III, born in 1948. Adding 10 to his birth year gives us 1958.",
        },
        {"type": "token", "content": "[DONE]"},
    ],
}

@pytest.fixture
def mock_httpx_stream():
    """Fixture to mock the HTTPX stream response."""
    with patch("httpx.stream") as mock_stream:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_stream.return_value.__enter__.return_value = mock_response
        yield mock_stream

@pytest.mark.parametrize("instruction", instructions)
def test_complex_multitool(instruction, mock_httpx_stream):
    """Test the Complex Multitool Agent with various instructions."""
    # Prepare the request payload
    payload = {"input": instruction, "stream_tokens": True}

    # Set the mock response data based on the instruction
    mock_httpx_stream.return_value.__enter__.return_value.iter_lines.return_value = [
        f"data: {json.dumps(data)}\n" for data in mock_responses[instruction]
    ]

    # Send the request and handle the streaming response
    with httpx.stream("POST", endpoint_url, json=payload) as response:
        assert response.status_code == 200, f"Failed to connect: {response.status_code}"

        # Collect and process the streamed events
        tool_start_detected = False
        tool_end_detected = False
        final_text_received = False

        for line in response.iter_lines():
            if line:
                # Parse the JSON data
                data = json.loads(line.split("data: ")[-1])
                message_type = data.get("type")
                content = data.get("content")

                # Check for tool start/end messages and final text
                if message_type == "message":
                    if content.get("type") == "ai" and content.get("tool_calls"):
                        tool_start_detected = True
                        assert any(call["name"] in ["WebSearch", "Calculator"] for call in content["tool_calls"]), "Expected tool call not detected"
                    elif content.get("type") == "tool":
                        tool_end_detected = True
                        # Adjusted to handle floating-point numbers
                        assert "snippet" in content["content"] or content["content"].replace('.', '', 1).isdigit(), "Tool result not as expected"
                elif message_type == "token":
                    if "[DONE]" in content:
                        final_text_received = True

        # Assert that all expected messages were detected
        assert tool_start_detected, "Tool start message not detected"
        assert tool_end_detected, "Tool end message not detected"
        assert final_text_received, "Final LLM text not received"
