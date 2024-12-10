from unittest.mock import patch

import pytest

from schema import AgentInfo, ServiceMetadata
from schema.models import OpenAIModelName


@pytest.fixture
def mock_agent_client(mock_env):
    """Fixture for creating a mock AgentClient with a clean environment."""

    mock_info = ServiceMetadata(
        default_agent="test-agent",
        agents=[
            AgentInfo(key="test-agent", description="Test agent"),
            AgentInfo(key="chatbot", description="Chatbot"),
        ],
        default_model=OpenAIModelName.GPT_4O,
        models=[OpenAIModelName.GPT_4O, OpenAIModelName.GPT_4O_MINI],
    )

    with patch("client.AgentClient") as mock_agent_client:
        mock_agent_client_instance = mock_agent_client.return_value
        mock_agent_client_instance.info = mock_info
        yield mock_agent_client_instance
