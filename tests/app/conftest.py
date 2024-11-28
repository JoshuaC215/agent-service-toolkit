from unittest.mock import patch

import pytest


@pytest.fixture
def mock_agent_client(mock_env):
    """Fixture for creating a mock AgentClient with a clean environment."""

    with patch("client.AgentClient") as mock_agent_client:
        mock_agent_client_instance = mock_agent_client.return_value
        yield mock_agent_client_instance
