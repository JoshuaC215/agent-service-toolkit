import pytest

from client import AgentClient


@pytest.fixture
def agent_client(mock_env):
    """Fixture for creating a test client with a clean environment."""
    return AgentClient(base_url="http://test", agent="test-agent")
