import pytest

from client import AgentClient


@pytest.fixture
def agent_client(mock_env):
    """Fixture for creating a test client with a clean environment."""
    ac = AgentClient(base_url="http://test", get_info=False)
    ac.update_agent("test-agent", verify=False)
    return ac
