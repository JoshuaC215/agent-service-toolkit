from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agents import DEFAULT_AGENT
from service import app


@pytest.fixture
def test_client():
    """Fixture to create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent that can be configured for different test scenarios."""
    agent_mock = AsyncMock()
    agent_mock.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Test response")]})
    agent_mock.get_state = Mock()  # Default empty mock for get_state
    with patch.dict("service.service.agents", {DEFAULT_AGENT: agent_mock}):
        yield agent_mock


@pytest.fixture
def mock_settings(mock_env):
    """Fixture to ensure settings are clean for each test."""
    with patch("service.service.settings") as mock_settings:
        yield mock_settings
