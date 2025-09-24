"""Tests for agent loading functionality."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents.agents import agents, get_agent, load_agent
from agents.lazy_agent import LazyLoadingAgent


class TestAgentLoading:
    """Test agent loading functionality."""

    @pytest.mark.asyncio
    async def test_load_agent_static_agent(self):
        """Test loading a static agent (no-op)."""
        # Static agents don't need loading
        await load_agent("chatbot")
        # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_load_agent_lazy_agent(self):
        """Test loading a lazy agent."""
        # Mock the GitHub MCP agent
        mock_agent = Mock(spec=LazyLoadingAgent)
        mock_agent.load = AsyncMock()

        with patch.dict(agents, {"test-lazy-agent": Mock(graph_like=mock_agent)}):
            await load_agent("test-lazy-agent")

        mock_agent.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_agent_nonexistent(self):
        """Test loading a non-existent agent."""
        with pytest.raises(KeyError):
            await load_agent("nonexistent-agent")

    def test_get_agent_static_agent(self):
        """Test getting a static agent."""
        agent = get_agent("chatbot")
        assert agent is not None

    def test_get_agent_lazy_agent_not_loaded(self):
        """Test getting a lazy agent that hasn't been loaded."""
        mock_agent = Mock(spec=LazyLoadingAgent)
        mock_agent._loaded = False

        with patch.dict(agents, {"test-lazy-agent": Mock(graph_like=mock_agent)}):
            with pytest.raises(
                RuntimeError, match="Agent test-lazy-agent not loaded. Call load\\(\\) first."
            ):
                get_agent("test-lazy-agent")

    def test_get_agent_lazy_agent_loaded(self):
        """Test getting a lazy agent that has been loaded."""
        mock_agent = Mock(spec=LazyLoadingAgent)
        mock_agent._loaded = True
        mock_graph = Mock()
        mock_agent.get_graph.return_value = mock_graph

        with patch.dict(agents, {"test-lazy-agent": Mock(graph_like=mock_agent)}):
            result = get_agent("test-lazy-agent")

        assert result == mock_graph
        mock_agent.get_graph.assert_called_once()

    def test_get_agent_nonexistent(self):
        """Test getting a non-existent agent."""
        with pytest.raises(KeyError):
            get_agent("nonexistent-agent")
