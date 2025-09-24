"""Tests for the LazyLoadingAgent base class."""

from unittest.mock import Mock

import pytest

from agents.lazy_agent import LazyLoadingAgent


class TestLazyLoadingAgent(LazyLoadingAgent):
    """Test implementation of LazyLoadingAgent."""

    def __init__(self):
        super().__init__()

    async def load(self) -> None:
        """Test load implementation."""
        self._loaded = True

    def _create_graph(self):
        """Test graph creation."""
        mock_graph = Mock()
        mock_graph.name = "test-graph"
        return mock_graph


class TestLazyLoadingAgentBase:
    """Test the LazyLoadingAgent base class functionality."""

    def test_initialization(self):
        """Test that agent initializes correctly."""
        agent = TestLazyLoadingAgent()
        assert not agent._loaded
        assert agent._graph is None

    @pytest.mark.asyncio
    async def test_load(self):
        """Test that load works correctly."""
        agent = TestLazyLoadingAgent()
        await agent.load()
        assert agent._loaded

    def test_get_graph_before_load(self):
        """Test that get_graph raises error before load."""
        agent = TestLazyLoadingAgent()
        with pytest.raises(RuntimeError, match="Agent not loaded"):
            agent.get_graph()

    def test_get_graph_after_load(self):
        """Test that get_graph works after load."""
        agent = TestLazyLoadingAgent()
        agent._loaded = True
        agent._graph = Mock()

        graph = agent.get_graph()
        assert graph is not None

    def test_get_graph_no_graph_created(self):
        """Test that get_graph raises error if no graph was created."""
        agent = TestLazyLoadingAgent()
        agent._loaded = True
        agent._graph = None

        with pytest.raises(RuntimeError, match="Agent graph not created"):
            agent.get_graph()
