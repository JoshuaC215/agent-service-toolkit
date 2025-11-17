"""GitHub MCP Agent - An agent that uses GitHub MCP tools for repository management."""

import logging
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langgraph.graph.state import CompiledStateGraph

from agents.lazy_agent import LazyLoadingAgent
from core import get_model, settings

logger = logging.getLogger(__name__)

current_date = datetime.now().strftime("%B %d, %Y")
prompt = f"""
You are GitHubBot, a specialized assistant for GitHub repository management and development workflows.
You have access to GitHub MCP tools that allow you to interact with GitHub repositories, issues, pull requests,
and other GitHub resources. Today's date is {current_date}.

Your capabilities include:
- Repository management (create, clone, browse)
- Issue management (create, list, update, close)
- Pull request management (create, review, merge)
- Branch management (create, switch, merge)
- File operations (read, write, search)
- Commit operations (create, view history)

Guidelines:
- Always be helpful and provide clear explanations of GitHub operations
- When creating or modifying content, ensure it follows best practices
- Be cautious with destructive operations (deletes, force pushes, etc.)
- Provide context about what you're doing and why
- Use appropriate commit messages and PR descriptions
- Respect repository permissions and access controls

NOTE: You have access to GitHub MCP tools that provide direct GitHub API access.
"""


class GitHubMCPAgent(LazyLoadingAgent):
    """GitHub MCP Agent with async initialization."""

    def __init__(self) -> None:
        super().__init__()
        self._mcp_tools: list[BaseTool] = []
        self._mcp_client: MultiServerMCPClient | None = None

    async def load(self) -> None:
        """Initialize the GitHub MCP agent by loading MCP tools."""
        if not settings.GITHUB_PAT:
            logger.info("GITHUB_PAT is not set, GitHub MCP agent will have no tools")
            self._mcp_tools = []
            self._graph = self._create_graph()
            self._loaded = True
            return

        try:
            # Initialize MCP client directly
            github_pat = settings.GITHUB_PAT.get_secret_value()
            connections = {
                "github": StreamableHttpConnection(
                    transport="streamable_http",
                    url=settings.MCP_GITHUB_SERVER_URL,
                    headers={
                        "Authorization": f"Bearer {github_pat}",
                    },
                )
            }

            self._mcp_client = MultiServerMCPClient(connections)
            logger.info("MCP client initialized successfully")

            # Get tools from the client
            self._mcp_tools = await self._mcp_client.get_tools()
            logger.info(f"GitHub MCP agent initialized with {len(self._mcp_tools)} tools")

        except Exception as e:
            logger.error(f"Failed to initialize GitHub MCP agent: {e}")
            self._mcp_tools = []
            self._mcp_client = None

        # Create and store the graph
        self._graph = self._create_graph()
        self._loaded = True

    def _create_graph(self) -> CompiledStateGraph:
        """Create the GitHub MCP agent graph."""
        model = get_model(settings.DEFAULT_MODEL)

        return create_agent(
            model=model,
            tools=self._mcp_tools,
            name="github-mcp-agent",
            system_prompt=prompt,
        )


# Create the agent instance
github_mcp_agent = GitHubMCPAgent()
