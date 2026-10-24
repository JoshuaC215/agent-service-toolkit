"""Search tools for the research assistant."""

import asyncio
from typing import Annotated, Any

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langgraph.prebuilt import InjectedState

from core import settings

PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
PROJECT_USER_AGENT = "agent-service-toolkit (https://github.com/JoshuaC215/agent-service-toolkit)"


async def _call_parallel(name: str, arguments: dict[str, Any]) -> Any:
    try:
        async with asyncio.timeout(settings.SEARCH_TIMEOUT_SECONDS):
            client = MultiServerMCPClient(
                {
                    "parallel": StreamableHttpConnection(
                        transport="streamable_http",
                        url=PARALLEL_MCP_URL,
                        headers={"User-Agent": PROJECT_USER_AGENT},
                    )
                }
            )
            tools = await client.get_tools()
            remote_tool = next(t for t in tools if t.name == name)
            remote_tool.handle_tool_error = False
            return await remote_tool.ainvoke(arguments)
    except TimeoutError as exc:
        raise ToolException("Parallel search timed out. Please try again.") from exc
    except Exception as exc:
        raise ToolException(f"Parallel search failed: {exc}") from exc


@tool("WebSearch")
async def parallel_search(
    objective: str,
    search_queries: list[str],
    session_id: Annotated[str, InjectedState("search_session_id")],
) -> Any:
    """Search the web for a focused objective with related keyword queries."""
    return await _call_parallel(
        "web_search",
        {"objective": objective, "search_queries": search_queries, "session_id": session_id},
    )


@tool("WebFetch")
async def parallel_fetch(
    urls: list[str],
    session_id: Annotated[str, InjectedState("search_session_id")],
    objective: str | None = None,
    search_queries: list[str] | None = None,
) -> Any:
    """Fetch excerpts from up to 20 HTTP(S) URLs, optionally focused on an objective."""
    return await _call_parallel(
        "web_fetch",
        {
            "urls": urls,
            "objective": objective,
            "search_queries": search_queries,
            "session_id": session_id,
        },
    )


parallel_search.handle_tool_error = True
parallel_fetch.handle_tool_error = True


def get_search_tools(config: RunnableConfig) -> list[BaseTool]:
    provider = settings.SEARCH_PROVIDER
    if provider != "none":
        provider = config.get("configurable", {}).get("search_provider", provider)
    match provider:
        case "duckduckgo":
            return [DuckDuckGoSearchResults(name="WebSearch")]
        case "parallel":
            return [parallel_search, parallel_fetch]
        case "none":
            return []
        case _:
            raise ValueError(f"Unknown search provider: {provider}")
