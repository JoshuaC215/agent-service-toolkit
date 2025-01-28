"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import os
from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from composio_langgraph import Action, ComposioToolSet, App

from .configuration import Configuration
import pdb

composio_toolset = ComposioToolSet(
    auth = {
        'apiKey': os.getenv("COMPOSIO_API_KEY"),
        'entityId': "default"
    }
)

apps = [
    App.GMAIL,
    App.GOOGLECALENDAR,
    App.LINEAR,
    App.TWITTER,
    App.WEBFLOW,
    App.NOTION,
    App.HUBSPOT,
    App.GOOGLE_ANALYTICS,
    App.LINKEDIN,
    App.CANVA,
    App.SLACK,
    App.REDDIT
]
# for extracting the tools and flattening the list afterwards
tools = [composio_toolset.get_tools(apps=[app]) for app in apps]
combined_tools = [tool for sublist in tools for tool in sublist]
# pdb.set_trace()

async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


# Combine the search tool with Composio tools using spread operator
TOOLS: List[Callable[..., Any]] = [search, *combined_tools]
