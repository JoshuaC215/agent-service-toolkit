"""This module dynamically loads tools based on the entity ID."""

from typing import Any, Callable, List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from composio_langgraph import Action, ComposioToolSet, App

from .configuration import Configuration

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
    App.REDDIT,
    App.GITHUB
]

async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results."""
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return result

def get_tools(entity_id: str) -> List[Callable[..., Any]]:
    """Retrieve tools configured for a specific entity."""
    composio_toolset = ComposioToolSet()
    combined_tools = []

    for app in apps:
        combined_tools.extend(composio_toolset.get_tools(entity_id=entity_id, apps=[app]))
    return [search, *combined_tools]
