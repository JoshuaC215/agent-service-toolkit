from typing import TYPE_CHECKING

__all__ = ["get_agent", "get_all_agent_info", "DEFAULT_AGENT", "AgentGraph", "agents"]

# Avoid importing heavy submodules (that may require external LLM env) at package import time.
# Expose names lazily via __getattr__ so "import agents.dwh_readiness_summary" does not
# pull in other agents.
if TYPE_CHECKING:
    # Only for type checkers; does not run at runtime.
    from agents.agents import DEFAULT_AGENT, AgentGraph, agents, get_agent, get_all_agent_info  # noqa: F401


def __getattr__(name: str):
    if name in {"get_agent", "get_all_agent_info", "DEFAULT_AGENT", "AgentGraph", "agents"}:
        from agents.agents import DEFAULT_AGENT, AgentGraph, agents, get_agent, get_all_agent_info

        mapping = {
            "get_agent": get_agent,
            "get_all_agent_info": get_all_agent_info,
            "DEFAULT_AGENT": DEFAULT_AGENT,
            "AgentGraph": AgentGraph,
            "agents": agents,
        }
        return mapping[name]
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
