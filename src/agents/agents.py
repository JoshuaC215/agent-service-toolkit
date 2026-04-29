from dataclasses import dataclass

from agents.packs import load_agent_specs
from agents.packs.types import AgentGraph
from schema import AgentInfo

DEFAULT_AGENT = "research-assistant"


@dataclass
class Agent:
    description: str
    graph: AgentGraph
    track: str = "core"
    stability: str = "stable"
    pack: str = "core"


def _build_agent_registry() -> dict[str, Agent]:
    registry: dict[str, Agent] = {}
    for spec in load_agent_specs():
        registry[spec.key] = Agent(
            description=spec.description,
            graph=spec.graph,
            track=spec.track,
            stability=spec.stability,
            pack=spec.pack,
        )
    return registry


agents: dict[str, Agent] = _build_agent_registry()


def get_agent(agent_id: str) -> AgentGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(
            key=agent_id,
            description=agent.description,
            track=agent.track,
            stability=agent.stability,
            pack=agent.pack,
        )
        for agent_id, agent in agents.items()
    ]
