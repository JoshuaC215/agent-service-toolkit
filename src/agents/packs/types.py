from dataclasses import dataclass
from typing import Literal

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

AgentGraph = CompiledStateGraph | Pregel
Track = Literal["core", "product", "experimental"]
Stability = Literal["stable", "beta", "experimental", "deprecated"]


@dataclass(frozen=True)
class AgentSpec:
    key: str
    description: str
    graph: AgentGraph
    track: Track = "core"
    stability: Stability = "stable"
    pack: str = "core"
