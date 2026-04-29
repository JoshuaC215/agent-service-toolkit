from agents.packs.core import get_core_agent_specs
from agents.packs.dwh import get_dwh_agent_specs
from agents.packs.skill import get_skill_agent_specs
from agents.packs.types import AgentSpec


def load_agent_specs() -> list[AgentSpec]:
    specs: list[AgentSpec] = []
    specs.extend(get_core_agent_specs())
    specs.extend(get_skill_agent_specs())
    specs.extend(get_dwh_agent_specs())

    seen: set[str] = set()
    for spec in specs:
        if spec.key in seen:
            raise ValueError(f"Duplicate agent key registered: {spec.key}")
        seen.add(spec.key)

    return specs
