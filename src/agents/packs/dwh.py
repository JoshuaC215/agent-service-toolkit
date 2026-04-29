from agents.dwh_readiness_summary import dwh_readiness_summary
from agents.packs.types import AgentSpec


def get_dwh_agent_specs() -> list[AgentSpec]:
    return [
        AgentSpec(
            key="dwh_readiness_summary",
            description="Generates summary and recommendations for DWH Readiness Check.",
            graph=dwh_readiness_summary,
            track="product",
            stability="stable",
            pack="dwh",
        )
    ]
