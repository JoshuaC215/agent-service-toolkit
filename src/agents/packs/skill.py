from agents.packs.types import AgentSpec
from agents.skillcompanion import skillcompanion
from agents.skillcompanion_interrupted import skillcompanion_interrupted


def get_skill_agent_specs() -> list[AgentSpec]:
    return [
        AgentSpec(
            key="skill-companion",
            description="An assistant to check your skills with AI.",
            graph=skillcompanion,
            track="product",
            stability="stable",
            pack="skill",
        ),
        AgentSpec(
            key="skillcompanion_interrupted",
            description="A Skill Companion agent with interrupt capability.",
            graph=skillcompanion_interrupted,
            track="product",
            stability="stable",
            pack="skill",
        ),
    ]
