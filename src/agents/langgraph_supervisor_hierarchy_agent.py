from langchain.agents import create_agent

from agents.langgraph_supervisor_agent import (
    SequentialToolCalls,
    add,
    agent_as_tool,
    multiply,
    web_search,
)
from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


def workflow(chosen_model, checkpointer=None):
    math_agent = create_agent(
        model=chosen_model,
        tools=[add, multiply],
        system_prompt="You are a math expert. Always use one tool at a time.",
    )

    research_agent = create_agent(
        model=chosen_model,
        tools=[
            web_search,
            agent_as_tool(math_agent, "math_expert", "Solve math problems with a calculator."),
        ],
        system_prompt=(
            "You are a world class researcher with access to web search. "
            "Do not do any math, you have a math expert for that."
        ),
        middleware=[SequentialToolCalls()],
    )

    return create_agent(
        model=chosen_model,
        tools=[
            agent_as_tool(
                research_agent,
                "research_expert",
                "Research current events with web search, with math capabilities.",
            )
        ],
        system_prompt=(
            "You are a team supervisor managing a research expert with math capabilities. "
            "For current events, use research_expert. "
            "Pass along the user's full request, including any math."
        ),
        middleware=[SequentialToolCalls()],
        checkpointer=checkpointer,
    )


langgraph_supervisor_hierarchy_agent = workflow(model)
