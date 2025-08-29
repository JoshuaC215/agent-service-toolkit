from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from agents.langgraph_supervisor_agent import add, multiply, web_search
from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


def workflow(chosen_model):
    math_agent = create_react_agent(
        model=chosen_model,
        tools=[add, multiply],
        name="sub-agent-math_expert",  # Identify the graph node as a sub-agent
        prompt="You are a math expert. Always use one tool at a time.",
    ).with_config(tags=["skip_stream"])

    research_agent = (
        create_supervisor(
            [math_agent],
            model=chosen_model,
            tools=[web_search],
            prompt="You are a world class researcher with access to web search. Do not do any math, you have a math expert for that. ",
            supervisor_name="supervisor-research_expert",  # Identify the graph node as a supervisor to the math agent
        )
        .compile(
            name="sub-agent-research_expert"
        )  # Identify the graph node as a sub-agent to the main supervisor
        .with_config(tags=["skip_stream"])
    )  # Stream tokens are ignored for sub-agents in the UI

    # Create supervisor workflow
    return create_supervisor(
        [research_agent],
        model=chosen_model,
        prompt=(
            "You are a team supervisor managing a research expert with math capabilities."
            "For current events, use research_agent. "
        ),
        add_handoff_back_messages=True,
        # UI now expects this to be True so we don't have to guess when a handoff back occurs
        output_mode="full_history",  # otherwise when reloading conversations, the sub-agents' messages are not included
    )  # default name for supervisor is "supervisor".


langgraph_supervisor_hierarchy_agent = workflow(model).compile()
