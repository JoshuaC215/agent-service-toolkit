from deepagents import create_deep_agent

from agents.langgraph_supervisor_agent import add, multiply, web_search
from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


def workflow(chosen_model, checkpointer=None, store=None):
    # The inner team is intentionally stateless: each delegation invokes it
    # with a fresh task message (isolated subagents), so it needs no
    # checkpointer of its own. checkpointer/store persist the outer thread,
    # which already contains the full delegation history.
    research_team = create_deep_agent(
        model=chosen_model,
        system_prompt=(
            "You are a world class researcher with access to web search. "
            "Do not do any math, you have a math expert for that."
        ),
        tools=[web_search],
        subagents=[
            {
                "name": "math_expert",
                "description": "Solve math problems with a calculator.",
                "system_prompt": "You are a math expert. Always use one tool at a time.",
                "tools": [add, multiply],
            },
        ],
    )
    return create_deep_agent(
        model=chosen_model,
        system_prompt=(
            "You are a team supervisor managing a research expert with math capabilities. "
            "For current events, delegate to the research_expert subagent."
        ),
        subagents=[
            {
                "name": "research_expert",
                "description": ("Research current events with web search, with math capabilities."),
                "runnable": research_team,
            },
        ],
        checkpointer=checkpointer,
        store=store,
    )


langgraph_supervisor_hierarchy_agent = workflow(model)
