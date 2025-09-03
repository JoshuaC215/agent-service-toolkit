from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="sub-agent-math_expert",
    prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="sub-agent-research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math.",
).with_config(tags=["skip_stream"])

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    ),
    add_handoff_back_messages=True,
    # UI now expects this to be True so we don't have to guess when a handoff back occurs
    output_mode="full_history",  # otherwise when reloading conversations, the sub-agents' messages are not included
)

langgraph_supervisor_agent = workflow.compile()
