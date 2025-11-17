# TODO: Update back to using langgraph-supervisor once it is compatible with langgraph 1.0
# Ref: https://github.com/langchain-ai/langgraph-supervisor-py/issues/242
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

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


math_agent: Any = create_agent(
    model=model,
    tools=[add, multiply],
    name="sub-agent-math_expert",
    system_prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])

research_agent: Any = create_agent(
    model=model,
    tools=[web_search],
    name="sub-agent-research_expert",
    system_prompt="You are a world class researcher with access to web search. Do not do any math.",
).with_config(tags=["skip_stream"])


@tool
def delegate_to_math_expert(request: str) -> str:
    """Use this for any math operations like addition, multiplication, or calculations.

    Input: Natural language math request (e.g., 'add 5 and 10')
    """
    result = math_agent.invoke({"messages": [{"role": "user", "content": request}]})
    last_message = result["messages"][-1]
    return last_message.content if hasattr(last_message, "content") else str(last_message)


@tool
def delegate_to_research_expert(request: str) -> str:
    """Use this for research tasks and information lookup.

    Input: Natural language research request (e.g., 'find information about companies')
    """
    result = research_agent.invoke({"messages": [{"role": "user", "content": request}]})
    last_message = result["messages"][-1]
    return last_message.content if hasattr(last_message, "content") else str(last_message)


# Create supervisor workflow
supervisor_agent: Any = create_agent(
    model=model,
    tools=[delegate_to_math_expert, delegate_to_research_expert],
    system_prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events and information lookup, use delegate_to_research_expert. "
        "For math problems and calculations, use delegate_to_math_expert. "
        "Break down user requests into appropriate tool calls and coordinate the results."
    ),
)

langgraph_supervisor_agent = supervisor_agent
