from deepagents import create_deep_agent

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


langgraph_supervisor_agent = create_deep_agent(
    model=model,
    system_prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, delegate to the research_expert subagent. "
        "For math problems, delegate to the math_expert subagent."
    ),
    subagents=[
        {
            "name": "research_expert",
            "description": (
                "Research current events with web search. "
                "Use for factual and current-events questions. Do not do any math."
            ),
            "system_prompt": ("You are a world class researcher with access to web search."),
            "tools": [web_search],
        },
        {
            "name": "math_expert",
            "description": "Solve math problems with a calculator.",
            "system_prompt": "You are a math expert. Always use one tool at a time.",
            "tools": [add, multiply],
        },
    ],
)
