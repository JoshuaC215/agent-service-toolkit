from ddgs import DDGS
from langchain.agents import create_agent
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
    cleaned_query = query.strip()
    if not cleaned_query:
        return "No query provided."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(cleaned_query, max_results=5))
    except Exception as exc:
        return f"Web search failed: {exc}"

    if not results:
        return "No web results found."

    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title") or "Untitled"
        url = item.get("href") or ""
        snippet = (item.get("body") or "").strip()
        if snippet:
            lines.append(f"{idx}. {title}\nURL: {url}\nSnippet: {snippet}")
        else:
            lines.append(f"{idx}. {title}\nURL: {url}")

    return "\n\n".join(lines)


math_agent = create_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    system_prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])

research_agent = create_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    system_prompt="You are a world class researcher with access to web search. Do not do any math.",
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
    add_handoff_back_messages=False,
)

langgraph_supervisor_agent = workflow.compile()
