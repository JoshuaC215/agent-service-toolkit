from collections.abc import Awaitable, Callable
from inspect import signature
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool

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


def agent_as_tool(agent: Any, name: str, description: str) -> BaseTool:
    """Expose a sub-agent to a supervisor as a tool that returns its final answer."""

    # Hide the sub-agent's tokens from the stream; its full messages still stream.
    tagged_agent = agent.with_config(tags=["skip_stream"])

    async def call_agent(request: str) -> str:
        result = await tagged_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        return result["messages"][-1].text

    return StructuredTool.from_function(coroutine=call_agent, name=name, description=description)


class SequentialToolCalls(AgentMiddleware):
    """Delegate to one sub-agent at a time so the UI can nest each sub-agent's messages.

    Only OpenAI and Anthropic models support disabling parallel tool calls.
    """

    def _sequential(self, request: ModelRequest) -> ModelRequest:
        if "parallel_tool_calls" not in signature(request.model.bind_tools).parameters:
            return request
        return request.override(
            model_settings={**request.model_settings, "parallel_tool_calls": False}
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        return handler(self._sequential(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._sequential(request))


math_agent = create_agent(
    model=model,
    tools=[add, multiply],
    system_prompt="You are a math expert. Always use one tool at a time.",
)

research_agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt="You are a world class researcher with access to web search. Do not do any math.",
)

langgraph_supervisor_agent = create_agent(
    model=model,
    tools=[
        agent_as_tool(
            research_agent,
            "research_expert",
            "Research current events with web search. Do not use for math.",
        ),
        agent_as_tool(math_agent, "math_expert", "Solve math problems with a calculator."),
    ],
    system_prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_expert. "
        "For math problems, use math_expert."
    ),
    middleware=[SequentialToolCalls()],
)
