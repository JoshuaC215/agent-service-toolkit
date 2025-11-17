# from langgraph_supervisor import create_supervisor --> not used in v1
from langchain.agents import create_agent
from langchain_core.tools import tool

from agents.langgraph_supervisor_agent import add, multiply, web_search

# IMPORTANT: Import only the base tools needed for nested agents
# Keep them in the function scope, not module scope, to prevent tool leakage
from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


def workflow(chosen_model):
    math_agent = create_agent(
        model=chosen_model,
        tools=[add, multiply],
        name="sub-agent-math_expert",  # Identify the graph node as a sub-agent
        system_prompt="You are a math expert. Always use one tool at a time.",
    ).with_config(tags=["skip_stream"])

    @tool
    def delegate_to_math_expert(request: str) -> str:
        """Use this for any math operations like addition, multiplication, or calculations.

        Input: Natural language math request (e.g., 'multiply 5 and 10')
        """

        result = math_agent.invoke({"messages": [{"role": "user", "content": request}]})
        # Extract the text content from the last message
        last_message = result["messages"][-1]
        return last_message.content if hasattr(last_message, "content") else str(last_message)

    research_supervisor = create_agent(
        model=chosen_model,
        tools=[web_search, delegate_to_math_expert],
        system_prompt=(
            "You are a world class researcher with access to web search. "
            "When you need to do math, use delegate_to_math_expert. "
            "For research and information lookup, use web_search directly."
        ),
    )

    @tool
    def delegate_to_research_expert(request: str) -> str:
        """Use this for research tasks, information lookup, and math operations.

        The research expert has access to web search and can delegate to a math expert.

        Input: Natural language request (e.g., 'find FAANG headcounts and calculate the average')
        """
        result = research_supervisor.invoke({"messages": [{"role": "user", "content": request}]})
        # Extract the text content from the last message
        last_message = result["messages"][-1]
        return last_message.content if hasattr(last_message, "content") else str(last_message)

    # Create supervisor workflow
    main_supervisor = create_agent(
        model=chosen_model,
        tools=[delegate_to_research_expert],
        system_prompt=(
            "You are a team supervisor managing a research expert with math capabilities. "
            "For current events, research, or any tasks requiring information lookup, "
            "use delegate_to_research_expert."
        ),
    )
    return main_supervisor


# Note: When used in service, a checkpointer must be attached for state operations
langgraph_supervisor_hierarchy_agent = workflow(model)
