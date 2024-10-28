from langgraph.graph.state import CompiledStateGraph

from agents.chatbot import chatbot
from agents.research_assistant import research_assistant

DEFAULT_AGENT = "research-assistant"


agents: dict[str, CompiledStateGraph] = {
    "chatbot": chatbot,
    "research-assistant": research_assistant,
}


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4

    from dotenv import load_dotenv
    from langchain_core.runnables import RunnableConfig

    load_dotenv()

    agent = agents[DEFAULT_AGENT]

    async def main() -> None:
        inputs = {"messages": [("user", "Find me a recipe for chocolate chip cookies")]}
        result = await agent.ainvoke(
            inputs,
            config=RunnableConfig(configurable={"thread_id": uuid4()}),
        )
        result["messages"][-1].pretty_print()

        # Draw the agent graph as png
        # requires:
        # brew install graphviz
        # export CFLAGS="-I $(brew --prefix graphviz)/include"
        # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
        # pip install pygraphviz
        #
        # agent.get_graph().draw_png("agent_diagram.png")

    asyncio.run(main())
