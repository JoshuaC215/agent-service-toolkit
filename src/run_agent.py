import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

load_dotenv()

from agents import DEFAULT_AGENT, get_agent  # noqa: E402

agent = get_agent(DEFAULT_AGENT)


async def main() -> None:
    thread_config = RunnableConfig(configurable={"thread_id": uuid4()})

    # get user input from console
    user_input = input(f"({agent.name}) Please enter your message: ")
    initial_input = {"messages": [("user", user_input)]}
    current_input = initial_input

    while True:
        try:
            # invoke the agent
            result = await agent.ainvoke(input=current_input, config=thread_config)
            # Get current state to check for interrupts
            state = agent.get_state(thread_config)
            # Create array of interrupted tasks
            interrupted_tasks = [
                task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
            ]
            if not interrupted_tasks:
                result["messages"][-1].pretty_print()
                break

            # we have interrupts, but print any other messages
            if len(result["messages"]) > 0:
                result["messages"][-1].pretty_print()

            # print value of first interrupted task
            print("================================ Interrupt =================================")
            print(f"Task: {interrupted_tasks[0].name}")
            print(f"Value:\n{interrupted_tasks[0].interrupts[0].value}")

            # get user input from console
            user_input = input(f"({agent.name}) Please enter your message: ")
            current_input = Command(resume=user_input)

        except Exception as e:
            print(f"An error occurred: {e}")

    # Draw the agent graph as png
    # requires:
    # brew install graphviz
    # export CFLAGS="-I $(brew --prefix graphviz)/include"
    # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    # pip install pygraphviz
    #
    # agent.get_graph().draw_png("agent_diagram.png")


if __name__ == "__main__":
    asyncio.run(main())
