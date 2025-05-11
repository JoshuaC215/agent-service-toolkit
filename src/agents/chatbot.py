from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore

from core import get_model, settings


@entrypoint(checkpointer=MemorySaver(), store=InMemoryStore())
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str, list[BaseMessage]],
    config: RunnableConfig,
):
    messages = inputs["messages"]
    if previous:
        messages = previous["messages"] + messages

    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await model.ainvoke(messages)
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    )
