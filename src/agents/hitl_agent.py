from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    RunnableConfig,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    status: Annotated[Literal["approve", "reject"], "Status must be either 'approve' or 'reject'"]


def question(state: AgentState, config: RunnableConfig) -> AgentState:
    return {
        "messages": [
            AIMessage(
                content='question Node> Please enter "approve" or "reject."',
            )
        ]
    }


def human_input(state: AgentState, config: RunnableConfig) -> AgentState:
    pass


def show_status(state: AgentState, config: RunnableConfig) -> AgentState:
    status = "reject"

    latest_message = state["messages"][-1]
    if latest_message.content == "approve":
        status = "approve"

    return {
        "messages": [
            AIMessage(
                content=f'show_status Node> status is "{status}"',
            )
        ],
        "status": status,
    }


def finish_message(state: AgentState, config: RunnableConfig) -> AgentState:
    return {
        "messages": [
            AIMessage(
                content="finish_message Node> see you",
            )
        ]
    }


def check_status(state: AgentState, config: RunnableConfig) -> AgentState:
    if state.get("status") == "approve":
        return "approve"

    return "reject"


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("question", question)
agent.add_node("human_input", human_input)
agent.add_node("show_status", show_status)
agent.add_node("finish_message", finish_message)

agent.set_entry_point("question")

agent.add_edge("question", "human_input")
agent.add_edge("human_input", "show_status")
agent.add_conditional_edges(
    "show_status", check_status, {"approve": "finish_message", "reject": "question"}
)
agent.set_finish_point("finish_message")

hitl_agent = agent.compile(checkpointer=MemorySaver(), interrupt_before=["human_input"])
