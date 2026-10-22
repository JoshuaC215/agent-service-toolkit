from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agents.order_tools import get_order
from agents.ticket_tools import (
    create_ticket,
    get_ticket,
    list_tickets_by_order,
)
from core import get_model, settings


tools = [
    get_order,
    create_ticket,
    get_ticket,
    list_tickets_by_order,
]


instructions = """
You are an order and after-sales support agent.

Rules:
- Answer in the user's language.
- Use get_order to retrieve real order information.
- Never invent order information or ticket IDs.
- If an order does not exist, clearly tell the user.
- For an explicit after-sales request, obtain the order ID
  and a clear description of the issue.
- If required information is missing, ask the user instead
  of creating an incomplete ticket.
- Verify the order using get_order before creating a ticket.
- Use create_ticket to record the request.
- Only claim ticket creation succeeded if the tool returns
  success=true.
- Report the ticket ID and status exactly as returned by
  the tool.
- Use get_ticket to retrieve an existing ticket.
- Use list_tickets_by_order to retrieve tickets for an order.
- Creating a ticket does not mean a refund is approved.
- Never claim to have issued a refund or changed an order
  status. Those operations require separate authorization
  and workflows.
"""

def wrap_model(model: BaseChatModel) -> RunnableSerializable:
    bound_model = model.bind_tools(tools)

    preprocessor = RunnableLambda(
        lambda state: [
            SystemMessage(content=instructions),
            *state["messages"],
        ]
    )

    return preprocessor | bound_model

async def call_model(
    state: MessagesState,
    config: RunnableConfig,
):
    model_name = config["configurable"].get(
        "model",
        settings.DEFAULT_MODEL,
    )

    model = get_model(model_name)

    model_runnable = wrap_model(model)

    response = await model_runnable.ainvoke(
        state,
        config,
    )

    return {
        "messages": [response]
    }

def pending_tool_calls(
    state: MessagesState,
) -> str:
    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        return "done"

    if last_message.tool_calls:
        return "tools"

    return "done"

agent = StateGraph(MessagesState)

agent.add_node("model", call_model)
agent.add_node("tools", ToolNode(tools))

agent.set_entry_point("model")

agent.add_conditional_edges(
    "model",
    pending_tool_calls,
    {
        "tools": "tools",
        "done": END,
    },
)

agent.add_edge(
    "tools",
    "model",
)

order_agent = agent.compile()