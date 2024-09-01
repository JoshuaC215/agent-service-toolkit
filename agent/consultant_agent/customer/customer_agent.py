from datetime import datetime
from typing import Annotated, TypedDict, Literal, Optional

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field

from ..fetch_information import fetch_project_info
from ..utils import create_tool_node_with_fallback
from template.constant import (
    CHATBOT_PROMPT,
    SUMMARY_AND_UPDATE_PROMPT,
    POP_DIALOG_STATE_TEMPLATE_ZH,
    TSOUA_ZH, TSOUA_REQUEST_ZH,
    COE_ZH, COE_E1_CANCEL, COE_E1_REASON_ZH, COE_E2_CANCEL, COE_E2_REASON_ZH, COE_E3_CANCEL, COE_E3_REASON_ZH,
    CREATE_ENTRY_SUMMARY_NODE_TEMPLATE_ZH
)

from .tools.project_manager import update_project, cancel_project

def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class State(TypedDict):
    messages: Annotated[list, add_messages]
    project_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "chatbot",          # 提出问题
                "project_manager",  # 记录问题
            ]
        ],
        update_dialog_stack,
    ]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "给出一个真实的输出结果。")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

class ToSummaryOrUpdateAssistant(BaseModel):
    TSOUA_ZH

    request: str = Field(
        description=TSOUA_REQUEST_ZH,
    )

class CompleteOrEscalate(BaseModel):
    COE_ZH

    cancel: bool = True
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "cancel": COE_E1_CANCEL,
                "reason": COE_E1_REASON_ZH,
            },
            "example 2": {
                "cancel": COE_E2_CANCEL,
                "reason": COE_E2_REASON_ZH,
            },
            "example 3": {
                "cancel": COE_E3_CANCEL,
                "reason": COE_E3_REASON_ZH,
            },
        }

model_GPT4 = ChatOpenAI(
    model="gpt-4-turbo-2024-04-09",
    openai_api_base="https://aihubmix.com/v1",
    max_tokens=1024
)

chatbot_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            CHATBOT_PROMPT,
        ),
        ("placeholder", "{messages}")
    ]
).partial(time=datetime.now())
chatbot_tools = [TavilySearchResults(max_results=2)]
chatbot_runnable = chatbot_prompt | model_GPT4.bind_tools(chatbot_tools + [ToSummaryOrUpdateAssistant])

update_project_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SUMMARY_AND_UPDATE_PROMPT,
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

update_project_safe_tools = [update_project, cancel_project]
# update_project_sensitive_tools = []
# update_project_tools = update_project_safe_tools + update_project_sensitive_tools
update_project_tools = update_project_safe_tools
update_project_runnable = update_project_prompt | model_GPT4.bind_tools(
    update_project_tools + [CompleteOrEscalate]
)

builder = StateGraph(State)

def project_info(state: State):
    info = fetch_project_info()
    return {"project_info": info}

builder.add_node("fetch_project_info", project_info)
builder.set_entry_point("fetch_project_info")
builder.add_node("chatbot", Assistant(chatbot_runnable))
builder.add_edge("fetch_project_info", "chatbot")
builder.add_node("chatbot_tools", create_tool_node_with_fallback(chatbot_tools))

def route_chatbot_assistant(
    state: State,
) -> Literal[
    "chatbot_tools",
    "enter_summary",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToSummaryOrUpdateAssistant.__name__:
            return "enter_summary"
        return "chatbot_tools"
    raise ValueError("Invalid route")

builder.add_conditional_edges(
    "chatbot",
    route_chatbot_assistant,
    {
        "chatbot_tools": "chatbot_tools",
        "enter_summary": "enter_summary",
        END: END,
    },
)
builder.add_edge("chatbot_tools", "chatbot")

def enter_summary(state: State) -> dict:
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    return {
        "messages": [
            ToolMessage(
                content=CREATE_ENTRY_SUMMARY_NODE_TEMPLATE_ZH,
                tool_call_id=tool_call_id,
            )
        ],
        "dialog_state": "project_manager",
    }

builder.add_node("enter_summary", enter_summary)
builder.add_node("project_manager", Assistant(update_project_runnable))
builder.add_edge("enter_summary", "project_manager")
builder.add_node("update_project_safe_tools", create_tool_node_with_fallback(update_project_safe_tools))

def route_project_manager(
    state: State,
) -> Literal[
    "update_project_safe_tools",
    "leave_skill",
]:
    # route = tools_condition(state)
    # if route == END:
    #     return END
    # tool_calls = state["messages"][-1].tool_calls
    # did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    # if did_cancel:
    #     return "leave_skill"
    # safe_toolnames = [t.name for t in update_project_safe_tools]
    # if all(tc["name"] in safe_toolnames for tc in tool_calls):
    #     return "update_project_safe_tools"
    # raise ValueError("Invalid route")
    route = tools_condition(state)
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in update_project_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "update_project_safe_tools"
    raise ValueError("Invalid route")

builder.add_edge("update_project_safe_tools", "project_manager")
builder.add_conditional_edges("project_manager", route_project_manager)

def pop_dialog_state(state: State) -> dict:
    messages = []
    if state["messages"][-1].tool_calls:
        messages.append(
            ToolMessage(
                content=POP_DIALOG_STATE_TEMPLATE_ZH,
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }

builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "chatbot")

# builder.add_node("summary", Assistant(summary_runnable))
# builder.add_edge("summary", END)

memory = SqliteSaver.from_conn_string(":memory:")
customer_graph = builder.compile(checkpointer=memory)

customer_graph.get_graph().draw_mermaid_png(output_file_path='./media/customer_graph.png')


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        