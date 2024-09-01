from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# 定义一个函数，用于打印事件信息
def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")   # 获取对话状态
    if current_state:
        print(f"Currently in: ", current_state[-1])     # 打印当前状态
    message = event.get("messages")     # 获取消息
    if message:
        if isinstance(message, list):
            message = message[-1]   # 如果消息是列表，取最后一个消息
        if message.id not in _printed:  # 如果消息ID未被打印过
            msg_repr = message.pretty_repr(html=True)   # 获取消息的HTML表示
            if len(msg_repr) > max_length:  # 如果消息太长
                msg_repr = msg_repr[:max_length] + " ... (truncated)"   # 截断消息
            print(msg_repr) # 打印消息
            _printed.add(message.id)    # 将消息ID添加到已打印集合中

