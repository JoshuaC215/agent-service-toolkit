import asyncio
import os
from typing import AsyncGenerator, List

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from client import AgentClient
from schema import ChatMessage


# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"

@st.cache_resource
def get_agent_client():
    agent_url = os.getenv("AGENT_URL", "http://localhost")
    return AgentClient(agent_url)


async def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    models = {
        "OpenAI GPT-4o-mini (streaming)": "gpt-4o-mini",
        "llama-3.1-70b on Groq": "llama-3.1-70b",
    }
    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        with st.popover(":material/settings: Settings", use_container_width=True):
            m = st.radio("LLM to use", options=models.keys())
            model = models[m]
            use_streaming = st.toggle("Stream results", value=True)

        @st.dialog("Architecture")
        def architecture_dialog():
            st.image("https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true")
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption("App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)")

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write("Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only.")

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption("Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland")

    # Draw existing messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: List[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. I may take a few seconds to boot up when you send your first message. Ask me anything!"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter():
        for m in messages: yield m
    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if input := st.chat_input():
        messages.append(ChatMessage(type="human", content=input))
        st.chat_message("human").write(input)
        agent_client = get_agent_client()
        if use_streaming:
            stream = agent_client.astream(
                message=input,
                model=model,
                thread_id=get_script_run_ctx().session_id,
            )
            await draw_messages(stream, is_new=True)
        else:
            response = await agent_client.ainvoke(
                message=input,
                model=model,
                thread_id=get_script_run_ctx().session_id,
            )
            messages.append(response)
            st.chat_message("ai").write(response.content)
        st.rerun() # Clear stale containers

    # If messages have been generated, show feedback widget
    if len(messages) > 0:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
        messages_agen: AsyncGenerator[ChatMessage | str, None],
        is_new=False,
    ):
    """
    绘制一组聊天消息，可以是重放现有消息或流式传输新消息。

    此函数具有处理流式令牌和工具调用的附加逻辑。
    - 使用占位符容器来渲染实时流式令牌。
    - 使用状态容器来渲染工具调用。跟踪工具输入和输出，并相应更新状态容器。

    此函数还需要在会话状态中跟踪最后一条消息容器，因为后续消息可以绘制到同一容器中。
    这也用于在最新聊天消息中绘制反馈小部件。

    参数:
        messages_aiter: 一个异步迭代器，提供要绘制的消息。
        is_new: 消息是否为新消息。
    """

    # 跟踪最后一条消息类型
    last_message_type = None
    st.session_state.last_message = None    # 初始化最后一条消息

    # 用于中间流式令牌的占位符
    streaming_content = ""
    streaming_placeholder = None    # 流式占位符

    # 迭代消息并绘制它们
    while msg := await anext(messages_agen, None):
        # str 消息表示正在流式传输的中间令牌
        if isinstance(msg, str):
            # 如果占位符为空，这是新消息的第一个令牌
            # 需要进行初始化。
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")   # 创建 AI 消息容器
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()  # 创建占位符

            streaming_content += msg    # 添加流式内容
            streaming_placeholder.write(streaming_content)  # 更新占位符内容
            continue
        # 检查消息类型是否为 ChatMessage
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")   # 处理意外消息类型
            st.write(msg)
            st.stop()
        match msg.type:     # 根据消息类型进行匹配
            # 来自用户的消息，最简单的情况
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)     # 绘制人类消息

            # 来自代理的消息是最复杂的情况，需要处理流式令牌和工具调用
            case "ai":
                # 如果是新消息，将消息存储在会话状态中
                if is_new:
                    st.session_state.messages.append(msg)

                # 如果最后一条消息类型不是 AI，创建新的聊天消息
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # 如果消息有内容，写出内容。
                    # 重置流式变量以准备下一条消息。
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)    # 更新内容
                            streaming_content = ""  # 重置流式内容
                            streaming_placeholder = None    # 清除占位符
                        else:
                            st.write(msg.content)   # 如果没有占位符，直接写内容

                    if msg.tool_calls:
                        # 检查是否有工具调用
                        # 为每个工具调用创建状态容器，并按 ID 存储状态容器，以确保结果映射到正确的状态容器。
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                    f"""Tool Call: {tool_call["name"]}""",
                                    state="running" if is_new else "complete",
                                )
                            call_results[tool_call["id"]] = status  # 存储状态容器
                            status.write("Input:")  # 写入工具调用输入
                            status.write(tool_call["args"])

                        # 对于每个工具调用，期望一个 ToolMessage
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)   # 获取工具结果消息
                            if not tool_result.type == "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")    # 处理意外消息类型
                                st.write(tool_result)
                                st.stop()

                            # 如果是新消息，记录消息，并更新正确的状态容器以显示结果
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            status = call_results[tool_result.tool_call_id]     # 获取对应状态容器
                            status.write("Output:")     # 写入输出
                            status.write(tool_result.content)   # 更新输出内容
                            status.update(state="complete")     # 更新状态为完成

            # 处理意外消息类型，记录错误并停止
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback():
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback and (latest_run_id, feedback) != st.session_state.last_feedback:

        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client = get_agent_client()
        await agent_client.acreate_feedback(
            run_id=latest_run_id,
            key="human-feedback-stars",
            score=normalized_score,
            kwargs=dict(
                comment="In-line human feedback",
            ),
        )
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())
