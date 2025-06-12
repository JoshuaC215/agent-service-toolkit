import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
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

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)

            # Display user ID (for debugging or user information)
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # Include both thread_id and user_id in the URL for sharing to maintain user identity
            chat_url = (
                f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            )
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        )

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AcmeTech's Employee Handbook.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())
