import asyncio
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import cast

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit.delta_generator import DeltaGenerator

from agents.utils import current_date_str
from client import AgentClient, AgentClientError
from client.auth import Auth
from schema import ChatMessage, VariantIdentifier
from schema.task_data import TaskData, TaskDataStatus
from streamlit_utils.evaluation_ui import _render_evaluation_content
from variants.variant_config import VariantConfig

logger = logging.getLogger(__name__)


# Helper: determine if current user has admin privileges
def _is_admin_user(user: dict) -> bool:
    """
    Return True if the given user dict indicates admin privileges.

    Avoids broad exception handling by validating types defensively.
    """
    if not isinstance(user, dict):
        return False

    # Direct role string
    role = user.get("role")
    if isinstance(role, str) and role.strip().lower() == "admin":
        return True

    # Common boolean/flag indicators
    for k in ("is_admin", "isAdmin", "admin"):
        v = user.get(k)
        if isinstance(v, bool) and v:
            return True
        if isinstance(v, int) and v == 1:
            return True
        if isinstance(v, str) and v.strip().lower() in {"1", "true", "yes", "y", "on", "admin"}:
            return True

    # Roles collection
    roles = user.get("roles")
    if isinstance(roles, (list, tuple, set)):  # noqa: UP038
        for r in roles:
            if isinstance(r, str) and r.strip().lower() == "admin":
                return True
    elif isinstance(roles, str):
        parts = [p.strip() for p in roles.replace(";", ",").split(",") if p.strip()]
        if any(p.lower() == "admin" for p in parts):
            return True

    # Permissions/scopes/authorities collections
    for key in ("permissions", "scopes", "authorities"):
        seq = user.get(key)
        if isinstance(seq, (list, tuple, set)):  # noqa: UP038
            for p in seq:
                if isinstance(p, str) and p.strip().lower() == "admin":
                    return True
        elif isinstance(seq, str):
            parts = [p.strip() for p in seq.replace(";", ",").split(",") if p.strip()]
            if any(p.lower() == "admin" for p in parts):
                return True

    return False


def _parse_boolish(value) -> bool:
    """
    Convert common truthy/falsey representations into a boolean.
    Accepts str, int, float, bool, list/tuple (uses first element).
    Never raises; unknown or None -> False.
    """
    # Resolve first element of list/tuple
    if isinstance(value, (list, tuple)):  # noqa: UP038
        value = value[0] if value else None

    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):  # noqa: UP038
        return value == 1 or value == 1.0

    if isinstance(value, str):
        v = value.strip().lower()
        return v in {"1", "true", "t", "yes", "y", "on"}

    return False


# Evaluation UI helpers are now imported from streamlit_utils.evaluation_ui
# See: from streamlit_utils.evaluation_ui import _render_evaluation_content

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


HIDE_SIDEBAR = True


# TODO: maybe hide input when check finished?
async def main(config: VariantConfig) -> None:
    st.set_page_config(
        page_title=config.get("title", "roosi SkillCompanion"),
        page_icon=config.get("app_icon", "🤖"),
        # collapse sidebar on default
        initial_sidebar_state="collapsed" if HIDE_SIDEBAR else "auto",
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

    # hide sidebar
    if HIDE_SIDEBAR:
        st.html(
            """
        <style>
            [data-testid="stSidebarCollapsedControl"] {
                display: none
            }
        </style>
        """
        )
    if "show_chat_input" not in st.session_state:
        st.session_state.show_chat_input = False
    if "show_slider" not in st.session_state:
        st.session_state.show_slider = True
    if "show_start_button" not in st.session_state:
        st.session_state.show_start_button = True
    if "show_title" not in st.session_state:
        st.session_state.show_title = True
    if "skill_progress" not in st.session_state:
        st.session_state.skill_progress = 0
    if st.session_state.show_title:
        st.title(config.get("title", "roosi SkillCompanion"))
    if "url_parameters" not in st.session_state:
        st.session_state.url_parameters = st.query_params.to_dict()
    # URL param control for Evaluation button visibility:
    # - Parameter: show_evaluation=true|false (default: false)
    #   Examples:
    #   - ?show_evaluation=false  -> hide Evaluation button initially (default)
    #   - ?show_evaluation=true   -> show Evaluation button initially
    if "show_eval_button" not in st.session_state:
        _raw_show_eval = st.session_state.url_parameters.get("show_evaluation")
        st.session_state.show_eval_button = _parse_boolish(_raw_show_eval)

    # Initialize AI avatar once from config and reuse everywhere
    if "ai_avatar" not in st.session_state:
        st.session_state.ai_avatar = (
            f"src/static/{config.get('ai_message_avatar_filename', 'roosi_logo.png')}"
        )
    # Evaluation dialog/session defaults and trigger button (always visible)
    if "evaluation_dialog_open" not in st.session_state:
        st.session_state.evaluation_dialog_open = False
    if "eval_from_date" not in st.session_state:
        st.session_state.eval_from_date = None
    if "eval_to_date" not in st.session_state:
        st.session_state.eval_to_date = None
    if "eval_running" not in st.session_state:
        st.session_state.eval_running = False
    if "eval_image_path" not in st.session_state:
        st.session_state.eval_image_path = None
    if "eval_error" not in st.session_state:
        st.session_state.eval_error = None
    # Evaluation requires explicit login inside the modal
    if "eval_require_login" not in st.session_state:
        st.session_state.eval_require_login = False
    # Initialize time inputs defaults (UTC)
    from datetime import time as _time

    if "eval_from_time" not in st.session_state:
        st.session_state.eval_from_time = _time(hour=0, minute=0)
    if "eval_to_time" not in st.session_state:
        st.session_state.eval_to_time = _time(hour=23, minute=59, second=59)

    # Persistent "Evaluation" button (hide after starting Skill-Companion)
    # Hide also when chat input is active
    if (
        st.session_state.get("show_start_button", True)
        and not st.session_state.get("show_chat_input", False)
        and st.session_state.get("show_eval_button", False)
    ):
        col_eval_a, col_eval_b = st.columns([4, 1])
        with col_eval_b:
            if st.button("Evaluation", key="open_evaluation"):
                st.session_state.evaluation_dialog_open = True
                # Force login flow inside modal before showing date/time
                st.session_state.eval_require_login = True
                # clear previous results/errors
                st.session_state.eval_image_path = None
                st.session_state.eval_error = None

    # Modal evaluation dialog (true modal overlay if Streamlit supports st.dialog)
    # Guard: if Evaluation button is disabled via URL parameter, make sure modal cannot be open
    if not st.session_state.get("show_eval_button", False) and st.session_state.get(
        "evaluation_dialog_open", False
    ):
        st.session_state.evaluation_dialog_open = False
    if st.session_state.evaluation_dialog_open:
        _dialog_callable = getattr(st, "dialog", None)
        # Ensure login/admin is required before showing evaluation UI
        if "owui-token" not in st.session_state or (
            "user" in st.session_state and not _is_admin_user(st.session_state.get("user", {}))
        ):
            st.session_state.eval_require_login = True
        if callable(_dialog_callable):

            @st.dialog("Evaluation")
            def _eval_dialog():
                _render_evaluation_content("dialog", is_admin_user=_is_admin_user)

            _eval_dialog()
        else:
            with st.container():  # fallback for older Streamlit versions
                _render_evaluation_content("fallback", is_admin_user=_is_admin_user)
    if not st.session_state.show_title:
        skill_bar = st.progress(
            st.session_state.skill_progress, text="Fortschritt im Skill-Companion"
        )

    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()
    if "question_limit" not in st.session_state:
        st.session_state.question_limit = config.get("question_limit", 10)

    auth = Auth(default_login=True)
    if not auth.is_logged_in():
        return
    # Ensure per-session run_id exists (hotfix for non-dialog login path)
    st.session_state.setdefault("run_id", str(uuid.uuid4()))

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("BACKEND_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")  # nosec B104: intentional bind on all interfaces for containerized/remote access
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                api_key = None
                if "owui-token" in st.session_state:
                    api_key = st.session_state["owui-token"]
                st.session_state.agent_client = AgentClient(
                    base_url=agent_url,
                    api_key=str(api_key or ""),
                    variant=VariantIdentifier(
                        streamlit_app_name="Skill_Companion", variant=os.getenv("VARIANT", None)
                    ),
                )
        except AgentClientError as e:
            st.error(f"Error connecting to agent service: {e}")
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
                messages = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    model = config.get("model", "owui/gpt-4o")
    if agent_client.info is None or model not in agent_client.info.models:
        st.error("Selected Model not found.")
        st.stop()

    agent_client.agent = config.get("agent", "skillcompanion_interrupted")
    if agent_client.info is None or agent_client.agent not in [
        a.key for a in agent_client.info.agents
    ]:
        st.error("Selected Agent not found.")
        st.stop()

    use_streaming = False

    # Draw existing messages
    messages = st.session_state.messages

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    if st.session_state.show_slider:
        tech_affinity = st.select_slider(
            "Wie technisch affin ist Ihre Firma (1=nicht technisch bzw. 5= sehr technisch affin)?",
            options=[1, 2, 3, 4, 5],
            value=3,
        )
        company_size = st.select_slider(
            "Wie viele Mitarbeiter hat ihre Firma?",
            options=["1-10", "11-50", "51-250", "251-1000", ">1000"],
            value="51-250",
        )
        st.session_state["company_size"] = company_size
        date_of_skill_check = current_date_str()
    user = st.session_state["user"]
    run_id = st.session_state["run_id"]
    if st.session_state.show_start_button:
        if st.button("Mit dem Skill-Companion beginnen"):
            # Ensure Evaluation UI is fully closed and its trigger is hidden
            st.session_state.evaluation_dialog_open = False
            st.session_state.eval_require_login = False
            st.session_state.eval_image_path = None
            st.session_state.eval_error = None

            toggle_chat_input()
            try:
                response = await agent_client.ainvoke(
                    message=f"Companysize: {company_size}, Datum: {date_of_skill_check} und technische Affinität: {tech_affinity}",
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user["id"],
                    run_id=run_id,
                    url_parameters=st.session_state["url_parameters"],
                )
                messages.append(response)
                st.chat_message(
                    "ai",
                    avatar=st.session_state.get("ai_avatar", "src/static/roosi_logo.png"),
                ).write(response.content)
            except Exception:
                logger.exception("Unhandled exception during initial ainvoke")
            st.rerun()

    # Generate new message if the user provided new input
    if st.session_state.show_chat_input and not st.session_state.show_slider:
        if user_input := st.chat_input(placeholder="Ihre Antwort..."):
            # Hide slider after first user input
            st.session_state.show_slider = False
            messages.append(ChatMessage(type="human", content=user_input))
            st.chat_message("human", avatar="src/static/user.png").write(user_input)

            try:
                if use_streaming:
                    stream = agent_client.astream(
                        message=user_input,
                        model=model,
                        thread_id=st.session_state.thread_id,
                        user_id=user["id"],
                        run_id=run_id,
                        url_parameters=st.session_state["url_parameters"],
                    )
                    await draw_messages(stream, is_new=True)
                else:
                    response = await agent_client.ainvoke(
                        message=user_input,
                        model=model,
                        thread_id=st.session_state.thread_id,
                        user_id=user["id"],
                        run_id=run_id,
                        url_parameters=st.session_state["url_parameters"],
                    )
                    messages.append(response)
                    st.chat_message(
                        "ai",
                        avatar=st.session_state.get("ai_avatar", "src/static/roosi_logo.png"),
                    ).write(response.content)
                    st.session_state.skill_progress = min(
                        st.session_state.skill_progress
                        + 1 / float(st.session_state.question_limit),
                        1.0,
                    )
                    skill_bar.progress(st.session_state.skill_progress)
                st.rerun()  # Clear stale containers
            except AgentClientError as e:
                st.error(f"Error generating response: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {type(e).__name__}: {e}")
                st.stop()


# TODO Refactoring draw_messages wird in mehreren Dateien öfter verwendet, wahrscheinlich wegen Copy and Paste => Prüfen ob nötig und sonst refactorn
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
                    st.session_state.last_message = st.chat_message(
                        "ai",
                        avatar=st.session_state.get("ai_avatar", "src/static/roosi_logo.png"),
                    )
                if st.session_state.last_message is not None:
                    last_container = cast(DeltaGenerator, st.session_state.last_message)
                    with last_container:
                        streaming_placeholder = st.empty()

            streaming_content += msg
            if streaming_placeholder is not None:
                ph = cast(DeltaGenerator, streaming_placeholder)
                ph.write(streaming_content)
            else:
                if st.session_state.last_message is not None:
                    last_container = cast(DeltaGenerator, st.session_state.last_message)
                    with last_container:
                        streaming_placeholder = st.empty()
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
                st.chat_message("human", avatar="src/static/user.png").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state‚
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message(
                        "ai",
                        avatar=st.session_state.get("ai_avatar", "src/static/roosi_logo.png"),
                    )

                if st.session_state.last_message is not None:
                    last_container = cast(DeltaGenerator, st.session_state.last_message)
                    with last_container:
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
                            tool_result_any = await anext(messages_agen)  # type: ignore[arg-type]
                            if not isinstance(tool_result_any, ChatMessage):
                                st.error(f"Unexpected message type: {type(tool_result_any)}")
                                st.write(tool_result_any)
                                st.stop()
                            tool_result = tool_result_any
                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
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
                    last_container = cast(DeltaGenerator, st.session_state.last_message)
                    with last_container:
                        st.session_state.task_status = TaskDataStatus()

                st.session_state.task_status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


def toggle_chat_input():
    st.session_state.show_chat_input = True
    st.session_state.show_slider = False
    st.session_state.show_start_button = False
    st.session_state.show_title = False


if __name__ == "__main__":
    config = VariantConfig(
        VariantIdentifier(
            streamlit_app_name="Skill_Companion", variant=os.getenv("VARIANT", "default")
        )
    )
    asyncio.run(main(config))
