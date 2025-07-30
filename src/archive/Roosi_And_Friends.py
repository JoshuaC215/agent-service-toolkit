import asyncio

import streamlit as st

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "roosi & friends 2025"
APP_ICON = "src/static/roosi_logo.png"
HIDE_SIDEBAR = True


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
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
        """
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

    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    st.markdown(
        """
        <style>
        /* Make all Streamlit buttons full width */
        .stLinkButton > a {
            width: 100%;
            margin-bottom: -5px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("Willkommen beim roosi and friends 2025 Assistenten")
    st.write(
        "Vielen Dank für Ihr Interesse am roosi and friends Assistenten. Die Veranstaltung liegt bereits in der Vergangenheit. Weitere Informationen zu roosi AIOS sowie einige Eindrücke der Veranstaltung finden sie hier:"
    )

    st.link_button("Informationen zu roosi AIOS", "https://news.roo.si/roosi-aios")
    st.link_button(
        "Bilder der Veranstaltung", "https://flickr.com/photos/roosigmbh/albums/72177720326420202/"
    )


#     auth = Auth()
#     if not auth.is_logged_in():
#         # keep as idea
#         #user = auth.login('demo@roosi.ai', 'XXX') #TODO!
#         user = {}
#         user['name'] = "roosi and Friends"
#         # messetuerk: roosiandfriends@roo.si auf demo.roosi.ai
#         if user:
#             st.session_state["owui-token"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImRhYjQyNjk4LTViNDctNGJkOC1iYWRmLTM3OGI2YTM4ZWYzMSJ9.HtoJftZ1bf4cvfcYRXYacyzwGMBxSd7BWOQ1rGmb72Y" #TODO!
#             st.session_state["user"] = user
#             st.rerun()

#     if "agent_client" not in st.session_state:
#         load_dotenv()
#         backend_url = os.getenv("BACKEND_URL")
#         try:
#             with st.spinner(f"Service wird verbunden..."):
#                 api_key = None
#                 if "owui-token" in st.session_state:
#                     api_key = st.session_state["owui-token"]
#                 st.session_state.agent_client = AgentClient(base_url=backend_url, api_key=api_key)
#         except AgentClientError as e:
#             st.error(f"Error connecting to agent service: {e}")
#             st.markdown("The service might be booting up. Try again in a few seconds.")
#             st.stop()
#     agent_client: AgentClient = st.session_state.agent_client

#     if "thread_id" not in st.session_state:
#         thread_id = st.query_params.get("thread_id")
#         if not thread_id:
#             thread_id = get_script_run_ctx().session_id
#             messages = []
#         else:
#             try:
#                 messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
#             except AgentClientError:
#                 st.error("No message history found for this Thread ID.")
#                 messages = []
#         st.session_state.messages = messages
#         st.session_state.thread_id = thread_id

#     # Config options
#     model = "owui/roosi--friends-2025"
#     use_streaming = False
#     agent_client.agent = 'chatbot'

#     # Draw existing messages
#     messages: list[ChatMessage] = st.session_state.messages

#     if len(messages) == 0:
#         WELCOME = "Hallo! Ich bin der Assistent zum roosi & friends 2025. Stelle mir gerne Fragen zur Veranstaltung, oder nutze die folgenden Vorschläge:"
#         with st.chat_message("ai", avatar="src/static/roosi_logo.png"):
#             st.write(WELCOME)

#     # draw_messages() expects an async iterator over messages
#     async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
#         for m in messages:
#             yield m

#     await draw_messages(amessage_iter())

#     st.markdown("""
#         <style>
#         /* Make all Streamlit buttons full width */
#         .stButton > button {
#             width: 100%;
#             margin-bottom: -5px;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     prompt_suggestions = [
#         "Was sind die Highlights der Veranstaltung?",
#         "Kannst du mir mehr über die Fachexperten erzählen?",
#         "Was gibt es zu essen?",
#     ]

#     for element in prompt_suggestions:
#         if st.button(element, key=element):
#             await send_message(element, messages, agent_client, use_streaming, model)


#     # Generate new message if the user provided new input
#     if user_input := st.chat_input(placeholder="Deine Nachricht"):
#         await send_message(user_input, messages, agent_client, use_streaming, model)

# async def send_message(user_input, messages, agent_client, use_streaming, model):
#     messages.append(ChatMessage(type="human", content=user_input))
#     st.chat_message("human", avatar="src/static/user.png").write(user_input)
#     try:
#         if use_streaming:
#             stream = agent_client.astream(
#                 message=user_input,
#                 model=model,
#                 thread_id=st.session_state.thread_id,
#             )
#             await draw_messages(stream, is_new=True)
#         else:
#             with st.spinner("roosi denkt nach..."):
#                 response = await agent_client.ainvoke(
#                     message=user_input,
#                     model=model,
#                     thread_id=st.session_state.thread_id,
#                 )
#             messages.append(response)
#             st.chat_message("ai", avatar="src/static/roosi_logo.png").write(response.content)
#         st.rerun()  # Clear stale containers
#     except AgentClientError as e:
#         st.error("Ohje, leider ist etwas schiefgelaufen. Bitte versuche es erneut.")
#         st.stop()

# async def draw_messages(
#     messages_agen: AsyncGenerator[ChatMessage | str, None],
#     is_new: bool = False,
# ) -> None:
#     """
#     Draws a set of chat messages - either replaying existing messages
#     or streaming new ones.

#     This function has additional logic to handle streaming tokens and tool calls.
#     - Use a placeholder container to render streaming tokens as they arrive.
#     - Use a status container to render tool calls. Track the tool inputs and outputs
#       and update the status container accordingly.

#     The function also needs to track the last message container in session state
#     since later messages can draw to the same container. This is also used for
#     drawing the feedback widget in the latest chat message.

#     Args:
#         messages_aiter: An async iterator over messages to draw.
#         is_new: Whether the messages are new or not.
#     """

#     # Keep track of the last message container
#     last_message_type = None
#     st.session_state.last_message = None

#     # Placeholder for intermediate streaming tokens
#     streaming_content = ""
#     streaming_placeholder = None


#     # Iterate over the messages and draw them
#     while msg := await anext(messages_agen, None):
#         # str message represents an intermediate token being streamed
#         if isinstance(msg, str):
#             # If placeholder is empty, this is the first token of a new message
#             # being streamed. We need to do setup.
#             if not streaming_placeholder:
#                 if last_message_type != "ai":
#                     last_message_type = "ai"
#                     st.session_state.last_message = st.chat_message("ai", avatar="src/static/roosi_logo.png")
#                 with st.session_state.last_message:
#                     streaming_placeholder = st.empty()

#             streaming_content += msg
#             streaming_placeholder.write(streaming_content)
#             continue
#         if not isinstance(msg, ChatMessage):
#             st.error(f"Unexpected message type: {type(msg)}")
#             st.write(msg)
#             st.stop()
#         match msg.type:
#             # A message from the user, the easiest case
#             case "human":
#                 last_message_type = "human"
#                 st.chat_message("human", avatar="src/static/user.png").write(msg.content)

#             # A message from the agent is the most complex case, since we need to
#             # handle streaming tokens and tool calls.
#             case "ai":
#                 # If we're rendering new messages, store the message in session state
#                 if is_new:
#                     st.session_state.messages.append(msg)

#                 # If the last message type was not AI, create a new chat message
#                 if last_message_type != "ai":
#                     last_message_type = "ai"
#                     st.session_state.last_message = st.chat_message("ai", avatar="src/static/roosi_logo.png")

#                 with st.session_state.last_message:
#                     # If the message has content, write it out.
#                     # Reset the streaming variables to prepare for the next message.
#                     if msg.content:
#                         if streaming_placeholder:
#                             streaming_placeholder.write(msg.content)
#                             streaming_content = ""
#                             streaming_placeholder = None
#                         else:
#                             st.write(msg.content)

#                     if msg.tool_calls:
#                         # Create a status container for each tool call and store the
#                         # status container by ID to ensure results are mapped to the
#                         # correct status container.
#                         call_results = {}
#                         for tool_call in msg.tool_calls:
#                             status = st.status(
#                                 f"""Tool Call: {tool_call["name"]}""",
#                                 state="running" if is_new else "complete",
#                             )
#                             call_results[tool_call["id"]] = status
#                             status.write("Input:")
#                             status.write(tool_call["args"])

#                         # Expect one ToolMessage for each tool call.
#                         for _ in range(len(call_results)):
#                             tool_result: ChatMessage = await anext(messages_agen)
#                             if tool_result.type != "tool":
#                                 st.error(f"Unexpected ChatMessage type: {tool_result.type}")
#                                 st.write(tool_result)
#                                 st.stop()

#                             # Record the message if it's new, and update the correct
#                             # status container with the result
#                             if is_new:
#                                 st.session_state.messages.append(tool_result)
#                             status = call_results[tool_result.tool_call_id]
#                             status.write("Output:")
#                             status.write(tool_result.content)
#                             status.update(state="complete")

#             case "custom":
#                 # CustomData example used by the bg-task-agent
#                 # See:
#                 # - src/agents/utils.py CustomData
#                 # - src/agents/bg_task_agent/task.py
#                 try:
#                     task_data: TaskData = TaskData.model_validate(msg.custom_data)
#                 except ValidationError:
#                     st.error("Ohje, leider ist etwas schiefgelaufen. Bitte versuche es erneut.")
#                     st.stop()

#                 if is_new:
#                     st.session_state.messages.append(msg)

#                 if last_message_type != "task":
#                     last_message_type = "task"
#                     st.session_state.last_message = st.chat_message(
#                         name="task", avatar=":material/manufacturing:"
#                     )
#                     with st.session_state.last_message:
#                         status = TaskDataStatus()

#                 status.add_and_draw_task_data(task_data)

#             # In case of an unexpected message type, log an error and stop
#             case _:
#                 st.error("Ohje, leider ist etwas schiefgelaufen. Bitte versuche es erneut.")
#                 st.stop()


if __name__ == "__main__":
    asyncio.run(main())
