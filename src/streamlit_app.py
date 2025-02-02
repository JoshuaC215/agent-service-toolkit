import asyncio
import os
import urllib.parse
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.agents.chatbot import chatbot  # Agora usamos o chatbot atualizado
from schema import ChatHistory, ChatMessage

APP_TITLE = "SaphyrAI - Direito do Consumidor"
APP_ICON = "⚖️"


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    st.sidebar.title(f"{APP_ICON} {APP_TITLE}")
    st.sidebar.write("Chatbot especializado no Código de Defesa do Consumidor.")

    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        WELCOME = "Bem-vindo ao SaphyrAI! Pergunte sobre seus direitos como consumidor."
        with st.chat_message("ai"):
            st.write(WELCOME)

    # Exibir mensagens anteriores
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Capturar a entrada do usuário e enviar para o chatbot
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)

        response = await chatbot.ainvoke({"messages": messages})

        messages.append(response["messages"][-1])
        st.chat_message("ai").write(response["messages"][-1].content)

        st.rerun()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """Desenha as mensagens no chat do Streamlit."""

    last_message_type = None
    st.session_state.last_message = None

    streaming_content = ""
    streaming_placeholder = None

    while msg := await anext(messages_agen, None):
        if isinstance(msg, str):
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
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)
            case "ai":
                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)


if __name__ == "__main__":
    asyncio.run(main())
