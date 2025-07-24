import asyncio
import os
import json

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient, AgentClientError
from schema import ChatHistory

from streamlit_extras.stylable_container import stylable_container 

from client.auth import Auth
import io
from pydub import AudioSegment
from audiorecorder import audiorecorder



APP_TITLE = "Meeting Mind"
APP_ICON = "🧠"


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

    auth = Auth()

    if not auth.is_logged_in():
        return

    st.title(APP_TITLE)


    container_main = st.container()
    container_single_result = st.container()

    # Load the audio file from the local file manager
    uploaded_file = container_main.file_uploader("Datei hochladen", type=["wav", "m4a", "mp3"])
    transcribed_file = None

    if "agent_client" not in st.session_state:
        load_dotenv()
        backend_url = os.getenv("BACKEND_URL")
        try:
            with st.spinner(f"Connecting to agent service..."):
                api_key = None
                if "owui-token" in st.session_state:
                    api_key = st.session_state["owui-token"]
                st.session_state.agent_client = AgentClient(base_url=backend_url, api_key=api_key)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client
    agent_client.agent = "meeting-mind"

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Upload oder Aufnahme: Audio erfassen, mergen und transkribieren
    st.markdown("### Oder direkt im Browser aufnehmen")
    recorded_audio = audiorecorder("Aufnahme starten", "Aufnahme stoppen", "Pause", key="meeting_recorder")
    audio_segments = []
    # Hochgeladene Datei verarbeiten
    if uploaded_file is not None:
        ext = uploaded_file.name.split('.')[-1]
        audio_segments.append(AudioSegment.from_file(io.BytesIO(uploaded_file.getvalue()), format=ext))
    # Aufgenommene Datei verarbeiten
    if recorded_audio and len(recorded_audio) > 0:
        audio_segments.append(recorded_audio)
    if not audio_segments:
        st.info("Bitte eine Datei hochladen oder eine Aufnahme starten.")
        return
    # Alle Segmente zusammenführen
    merged = audio_segments[0]
    for seg in audio_segments[1:]:
        merged += seg
    # Export als WAV
    buffer = io.BytesIO()
    merged.export(buffer, format="wav")
    buffer.seek(0)
    with st.spinner("Transkribiere Audio..."):
        transcript = agent_client.transcribe("merged_audio.wav", buffer.read())
    with st.spinner("Meeting zusammenfassen..."):
        response = await agent_client.ainvoke(
            message=transcript,
            model="owui/basismodell",
            thread_id=st.session_state.thread_id,
        )
        parsed = json.loads(response.content)

        if "Breakdown" in parsed:
            container_single_result.subheader(parsed["Breakdown"]["meeting_name"])
            container_single_result.markdown(parsed["Breakdown"]["summary"])

            container_css = """{
                                background-color: rgb(38, 39, 48);
                                padding: 20px;
                                border-radius: 5%;

                                div {
                                    width: calc(100% - 40px);
                                }
                            }
                            """

            col1, col2 = container_single_result.columns(2)

            if (len(parsed["Breakdown"]["agenda"]) > 0):
                with col1:
                    with stylable_container('agenda', css_styles=container_css):
                        st.subheader("Agenda")
                        st.markdown("\n".join([f"- {item}" for item in parsed["Breakdown"]["agenda"]]))

            if (len(parsed["Breakdown"]["decisions"]) > 0):
                with col2:
                    with stylable_container('decisions', css_styles=container_css):
                        st.subheader("Beschlüsse")
                        for decision in parsed["Breakdown"]["decisions"]:
                            st.markdown(f"- {decision['description']}")

            if (len(parsed["Breakdown"]["attendees"]) > 0):
                with col1:
                    with stylable_container('attendees', css_styles=container_css):
                        st.subheader("Anwesende Personen")
                        st.markdown(", ".join([attendee["name"] for attendee in parsed["Breakdown"]["attendees"]]))
            
            if (len(parsed["Breakdown"]["questions"]) > 0):
                with col2:
                    with stylable_container('questions', css_styles=container_css):
                        st.subheader("Fragen")
                        for question in parsed["Breakdown"]["questions"]:
                            st.markdown(f"- **{question['question']}** (Raised by: {question['raised_by']}, Status: {question['status']})")
                            if "answer" in question:
                                st.markdown(f"  - Answer: {question['answer']}")

            if (len(parsed["Breakdown"]["insights"]) > 0):
                with col1:
                    with stylable_container('insights', css_styles=container_css):
                        st.subheader("Einblicke")
                        for insight in parsed["Breakdown"]["insights"]:
                            st.markdown(f"- {insight['description']}")

            if (len(parsed["Breakdown"]["tasks"]) > 0):
                with col2:
                    with stylable_container('insights', css_styles=container_css):
                        st.subheader("Aufgaben")
                        for task in parsed["Breakdown"]["tasks"]:
                            st.markdown(f"- **{task['description']}** (Assigned to: {task['assigned_to']}, Priority: {task['priority']})")

            if (len(parsed["Breakdown"]["deadlines"]) > 0):
                with col1:
                    with stylable_container('deadlines', css_styles=container_css):
                        st.subheader("Termine")
                        for deadline in parsed["Breakdown"]["deadlines"]:
                            st.markdown(f"- **{deadline['description']}** (Date: {deadline['date']})")

            if (len(parsed["Breakdown"]["follow_ups"]) > 0):
                with col2:
                    with stylable_container('follow_ups', css_styles=container_css):
                        st.subheader("Follow-ups")
                        for follow_up in parsed["Breakdown"]["follow_ups"]:
                            st.markdown(f"- **{follow_up['description']}** (Owner: {follow_up['owner']}, Due Date: {follow_up['due_date']})")

            if (len(parsed["Breakdown"]["risks"]) > 0):
                with col1:
                    with stylable_container('follow_ups', css_styles=container_css):
                        st.subheader("Risiken")
                        for risk in parsed["Breakdown"]["risks"]:
                            st.markdown(f"- {risk['description']}")

            # Display summary, or not. keep for list?
            container_single_result.markdown(parsed["Breakdown"]["description"])


if __name__ == "__main__":
    asyncio.run(main())
