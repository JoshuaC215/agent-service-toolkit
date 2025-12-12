"""VoiceManager - Streamlit integration layer.

This module provides Streamlit-specific UI integration for voice features.
All Streamlit dependencies are isolated here.
"""

import logging
from typing import Optional

import streamlit as st

from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


class VoiceManager:
    """Streamlit convenience layer for voice features.

    This class provides Streamlit-specific methods for voice input/output.
    It handles UI feedback (spinners, errors) while delegating actual
    voice processing to STT and TTS modules.

    Example:
        >>> voice = VoiceManager.from_env()
        >>>
        >>> if voice:
        ...     user_input = voice.get_chat_input()
        ...     if user_input:
        ...         with st.chat_message("ai"):
        ...             voice.render_message("Hello!")
    """

    def __init__(self, stt: SpeechToText | None = None, tts: TextToSpeech | None = None):
        """Initialize VoiceManager.

        Args:
            stt: SpeechToText instance (None to disable STT)
            tts: TextToSpeech instance (None to disable TTS)
        """
        self.stt = stt
        self.tts = tts

        logger.info(
            f"VoiceManager: STT={'enabled' if stt else 'disabled'}, "
            f"TTS={'enabled' if tts else 'disabled'}"
        )

    @classmethod
    def from_env(cls) -> Optional["VoiceManager"]:
        """Create VoiceManager from environment variables.

        Reads VOICE_STT_PROVIDER and VOICE_TTS_PROVIDER to configure
        speech-to-text and text-to-speech providers.

        Returns:
            VoiceManager if either STT or TTS is configured, None otherwise

        Example:
            >>> # In .env:
            >>> # VOICE_STT_PROVIDER=openai
            >>> # VOICE_TTS_PROVIDER=openai
            >>>
            >>> voice = VoiceManager.from_env()
            >>> # Returns configured VoiceManager or None if disabled
        """
        # Create STT and TTS from environment
        stt = SpeechToText.from_env()
        tts = TextToSpeech.from_env()

        # If both disabled, return None (no voice features)
        if not stt and not tts:
            logger.debug("Voice features not configured")
            return None

        return cls(stt=stt, tts=tts)

    def _transcribe_audio(self, audio) -> str | None:
        """Transcribe audio with UI feedback.

        Shows spinner during transcription and error message on failure.

        Args:
            audio: Audio file object from Streamlit chat input

        Returns:
            Transcribed text, or None if transcription failed
        """
        # Defensive check (should not happen if called correctly)
        if not self.stt:
            st.error("⚠️ Speech-to-text not configured.")
            return None

        # Show spinner while transcribing
        with st.spinner("🎤 Transcribing audio..."):
            transcribed = self.stt.transcribe(audio)

        # Check if transcription succeeded
        if not transcribed:
            st.error("⚠️ Transcription failed. Please try again or type your message.")
            return None

        return transcribed

    def get_chat_input(self, placeholder: str = "Your message") -> str | None:
        """Get chat input with optional voice transcription.

        Handles Streamlit UI including audio input widget and transcription
        feedback (spinner, errors).

        Args:
            placeholder: Placeholder text for input

        Returns:
            User's message (transcribed if audio, otherwise text), or None if no input
        """
        # No STT - use regular text input
        if not self.stt:
            return st.chat_input(placeholder)

        # STT enabled - use audio-capable input
        chat_value = st.chat_input(placeholder, accept_audio=True)

        if not chat_value:
            return None

        # Handle string return (text-only input)
        if isinstance(chat_value, str):
            return chat_value

        # Handle object/dict return (audio-capable input)
        # Extract text - support both attribute and dict access
        text_content = None
        if hasattr(chat_value, "text"):
            text_content = chat_value.text
        elif isinstance(chat_value, dict):
            text_content = chat_value.get("text", "")

        # Extract audio - support both attribute and dict access
        audio_content = None
        if hasattr(chat_value, "audio"):
            audio_content = chat_value.audio
        elif isinstance(chat_value, dict):
            audio_content = chat_value.get("audio")

        # If audio is provided, transcribe it
        if audio_content:
            return self._transcribe_audio(audio_content)

        # If no audio, return the text content
        if text_content:
            return text_content

        # No text or audio provided
        return None

    def render_message(self, content: str, container=None, audio_only: bool = False) -> None:
        """Render message with optional TTS audio.

        Handles Streamlit UI including text display and audio player.
        Saves generated audio in session state so it persists across reruns.

        Args:
            content: Message content to display
            container: Streamlit container (defaults to current context)
            audio_only: If True, only render audio (text already displayed)
        """
        if container is None:
            container = st

        # Show text unless audio_only mode (for streaming where text is already shown)
        if not audio_only:
            container.write(content)

        # Add audio if TTS enabled and content is not empty
        if self.tts and content.strip():
            # Show placeholder while generating audio
            placeholder = container.empty()
            with placeholder:
                st.caption("🎙️ Generating audio...")

            # Generate TTS audio
            audio = self.tts.generate(content)

            # Save audio in session state for the last AI message
            # This allows it to persist across st.rerun() calls
            if audio:
                st.session_state.last_audio = {"data": audio, "format": self.tts.get_format()}

            # Replace placeholder with audio player or error message
            if audio:
                placeholder.audio(audio, format=self.tts.get_format())
            else:
                placeholder.caption("🔇 Audio generation unavailable")
