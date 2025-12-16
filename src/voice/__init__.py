"""Voice input/output module.

Provides speech-to-text and text-to-speech capabilities with support
for multiple providers.

Modules:
    - SpeechToText: STT handler (can be used standalone)
    - TextToSpeech: TTS handler (can be used standalone)
    - VoiceManager: Streamlit convenience wrapper

Quick Start:
    >>> from voice import VoiceManager
    >>>
    >>> # Easy way: create from environment
    >>> voice = VoiceManager.from_env()
    >>>
    >>> # Use in Streamlit
    >>> if voice:
    ...     user_input = voice.get_chat_input()
    ...     # ... process input ...
    ...     with st.chat_message("ai"):
    ...         voice.render_message(response)

Advanced Usage:
    >>> from voice import SpeechToText, TextToSpeech, VoiceManager
    >>>
    >>> # Mix providers: OpenAI STT + custom TTS
    >>> stt = SpeechToText(provider="openai")
    >>> tts = TextToSpeech(provider="openai", voice="nova")
    >>> voice = VoiceManager(stt=stt, tts=tts)
"""

from voice.manager import VoiceManager
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

__all__ = ["VoiceManager", "SpeechToText", "TextToSpeech"]
