"""OpenAI Whisper speech-to-text implementation."""

import logging
from typing import BinaryIO

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAISTT:
    """OpenAI Whisper STT provider."""

    def __init__(self, api_key: str | None = None):
        """Initialize OpenAI STT.

        Args:
            api_key: OpenAI API key (uses env var if not provided)

        Raises:
            Exception: If OpenAI client initialization fails
        """
        # Create OpenAI client with provided key or from environment
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        logger.info("OpenAI STT initialized")

    def transcribe(self, audio_file: BinaryIO) -> str:
        """Transcribe audio using OpenAI Whisper.

        Args:
            audio_file: Binary audio file

        Returns:
            Transcribed text (empty string on failure)

        Note:
            Errors are logged but not raised - returns empty string instead.
            This allows graceful degradation in user-facing applications.
        """
        try:
            # Reset file pointer to beginning (may have been read elsewhere)
            audio_file.seek(0)

            # Call OpenAI Whisper API for transcription
            result = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )

            # Clean up whitespace from result
            transcribed = result.strip()
            logger.info(f"OpenAI STT: transcribed {len(transcribed)} chars")
            return transcribed

        except Exception as e:
            # Log error with full traceback for debugging
            logger.error(f"OpenAI STT failed: {e}", exc_info=True)
            # Return empty string to allow graceful degradation
            return ""
