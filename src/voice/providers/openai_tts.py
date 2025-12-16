"""OpenAI text-to-speech implementation."""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAITTS:
    """OpenAI TTS provider."""

    # API constraints
    MAX_TEXT_LENGTH = 4096
    MIN_TEXT_LENGTH = 3

    # Available configuration options
    VALID_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    VALID_MODELS = ["tts-1", "tts-1-hd"]

    def __init__(self, api_key: str | None = None, voice: str = "alloy", model: str = "tts-1"):
        """Initialize OpenAI TTS.

        Args:
            api_key: OpenAI API key (uses env var if not provided)
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
            model: Model name (tts-1 or tts-1-hd)

        Raises:
            ValueError: If voice or model is invalid
            Exception: If OpenAI client initialization fails
        """
        # Validate voice parameter
        if voice not in self.VALID_VOICES:
            raise ValueError(f"Invalid voice '{voice}'. Must be one of {self.VALID_VOICES}")

        # Validate model parameter
        if model not in self.VALID_MODELS:
            raise ValueError(f"Invalid model '{model}'. Must be one of {self.VALID_MODELS}")

        # Create OpenAI client with provided key or from environment
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.voice = voice
        self.model = model

        logger.info(f"OpenAI TTS initialized: voice={voice}, model={model}")

    def _validate_and_prepare_text(self, text: str) -> str | None:
        """Validate and prepare text for TTS generation.

        Args:
            text: Raw text input

        Returns:
            Prepared text ready for TTS, or None if text is too short

        Note:
            - Strips whitespace
            - Returns None if text is below minimum length
            - Truncates text if above maximum length
        """
        # Remove leading/trailing whitespace
        text = text.strip()

        # Skip very short text (not worth API call)
        if len(text) < self.MIN_TEXT_LENGTH:
            logger.debug(f"OpenAI TTS: skipping short text ({len(text)} chars)")
            return None

        # Truncate to API limit if needed
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(
                f"OpenAI TTS: truncating from {len(text)} to {self.MAX_TEXT_LENGTH} chars"
            )
            text = text[: self.MAX_TEXT_LENGTH]

        return text

    def generate(self, text: str) -> bytes | None:
        """Generate speech from text.

        Args:
            text: Text to convert to speech

        Returns:
            MP3 audio bytes, or None if text is too short or generation fails

        Note:
            - Text shorter than 3 chars returns None
            - Text longer than 4096 chars is truncated
            - Errors are logged but not raised - returns None instead
        """
        # Validate and prepare text
        prepared_text = self._validate_and_prepare_text(text)
        if not prepared_text:
            return None

        try:
            # Call OpenAI TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=prepared_text,
                response_format="mp3",
            )

            # Extract audio bytes from response
            audio_bytes = response.content
            logger.info(f"OpenAI TTS: generated {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            # Log error with full traceback for debugging
            logger.error(f"OpenAI TTS failed: {e}", exc_info=True)
            # Return None to allow graceful degradation
            return None

    def get_format(self) -> str:
        """Get audio format (MIME type).

        Returns:
            MIME type string for generated audio
        """
        return "audio/mp3"
