"""Speech-to-text factory.

This module provides a factory class that loads the appropriate STT provider
based on configuration.
"""

import logging
import os
from typing import BinaryIO, Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "deepgram"]


class SpeechToText:
    """Speech-to-text factory.

    Loads and delegates to specific STT provider implementations.

    Example:
        >>> stt = SpeechToText(provider="openai")
        >>> text = stt.transcribe(audio_file)
        >>>
        >>> # Or from environment
        >>> stt = SpeechToText.from_env()
        >>> if stt:
        ...     text = stt.transcribe(audio_file)
    """

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """Initialize STT with specified provider.

        Args:
            provider: Provider name ("openai", "deepgram", etc.)
            api_key: API key (uses env var if not provided)
            **config: Provider-specific configuration

        Raises:
            ValueError: If provider is unknown
        """
        self._provider_name = provider

        # Resolve API key from parameter or environment
        resolved_api_key = self._get_api_key(provider, api_key)

        # Load and configure the provider
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"SpeechToText created with provider={provider}")

    def _get_api_key(self, provider: Provider, api_key: str | None) -> str | None:
        """Get API key from parameter or environment.

        Args:
            provider: Provider name
            api_key: API key from parameter (takes precedence)

        Returns:
            Resolved API key or None
        """
        # If API key provided explicitly, use it
        if api_key:
            return api_key

        # Otherwise, get from environment based on provider
        match provider:
            case "openai":
                return os.getenv("OPENAI_API_KEY")
            case "deepgram":
                return os.getenv("DEEPGRAM_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """Load the appropriate STT provider implementation.

        Args:
            provider: Provider name
            api_key: Resolved API key
            config: Provider-specific configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is unknown
            NotImplementedError: If provider not yet implemented
        """
        match provider:
            case "openai":
                from voice.providers.openai_stt import OpenAISTT

                return OpenAISTT(api_key=api_key, **config)

            case "deepgram":
                # Example for future extensions: to add Deepgram support, implement DeepgramSTT provider and uncomment:
                # from voice.providers.deepgram_stt import DeepgramSTT
                # return DeepgramSTT(api_key=api_key, **config)
                raise NotImplementedError("Deepgram STT provider not yet implemented")

            case _:
                # Catch-all for unknown providers
                raise ValueError(f"Unknown STT provider: {provider}. Available providers: openai")

    @property
    def provider(self) -> str:
        """Get the provider name.

        Returns:
            Provider name string
        """
        return self._provider_name

    @classmethod
    def from_env(cls) -> "SpeechToText | None":
        """Create STT from environment variables.

        Reads VOICE_STT_PROVIDER env var to determine which provider to use.
        Returns None if not configured.

        Returns:
            SpeechToText instance or None

        Example:
            >>> # In .env: VOICE_STT_PROVIDER=openai
            >>> stt = SpeechToText.from_env()
            >>> if stt:
            ...     text = stt.transcribe(audio_file)
        """
        provider = os.getenv("VOICE_STT_PROVIDER")

        # If provider not set, voice features are disabled
        if not provider:
            logger.debug("VOICE_STT_PROVIDER not set, STT disabled")
            return None

        try:
            # Create instance with provider from environment
            # Validates provider and raises ValueError if invalid
            return cls(provider=cast(Provider, provider))
        except Exception as e:
            # Log error but don't crash - allow app to continue without voice
            logger.error(f"Failed to create STT provider: {e}", exc_info=True)
            return None

    def transcribe(self, audio_file: BinaryIO) -> str:
        """Transcribe audio to text.

        Delegates to the underlying provider implementation.

        Args:
            audio_file: Binary audio file

        Returns:
            Transcribed text (empty string on failure)
        """
        return self._provider.transcribe(audio_file)
