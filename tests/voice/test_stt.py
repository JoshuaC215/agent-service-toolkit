"""Tests for SpeechToText factory class."""

import os
from unittest.mock import patch

import pytest

from voice.stt import SpeechToText


def test_init_with_openai_provider(mock_openai_client):
    """Test creating STT with openai provider and explicit API key."""
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = SpeechToText(provider="openai", api_key="test-key")
        # Passes if provider property returns "openai"
        assert stt.provider == "openai"


def test_init_with_invalid_provider():
    """Test that invalid provider raises ValueError."""
    # Passes if ValueError is raised with expected message
    with pytest.raises(ValueError, match="Unknown STT provider: invalid"):
        SpeechToText(provider="invalid", api_key="test-key")


def test_init_with_unimplemented_provider():
    """Test that unimplemented provider raises NotImplementedError."""
    # Passes if NotImplementedError is raised (deepgram not yet implemented)
    with pytest.raises(NotImplementedError, match="Deepgram STT provider not yet implemented"):
        SpeechToText(provider="deepgram", api_key="test-key")


def test_from_env_provider_not_set():
    """Test from_env returns None when VOICE_STT_PROVIDER is not set."""
    with patch.dict(os.environ, {}, clear=True):
        result = SpeechToText.from_env()
        # Passes if None is returned when env var not set
        assert result is None


def test_from_env_valid_provider(mock_openai_client):
    """Test from_env creates STT instance with valid provider."""
    with patch.dict(os.environ, {"VOICE_STT_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
        with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
            stt = SpeechToText.from_env()
            # Passes if SpeechToText instance is created with correct provider
            assert stt is not None
            assert stt.provider == "openai"


def test_from_env_invalid_provider_returns_none():
    """Test from_env returns None (and logs error) for invalid provider."""
    with patch.dict(os.environ, {"VOICE_STT_PROVIDER": "invalid"}):
        result = SpeechToText.from_env()
        # Passes if None is returned instead of crashing
        assert result is None


def test_get_api_key_from_param(mock_openai_client):
    """Test that explicit api_key parameter takes precedence over env var."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        with patch(
            "voice.providers.openai_stt.OpenAI", return_value=mock_openai_client
        ) as mock_openai:
            SpeechToText(provider="openai", api_key="param-key")
            # Passes if OpenAI client is initialized with param key (not env key)
            mock_openai.assert_called_once_with(api_key="param-key")


def test_get_api_key_from_env(mock_openai_client):
    """Test that API key is loaded from environment when not provided."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        with patch(
            "voice.providers.openai_stt.OpenAI", return_value=mock_openai_client
        ) as mock_openai:
            SpeechToText(provider="openai")
            # Passes if OpenAI client is initialized with env key
            mock_openai.assert_called_once_with(api_key="env-key")
