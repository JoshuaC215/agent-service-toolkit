"""Tests for OpenAI TTS provider."""

from unittest.mock import patch

import pytest

from voice.providers.openai_tts import OpenAITTS


def test_init_with_valid_params(mock_openai_client):
    """Test creating OpenAITTS with valid parameters."""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key", voice="nova", model="tts-1")
        # Passes if client is initialized without errors
        assert tts.client == mock_openai_client


def test_init_with_invalid_voice():
    """Test that invalid voice raises ValueError."""
    # Passes if ValueError is raised for invalid voice
    with pytest.raises(ValueError, match="Invalid voice"):
        OpenAITTS(api_key="test-key", voice="invalid", model="tts-1")


def test_init_with_invalid_model():
    """Test that invalid model raises ValueError."""
    # Passes if ValueError is raised for invalid model
    with pytest.raises(ValueError, match="Invalid model"):
        OpenAITTS(api_key="test-key", voice="nova", model="invalid")


def test_validate_text_too_short(mock_openai_client):
    """Test that text shorter than MIN_LENGTH returns None."""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        result = tts._validate_and_prepare_text("ab")  # 2 chars < MIN_LENGTH (3)
        # Passes if None is returned for too-short text
        assert result is None


def test_validate_text_too_long(mock_openai_client):
    """Test that text longer than MAX_LENGTH is truncated."""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        long_text = "a" * 5000  # > MAX_LENGTH (4096)
        result = tts._validate_and_prepare_text(long_text)
        # Passes if text is truncated to MAX_LENGTH
        assert result is not None
        assert len(result) == 4096


def test_generate_success(mock_openai_client):
    """Test successful audio generation."""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        result = tts.generate("Hello world")
        # Passes if audio bytes are returned
        assert result == b"fake audio data"


def test_generate_api_error(mock_openai_client):
    """Test that API errors are handled gracefully."""
    # Make the mock raise an exception
    mock_openai_client.audio.speech.create.side_effect = Exception("API Error")
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        result = tts.generate("Hello world")
        # Passes if None is returned instead of crashing
        assert result is None
