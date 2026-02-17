"""Shared fixtures for voice module tests."""

import io
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for TTS/STT tests."""
    client = Mock()

    # Mock TTS response (returns object with .content attribute)
    mock_audio_response = Mock()
    mock_audio_response.content = b"fake audio data"
    client.audio.speech.create.return_value = mock_audio_response

    # Mock STT response (returns string directly when response_format="text")
    client.audio.transcriptions.create.return_value = "transcribed text"

    return client


@pytest.fixture
def mock_audio_file():
    """BytesIO mock audio file for STT tests."""
    return io.BytesIO(b"fake audio bytes")
