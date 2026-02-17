"""Tests for OpenAI STT provider."""

from unittest.mock import patch

from voice.providers.openai_stt import OpenAISTT


def test_init_with_api_key(mock_openai_client):
    """Test creating OpenAISTT with API key."""
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        # Passes if client is initialized without errors
        assert stt.client == mock_openai_client


def test_transcribe_success(mock_openai_client, mock_audio_file):
    """Test successful audio transcription."""
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        result = stt.transcribe(mock_audio_file)
        # Passes if transcribed text is returned (stripped)
        assert result == "transcribed text"


def test_transcribe_seeks_file_to_beginning(mock_openai_client, mock_audio_file):
    """Test that transcribe seeks file to beginning before reading."""
    # Move file pointer to simulate already-read file
    mock_audio_file.seek(100)
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        stt.transcribe(mock_audio_file)
        # Passes if file position was reset to 0 before transcription
        assert mock_audio_file.tell() == 0


def test_transcribe_strips_whitespace(mock_openai_client, mock_audio_file):
    """Test that transcription result has whitespace stripped."""
    # Mock API returns text with surrounding whitespace
    mock_openai_client.audio.transcriptions.create.return_value = "  text with spaces  "
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        result = stt.transcribe(mock_audio_file)
        # Passes if whitespace is stripped from result
        assert result == "text with spaces"


def test_transcribe_api_error(mock_openai_client, mock_audio_file):
    """Test that API errors are handled gracefully."""
    # Make the mock raise an exception
    mock_openai_client.audio.transcriptions.create.side_effect = Exception("API Error")
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        result = stt.transcribe(mock_audio_file)
        # Passes if empty string is returned instead of crashing
        assert result == ""
