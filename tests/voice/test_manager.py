"""Tests for VoiceManager core logic (no Streamlit UI tests)."""

from unittest.mock import Mock, patch

from voice.manager import VoiceManager


def test_init_with_both_stt_and_tts():
    """Test creating VoiceManager with both STT and TTS."""
    mock_stt = Mock()
    mock_tts = Mock()
    manager = VoiceManager(stt=mock_stt, tts=mock_tts)
    # Passes if both STT and TTS are assigned correctly
    assert manager.stt == mock_stt
    assert manager.tts == mock_tts


def test_init_with_only_tts():
    """Test creating VoiceManager with only TTS (STT=None)."""
    mock_tts = Mock()
    manager = VoiceManager(stt=None, tts=mock_tts)
    # Passes if partial voice features work (TTS only)
    assert manager.stt is None
    assert manager.tts == mock_tts


def test_from_env_both_configured():
    """Test from_env when both STT and TTS are configured."""
    mock_stt = Mock()
    mock_tts = Mock()
    with patch("voice.manager.SpeechToText.from_env", return_value=mock_stt):
        with patch("voice.manager.TextToSpeech.from_env", return_value=mock_tts):
            manager = VoiceManager.from_env()
            # Passes if VoiceManager is created with both STT and TTS
            assert manager is not None
            assert manager.stt == mock_stt
            assert manager.tts == mock_tts


def test_from_env_only_tts_configured():
    """Test from_env when only TTS is configured."""
    mock_tts = Mock()
    with patch("voice.manager.SpeechToText.from_env", return_value=None):
        with patch("voice.manager.TextToSpeech.from_env", return_value=mock_tts):
            manager = VoiceManager.from_env()
            # Passes if VoiceManager is created with TTS only (STT=None acceptable)
            assert manager is not None
            assert manager.stt is None
            assert manager.tts == mock_tts


def test_from_env_neither_configured():
    """Test from_env when neither STT nor TTS are configured."""
    with patch("voice.manager.SpeechToText.from_env", return_value=None):
        with patch("voice.manager.TextToSpeech.from_env", return_value=None):
            manager = VoiceManager.from_env()
            # Passes if None is returned when voice features not configured
            assert manager is None


def test_transcribe_audio_stt_not_configured():
    """Test _transcribe_audio returns None when STT not configured."""
    manager = VoiceManager(stt=None, tts=Mock())
    mock_audio = Mock()
    # Mock Streamlit's st.error to avoid actual UI calls
    with patch("voice.manager.st.error"):
        result = manager._transcribe_audio(mock_audio)
        # Passes if None is returned (defensive check when STT not configured)
        assert result is None
