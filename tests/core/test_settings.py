import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from core.settings import Settings, check_str_is_http
from schema.models import AnthropicModelName, OpenAIModelName


def test_check_str_is_http():
    # Test valid HTTP URLs
    assert check_str_is_http("http://example.com/") == "http://example.com/"
    assert check_str_is_http("https://api.test.com/") == "https://api.test.com/"

    # Test invalid URLs
    with pytest.raises(ValidationError):
        check_str_is_http("not_a_url")
    with pytest.raises(ValidationError):
        check_str_is_http("ftp://invalid.com")


def test_settings_default_values():
    settings = Settings(_env_file=None)
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8080
    assert settings.USE_AWS_BEDROCK is False
    assert settings.USE_FAKE_MODEL is False


def test_settings_no_api_keys():
    # Test that settings raises error when no API keys are provided
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
            _ = Settings(_env_file=None)


def test_settings_with_openai_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_4O_MINI
        assert settings.AVAILABLE_MODELS == set(OpenAIModelName)


def test_settings_with_anthropic_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.ANTHROPIC_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == AnthropicModelName.HAIKU_3
        assert settings.AVAILABLE_MODELS == set(AnthropicModelName)


def test_settings_with_multiple_api_keys():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_openai_key")
        assert settings.ANTHROPIC_API_KEY == SecretStr("test_anthropic_key")
        # When multiple providers are available, OpenAI should be the default
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_4O_MINI
        # Available models should include exactly all OpenAI and Anthropic models
        expected_models = set(OpenAIModelName)
        expected_models.update(set(AnthropicModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_base_url():
    settings = Settings(HOST="0.0.0.0", PORT=8000, _env_file=None)
    assert settings.BASE_URL == "http://0.0.0.0:8000"


def test_settings_is_dev():
    settings = Settings(MODE="dev", _env_file=None)
    assert settings.is_dev() is True

    settings = Settings(MODE="prod", _env_file=None)
    assert settings.is_dev() is False
