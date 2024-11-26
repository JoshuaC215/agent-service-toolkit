from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import BeforeValidator, HttpUrl, SecretStr, TypeAdapter, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OpenAIModelName,
    Provider,
)


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "0.0.0.0"
    PORT: int = 80

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    USE_AWS_BEDROCK: bool = False
    USE_FAKE_MODEL: bool = False

    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]

    OPENWEATHERMAP_API_KEY: SecretStr | None = None

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.AWS: self.USE_AWS_BEDROCK,
            Provider.FAKE: self.USE_FAKE_MODEL,
        }
        active_keys = {k for k, v in api_keys.items() if v}
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        if self.DEFAULT_MODEL is None:
            first_provider = next(iter(active_keys))
            match first_provider:
                case Provider.OPENAI:
                    self.DEFAULT_MODEL = OpenAIModelName.GPT_4O_MINI
                case Provider.ANTHROPIC:
                    self.DEFAULT_MODEL = AnthropicModelName.HAIKU_3
                case Provider.GOOGLE:
                    self.DEFAULT_MODEL = GoogleModelName.GEMINI_15_FLASH
                case Provider.GROQ:
                    self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B
                case Provider.AWS:
                    self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                case Provider.FAKE:
                    self.DEFAULT_MODEL = FakeModelName.FAKE
                case _:
                    raise ValueError(f"Unknown provider: {first_provider}")

    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()
