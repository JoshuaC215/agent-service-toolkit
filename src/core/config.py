from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import BeforeValidator, HttpUrl, SecretStr, TypeAdapter, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.llm import OpenAIModelName


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )
    MODE: str | None = None

    HOST: str = "0.0.0.0"
    PORT: int = 80

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None

    DEFAULT_MODEL: OpenAIModelName = OpenAIModelName.GPT_4O_MINI

    OPENWEATHERMAP_API_KEY: SecretStr | None = None

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    def model_post_init(self, __context: Any) -> None:
        if not (
            self.OPENAI_API_KEY
            or self.ANTHROPIC_API_KEY
            or self.GOOGLE_API_KEY
            or self.GROQ_API_KEY
        ):
            raise ValueError("Please set API keys to enable at least one LLM.")

        if self.is_dev():
            print("Running in dev mode. host=127.0.0.1, port=8000")
            self.HOST = "127.0.0.1"
            self.PORT = 8000

    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()
