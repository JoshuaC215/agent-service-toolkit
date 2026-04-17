import os
from functools import cache
from typing import TypeAlias
from urllib.parse import urlparse, urlunparse

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from core.settings import settings
from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    OpenwebuiModelName,
    VertexAIModelName,
)

# Map Enum -> API model string
_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in AzureOpenAIModelName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in VertexAIModelName}
    | {m: m.value for m in GroqModelName}
    | {m: m.value for m in AWSModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in OpenRouterModelName}
    | {m: m.value for m in FakeModelName}
    | {m: m.value for m in OpenwebuiModelName}
)


class FakeToolModel(FakeListChatModel):
    """Fake model that tolerates .bind_tools, used in tests."""

    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        return self


ModelT: TypeAlias = (
    AzureChatOpenAI
    | ChatOpenAI
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | ChatVertexAI
    | ChatGroq
    | ChatBedrock
    | ChatOllama
    | FakeToolModel
)


def _normalize_base_url(raw_url: str) -> str:
    """Normalize base URL by collapsing duplicate path slashes and trimming trailing slash."""
    parsed = urlparse(raw_url.strip())
    # Keep exactly one leading slash, collapse duplicate slashes in the rest
    path = "/" + "/".join(p for p in parsed.path.split("/") if p)
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return normalized.rstrip("/")


def _build_openai_compatible_chat_model(
    *,
    model_name: str,
    base_url: str,
    api_key: str,
) -> ChatOpenAI:
    """
    Build ChatOpenAI for OpenAI-compatible providers.

    Some providers (including OpenWebUI deployments) do not implement the newer
    OpenAI "responses" API route and return HTTP 405. We explicitly disable it
    when supported by the installed langchain_openai version.
    """
    kwargs = {
        "model_name": model_name,
        "temperature": 0.5,
        "streaming": True,
        "openai_api_base": base_url,
        "openai_api_key": api_key,
    }

    try:
        return ChatOpenAI(  # type: ignore[call-arg]
            **kwargs,
            use_responses_api=False,
        )
    except TypeError:
        return ChatOpenAI(**kwargs)  # type: ignore[call-arg]


@cache
def get_model(model_name: AllModelEnum, owui_api_key: str = "") -> ModelT:
    """
    Return a configured chat model client for the given model enum.

    Important compatibility notes:
    - For OpenAI-compatible providers (Deepseek, Custom Compat, OpenWebUI), we prefer
      configuring credentials via environment variables and only pass `model_name`
      to ChatOpenAI. This avoids signature drift across langchain_openai versions.
    """
    api_model_name = _MODEL_TABLE.get(model_name)

    # OpenAI
    if model_name in OpenAIModelName:
        if api_model_name is None:
            raise ValueError(f"Model mapping missing for {model_name}")
        return ChatOpenAI(model_name=api_model_name, temperature=0.5, streaming=True)

    # OpenAI-Compatible (custom base URL)
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")
        # Configure via environment to avoid constructor param variance
        os.environ.setdefault("OPENAI_API_KEY", str(settings.COMPATIBLE_API_KEY or ""))
        compatible_base_url = _normalize_base_url(settings.COMPATIBLE_BASE_URL)
        os.environ["OPENAI_API_BASE"] = compatible_base_url
        os.environ["OPENAI_BASE_URL"] = compatible_base_url
        return ChatOpenAI(model_name=settings.COMPATIBLE_MODEL, temperature=0.5, streaming=True)

    # Azure OpenAI
    if model_name in AzureOpenAIModelName:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")
        if api_model_name is None:
            raise ValueError(f"Model mapping missing for {model_name}")
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=api_model_name,
            temperature=0.5,
            streaming=True,
            max_retries=3,
        )

    # Deepseek (OpenAI-compatible)
    if model_name in DeepseekModelName:
        if api_model_name is None:
            raise ValueError(f"Model mapping missing for {model_name}")
        os.environ.setdefault("OPENAI_API_KEY", str(settings.DEEPSEEK_API_KEY or ""))
        os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"
        os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
        return ChatOpenAI(model_name=api_model_name, temperature=0.5, streaming=True)

    # Anthropic
    if model_name in AnthropicModelName:
        if api_model_name is None:
            raise ValueError(f"Model mapping missing for {model_name}")
        # Some versions accept model_name, others model; ChatAnthropic exposes .model
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)  # type: ignore[call-arg]

    # Google
    if model_name in GoogleModelName:
        if api_model_name is None:
            raise ValueError(f"Model mapping missing for {model_name}")
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5)

    # Vertex AI
    if model_name in VertexAIModelName:
        if api_model_name is None:
            raise ValueError(f"Model mapping missing for {model_name}")
        return ChatVertexAI(model=api_model_name, temperature=0.5, streaming=True)

    # Groq
    if model_name in GroqModelName:
        if api_model_name is None:
            raise ValueError(f"Model mapping missing for {model_name}")
        if model_name == GroqModelName.LLAMA_GUARD_4_12B:
            return ChatGroq(model_name=api_model_name, temperature=0.0)
        return ChatGroq(model_name=api_model_name, temperature=0.5)

    # AWS Bedrock
    if model_name in AWSModelName:
        if api_model_name is None:
            raise ValueError(f"Model mapping missing for {model_name}")
        # Some versions expect 'model', others 'model_id'; keep ignore for types
        return ChatBedrock(model=api_model_name, temperature=0.5)  # type: ignore[call-arg]

    # Ollama
    if model_name in OllamaModelName:
        model_id = settings.OLLAMA_MODEL or "llama3.3"
        if settings.OLLAMA_BASE_URL:
            return ChatOllama(model=model_id, temperature=0.5, base_url=settings.OLLAMA_BASE_URL)
        return ChatOllama(model=model_id, temperature=0.5)

    # OpenWebUI (OpenAI-compatible chat API)
    if model_name in OpenwebuiModelName:
        base_url = os.getenv("OWUI_CHAT_API_URL")
        if not base_url:
            owui_base = os.getenv("OWUI_BASE_URL")
            if owui_base:
                base_url = owui_base.rstrip("/") + "/api"
        if not base_url:
            raise ValueError(
                "OpenWebUI base URL not configured. Set OWUI_CHAT_API_URL or OWUI_BASE_URL."
            )
        base_url = _normalize_base_url(base_url)
        if api_model_name is None:
            raise ValueError(f"Model mapping missing for {model_name}")
        trimmed_model = api_model_name.split("/", 1)[1] if "/" in api_model_name else api_model_name
        # Configure env (defensive) AND pass explicit params for widest compatibility
        os.environ.setdefault("OPENAI_API_KEY", owui_api_key or "")
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
        return _build_openai_compatible_chat_model(
            model_name=trimmed_model,
            base_url=base_url,
            api_key=owui_api_key,
        )

    # Fake (tests)
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    # Fallback: OWUI default if only base URL provided
    if not api_model_name:
        owui_base = os.getenv("OWUI_BASE_URL")
        if not owui_base:
            raise ValueError(f"Unsupported model: {model_name}")
        base_url = _normalize_base_url(
            os.getenv("OWUI_CHAT_API_URL") or owui_base.rstrip("/") + "/api"
        )
        os.environ.setdefault("OPENAI_API_KEY", owui_api_key or "")
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
        return _build_openai_compatible_chat_model(
            model_name="gpt-4o",
            base_url=base_url,
            api_key=owui_api_key,
        )

    # Should not reach here
    raise ValueError(f"Unsupported model: {model_name}")
