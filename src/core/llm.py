from enum import StrEnum, auto
from functools import cache
from typing import TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


class Provider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    GROQ = auto()
    AWS = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


class AnthropicModelName(StrEnum):
    """https://docs.anthropic.com/en/docs/about-claude/models#model-names"""

    HAIKU_3 = "claude-3-haiku"
    HAIKU_35 = "claude-3.5-haiku"
    SONNET_35 = "claude-3.5-sonnet"


class GoogleModelName(StrEnum):
    GEMINI_15_FLASH = "gemini-1.5-flash"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    LLAMA_31_8B = "groq-llama-3.1-8b"
    LLAMA_31_70B = "groq-llama-3.1-70b"

    LlAMA_GUARD_3_8B = "groq-llama-guard-3-8b"


class AWSModelName(StrEnum):
    BEDROCK_HAIKU = "bedrock-3.5-haiku"


_MODEL_TABLE = {
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",
    AnthropicModelName.HAIKU_3: "claude-3-haiku-20240307",
    AnthropicModelName.HAIKU_35: "claude-3-5-haiku-latest",
    AnthropicModelName.SONNET_35: "claude-3-5-sonnet-latest",
    GoogleModelName.GEMINI_15_FLASH: "gemini-1.5-flash",
    GroqModelName.LLAMA_31_8B: "llama-3.1-8b-instant",
    GroqModelName.LLAMA_31_70B: "llama-3.1-70b-versatile",
    GroqModelName.LlAMA_GUARD_3_8B: "llama-guard-3-8b",
    AWSModelName.BEDROCK_HAIKU: "anthropic.claude-3-5-haiku-20241022-v1:0",
}

AllModelEnum: TypeAlias = (
    OpenAIModelName | AnthropicModelName | GoogleModelName | GroqModelName | AWSModelName
)
ModelT: TypeAlias = ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI | ChatGroq | ChatBedrock


@cache
def get_model(model_name: str) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)

    if model_name in OpenAIModelName._value2member_map_:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in AnthropicModelName._value2member_map_:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleModelName._value2member_map_:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GroqModelName._value2member_map_:
        if model_name == GroqModelName.LlAMA_GUARD_3_8B:
            return ChatGroq(model=api_model_name, temperature=0.0).with_config(tags=["llama_guard"])
        return ChatGroq(model=api_model_name, temperature=0.5)
    if model_name in AWSModelName._value2member_map_:
        return ChatBedrock(model_id=api_model_name, temperature=0.5)
    raise ValueError(f"Unsupported model: {model_name}")
