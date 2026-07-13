from enum import StrEnum, auto


class Provider(StrEnum):
    OPENAI = auto()
    OPENAI_COMPATIBLE = auto()
    AZURE_OPENAI = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    VERTEXAI = auto()
    GROQ = auto()
    AWS = auto()
    OLLAMA = auto()
    OPENROUTER = auto()
    FAKE = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_1 = "gpt-5.1"
    GPT_56_LUNA = "gpt-5.6-luna"
    GPT_56_TERRA = "gpt-5.6-terra"
    GPT_56_SOL = "gpt-5.6-sol"


class AzureOpenAIModelName(StrEnum):
    """Azure OpenAI model names"""

    AZURE_GPT_4O = "azure-gpt-4o"
    AZURE_GPT_4O_MINI = "azure-gpt-4o-mini"


class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""

    DEEPSEEK_V4_FLASH = "deepseek-v4-flash"


class AnthropicModelName(StrEnum):
    """https://docs.anthropic.com/en/docs/about-claude/models#model-names"""

    HAIKU_45 = "claude-haiku-4-5"
    SONNET_45 = "claude-sonnet-4-5"
    SONNET_5 = "claude-sonnet-5"


class GoogleModelName(StrEnum):
    """https://ai.google.dev/gemini-api/docs/models/gemini"""

    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_31_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_35_FLASH = "gemini-3.5-flash"
    GEMINI_30_PRO = "gemini-3-pro-preview"


class VertexAIModelName(StrEnum):
    """https://cloud.google.com/vertex-ai/generative-ai/docs/models"""

    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_31_FLASH_LITE = "models/gemini-3.1-flash-lite"
    GEMINI_35_FLASH = "models/gemini-3.5-flash"
    GEMINI_30_PRO = "gemini-3-pro-preview"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    LLAMA_31_8B = "llama-3.1-8b-instant"
    LLAMA_33_70B = "llama-3.3-70b-versatile"

    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_OSS_SAFEGUARD_20B = "openai/gpt-oss-safeguard-20b"


class AWSModelName(StrEnum):
    """https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

    Values are Bedrock cross-region inference profile IDs, not bare foundation
    model IDs. The latest Claude models on Bedrock reject on-demand invocation of
    the base model ID ("on-demand throughput isn't supported") and must be called
    through an inference profile prefixed with a geo (``us.``/``eu.``/``apac.``)
    or ``global.``. ``global.`` routes dynamically and is the most portable
    default; single-region deployments not enrolled in Global cross-Region
    inference should swap it for their geo prefix (e.g. ``us.``).
    """

    BEDROCK_HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    BEDROCK_SONNET = "global.anthropic.claude-sonnet-5"


class OllamaModelName(StrEnum):
    """https://ollama.com/search"""

    OLLAMA_GENERIC = "ollama"


class OpenRouterModelName(StrEnum):
    """https://openrouter.ai/models"""

    GEMINI_35_FLASH = "google/gemini-3.5-flash"


class OpenAICompatibleName(StrEnum):
    """https://platform.openai.com/docs/guides/text-generation"""

    OPENAI_COMPATIBLE = "openai-compatible"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


type AllModelEnum = (
    OpenAIModelName
    | OpenAICompatibleName
    | AzureOpenAIModelName
    | DeepseekModelName
    | AnthropicModelName
    | GoogleModelName
    | VertexAIModelName
    | GroqModelName
    | AWSModelName
    | OllamaModelName
    | OpenRouterModelName
    | FakeModelName
)
