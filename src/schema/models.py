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
    """https://developers.openai.com/api/docs/models"""

    GPT_6_LUNA = "gpt-6-luna"
    GPT_61_TERRA = "gpt-6.1-terra"


class AzureOpenAIModelName(StrEnum):
    """https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure"""

    # Values double as Azure deployment names / required_models keys; renaming breaks existing deployment maps.
    AZURE_GPT_5 = "azure-gpt-5"
    AZURE_GPT_5_MINI = "azure-gpt-5-mini"


class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""

    DEEPSEEK_FLASH = "deepseek-flash"
    DEEPSEEK_V4_PRO = "deepseek-v4-pro"


class AnthropicModelName(StrEnum):
    """https://platform.claude.com/docs/en/docs/about-claude/models"""

    HAIKU_55 = "claude-haiku-5-5"
    SONNET_5 = "claude-sonnet-5"
    SONNET_55 = "claude-sonnet-5-5"


class GoogleModelName(StrEnum):
    """https://ai.google.dev/gemini-api/docs/models/gemini"""

    GEMINI_31_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_35_FLASH = "gemini-3.5-flash"
    GEMINI_35_FLASH_LITE = "gemini-3.5-flash-lite"
    GEMINI_36_FLASH = "gemini-3.6-flash"
    GEMINI_37_FLASH = "gemini-3.7-flash"
    GEMINI_38_FLASH = "gemini-3.8-flash"


class VertexAIModelName(StrEnum):
    """https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models"""

    # The models/ prefix keeps every value distinct from its GoogleModelName twin;
    # Vertex resolves it to the same resource path as the bare name.
    GEMINI_31_FLASH_LITE = "models/gemini-3.1-flash-lite"
    GEMINI_35_FLASH = "models/gemini-3.5-flash"
    GEMINI_35_FLASH_LITE = "models/gemini-3.5-flash-lite"
    GEMINI_36_FLASH = "models/gemini-3.6-flash"
    GEMINI_37_FLASH = "models/gemini-3.7-flash"
    GEMINI_38_FLASH = "models/gemini-3.8-flash"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_OSS_SAFEGUARD_20B = "openai/gpt-oss-safeguard-20b"


class AWSModelName(StrEnum):
    """https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"""

    # Values are global cross-region inference profile IDs; latest Claude on Bedrock rejects bare on-demand model IDs. Single-region deployments may need a us./eu. prefix instead.
    BEDROCK_HAIKU = "global.anthropic.claude-haiku-5-5"
    BEDROCK_SONNET = "global.anthropic.claude-sonnet-5"


class OllamaModelName(StrEnum):
    """https://ollama.com/search"""

    OLLAMA_GENERIC = "ollama"


class OpenRouterModelName(StrEnum):
    """https://openrouter.ai/models"""

    GEMINI_35_FLASH = "google/gemini-3.5-flash"
    GEMINI_36_FLASH = "google/gemini-3.6-flash"
    GEMINI_37_FLASH = "google/gemini-3.7-flash"
    GEMINI_38_FLASH = "google/gemini-3.8-flash"


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
