from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field, SerializeAsAny

from schema.models import AllModelEnum, AnthropicModelName, OpenAIModelName


class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant for generating research papers."],
    )
    track: Literal["core", "product", "experimental"] = Field(
        description="Classification track for the agent.",
        examples=["core", "product"],
        default="core",
    )
    stability: Literal["stable", "beta", "experimental", "deprecated"] = Field(
        description="Stability level for the agent.",
        examples=["stable", "beta"],
        default="stable",
    )
    pack: str = Field(
        description="Pack identifier that owns the agent.",
        examples=["core", "skill", "dwh"],
        default="core",
    )


class VariantIdentifier(TypedDict):
    streamlit_app_name: str
    variant: str | None


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    models: list[AllModelEnum] = Field(
        description="List of available LLMs.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["research-assistant"],
    )
    default_model: AllModelEnum = Field(
        description="Default model used when none is specified.",
    )


class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="Model",
        description="LLM Model to use for the agent.",
        default=OpenAIModelName.GPT_4O_MINI,
        examples=[OpenAIModelName.GPT_4O_MINI, AnthropicModelName.HAIKU_35],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    run_id: str | None = Field(
        description="Run ID to persist and continue a trace in Langfuse.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )
    api_key: str | None = Field(
        description="custom chat api key, e.g. JWT token for openwebui",
        default=None,
        examples=["difazlvgduhijfagäfoö.rtphrqefwizurwohtij45"],
    )
    url_parameters: dict[str, Any] | None = Field(
        description="Optionale URL-Parameter für dynamische Link-Generierung, z.B. {'hubspot_id': '1234'}",
        default=None,
        examples=[{"hubspot_id": "1234"}],
    )
    variant: VariantIdentifier | None = Field(
        description="custom variant identifier, containing streamlit app name and variant id",
        default=None,
        examples=[{"streamlit_app_name": "Skill_Companion", "variant": "default"}],
    )

    @classmethod
    def __get_validators__(cls):
        yield from super().__get_validators__()
        yield cls._coerce_model_enum

    @classmethod
    def _coerce_model_enum(cls, values):
        # Only coerce if 'model' is present and is a string
        if isinstance(values, dict) and "model" in values:
            model_val = values["model"]
            if isinstance(model_val, str):
                # Try all enums in AllModelEnum
                for enum_type in AllModelEnum.__args__:
                    try:
                        values["model"] = enum_type(model_val)
                        break
                    except ValueError:
                        continue
        return values


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class Feedback(BaseModel):  # type: ignore[no-redef]
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_id: str | None = Field(
        description="Optional agent ID for agent-specific history retrieval.",
        default=None,
        examples=["research-assistant"],
    )


class ChatHistory(BaseModel):
    messages: list[ChatMessage]
