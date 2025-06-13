from collections.abc import AsyncGenerator
from typing import Any, cast
from uuid import uuid4

from ag_ui.core.events import (
    CustomEvent,
    EventEncoder,
    EventType,
    RawEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

from schema import ChatMessage


def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=message.content[0],
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


async def convert_message_to_agui_events(
    message: AnyMessage, encoder: EventEncoder
) -> AsyncGenerator[str, None]:
    """Convert a single LangChain message to AG-UI events."""
    message_id = str(uuid4())

    if isinstance(message, AIMessage):
        # Handle tool calls first
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_id = tool_call.get("id", str(uuid4()))
                yield encoder.encode(
                    ToolCallStartEvent(
                        type=EventType.TOOL_CALL_START,
                        tool_call_id=tool_call_id,
                        tool_call_name=tool_call.get("name", "unknown"),
                    )
                )
                yield encoder.encode(
                    ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=tool_call_id,
                        delta=str(tool_call.get("args", {})),
                    )
                )
                yield encoder.encode(
                    ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=tool_call_id)
                )

        # Handle text content
        content = remove_tool_calls(message.content)
        if content:
            yield encoder.encode(
                TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START, message_id=message_id, role="assistant"
                )
            )
            yield encoder.encode(
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id,
                    delta=convert_message_content_to_string(content),
                )
            )
            yield encoder.encode(
                TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id)
            )

    elif isinstance(message, ToolMessage):
        """
        AG-UI protocol does not support tool messages.
        """
        pass

    elif isinstance(message, LangchainChatMessage) and message.role == "custom":
        # m is TaskData in the form of a dict
        m = cast(dict[str, Any], message.content[0])

        # Handle custom messages as raw events
        yield encoder.encode(
            CustomEvent(
                type=EventType.CUSTOM,
                name=m.get("name", "custom"),
                value=m,
            )
        )

    else:
        # Handle other message types as raw events
        yield encoder.encode(
            RawEvent(
                type=EventType.RAW,
                event=str(message.content) if hasattr(message, "content") else str(message),
            )
        )
