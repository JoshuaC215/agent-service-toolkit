from typing import Dict, Any, List, Literal
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage,
    ToolMessage, ToolCall,
    message_to_dict, messages_from_dict,
)
from pydantic import BaseModel, Field


class UserInput(BaseModel):
    """Basic user input for the agent."""
    """用户输入的基本信息，用于代理。"""
    message: str = Field(
        description="用户发送给代理的信息。",
        examples=["东京的天气怎么样？"],
    )
    model: str = Field(
        description="用于代理的LLM模型。",
        default="gpt-4o-mini",
        examples=["gpt-4o-mini", "llama-3.1-70b"],
    )
    thread_id: str | None = Field(
        description="用于保持和继续多轮对话的线程 ID。",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class StreamInput(UserInput):
    """用于流式传输代理响应的用户输入。"""
    stream_tokens: bool = Field(
        description="是否将 LLM 令牌流式传输给客户端。",
        default=True,
    )


class AgentResponse(BaseModel):
    """通过 /invoke 调用代理时的响应。"""
    message: Dict[str, Any] = Field(
        description="代理的最终响应，序列化为 LangChain 消息。",
        examples=[{'message':
                   {'type': 'ai', 'data':
                     {'content': '东京的天气是 70 度。', 'type': 'ai'}
                   }
                 }],
    )


class ChatMessage(BaseModel):
    """聊天中的消息。"""
    type: Literal["human", "ai", "tool"] = Field(
        description="消息的角色。",
        examples=["human", "ai", "tool"],
    )
    content: str = Field(
        description="消息的内容。",
        examples=["Hello, world!"],
    )
    tool_calls: List[ToolCall] = Field(
        description="消息中的工具调用。",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="此消息响应的工具调用 ID。",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="消息的运行 ID。",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    original: Dict[str, Any] = Field(
        description="原始的 LangChain 消息，序列化形式。",
        default={},
    )

    @classmethod
    def from_langchain(cls, message: BaseMessage) -> "ChatMessage":
        """从 LangChain 消息创建 ChatMessage。"""
        original = message_to_dict(message) # 将消息转换为字典形式
        match message:
            case HumanMessage():
                # 处理人类消息
                human_message = cls(type="human", content=message.content, original=original)
                return human_message
            case AIMessage():
                # 处理 AI 消息
                ai_message = cls(type="ai", content=message.content, original=original)
                if message.tool_calls:
                    ai_message.tool_calls = message.tool_calls  # 包含工具调用
                return ai_message
            case ToolMessage():
                # 处理工具消息
                tool_message = cls(
                    type="tool",
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                    original=original,
                )
                return tool_message
            case _:
                # 抛出错误处理不支持的消息类型
                raise ValueError(f"Unsupported message type: {message.__class__.__name__}")

    def to_langchain(self) -> BaseMessage:
        """将 ChatMessage 转换为 LangChain 消息。"""
        if self.original:
            return messages_from_dict([self.original])[0]   # 从原始字典恢复消息
        match self.type:
            case "human":
                return HumanMessage(content=self.content)   # 返回人类消息
            case _:
                # 不支持的消息类型
                raise NotImplementedError(f"Unsupported message type: {self.type}")

    def pretty_print(self) -> None:
        """美化打印 ChatMessage。"""
        lc_msg = self.to_langchain()    # 转换为 LangChain 消息
        lc_msg.pretty_print()   # 打印消息


class Feedback(BaseModel):
    """对运行的反馈，记录到 LangSmith。"""
    run_id: str = Field(
        description="记录反馈的运行 ID。",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="反馈的键。",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="反馈分数。",
        examples=[0.8],
    )
    kwargs: Dict[str, Any] = Field(
        description="额外的反馈参数，传递给 LangSmith。",
        default={},
        examples=[{'comment': '实时人类反馈'}],
    )
