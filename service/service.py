import asyncio
from contextlib import asynccontextmanager
import json
import os
from typing import AsyncGenerator, Dict, Any, Tuple
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.graph import CompiledGraph
from langsmith import Client as LangsmithClient

from agent import research_assistant
from schema import ChatMessage, Feedback, UserInput, StreamInput


class TokenQueueStreamingHandler(AsyncCallbackHandler):
    """LangChain 的回调处理器，用于将 LLM 令牌流式传输到 asyncio 队列。"""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """当接收到新的 LLM 令牌时，将其放入队列。"""
        if token:
            await self.queue.put(token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序的生命周期管理器，用于初始化和清理资源。"""
    # 使用 Sqlite 检查点构造代理
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        research_assistant.checkpointer = saver # 设置检查点保存器
        app.state.agent = research_assistant    # 将代理存储在应用状态中
        yield
    # 上下文管理器在退出时将清理 AsyncSqliteSaver

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def check_auth_header(request: Request, call_next):
    """中间件，用于检查请求中的身份验证头。"""
    if auth_secret := os.getenv("AUTH_SECRET"):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(status_code=401, content="Missing or invalid token")
        if auth_header[7:] != auth_secret:
            return Response(status_code=401, content="Invalid token")
    return await call_next(request)

def _parse_input(user_input: UserInput) -> Tuple[Dict[str, Any], str]:
    """解析用户输入，生成处理所需的参数和运行 ID。"""
    run_id = uuid4()    # 生成唯一的运行 ID
    thread_id = user_input.thread_id or str(uuid4())    # 获取线程 ID
    input_message = ChatMessage(type="human", content=user_input.message)   # 创建聊天消息
    kwargs = dict(
        input={"messages": [input_message.to_langchain()]},
        config=RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model},
            run_id=run_id,
        ),
    )
    return kwargs, run_id

@app.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    """
    使用用户输入调用代理以检索最终响应。

    使用 thread_id 来持久化并继续多轮对话。运行 ID 也附加到消息中以记录反馈。
    """
    agent: CompiledGraph = app.state.agent  # 获取代理实例
    kwargs, run_id = _parse_input(user_input)   # 解析输入
    try:
        response = await agent.ainvoke(**kwargs)    # 调用代理
        output = ChatMessage.from_langchain(response["messages"][-1])   # 从响应中获取消息
        output.run_id = str(run_id) # 设置运行 ID
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) # 捕获异常并返回错误

async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
    """
    从代理生成消息流。

    这是 /stream 端点的工作方法。
    """
    agent: CompiledGraph = app.state.agent  # 获取代理实例
    kwargs, run_id = _parse_input(user_input)   # 解析输入

    # 使用 asyncio 队列按时间顺序处理消息和令牌
    output_queue = asyncio.Queue(maxsize=10)
    if user_input.stream_tokens:
        kwargs["config"]["callbacks"] = [TokenQueueStreamingHandler(queue=output_queue)]

    # 在单独的任务中运行代理的消息流
    async def run_agent_stream():
        async for s in agent.astream(**kwargs, stream_mode="updates"):
            await output_queue.put(s)   # 将消息放入输出队列
        await output_queue.put(None)    # 结束信号
    stream_task = asyncio.create_task(run_agent_stream())

    # 处理队列并通过 SSE 流发送消息
    while s := await output_queue.get():
        if isinstance(s, str):
            # 如果 s 是字符串，则表示 LLM 令牌
            yield f"data: {json.dumps({'type': 'token', 'content': s})}\n\n"
            continue

        # 否则，s 应该是图中每个节点的状态更新字典
        new_messages = []
        for _, state in s.items():
            new_messages.extend(state["messages"])  # 获取新的消息
        for message in new_messages:
            try:
                chat_message = ChatMessage.from_langchain(message)  # 从消息中创建 ChatMessage
                chat_message.run_id = str(run_id)   # 设置运行 ID
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                continue
            # LangGraph 重新发送输入消息，感觉奇怪，所以丢弃它
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.dict()})}\n\n"

    await stream_task   # 等待流任务完成
    yield "data: [DONE]\n\n"    # 结束信号

@app.post("/stream")
async def stream_agent(user_input: StreamInput):
    """
    流式传输代理对用户输入的响应，包括中间消息和令牌。

    使用 thread_id 来持久化并继续多轮对话。运行 ID 也附加到所有消息中以记录反馈。
    """
    return StreamingResponse(message_generator(user_input), media_type="text/event-stream")

@app.post("/feedback")
async def feedback(feedback: Feedback):
    """
    记录对运行的反馈到 LangSmith。

    这是 LangSmith create_feedback API 的简单封装，
    以便在服务中存储和管理凭据，而不是在客户端中。
    参见: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()  # 创建 Langsmith 客户端
    kwargs = feedback.kwargs or {}  # 获取反馈参数
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return {"status": "success"}    # 返回成功状态
