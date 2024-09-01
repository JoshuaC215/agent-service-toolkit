import aiohttp
import json
import os
from typing import AsyncGenerator, Dict, Any, Generator
import requests
from schema import ChatMessage, UserInput, StreamInput, Feedback

class AgentClient:
    """与代理服务交互的客户端。"""

    def __init__(self, base_url: str = "http://localhost:80"):
        """
        初始化客户端。

        Args:
            base_url (str): 代理服务的基本 URL。
        """
        self.base_url = base_url    # 设置基础 URL
        self.auth_secret = os.getenv("AUTH_SECRET") # 从环境变量中获取认证密钥

    @property
    def _headers(self):
        headers = {}    # 初始化请求头
        if self.auth_secret:    # 如果存在认证密钥
            headers["Authorization"] = f"Bearer {self.auth_secret}" # 添加认证头
        return headers  # 返回请求头

    async def ainvoke(self, message: str, model: str|None = None, thread_id: str|None = None) -> ChatMessage:
        """
        异步调用代理。只返回最终消息。

        Args:
            message (str): 要发送给代理的消息。
            model (str, optional): 要使用的 LLM 模型。
            thread_id (str, optional): 继续对话的线程 ID。

        Returns:
            AnyMessage: 代理的响应。
        """
        async with aiohttp.ClientSession() as session:  # 创建异步 HTTP 会话
            request = UserInput(message=message)    # 创建请求对象
            if thread_id:   # 如果提供了线程 ID
                request.thread_id = thread_id   # 设置线程 ID
            if model:   # 如果提供了模型
                request.model = model   # 设置模型
            async with session.post(f"{self.base_url}/invoke", json=request.dict(), headers=self._headers) as response:
                if response.status == 200:  # 如果请求成功
                    result = await response.json()  # 解析 JSON 响应
                    return ChatMessage.parse_obj(result)    # 返回解析后的消息
                else:
                    raise Exception(f"Error: {response.status} - {await response.text()}")  # 抛出异常

    def invoke(self, message: str, model: str|None = None, thread_id: str|None = None) -> ChatMessage:
        """
        同步调用代理。只返回最终消息。

        Args:
            message (str): 要发送给代理的消息。
            model (str, optional): 要使用的 LLM 模型。
            thread_id (str, optional): 继续对话的线程 ID。

        Returns:
            ChatMessage: 代理的响应。
        """
        request = UserInput(message=message)    # 创建请求对象
        if thread_id:   # 如果提供了线程 ID
            request.thread_id = thread_id   # 设置线程 ID
        if model:   # 如果提供了模型
            request.model = model   # 设置模型
        response = requests.post(f"{self.base_url}/invoke", json=request.dict(), headers=self._headers) # 发送 POST 请求
        if response.status_code == 200: # 如果请求成功
            return ChatMessage.parse_obj(response.json())   # 返回解析后的消息
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}") # 抛出异常

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.decode('utf-8').strip() # 解码并去除首尾空白
        if line.startswith("data: "):   # 如果以 "data: " 开头
            data = line[6:]     # 获取数据部分
            if data == "[DONE]":    # 如果数据为 "[DONE]"
                return None     # 返回 None，表示结束
            try:
                parsed = json.loads(data)   # 尝试解析 JSON 数据
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}") # 抛出解析错误
            match parsed["type"]:   # 根据类型进行匹配
                case "message":     # 如果类型为 "message"
                    # Convert the JSON formatted message to an AnyMessage
                    try:
                        return ChatMessage.parse_obj(parsed["content"])     # 返回解析后的消息
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")    # 抛出无效消息错误
                case "token":   # 如果类型为 "token"
                    # Yield the str token directly
                    return parsed["content"]    # 返回内容令牌
                case "error":   # 如果类型为 "error"
                    raise Exception(parsed["content"])  # 抛出错误信息

    def stream(
            self,
            message: str,
            model: str|None = None,
            thread_id: str|None = None,
            stream_tokens: bool = True
        ) -> Generator[ChatMessage | str, None, None]:
        """
        同步流式响应代理的消息。

        每个中间消息作为 ChatMessage 生成。如果 stream_tokens 为 True（默认值），
        响应还将逐步生成内容令牌。

        Args:
            message (str): 要发送给代理的消息。
            model (str, optional): 要使用的 LLM 模型。
            thread_id (str, optional): 继续对话的线程 ID。
            stream_tokens (bool, optional): 是否流式传输生成的令牌
                默认: True

        Returns:
            Generator[ChatMessage | str, None, None]: 代理的响应。
        """
        request = StreamInput(message=message, stream_tokens=stream_tokens)     # 创建流请求对象
        if thread_id:   # 如果提供了线程 ID
            request.thread_id = thread_id   # 设置线程 ID
        if model:   # 如果提供了模型
            request.model = model   # 设置模型
        response = requests.post(f"{self.base_url}/stream", json=request.dict(), headers=self._headers, stream=True)    # 发送流式 POST 请求
        if response.status_code != 200:     # 如果请求失败
            raise Exception(f"Error: {response.status_code} - {response.text}") # 抛出异常

        for line in response.iter_lines():  # 逐行迭代响应
            if line:    # 如果行不为空
                parsed = self._parse_stream_line(line)  # 解析行
                if parsed is None:  # 如果解析结果为 None
                    break   # 结束迭代
                yield parsed    # 生成解析结果

    async def astream(
            self,
            message: str,
            model: str|None = None,
            thread_id: str|None = None,
            stream_tokens: bool = True
        ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        异步流式响应代理的消息。

        每个中间消息作为 AnyMessage 生成。如果 stream_tokens 为 True（默认值），
        响应还将逐步生成内容令牌。

        Args:
            message (str): 要发送给代理的消息。
            model (str, optional): 要使用的 LLM 模型。
            thread_id (str, optional): 继续对话的线程 ID。
            stream_tokens (bool, optional): 是否流式传输生成的令牌
                默认: True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: 代理的响应。
        """
        async with aiohttp.ClientSession() as session:  # 创建异步 HTTP 会话
            request = StreamInput(message=message, stream_tokens=stream_tokens) # 创建流请求对象
            if thread_id:   # 如果提供了线程 ID
                request.thread_id = thread_id   # 设置线程 ID
            if model:   # 如果提供了模型
                request.model = model   # 设置模型
            async with session.post(f"{self.base_url}/stream", json=request.dict(), headers=self._headers) as response:
                if response.status != 200:  # 如果请求失败
                    raise Exception(f"Error: {response.status} - {await response.text()}")  # 抛出异常
                # 使用 SSE 协议解析传入事件
                async for line in response.content: # 异步逐行读取内容
                    if line.decode('utf-8').strip():    # 如果行不为空
                        parsed = self._parse_stream_line(line)  # 解析行
                        if parsed is None:  # 如果解析结果为 None
                            break
                        yield parsed    # 生成解析结果

    async def acreate_feedback(
            self,
            run_id: str,
            key: str,
            score: float,
            kwargs: Dict[str, Any] = {}
        ):
        """
        为运行创建反馈记录。

        这是 LangSmith create_feedback API 的简单包装，以便在服务中存储和管理凭据，而不是客户端。
        参见: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        async with aiohttp.ClientSession() as session:  # 创建异步 HTTP 会话
            request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)      # 创建反馈请求对象
            async with session.post(f"{self.base_url}/feedback", json=request.dict(), headers=self._headers) as response:
                if response.status != 200:  # 如果请求失败
                    raise Exception(f"Error: {response.status} - {await response.text()}")  # 抛出异常
                await response.json()   # 解析 JSON 响应

