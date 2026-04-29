import json
import logging
import os
import time
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
    VariantIdentifier,
)

logging.basicConfig(level=logging.INFO)


class AgentClientError(Exception):
    pass


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
        api_key: str | None = None,
        variant: VariantIdentifier | None = None,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
            get_info (bool, optional): Whether to fetch agent information on init.
                Default: True
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        self.api_key = api_key
        self.variant = variant
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)
        self.verbose = os.getenv("CLIENT_VERBOSE_ERRORS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        attempts = 8
        delay = 0.5
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.get(
                    f"{self.base_url}/info",
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                content = response.content.strip() if response.content else b""
                if not content:
                    logging.warning(
                        "Received empty body from /info; using empty metadata fallback."
                    )
                    data = {
                        "agents": [],
                        "models": [],
                        "default_agent": "",
                        "default_model": "",
                    }
                    break
                if "application/json" not in content_type.lower():
                    logging.warning(
                        "Non-JSON Content-Type from /info (%s); using empty metadata fallback.",
                        content_type,
                    )
                    data = {
                        "agents": [],
                        "models": [],
                        "default_agent": "",
                        "default_model": "",
                    }
                    break
                try:
                    data = response.json()
                except json.JSONDecodeError as je:
                    logging.warning(
                        "Invalid JSON from /info: %s; using empty metadata fallback.", je
                    )
                    data = {
                        "agents": [],
                        "models": [],
                        "default_agent": "",
                        "default_model": "",
                    }
                break
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt == attempts:
                    raise AgentClientError(f"Error getting service info: {e}") from e
                logging.warning(
                    "Failed to fetch /info (attempt %s/%s): %s. Retrying in %.1fs",
                    attempt,
                    attempts,
                    e,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 1.5, 4.0)
        else:
            raise AgentClientError(f"Error getting service info: {last_error}")

        self.info = ServiceMetadata.model_validate(data)
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
        url_parameters: dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            AnyMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message, api_key=self.api_key, variant=self.variant)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        if run_id:
            request.run_id = run_id
        if url_parameters:
            request.url_parameters = url_parameters

        print(f"SENDING TO {self.base_url}/{self.agent}/invoke: {request.dict()}")
        # Build a set of candidate URLs to be robust against trailing slash or proxy prefixes.
        candidate_urls = [
            f"{self.base_url.rstrip('/')}/{self.agent}/invoke",
        ]
        # Trailing-slash variant
        if not candidate_urls[0].endswith("/"):
            candidate_urls.append(candidate_urls[0] + "/")
        # Optional /api/v1 prefix variant (useful if the service is mounted behind a proxy prefix)
        _base = self.base_url.rstrip("/")
        if os.getenv("ALLOW_API_V1_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}:
            if "/api/" not in _base and not _base.endswith("/api") and "/v1" not in _base:
                candidate_urls.append(f"{_base}/api/v1/{self.agent}/invoke")

        last_err: Exception | None = None
        response: httpx.Response | None = None
        async with httpx.AsyncClient(follow_redirects=True) as client:
            for url in candidate_urls:
                try:
                    resp = await client.post(
                        url,
                        json=request.dict(),
                        headers=self._headers,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    response = resp
                    break
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if e.response is not None else "?"
                    if self.verbose:
                        body = e.response.text[:500] if e.response is not None else "<no body>"
                        # Include contiguous "{status} {body}" for compatibility with tests when verbose
                        last_err = AgentClientError(
                            f"HTTP {status} POST {url}: {body} | {status} {body}"
                        )
                    else:
                        reason = e.response.reason_phrase if e.response is not None else ""
                        last_err = AgentClientError(f"HTTP {status} {reason}")
                except httpx.HTTPError as e:
                    last_err = AgentClientError(f"Error POST {url}: {e}")

        if response is None:
            raise last_err or AgentClientError("Unknown HTTP error invoking agent")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
        url_parameters: dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent synchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            ChatMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message, api_key=self.api_key)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        if run_id:
            request.run_id = run_id
        if url_parameters:
            request.url_parameters = url_parameters

        # Build a set of candidate URLs to be robust against trailing slash or proxy prefixes.
        _base = self.base_url.rstrip("/")
        candidate_urls = [
            f"{_base}/{self.agent}/invoke",
        ]
        # Trailing-slash variant
        if not candidate_urls[0].endswith("/"):
            candidate_urls.append(candidate_urls[0] + "/")
        # Optional /api/v1 prefix variant (useful if the service is mounted behind a proxy prefix)
        if os.getenv("ALLOW_API_V1_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}:
            if "/api/" not in _base and not _base.endswith("/api") and "/v1" not in _base:
                candidate_urls.append(f"{_base}/api/v1/{self.agent}/invoke")

        last_err: Exception | None = None
        response: httpx.Response | None = None
        for url in candidate_urls:
            try:
                resp = httpx.post(
                    url,
                    json=request.dict(),
                    headers=self._headers,
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                response = resp
                break
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else "?"
                if self.verbose:
                    body = e.response.text[:500] if e.response is not None else "<no body>"
                    # Include contiguous "{status} {body}" for compatibility with tests when verbose
                    last_err = AgentClientError(
                        f"HTTP {status} POST {url}: {body} | {status} {body}"
                    )
                else:
                    reason = e.response.reason_phrase if e.response is not None else ""
                    last_err = AgentClientError(f"HTTP {status} {reason}")
            except httpx.HTTPError as e:
                last_err = AgentClientError(f"Error POST {url}: {e}")

        if response is None:
            raise last_err or AgentClientError("Unknown HTTP error invoking agent")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # Convert the JSON formatted message to an AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # Yield the str token directly
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
        run_id: str | None = None,
        url_parameters: dict[str, Any] | None = None,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        Stream the agent's response synchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens, api_key=self.api_key)
        if thread_id:
            request.thread_id = thread_id
        if user_id:
            request.user_id = user_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if run_id:
            request.run_id = run_id
        if url_parameters:
            request.url_parameters = url_parameters
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.dict(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
        run_id: str | None = None,
        url_parameters: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        Stream the agent's response asynchronously.

        Each intermediate message of the agent process is yielded as an AnyMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming modelsas they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens, api_key=self.api_key)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        if run_id:
            request.run_id = run_id
        if url_parameters:
            request.url_parameters = url_parameters
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.dict(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        Create a feedback record for a run.

        This is a simple wrapper for the LangSmith create_feedback API, so the
        credentials can be stored and managed in the service rather than the client.
        See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.dict(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str, agent_id: str | None = None) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
        """
        request = ChatHistoryInput(thread_id=thread_id, agent_id=agent_id)
        try:
            response = httpx.post(
                f"{self.base_url}/history",
                json=request.dict(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())


def transcribe(self, filename: str, file: bytes) -> str:
    if self.api_key is None:
        raise AgentClientError("API Key required")

    allowed_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".m4a": "audio/x-m4a",
    }

    filetype = Path(filename).suffix
    if filetype not in allowed_types:
        raise AgentClientError(f"File type {filetype} not allowed.")

    try:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        files = {"file": (filename, file, allowed_types[filetype])}
        response = httpx.post(
            f"{os.getenv('OWUI_AUDIO_API_URL')}/transcriptions",
            files=files,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        transcription = response.json()
        return transcription["text"]
    except Exception as e:
        raise AgentClientError(f"Error: {e}")
