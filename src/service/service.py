import json
import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangsmithClient

from agent import research_assistant
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput,
    convert_message_content_to_string,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.")),
    ],
) -> None:
    if http_auth.credentials != os.getenv("AUTH_SECRET"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


bearer_depend = [Depends(verify_bearer)] if os.getenv("AUTH_SECRET") else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        research_assistant.checkpointer = saver
        app.state.agent = research_assistant
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=bearer_depend)


def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    input_message = ChatMessage(type="human", content=user_input.message)
    kwargs = {
        "input": {"messages": [input_message.to_langchain()]},
        "config": RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model}, run_id=run_id
        ),
    }
    return kwargs, run_id


def _remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


@router.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    agent: CompiledStateGraph = app.state.agent
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = ChatMessage.from_langchain(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: CompiledStateGraph = app.state.agent
    kwargs, run_id = _parse_input(user_input)

    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events(**kwargs, version="v2"):
        if not event:
            continue

        # Yield messages written to the graph state after node execution finishes.
        if (
            event["event"] == "on_chain_end"
            # on_chain_end gets called a bunch of times in a graph execution
            # This filters out everything except for "graph node finished"
            and any(t.startswith("graph:step:") for t in event.get("tags", []))
            and "messages" in event["data"]["output"]
        ):
            new_messages = event["data"]["output"]["messages"]
            for message in new_messages:
                try:
                    chat_message = ChatMessage.from_langchain(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

        # Yield tokens streamed from LLMs.
        if (
            event["event"] == "on_chat_model_stream"
            and user_input.stream_tokens
            and "llama_guard" not in event.get("tags", [])
        ):
            content = _remove_tool_calls(event["data"]["chunk"].content)
            if content:
                # Empty content in the context of OpenAI usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content.
                yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
            continue

    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream_agent(user_input: StreamInput) -> StreamingResponse:
    """
    Stream the agent's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    return StreamingResponse(message_generator(user_input), media_type="text/event-stream")


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    agent: CompiledStateGraph = app.state.agent
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = []
        for message in messages:
            chat_messages.append(ChatMessage.from_langchain(message))
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)
