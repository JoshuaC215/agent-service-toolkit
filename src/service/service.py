import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langsmith import Client as LangsmithClient

from agents import DEFAULT_AGENT, agents
from core import settings
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    InterruptMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
    interrupt_from_call_tool
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Construct agent with Sqlite checkpointer
    # TODO: It's probably dangerous to share the same checkpointer on multiple agents
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        for a in agents.values():
            a.checkpointer = saver
        yield
    # context manager will clean up the AsyncSqliteSaver on exit


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])


def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], UUID]:
    run_id = user_input.run_id or uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    if user_input.type == "user":
        inputU = {"messages": [HumanMessage(content=user_input.message)]}
    else:
        if user_input.message == "aproved_tool":
            inputU = None # This means to continue graph execution after interruption
        elif user_input.message == "canceled_tool":
            inputU = {"messages": [ToolMessage(
                        tool_call_id=user_input.tool_call_id,
                        content=f"API call denied by user. Ask them about de reason or what else can you help them with.",
                        )]}
        else:
            inputU = {"messages": [
                            ToolMessage(
                                tool_call_id=user_input.tool_call_id,
                                content=f"API call denied by user.",
                                ),
                            HumanMessage(content=user_input.message)
                        ]}

    kwargs = {
            "input": inputU,
            "config": RunnableConfig(
                configurable={"thread_id": thread_id, "model": user_input.model}, run_id=run_id
            ),
        }
    return kwargs, run_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    agent: CompiledStateGraph = agents[agent_id]
    kwargs, run_id = _parse_input(user_input)
    try:
        response = await agent.ainvoke(**kwargs)
        output = langchain_to_chat_message(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: CompiledStateGraph = agents[agent_id]
    kwargs, run_id = _parse_input(user_input)

    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events(**kwargs, version="v2"):
        if not event:
            continue

        new_messages = []
        # Yield messages written to the graph state after node execution finishes.
        if (
            event["event"] == "on_chain_end"
            # on_chain_end gets called a bunch of times in a graph execution
            # This filters out everything except for "graph node finished"
            and any(t.startswith("graph:step:") for t in event.get("tags", []))
            and "messages" in event["data"]["output"]
        ):
            new_messages = event["data"]["output"]["messages"]

        # Also yield intermediate messages from agents.utils.CustomData.adispatch().
        if event["event"] == "on_custom_event" and "custom_data_dispatch" in event.get("tags", []):
            new_messages = [event["data"]]

        for message in new_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                logger.error(f"Error parsing message: {e}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
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
            content = remove_tool_calls(event["data"]["chunk"].content)
            if content:
                # Empty content in the context of OpenAI usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content.
                yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
            continue
    
    # Handle interruptions
    snapshot = await agent.aget_state(kwargs["config"])
    if snapshot.next: 
        try:
            ai_message = langchain_to_chat_message(snapshot.values["messages"][-1])
            ichat_message = interrupt_from_call_tool(call_tool=ai_message.tool_calls[0])
            ichat_message.run_id = str(run_id)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error parsing interrupt message: {e}'})}\n\n"
        yield f"data: {json.dumps({'type': 'interrupt', 'content': ichat_message.model_dump()})}\n\n"

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


@router.post(
    "/{agent_id}/stream", response_class=StreamingResponse, responses=_sse_response_example()
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


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
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: CompiledStateGraph = agents[DEFAULT_AGENT]
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(router)
