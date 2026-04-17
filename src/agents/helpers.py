from asyncio import sleep, wait_for
from collections.abc import Callable, Sequence
from os import getenv
from typing import Any, Literal
from urllib.parse import urlencode, urlparse, urlunparse

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSerializable
from langgraph.types import interrupt

from agents.llama_guard import LlamaGuardOutput, SafetyAssessment


def pending_tool_calls(state) -> Literal["tools", "done"]:
    """
    Route based on presence of tool calls in the last AIMessage.
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "done"


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    """
    Build a standardized AIMessage when unsafe content is detected.
    """
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


def check_safety(state) -> Literal["unsafe", "safe"]:
    """
    Conditional routing helper for LlamaGuard safety assessment.
    Expects `state["safety"]` to be a LlamaGuardOutput.
    """
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


def wrap_model(
    model: BaseChatModel | Runnable[LanguageModelInput, Any],
    system: str | BaseMessage | Callable[[dict], BaseMessage | Sequence[BaseMessage]] | None = None,
    *,
    tools: list[Any] | None = None,
    name: str = "StateModifier",
) -> RunnableSerializable[Any, Any]:
    """
    Universal wrapper that:
    - Optionally binds tools to a BaseChatModel.
    - Prepends either:
        - a static string SystemMessage,
        - a BaseMessage, or
        - messages returned by a callable(system) based on state.
    Returns a Runnable pipeline: preprocessor | model.
    """
    bound = model
    # Bind tools if available and model supports it
    if tools:
        try:
            if hasattr(model, "bind_tools"):
                bound = model.bind_tools(tools)  # type: ignore[attr-defined]
        except Exception:
            bound = model

    def build_messages(state: dict) -> list[BaseMessage]:
        # Determine prefix messages from 'system'
        if callable(system):
            prefix = system(state)
            # Validate callable return type strictly: BaseMessage or Sequence[BaseMessage]
            if isinstance(prefix, BaseMessage):
                return [prefix] + state["messages"]
            if isinstance(prefix, Sequence):
                # Ensure all elements in the sequence are BaseMessage instances
                if not all(isinstance(m, BaseMessage) for m in prefix):
                    raise TypeError(
                        "Callable 'system' must return a BaseMessage or a Sequence[BaseMessage]. "
                        "Found a sequence containing non-BaseMessage elements."
                    )
                return list(prefix) + state["messages"]
            raise TypeError(
                f"Callable 'system' must return a BaseMessage or a Sequence[BaseMessage], got {type(prefix)}"
            )
        elif system is None:
            return state["messages"]
        else:
            sys_msg = SystemMessage(content=system) if isinstance(system, str) else system
            if not isinstance(sys_msg, BaseMessage):
                raise TypeError(
                    f"'system' must be str or BaseMessage when not callable, got {type(system)}"
                )
            return [sys_msg] + state["messages"]

    preprocessor = RunnableLambda(build_messages, name=name)
    return preprocessor | bound


async def run_llamaguard(
    role: str,
    messages: list[AnyMessage],
    *,
    guard: Any | None = None,
    timeout: float | None = None,
    retries: int = 2,
    retry_backoff_seconds: float = 0.5,
) -> LlamaGuardOutput:
    """
    Run LlamaGuard asynchronously for a given role ("User" or "Agent") and message history.

    Enhancements:
    - Allows injecting a guard instance (for tests/mocking).
    - Optional timeout via asyncio.wait_for.
    - Retries on asyncio.TimeoutError with exponential backoff.
    """

    if guard is None:
        from agents.llama_guard import LlamaGuard

        guard = LlamaGuard()

    attempt = 0
    while True:
        try:
            coro = guard.ainvoke(role, messages)
            if timeout is not None and timeout > 0:
                return await wait_for(coro, timeout)
            return await coro
        except TimeoutError:
            if attempt >= retries:
                raise
            await sleep(min(retry_backoff_seconds * (2**attempt), 5.0))
            attempt += 1


def should_block(safety: LlamaGuardOutput) -> bool:
    """
    Convenience predicate to check whether content should be blocked.
    """
    return safety.safety_assessment == SafetyAssessment.UNSAFE


def interrupt_and_append(state: dict, prompt: str) -> str:
    """
    Trigger a LangGraph interrupt with the given prompt and append the user's response
    as a HumanMessage to the state's message list. Returns the user's input string.
    """
    user_input = interrupt(prompt)
    state["messages"].append(HumanMessage(user_input))
    return user_input


def built_finish_msg_and_link(category: str, url_parameters: dict | None = None) -> tuple[str, str]:
    """
    Build a standardized finish message prefix and a markdown link, handling HubSpot and Bento overrides.
    Priority:
    1) If url_parameters contains non-empty 'hubspot_id', use HUBSPOT_URL env (default 'https://hubspot.de')
       and build a link with ?hubspot_id=...
    2) Else build Bento link using BENTO_URL env (default 'https://bento.roosi.ai') and kiskill param:
       - If url_parameters contains non-empty 'kiskill', use it
       - Otherwise fallback to the provided category
    Returns:
      (base_message, md_link)
    """

    def _normalize_base_url(raw: str, fallback: str) -> str:
        """
        Ensure a sane http(s) base URL:
        - Trim spaces
        - Prepend https:// if scheme is missing
        - Validate scheme is http/https; otherwise fallback
        - Ensure path is at least '/'
        """
        raw = (raw or "").strip() or fallback
        # Prepend https:// if missing scheme
        parsed = urlparse(raw if "://" in raw else f"https://{raw.lstrip('/')}")
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            parsed = urlparse(fallback)
        path = parsed.path or "/"
        return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))

    def _build_url(base: str, params: dict[str, str], fallback: str) -> str:
        base_norm = _normalize_base_url(base, fallback)
        p = urlparse(base_norm)
        query = urlencode(params, doseq=False)
        return urlunparse((p.scheme, p.netloc, p.path or "/", "", query, ""))

    base_message = (
        "Vielen Dank für Ihre Teilnahme!\n\n"
        f"Basierend auf Ihren Angaben ordne ich Sie der Kategorie **{category}** zu. "
        "Diese Information wird nun an unser System weitergeleitet, um Ihnen passende Ressourcen bereitzustellen."
    )

    url = ""
    # 1) HubSpot override
    if isinstance(url_parameters, dict):
        hubspot_id = str(url_parameters.get("hubspot_id", "") or "").strip()
        if hubspot_id:
            hubspot_base = getenv("HUBSPOT_URL", "https://hubspot.de")
            url = _build_url(hubspot_base, {"hubspot_id": hubspot_id}, "https://hubspot.de")

    # 2) Bento default (with optional kiskill from url_parameters)
    if not url:
        bento_base = getenv("BENTO_URL", "https://bento.roosi.ai")
        kiskill_value: str | None = None
        if isinstance(url_parameters, dict):
            kiskill_value = str(url_parameters.get("kiskill", "") or "").strip() or None
        kiskill_value = kiskill_value or category
        # Use quote for path-like but we put it as a query parameter; urlencode handles quoting
        url = _build_url(bento_base, {"kiskill": kiskill_value}, "https://bento.roosi.ai")

    md_link = f"[Weitere Informationen]({url})"
    return base_message, md_link
