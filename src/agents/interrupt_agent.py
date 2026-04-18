import logging
from datetime import UTC, datetime

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from agents.helpers import interrupt_and_append, wrap_model
from core import get_model, settings

# Added logger
logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    birthdate: datetime | None


background_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
Provide a one sentence summary of the origin of zodiac signs.
Don't tell the user what their sign is, you are just demonstrating your knowledge on the topic.
""")


async def background(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node is to demonstrate doing work before the interrupt"""

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, background_prompt.format())
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


birthdate_extraction_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert at extracting birthdates from conversational text.

Rules for extraction:
- Look for user messages that mention birthdates
- Consider various date formats (MM/DD/YYYY, YYYY-MM-DD, Month Day, Year)
- Validate that the date is reasonable (not in the future)
- If no clear birthdate was provided by the user, return None
""")


class BirthdateExtraction(BaseModel):
    birthdate: str | None = Field(
        description="The extracted birthdate in YYYY-MM-DD format. If no birthdate is found, this should be None."
    )
    reasoning: str = Field(
        description="Explanation of how the birthdate was extracted or why no birthdate was found"
    )


def normalize_birthdate(date_str: str) -> datetime:
    """
    Parse and validate a birthdate string, normalizing to a naive datetime at midnight.
    - Accepts ISO 8601 (YYYY-MM-DD or full datetime, with or without timezone)
    - Also tries common formats: MM/DD/YYYY, 'Month Day, Year', 'Mon Day, Year'
    - Ensures the date is not in the future
    Raises ValueError for invalid formats or future dates.
    """
    if date_str is None:
        raise ValueError("No date string provided")

    s = str(date_str).strip()
    dt: datetime | None = None

    # 1) ISO 8601 first (handles YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS[.fff][+TZ])
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        dt = None

    # 2) Try common human formats if ISO failed
    if dt is None:
        from datetime import datetime as _dt

        for fmt in ("%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                dt = _dt.strptime(s, fmt)
                break
            except Exception:
                continue  # nosec B112

    if dt is None:
        raise ValueError(f"Unrecognized date format: {s!r}")

    # Normalize timezone: convert tz-aware to UTC then drop tzinfo; keep only date component
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)

    # Truncate to midnight of that date
    normalized = datetime(dt.year, dt.month, dt.day)

    # Validate not in the future
    today = datetime.now(UTC).date()
    if normalized.date() > today:
        raise ValueError("Birthdate cannot be in the future")

    return normalized


MAX_BIRTHDATE_ATTEMPTS = 3


async def determine_birthdate(
    state: AgentState, config: RunnableConfig, store: BaseStore
) -> AgentState:
    """This node examines the conversation history to determine user's birthdate, checking store first."""

    # Attempt to get user_id for unique storage per user
    user_id = config["configurable"].get("user_id")
    logger.debug(f"[determine_birthdate] Extracted user_id: {user_id}")
    namespace = None
    key = "birthdate"
    birthdate = None  # Initialize birthdate

    if user_id:
        # Use user_id in the namespace to ensure uniqueness per user
        namespace = (user_id,)

        # Check if we already have the birthdate in the store for this user
        try:
            result = await store.aget(namespace, key=key)
            # Handle cases where store.aget might return Item directly or a list
            user_data = None
            if result:  # Check if anything was returned
                if isinstance(result, list):
                    if result:  # Check if list is not empty
                        user_data = result[0]
                else:  # Assume it's the Item object directly
                    user_data = result

            if user_data:
                value = getattr(user_data, "value", None)
                if isinstance(value, dict) and value.get("birthdate"):
                    birthdate_str = value["birthdate"]
                    birthdate = normalize_birthdate(birthdate_str)
                    # We already have the birthdate, return it
                    logger.info(
                        f"[determine_birthdate] Found birthdate in store for user {user_id}"
                    )
                    return {
                        "birthdate": birthdate,
                        "messages": [],
                    }
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.warning(
                f"[determine_birthdate] Store read failed for namespace={namespace}, key={key}: {type(e).__name__}: {e}"
            )
        except Exception as e:
            logger.error(
                f"[determine_birthdate] Unexpected error reading store for namespace={namespace}, key={key}: {type(e).__name__}: {e}",
                exc_info=True,
            )
    else:
        # If no user_id, we cannot reliably store/retrieve user-specific data.
        # Consider logging this situation.
        logger.warning(
            "Warning: user_id not found in config. Skipping persistent birthdate storage/retrieval for this run."
        )

    # If birthdate wasn't retrieved from store, proceed with extraction
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m.with_structured_output(BirthdateExtraction), birthdate_extraction_prompt.format()
    ).with_config(tags=["skip_stream"])
    birthdate = None
    for attempt in range(MAX_BIRTHDATE_ATTEMPTS):
        response: BirthdateExtraction = await model_runnable.ainvoke(state, config)
        # If no birthdate found after extraction attempt, interrupt
        if response.birthdate is None:
            interrupt_and_append(
                state, f"{response.reasoning}\nPlease tell me your birthdate in YYYY-MM-DD format."
            )
            continue
        # Birthdate found - normalize and validate
        try:
            birthdate = normalize_birthdate(response.birthdate)
            break
        except ValueError:
            # If parsing/validation fails, ask for clarification
            interrupt_and_append(
                state,
                "I couldn't understand the date. Please provide your birthdate in YYYY-MM-DD format (e.g., 1990-04-23).",
            )
            logger.info("birthdate validation error")
            continue  # nosec B112: iterative retry is intentional; handled with user prompt and bounded by MAX_BIRTHDATE_ATTEMPTS

    # Give up after max attempts without raising or recursing
    if birthdate is None:
        logger.info(
            f"[determine_birthdate] Max attempts reached without valid birthdate for user_id={user_id}"
        )
        return {
            "birthdate": None,
            "messages": [],
        }

    # Store the newly extracted birthdate only if we have a user_id
    if user_id and namespace:
        # Convert datetime to ISO format string for JSON serialization
        birthdate_str = birthdate.isoformat() if birthdate else None
        try:
            await store.aput(namespace, key, {"birthdate": birthdate_str})
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.warning(
                f"[determine_birthdate] Store write failed for namespace={namespace}, key={key}: {type(e).__name__}: {e}"
            )
        except Exception as e:
            logger.error(
                f"[determine_birthdate] Unexpected error writing store for namespace={namespace}, key={key}: {type(e).__name__}: {e}",
                exc_info=True,
            )

    # Return the determined birthdate (either from store or extracted)
    logger.info(f"[determine_birthdate] Returning birthdate {birthdate} for user {user_id}")
    return {
        "birthdate": birthdate,
        "messages": [],
    }


response_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant.

Known information:
- The user's birthdate is {birthdate_str}

User's latest message: "{last_user_message}"

Based on the known information and the user's message, provide a helpful and relevant response.
If the user asked for their birthdate, confirm it.
If the user asked for their zodiac sign, calculate it and tell them.
Otherwise, respond conversationally based on their message.
""")


async def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generates the final response based on the user's query and the available birthdate."""
    birthdate = state.get("birthdate")
    if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        last_user_message = state["messages"][-1].content
    else:
        last_user_message = ""

    if not birthdate:
        # This should ideally not be reached if determine_birthdate worked correctly and possibly interrupted.
        # Handle cases where birthdate might still be missing.
        return {
            "messages": [
                AIMessage(
                    content="I couldn't determine your birthdate. Could you please provide it?"
                )
            ]
        }

    birthdate_str = birthdate.strftime("%B %d, %Y")  # Format for display

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m, response_prompt.format(birthdate_str=birthdate_str, last_user_message=last_user_message)
    )
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("background", background)
agent.add_node("determine_birthdate", determine_birthdate)
agent.add_node("generate_response", generate_response)

agent.set_entry_point("background")
agent.add_edge("background", "determine_birthdate")
agent.add_edge("determine_birthdate", "generate_response")
agent.add_edge("generate_response", END)

interrupt_agent = agent.compile()
interrupt_agent.name = "interrupt-agent"
