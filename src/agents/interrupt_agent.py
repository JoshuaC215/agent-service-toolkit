import logging
from datetime import datetime
from typing import Any

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings

# Added logger
logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    birthdate: datetime | None


def wrap_model(
    model: BaseChatModel | Runnable[LanguageModelInput, Any], system_prompt: BaseMessage
) -> RunnableSerializable[AgentState, Any]:
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


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


async def determine_birthdate(
    state: AgentState, config: RunnableConfig, store: BaseStore
) -> AgentState:
    """This node examines the conversation history to determine user's birthdate, checking store first."""

    # Attempt to get user_id for unique storage per user
    user_id = config["configurable"].get("user_id")
    logger.info(f"[determine_birthdate] Extracted user_id: {user_id}")
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

            if user_data and user_data.value.get("birthdate"):
                # Convert ISO format string back to datetime object
                birthdate_str = user_data.value["birthdate"]
                birthdate = datetime.fromisoformat(birthdate_str) if birthdate_str else None
                # We already have the birthdate, return it
                logger.info(
                    f"[determine_birthdate] Found birthdate in store for user {user_id}: {birthdate}"
                )
                return {
                    "birthdate": birthdate,
                    "messages": [],
                }
        except Exception as e:
            # Log the error or handle cases where the store might be unavailable
            logger.error(f"Error reading from store for namespace {namespace}, key {key}: {e}")
            # Proceed with extraction if read fails
            pass
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
    response: BirthdateExtraction = await model_runnable.ainvoke(state, config)

    # If no birthdate found after extraction attempt, interrupt
    if response.birthdate is None:
        birthdate_input = interrupt(f"{response.reasoning}\nPlease tell me your birthdate?")
        # Re-run extraction with the new input
        state["messages"].append(HumanMessage(birthdate_input))
        # Note: Recursive call might need careful handling of depth or state updates
        return await determine_birthdate(state, config, store)

    # Birthdate found - convert string to datetime
    try:
        birthdate = datetime.fromisoformat(response.birthdate)
    except ValueError:
        # If parsing fails, ask for clarification
        birthdate_input = interrupt(
            "I couldn't understand the date format. Please provide your birthdate in YYYY-MM-DD format."
        )
        # Re-run extraction with the new input
        state["messages"].append(HumanMessage(birthdate_input))
        # Note: Recursive call might need careful handling of depth or state updates
        return await determine_birthdate(state, config, store)

    # Store the newly extracted birthdate only if we have a user_id
    if user_id and namespace:
        # Convert datetime to ISO format string for JSON serialization
        birthdate_str = birthdate.isoformat() if birthdate else None
        try:
            await store.aput(namespace, key, {"birthdate": birthdate_str})
        except Exception as e:
            # Log the error or handle cases where the store write might fail
            logger.error(f"Error writing to store for namespace {namespace}, key {key}: {e}")

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
