from datetime import datetime
from typing import cast

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    birthdate: datetime | None


def wrap_model(
    model: BaseChatModel, system_prompt: SystemMessage
) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


background_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
Provide a one paragraph summary of the origin of zodiac signs.
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
    birthdate: datetime | None = Field(
        description="The extracted birthdate. If no birthdate is found, this should be None."
    )
    reasoning: str = Field(
        description="Explanation of how the birthdate was extracted or why no birthdate was found"
    )


async def determine_birthdate(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node examines the conversation history to determine user's birthdate.  If no birthdate is found, it will perform an interrupt before proceeding."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m.with_structured_output(BirthdateExtraction), birthdate_extraction_prompt.format()
    )
    response = await model_runnable.ainvoke(state, config)
    response = cast(BirthdateExtraction, response)

    # If no birthdate found, interrupt
    if response.birthdate is None:
        birthdate_input = interrupt(f"{response.reasoning}\n" "Please tell me your birthdate?")
        # Re-run extraction with the new input
        state["messages"].append(HumanMessage(birthdate_input))
        return await determine_birthdate(state, config)

    # Birthdate found
    return {
        "birthdate": response.birthdate,
    }


sign_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
What is the sign of somebody born on {birthdate}?
""")


async def determine_sign(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node determines the zodiac sign of the user based on their birthdate."""
    if not state.get("birthdate"):
        raise ValueError("No birthdate found in state")

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m, sign_prompt.format(birthdate=state["birthdate"].strftime("%Y-%m-%d"))
    )
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("background", background)
agent.add_node("determine_birthdate", determine_birthdate)
agent.add_node("determine_sign", determine_sign)

agent.set_entry_point("background")
agent.add_edge("background", "determine_birthdate")
agent.add_edge("determine_birthdate", "determine_sign")
agent.add_edge("determine_sign", END)

interrupt_agent = agent.compile(
    checkpointer=MemorySaver(),
)
interrupt_agent.name = "interrupt-agent"
