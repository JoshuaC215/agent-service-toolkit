import json
import logging
import os
from typing import Any, cast

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import Field

from core import get_model
from variants.variant_config import VariantConfig

from .utils import load_prompt

# Logger setup
logger = logging.getLogger(__name__)


class AgentState(MessagesState):
    questions: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    category: str | None
    finished: bool = False
    llm_model: Any | None


def wrap_model(
    model: BaseChatModel | Runnable[LanguageModelInput, Any], system_prompt: BaseMessage
) -> RunnableSerializable[AgentState, Any]:
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def ask_question(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Sequentially asks questions, waits for user responses, and stores the answers.

    Args:
        state: The current AgentState containing messages, questions, and answers.
        config: The RunnableConfig for the model invocation.

    Returns:
        AgentState: Updated state with new question, user answer, and messages.
    """
    answers = state.get("answers", [])

    variant_config = VariantConfig(config["configurable"].get("variant", None))
    questions_json = json.dumps(variant_config.get_or_fail("skill_questions"), ensure_ascii=False)
    question_count = variant_config.get_or_fail("question_limit")
    promptfile = variant_config.get_or_fail("ask_question_prompt_filename")

    answers_json = json.dumps(answers, ensure_ascii=False)
    model = state.get("llm_model", None)
    if model is None:
        state["llm_model"] = get_model(
            config["metadata"]["model"], config["metadata"]["owui_token"]
        )
    llm = cast(BaseChatModel | Runnable[LanguageModelInput, Any], state.get("llm_model"))
    # Use prompts directory instead of agents_questions
    prompt_template = load_prompt(f"../prompts/{promptfile}")
    system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
    try:
        model_runnable = wrap_model(
            llm,
            system_prompt.format(
                question_count=question_count,
                questions_json=questions_json,
                answers_json=answers_json,
            ),
        )
        llm_response = await model_runnable.ainvoke(state, config)
    except Exception as e:
        logger.error(f"Exception: {e}")
    # Include company_size and tech_affinity in the LLM invocation
    question_text = llm_response.content.strip()
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    questions.append(question_text)
    state["messages"].append(AIMessage(content=question_text))
    user_input = interrupt(question_text)
    state["llm_model"] = get_model(config["metadata"]["model"], config["metadata"]["owui_token"])
    answers.append(user_input)
    state["messages"].append(HumanMessage(user_input))
    return {
        "messages": state.get("messages", []),
        "questions": questions,
        "answers": answers,
        "llm_model": state.get("llm_model", None),
        "category": None,
        "finished": False,
    }


async def categorize_user(state: AgentState, config: RunnableConfig):
    """
    Categorizes the user based on their answers after all questions have been answered.

    Args:
        state: The current AgentState containing questions and answers.
        config: The RunnableConfig for the model invocation.

    Returns:
        dict: Updated state with the assigned category and finished flag if complete,
              otherwise returns the unchanged state.
    """
    answers_of_user = state.get("answers", [])
    questions_of_system = state.get("questions", [])

    variant_config = VariantConfig(config["configurable"].get("variant", None))
    question_limit = variant_config.get_or_fail("question_limit")

    if len(answers_of_user) == question_limit:
        if len(questions_of_system) != len(answers_of_user):
            raise ValueError("The number of answers and questions must be equal!")
        else:
            pairs_of_question_answers = {
                frage: antwort for frage, antwort in zip(questions_of_system, answers_of_user)
            }
            pairs_of_question_answers_json = json.dumps(
                pairs_of_question_answers, ensure_ascii=False, indent=4
            )

            prompt_template = load_prompt("../prompts/categorize_user_prompt.txt")
            system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
            llm = cast(BaseChatModel | Runnable[LanguageModelInput, Any], state.get("llm_model"))
            try:
                model_runnable = wrap_model(
                    llm,
                    system_prompt.format(
                        pairs_of_question_answers_json=pairs_of_question_answers_json,
                    ),
                )
                llm_response = await model_runnable.ainvoke(state, config)
            except Exception as e:
                logger.error(f"Exception: {e}")
            catagory = llm_response.content.strip()
            return {"category": catagory, "finished": True}
    return state


async def finish_skill_check(state: AgentState, config: RunnableConfig) -> dict:
    """
    Finalizes the skill check and presents the result to the user.

    Args:
        state: The current AgentState containing the assigned category.

    Returns:
        dict: Contains the final AIMessage with the result and resource link.
    """
    # Normalize category: treat None or empty as 'Beginner'
    cat_raw = state.get("category")
    if isinstance(cat_raw, str):
        category = cat_raw.strip() or "Beginner"
    else:
        category = "Beginner"

    url_parameters = config.get("configurable", {}).get("url_parameters")
    url = f"https://bento.roosi.ai/?kiskill={category}"
    if isinstance(url_parameters, dict):
        hubspot_id = url_parameters.get("hubspot_id")
        hubspot_id = str(hubspot_id).strip() if hubspot_id is not None else ""
        if hubspot_id:
            hubspot_url = os.getenv("HUBSPOT_URL", "https://hubspot.de")
            url = f"{hubspot_url.rstrip('/')}/?hubspot_id={hubspot_id}"
    md_link = f"[Weitere Informationen]({url})"

    msg = (
        f"Vielen Dank für Ihre Teilnahme!\n\n"
        f"Basierend auf Ihren Angaben ordne ich Sie der Kategorie **{category}** zu. "
        f"Diese Information wird nun an unser System weitergeleitet, um Ihnen passende Ressourcen bereitzustellen.\n\n"
    )

    teaser_message = f"{md_link}\n\nEs folgen entsprechende nächste Schritte."

    variant_config = VariantConfig(config["configurable"].get("variant", None))
    msg = msg + teaser_message if (variant_config.get("teaser_active", True)) else msg

    return {"messages": [AIMessage(content=msg)]}


async def ask_further_questions(state: AgentState) -> str:
    """
    Determines the next node after processing an answer.

    Args:
        state: The current AgentState containing the finished flag.

    Returns:
        str: The name of the next node ("finished" or "ask_question").
    """
    finished: bool = state.get("finished", False)
    if finished:
        return "finished"
    else:
        return "ask_question"


skill_companion_graph = StateGraph(AgentState)
skill_companion_graph.add_node("ask_question", ask_question)
skill_companion_graph.add_node("categorize", categorize_user)
skill_companion_graph.add_node("finish", finish_skill_check)

skill_companion_graph.add_edge(START, "ask_question")
skill_companion_graph.add_edge("ask_question", "categorize")
skill_companion_graph.add_edge("finish", END)
skill_companion_graph.add_conditional_edges(
    "categorize", ask_further_questions, {"finished": "finish", "ask_question": "ask_question"}
)

skillcompanion_interrupted = skill_companion_graph.compile()
skillcompanion_interrupted.name = "skillcompanion_interrupted"
