import json
import logging
from typing import Any, cast

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

from agents.helpers import built_finish_msg_and_link, interrupt_and_append, wrap_model
from core import get_model
from core.settings import settings
from variants.variant_config import VariantConfig

from .utils import load_prompt

# Logger setup
logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    questions: list[str]
    answers: list[str]
    category: str | None
    finished: bool
    llm_model: Any | None


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
        cfg = config.get("configurable", {})
        state["llm_model"] = get_model(
            cfg.get("model", settings.DEFAULT_MODEL),
            cfg.get("owui_token", ""),
        )
    llm = cast(BaseChatModel | Runnable[LanguageModelInput, Any], state.get("llm_model"))
    # Use prompts directory instead of agents_questions
    prompt_template = load_prompt(promptfile)
    system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
    llm_response = None
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
    if llm_response is None:
        question_text = "Es gab ein Problem bei der Fragegenerierung. Bitte beschreiben Sie kurz Ihr aktuelles Niveau."
    else:
        question_text = llm_response.content.strip()
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    questions.append(question_text)
    state["messages"].append(AIMessage(content=question_text))
    user_input = interrupt_and_append(state, question_text)
    cfg2 = config.get("configurable", {})
    state["llm_model"] = get_model(
        cfg2.get("model", settings.DEFAULT_MODEL),
        cfg2.get("owui_token", ""),
    )
    answers.append(user_input)
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

            prompt_template = load_prompt("categorize_user_prompt.txt")
            system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
            llm = cast(BaseChatModel | Runnable[LanguageModelInput, Any], state.get("llm_model"))
            llm_response = None
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
            if llm_response is None:
                return {"category": "Beginner", "finished": True}
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
    base, md_link = built_finish_msg_and_link(category, url_parameters)
    msg = f"{base}\n\n{md_link}"

    # Teaser toggling via VariantConfig if provided; default True when absent/invalid
    teaser_active = True
    try:
        variant = config.get("configurable", {}).get("variant")
        if variant:
            teaser_active = VariantConfig(variant).get("teaser_active", True)
    except Exception:
        teaser_active = True

    if teaser_active:
        msg = msg + "\n\nEs folgen entsprechende nächste Schritte."

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
