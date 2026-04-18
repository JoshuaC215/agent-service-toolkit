import json
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from agents.helpers import built_finish_msg_and_link, interrupt_and_append, wrap_model
from core import get_model
from schema.models import OpenwebuiModelName

logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    testid: int
    question: str
    question_index: int
    answers: list[dict[str, Any]]
    category: str | None
    finished: bool


memory = MemorySaver()

# Lazy LLM model initialization to avoid import-time failures when env is not ready
_llm_model: BaseChatModel | None = None


def _get_llm_model() -> BaseChatModel:
    """
    Lazily initialize and cache the LLM model.
    Avoids raising at import time when environment variables (e.g., OpenWebUI URL/token) are not yet set.
    """
    global _llm_model
    if _llm_model is None:
        _llm_model = get_model(OpenwebuiModelName.GPT_4O)
    return _llm_model


# --- Questions (mix of open/closed, 3-4 about KI use cases) ---
QUESTIONS = [
    # General AI knowledge
    {
        "text": "Haben Sie bereits praktische Erfahrungen mit KI-Technologien gesammelt? (Ja/Nein)",
        "type": "closed",
    },
    {
        "text": "Welche Anwendungen von KI finden Sie in Ihrem aktuellen Arbeitsbereich besonders interessant?",
        "type": "open_usecase",
    },
    {
        "text": "Können Sie ein Projekt beschreiben, bei dem Sie KI-Technologien eingesetzt haben? Falls nicht, geben Sie bitte an, welche Anwendungsfälle Sie interessieren würden.",
        "type": "open_usecase",
    },
    {
        "text": "Welche zukünftigen KI-Trends halten Sie für besonders relevant in Ihrer Branche?",
        "type": "open_usecase",
    },
    {
        "text": "Wie schätzen Sie Ihr theoretisches Wissen über KI ein? (Anfänger, Fortgeschritten, Experte)",
        "type": "closed",
    },
    {
        "text": "Welche Herausforderungen sehen Sie beim Einsatz von KI in Ihrem Arbeitsumfeld?",
        "type": "open",
    },
    {
        "text": "Nutzen Sie KI-Tools wie ChatGPT, Midjourney oder ähnliche regelmäßig?",
        "type": "closed",
    },
    {
        "text": "Falls Sie noch keine konkreten KI-Anwendungsfälle kennen: Interessieren Sie sich eher für Automatisierung, Datenanalyse, Kreativitätsunterstützung oder etwas anderes? (Bitte wählen Sie eine Option oder beschreiben Sie Ihr Interesse)",
        "type": "guided_usecase",
    },
    {
        "text": "Gibt es weitere Aspekte oder Fragen zu KI, die Sie besonders interessieren?",
        "type": "open",
    },
]


graph_builder = StateGraph(State)


async def ask_question(state: State) -> dict:
    """
    Generate and ask the next relevant question in the KI Skill Check dialog.

    This function constructs a dynamic system prompt based on the current state,
    including previous answers and the list of questions. It invokes the LLM to
    generate the next question, updates the state with the new question, and
    increments the testid counter.

    Args:
        state (State): The current state of the dialog.

    Returns:
        dict: {
            "messages": [AIMessage],  # The new AIMessage containing the next question
            "question": str,          # The text of the next question
            "testid": int             # The incremented test/question index
        }
    """
    # Always prepend a system prompt about KI
    # "Du bist ein KI-Skill-Check-Bot. "
    # "Stelle die nächste sinnvolle Frage aus dieser Liste, die noch nicht beantwortet wurde, "
    # "oder formuliere sie leicht um, falls sinnvoll. "
    # "Berücksichtige dabei explizit die letzte Antwort des Nutzers und stelle eine möglichst relevante, darauf aufbauende Frage aus der Liste "
    # "oder passe sie entsprechend an, sodass ein echter Dialog entsteht. "
    # "Wenn sinnvoll, stelle eine Rückfrage oder gehe auf Details aus der letzten Antwort ein. "
    idx = state.get("testid", 0)
    answers = state.get("answers", [])
    logger.info(f"LLM response: {state}")
    if not state.get("messages"):
        prev_answers = []
    else:
        prev_answers = answers
    questions_json = json.dumps(QUESTIONS, ensure_ascii=False)
    answers_json = json.dumps(prev_answers, ensure_ascii=False)
    system_prompt = SystemMessagePromptTemplate.from_template(
        """
        Du bist ein KI Skill Check Bot, der dazu entwickelt wurde, die Fähigkeiten und Kenntnisse der Nutzer:innen im Bereich Künstliche Intelligenz zu bewerten. Deine Aufgaben umfassen:

        1. **Start**
            Am Anfang bekommst du immer vom User drei Zahlen, die eine beschreibt, die Unternehmensgröße,
            die andere beschreibt die technische Affinität, wo bei (1=nicht technisch, 5=sehr technisch) der Firma.
            Die letzte ist das Datum von dem Skillcheck.

            Nimm folgenden Text und baue dort die technische Affinität, das Datum und die Firmengröße ein. 
            "Willkommen zum KI Skill Check mit dem roosi Skill-Companion! Ich werde Ihnen einige Fragen stellen, um Ihre Kenntnisse und Erfahrungen im Bereich Künstliche Intelligenz besser einschätzen zu können.\n\nFangen wir an: Wie würden Sie Künstliche Intelligenz in eigenen Worten beschreiben?"

        2. **Fragenstellung:**
            - Stelle 9 durchdachte Fragen, um das KI-Wissen und die praktischen Erfahrungen der Nutzer:innen zu ermitteln.
            - Kombiniere offene und geschlossenen Fragen auf eine sinnvolle Art und Weise.
            - Integriere 3-4 Fragen, die darauf abzielen, interessante KI-Anwendungsfälle der Nutzer:innen zu identifizieren. Beispiele hierfür könnten sein:
            - "Welche Anwendungen von KI finden Sie in Ihrem aktuellen Arbeitsbereich besonders interessant?"
            - "Können Sie ein Projekt beschreiben, bei dem Sie KI-Technologien eingesetzt haben?"
            - "Welche zukünftigen KI-Trends halten Sie für besonders relevant in Ihrer Branche?"
        - Sollte der Nutzer keine klare Vorstellung haben bzgl. Anwendungen, dann führe ihn mit Vorschlägen zu Interessanten und konkreten KI Anwendungsfällen durch und versuche eine Präferenz zu des Nutzers festzustellen. 

        3. **Kontext und Struktur:**
        - Sorge für eine klare und logische Abfolge der Fragen.
        - Stelle nicht alle Fragen auf einmal. Sondern Du stellst eine Frage und wartest die Antwort ab. Danach stellst Du erst die nächste Frage.
        - Gib den Nutzer:innen bei Bedarf zusätzliche Kontextinformationen, um Missverständnisse zu vermeiden.
        """
        + "Hier ist die Liste von möglichen Fragen (mit Typ):\n"
        f"{questions_json}\n\n"
        "Hier sind die bisherigen Antworten:\n"
        f"{answers_json}\n\n"
    )
    idx = idx + 1
    logger.info(f"idx: {idx}")
    config = RunnableConfig()
    llm_response = None
    try:
        model = _get_llm_model()
        try:
            model_runnable = wrap_model(model, str(system_prompt))
            llm_response = await model_runnable.ainvoke(state, config)
        except TypeError:
            logger.warning(
                "ask_question: wrap_model not supported for current model; falling back to direct ainvoke"
            )
            llm_response = await model.ainvoke(
                [SystemMessage(content=str(system_prompt)), *state.get("messages", [])],
                config=config,
            )
    except Exception:
        logger.exception("ask_question: LLM invocation failed")
    # Include company_size and tech_affinity in the LLM invocation
    question_text = (
        llm_response.content.strip()
        if llm_response and getattr(llm_response, "content", None)
        else "Können Sie bitte Ihre letzte Aussage präzisieren, damit ich die nächste Frage passend formulieren kann?"
    )
    return {
        "messages": [AIMessage(content=question_text)],
        "question": question_text,
        "testid": idx,
    }


async def process_answer(state: State) -> dict:
    """
    Process the user's answer to the last question.

    This function extracts the most recent HumanMessage from the state,
    appends the answer to the answers list, and returns the updated state.

    Args:
        state (State): The current state of the dialog.

    Returns:
        dict: {
            "answers": List[Dict[str, Any]]  # The updated list of answers with the new answer appended
        }
    """
    answers = state.get("answers", [])
    last_msgs = state.get("messages", [])
    question = state.get("question", "")
    logger.info(f"Answers: {answers}")
    user_msg = None
    for m in reversed(last_msgs):
        if isinstance(m, HumanMessage):
            user_msg = m
            break
    answer_text = user_msg.content if user_msg else ""
    answers.append(
        {
            "question": question,
            "answer": answer_text,
        }
    )
    logger.info(f"LLM response: {answers}")
    return {
        "answers": answers,
    }


async def assign_category(state: State) -> dict:
    """
    Assign a skill category ('Beginner', 'Advanced', 'Expert') based on user answers.

    This function formats all collected answers, builds a classification prompt,
    and invokes the LLM to assign a category. The result is added to the state.

    Args:
        state (State): The current state of the dialog.

    Returns:
        dict: {
            "category": str  # The assigned skill category ("Beginner", "Advanced", or "Expert")
        }
    """
    """
    Weist eine Kategorie ('Anfänger', 'Fortgeschritten', 'Experte') basierend auf den Antworten zu,
    indem ein LLM befragt wird.
    """
    # Antworten als Fließtext für das Prompt formatieren
    answers = state.get("answers", [])
    answer_texts = []
    for a in answers:
        q = a.get("question", "")
        ans = a.get("answer", "")
        answer_texts.append(f"Frage: {q}\nAntwort: {ans}")
    joined_answers = "\n\n".join(answer_texts)

    prompt = (
        "Du bist ein KI-Experte. Analysiere die folgenden Antworten eines Nutzers auf einen KI-Skill-Check "
        "und ordne den Nutzer einer der folgenden Kategorien zu: Beginner, Advanced, Expert. "
        "Antworte ausschließlich mit einer dieser Kategorien und ohne weitere Erklärung.\n\n"
        f"{joined_answers}\n\n"
        "Kategorie:"
    )
    config = RunnableConfig()
    messages = [SystemMessage(content=prompt)]
    try:
        model = _get_llm_model()
        llm_response = await model.ainvoke(messages, config=config)
        category = llm_response.content.strip()
    except Exception:
        logger.exception("assign_category: LLM invocation failed")
        category = "Beginner"
    return {"category": category}


async def finish(state: State, config: RunnableConfig) -> dict:
    """
    Finalize the skill check and present the result to the user.

    This function generates a closing message based on the assigned category,
    provides a resource link, and sets the finished flag in the state.
    """
    category = state.get("category", "Beginner")
    url_parameters = config.get("configurable", {}).get("url_parameters")
    base, md_link = built_finish_msg_and_link(category, url_parameters)
    msg = f"{base}\n\n{md_link}\n\nEs folgen entsprechende nächste Schritte."
    return {
        "messages": [AIMessage(content=msg)],
        "finished": True,
    }


async def after_process_answer(state: State) -> str:
    """
    Determine the next node after processing an answer.

    If fewer than 9 questions have been asked, route to 'interrupt_process'.
    Otherwise, route to 'categorize'.

    Args:
        state (State): The current state of the dialog.

    Returns:
        str: The name of the next node ("interrupt_process" or "categorize").
    """
    idx = state.get("testid", 0)
    if idx < 9:
        return "interrupt_process"
    elif idx >= 9:
        return "categorize"
    else:
        # Defensive fallback: always return a valid key
        import logging

        logging.getLogger(__name__).error(f"Unexpected idx value in after_process_answer: {idx}")
        return "categorize"


async def interrupt_process(state: State) -> dict:
    """
    Interrupt the current process and inform the user, following the pattern of interrupt_agent.

    This function triggers a LangGraph interrupt, appends a HumanMessage to the state
    with a user-friendly prompt, and returns the updated state to loop back to the next question.

    Args:
        state (State): The current state of the dialog.

    Returns:
        dict: The updated state with an interrupt message appended to "messages".
    """
    logger.info("Interrupting")
    # Use a user-friendly prompt for clarification, similar to interrupt_agent
    _ = interrupt_and_append(
        state,
        "Bitte präzisieren Sie Ihre letzte Antwort oder wählen Sie eine Option, damit ich fortfahren kann.",
    )
    return state


graph_builder.add_node("ask_question", ask_question)
graph_builder.add_node("process_answer", process_answer)
graph_builder.add_node("interrupt_process", interrupt_process)
graph_builder.add_node("categorize", assign_category)
graph_builder.add_node("finish", finish)

graph_builder.add_edge("ask_question", "process_answer")
graph_builder.add_edge("interrupt_process", "ask_question")
graph_builder.add_edge("categorize", "finish")

graph_builder.set_entry_point("ask_question")
graph_builder.add_conditional_edges(
    "process_answer",
    after_process_answer,
    {"interrupt_process": "interrupt_process", "categorize": "categorize"},
)

skillcompanion = graph_builder.compile(checkpointer=memory)
