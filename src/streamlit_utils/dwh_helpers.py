import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import streamlit as st

# Modulkonstante für die allgemeine Fehleranzeige beim Laden der Fragen
ERROR_LOADING_QUESTIONS_MSG = "Beim Laden der Fragen ist ein unerwarteter Fehler aufgetreten."

# Set up module-level logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Question limit for questionnaires (configurable via environment variable)
QUESTION_LIMIT = int(os.environ.get("DWH_QUESTION_LIMIT", 10))


@dataclass
class DWHQuestion:
    """
    Represents a single DWH readiness question.

    Args:
        text (str): The question text.
    """

    text: str

    @staticmethod
    def from_dict(obj: dict) -> "DWHQuestion":
        """
        Create a DWHQuestion instance from a dictionary.

        Args:
            obj (dict): Dictionary with a 'text' field.

        Returns:
            DWHQuestion: The created question object.

        Raises:
            ValueError: If the input is not valid.
        """
        if not isinstance(obj, dict) or "text" not in obj or not isinstance(obj["text"], str):
            raise ValueError("Each question must be a dict with a 'text' field of type str.")
        return DWHQuestion(text=obj["text"])


@dataclass
class DWHReadinessCheckObject:
    """
    Represents a DWH readiness check object with title, scale, and questions.

    Args:
        title (str): The title of the check.
        description (str): The description of the check.
        scale (Literal["normal", "inverse"]): The scoring scale.
        questions (list[DWHQuestion]): List of questions.
    """

    title: str
    description: str
    scale: Literal["normal", "inverse"]
    questions: list[DWHQuestion] = field(default_factory=list)

    @staticmethod
    def from_dict(obj: dict) -> "DWHReadinessCheckObject":
        """
        Create a DWHReadinessCheckObject from a dictionary.

        Args:
            obj (dict): Dictionary with keys 'title', 'description', 'scale', and 'questions'.

        Returns:
            DWHReadinessCheckObject: The created readiness check object.

        Raises:
            ValueError: If the input is not valid.
        """
        if not isinstance(obj, dict):
            raise ValueError("Root object must be a dict.")
        title = obj.get("title", "")
        description = obj.get("description", "")
        if not isinstance(title, str):
            title = ""
        scale = obj.get("scale", "normal")
        if scale not in ("normal", "inverse"):
            scale = "normal"
        questions = obj.get("questions", [])
        if not isinstance(questions, list):
            raise ValueError("questions must be a list.")
        if len(questions) != QUESTION_LIMIT:
            raise ValueError(f"questions must have {QUESTION_LIMIT} items, got {len(questions)}")
        question_objs = [DWHQuestion.from_dict(q) for q in questions]
        return DWHReadinessCheckObject(
            title=title, description=description, scale=scale, questions=question_objs
        )


def load_questions_from_json(json_filename: str) -> DWHReadinessCheckObject | None:
    """
    Load a DWHReadinessCheckObject from a JSON file.

    Args:
        json_filename (str): The filename of the JSON file containing questions.

    Returns:
        DWHReadinessCheckObject or None: The questionnaire object, or None if loading/validation fails.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    questions_path = os.path.join(base_dir, "agents", "agents_questions", json_filename)
    try:
        with open(questions_path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded questions from {questions_path}")
        return DWHReadinessCheckObject.from_dict(data)
    except ValueError as ve:
        logger.error(f"Invalid JSON in {json_filename}: {ve}")
        st.error(ERROR_LOADING_QUESTIONS_MSG)
        return None
    except Exception as e:
        logger.error(f"Error loading questions from {json_filename}: {e}")
        st.error(ERROR_LOADING_QUESTIONS_MSG)
        return None


questions_for_dwh_readiness_challenge: DWHReadinessCheckObject | None = load_questions_from_json(
    "dwh_readiness_challenge_questions.json"
)
questions_for_dwh_readiness_satisfaction: DWHReadinessCheckObject | None = load_questions_from_json(
    "dwh_readiness_satisfaction_questions.json"
)


challenge_eval_map: dict[tuple[int, int], tuple[str, str]] = {
    (0, 7): (
        "warning",
        "**Hoher Handlungsbedarf:** Es gibt massive Engpässe im aktuellen Datenmanagement. Das Unternehmen ist stark eingeschränkt und kann aus Daten kaum Mehrwert ziehen. Eine grundlegende Neuausrichtung ist erforderlich.",
    ),
    (8, 14): (
        "info",
        "**Mäßiger Handlungsbedarf:** Einige Herausforderungen bestehen, aber das Unternehmen ist teilweise in der Lage, mit Daten zu arbeiten. Optimierungen sind notwendig, um Datenmanagement und Analysen effizienter zu gestalten.",
    ),
    (15, 20): (
        "success",
        "**Geringer Handlungsbedarf:** Es gibt nur wenige Probleme mit der Datenverarbeitung. Das Unternehmen ist bereits gut aufgestellt, sollte aber spezifische Engpässe weiter verbessern.",
    ),
}
satisfaction_eval_map: dict[tuple[int, int], tuple[str, str]] = {
    (0, 7): (
        "warning",
        "**Schwache Dateninfrastruktur:** Die bestehenden Systeme sind nicht ausreichend, um eine datengetriebene Strategie zu unterstützen. Prozesse sind ineffizient, und Entscheidungen basieren oft auf unvollständigen oder fehlerhaften Daten.",
    ),
    (8, 14): (
        "info",
        "**Mittlere Dateninfrastruktur:** Es gibt bereits gute Ansätze, aber die bestehenden Lösungen zeigen Schwächen auf. Skalierbarkeit, Datenintegration und Automatisierung müssen verbessert werden, um nachhaltige Erfolge zu erzielen..",
    ),
    (15, 20): (
        "success",
        "**Starke Dateninfrastruktur:** Das Unternehmen verfügt über eine solide Infrastruktur. Es gibt noch Optimierungs-potenziale, aber grundsätzlich sind die Systeme leistungsfähig und zukunftssicher.",
    ),
}


def compute_and_show_results(score: int, eval_map: dict[tuple[int, int], tuple[str, str]]) -> None:
    """
    Compute the evaluation result based on the score and show the result using Streamlit.

    Args:
        score (int): The total score to evaluate.
        eval_map (dict[tuple[int, int], tuple[str, str]]): Mapping of score ranges to (message type, message).

    Returns:
        None
    """
    st.markdown("---")
    st.write(f"### Ihre Gesamtpunktzahl: {score}")
    selected_msg = None
    for (low, high), (msg_type, msg) in eval_map.items():
        if low <= score <= high:
            selected_msg = msg
            if msg_type == "warning":
                st.warning(msg)
            elif msg_type == "info":
                st.info(msg)
            elif msg_type == "success":
                st.success(msg)
            else:
                st.write(msg)
            break
    else:
        st.write("Punktzahl außerhalb des erwarteten Bereichs")
    if selected_msg is not None:
        if "Dateninfrastruktur" in selected_msg:
            st.session_state["satisfaction_final_result"] = selected_msg
        else:
            st.session_state["challenge_final_result"] = selected_msg


def validate_and_extract_questions(
    dwhReadinessCheckObject: DWHReadinessCheckObject | None,
) -> tuple[str, str, str, list[str]]:
    """
    Validate the questionnaire object and extract title, description, scale, and question texts.

    Args:
        dwhReadinessCheckObject (DWHReadinessCheckObject | None): The questionnaire object to display.

    Returns:
        tuple: (title, description, scale, question_texts)

    Raises:
        ValueError: If the questionnaire object is invalid.
    """
    if not isinstance(dwhReadinessCheckObject, DWHReadinessCheckObject):
        logger.error("Ungültiges Fragenformat übergeben. Erwartet: DWHReadinessCheckObject.")
        st.warning(ERROR_LOADING_QUESTIONS_MSG)
        raise ValueError("Invalid questionnaire object.")

    title = dwhReadinessCheckObject.title
    description = dwhReadinessCheckObject.description
    scale = dwhReadinessCheckObject.scale
    question_objs = dwhReadinessCheckObject.questions
    question_texts = [q.text for q in question_objs]

    if not question_texts or len(question_texts) != QUESTION_LIMIT:
        logger.error(
            f"Ungültige Anzahl an Fragen: Erwartet {QUESTION_LIMIT}, erhalten: {len(question_texts) if question_texts else 0}."
        )
        st.warning(
            f"Ungültige Anzahl an Fragen gefunden. Erwartet: {QUESTION_LIMIT}, erhalten: {len(question_texts) if question_texts else 0}. "
            "Bitte laden Sie die Seite neu oder wenden Sie sich an den Administrator."
        )
        raise ValueError("Invalid number of questions.")

    return title, description, scale, question_texts


def show_questionnaire_title_and_description(title: str, description: str) -> None:
    """
    Show the questionnaire title and description in Streamlit.

    Args:
        title (str): The questionnaire title.
        description (str): The questionnaire description.
    """
    if title:
        st.title(title)
        st.write(description)


def invert_points_dict_if_needed(scale: str, points_dict: dict[str, int]) -> dict[str, int]:
    """
    Invert the points dictionary if the scale is 'inverse'.

    Args:
        scale (str): The scoring scale.
        points_dict (dict[str, int]): The points mapping.

    Returns:
        dict[str, int]: The (possibly inverted) points mapping.
    """
    if scale == "inverse":
        return {k: 2 - v for k, v in points_dict.items()}
    return points_dict


def initialize_session_state(prefix: str, question_texts: list[str]) -> str:
    """
    Initialize session state for the questionnaire.

    Args:
        prefix (str): Prefix for session state keys.
        question_texts (list[str]): List of question texts.

    Returns:
        str: The question index key.
    """
    question_index_key = f"question_index_{prefix}"
    if question_index_key not in st.session_state:
        st.session_state[question_index_key] = 0
    for i in range(len(question_texts)):
        key = f"{prefix}_{i}"
        if key not in st.session_state:
            st.session_state[key] = None
    return question_index_key


def advance_index(prefix: str, question_index_key: str, question_texts: list[str]) -> None:
    """
    Advance the current question index if the current question is answered.

    Args:
        prefix (str): Prefix for session state keys.
        question_index_key (str): The session state key for the question index.
        question_texts (list[str]): List of question texts.
    """
    i = st.session_state.current_radio_key
    if st.session_state[f"{prefix}_{i}"] is not None:
        if st.session_state[question_index_key] < len(question_texts) - 1:
            st.session_state[question_index_key] += 1


def render_questions(
    prefix: str,
    question_index_key: str,
    question_texts: list[str],
    answer_options: list[str],
) -> None:
    """
    Render the questions and answer options in Streamlit.

    Args:
        prefix (str): Prefix for session state keys.
        question_index_key (str): The session state key for the question index.
        question_texts (list[str]): List of question texts.
        answer_options (list[str]): List of answer options.
    """
    for i in range(st.session_state[question_index_key] + 1):
        col1, col2 = st.columns([5, 4])
        with col1:
            st.write(question_texts[i])
        with col2:
            st.session_state.current_radio_key = i
            st.radio(
                label=f"Antwort für Frage {i + 1}",
                options=answer_options,
                horizontal=True,
                key=f"{prefix}_{i}",
                label_visibility="collapsed",
                on_change=lambda: advance_index(prefix, question_index_key, question_texts),
            )


def all_questions_answered(prefix: str, question_texts: list[str]) -> bool:
    """
    Check if all questions have been answered.

    Args:
        prefix (str): Prefix for session state keys.
        question_texts (list[str]): List of question texts.

    Returns:
        bool: True if all questions are answered, False otherwise.
    """
    return all(st.session_state[f"{prefix}_{i}"] is not None for i in range(len(question_texts)))


def calculate_total_points(
    prefix: str, question_texts: list[str], points_dict: dict[str, int]
) -> int:
    """
    Calculate the total points for the questionnaire.

    Args:
        prefix (str): Prefix for session state keys.
        question_texts (list[str]): List of question texts.
        points_dict (dict[str, int]): Mapping of answers to points.

    Returns:
        int: The total points.
    """
    return sum(
        points_dict.get(st.session_state[f"{prefix}_{i}"], 0) for i in range(len(question_texts))
    )


def render_questionnaire(
    dwhReadinessCheckObject: DWHReadinessCheckObject | None,
    prefix: str,
    toggle_next: Callable[..., None] | None = None,
    eval_map: dict = {},
    next_label: str = "",
    answer_options: list[str] = ["Ja", "Teilweise", "Nein"],
    points_dict: dict[str, int] = {"Ja": 2, "Teilweise": 1, "Nein": 0},
) -> None:
    """
    Render a questionnaire in the Streamlit UI.

    Args:
        dwhReadinessCheckObject (DWHReadinessCheckObject | None): The questionnaire object to display.
        prefix (str): Prefix for session state keys.
        toggle_next (Callable[..., None] | None, optional): Function to call when moving to the next section.
        eval_map (dict, optional): Evaluation map for scoring.
        next_label (str, optional): Label for the next button.
        answer_options (list[str], optional): List of answer options.
        points_dict (dict[str, int], optional): Mapping of answers to points.

    Returns:
        None
    """
    try:
        title, description, scale, question_texts = validate_and_extract_questions(
            dwhReadinessCheckObject
        )
    except ValueError:
        return

    show_questionnaire_title_and_description(title, description)
    points_dict = invert_points_dict_if_needed(scale, points_dict)
    question_index_key = initialize_session_state(prefix, question_texts)
    render_questions(prefix, question_index_key, question_texts, answer_options)

    if all_questions_answered(prefix, question_texts):
        total_points = calculate_total_points(prefix, question_texts, points_dict)
        if eval_map:
            compute_and_show_results(total_points, eval_map)
        if isinstance(next_label, str) and next_label != "":
            if st.button(next_label):
                if toggle_next is not None:
                    toggle_next()
                st.rerun()


def render_slider_section() -> None:
    """
    Render the slider section for technical affinity and company size in the Streamlit UI.

    Returns:
        None
    """
    st.write(
        "Vor Beginn des DWH Readiness-Checks bitten wir Sie, die technische Affinität und Mitarbeiteranzahl "
        "Ihres Unternehmens anzugeben. Diese Informationen helfen uns, die Ergebnisse optimal auf Ihre "
        "Bedürfnisse zuzuschneiden und den Weg zu einer verbesserten Datenstrategie zu ebnen."
    )
    tech_affinity = st.select_slider(
        "Wie technisch affin ist Ihre Firma (1=nicht technisch bzw. 5= sehr technisch affin)?",
        options=[1, 2, 3, 4, 5],
        value=3,
    )
    st.write(f"Die technische Affinität Ihrer Firma ist: {tech_affinity}")
    company_size = st.slider("Wie viele Mitarbeiter hat ihre Firma?", 0, 1000, 500)
    st.write(f"Die Größe ihres Unternehmens beträgt {company_size} Mitarbeiter!")
    st.session_state["tech_affinity"] = tech_affinity
    st.session_state["company_size"] = company_size
