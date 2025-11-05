# --- Load question lists from JSON files ---
import json
import os
from collections.abc import Callable

import streamlit as st


def load_questions_from_json(json_filename: str) -> list[str]:
    """
    Load a list of questions from a JSON file.

    Args:
        json_filename (str): The filename of the JSON file containing questions.

    Returns:
        list[str]: A list of question texts. Returns an empty list if loading fails.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    questions_path = os.path.join(base_dir, "agents", "agents_questions", json_filename)
    try:
        with open(questions_path, encoding="utf-8") as f:
            data = json.load(f)
        questions = [q["text"] for q in data]
        if len(questions) != 10:
            error_msg = f"Die Fragenliste in {json_filename} enthält {len(questions)} Elemente, erwartet werden 10."
            st.error(error_msg)
            raise ValueError(error_msg)
        return questions
    except ValueError as ve:
        st.error(f"Verwendetes Json-File enthält zu wenige Fragen: {json_filename}: {ve}")
        return []
    except Exception as e:
        st.error(f"Fehler beim Laden der Fragen aus {json_filename}: {e}")
        return []

questions_for_dwh_readiness_challenge = load_questions_from_json("dwh_readiness_challenge_questions.json")
questions_for_dwh_readiness_satisfaction = load_questions_from_json("dwh_readiness_satisfaction_questions.json")

challenge_eval_map:dict[tuple[int,int],tuple[str,str]] = {
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
satisfaction_eval_map:dict[tuple[int,int],tuple[str,str]] = {
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
    for (low, high), (msg_type, msg) in eval_map.items():
        if low <= score <= high:
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
    if "Dateninfrastruktur" in msg:
        st.session_state["satisfaction_final_result"]= msg
    else:
        st.session_state["challenge_final_result"] = msg

def render_slider_section() -> None:
    """
    Render the slider section for technical affinity and company size in the Streamlit UI.

    Returns:
        None
    """
    st.write("Vor Beginn des DWH Readiness-Checks bitten wir Sie, die technische Affinität und Mitarbeiteranzahl Ihres Unternehmens anzugeben. Diese Informationen helfen uns, die Ergebnisse optimal auf Ihre Bedürfnisse zuzuschneiden und den Weg zu einer verbesserten Datenstrategie zu ebnen.")
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

def render_questionnaire(
    questions: list[str],
    prefix: str,
    toggle_next: Callable[..., None] | None = None,
    eval_map: dict | None = None,
    next_label: str | None = None,
    answer_options: list[str] | None = None,
    points_dict: dict[str, int] | None = None,
) -> None:
    """
    Render a questionnaire in the Streamlit UI.

    Args:
        questions (list[str]): List of questions to display.
        prefix (str): Prefix for session state keys.
        toggle_next (Callable[..., None] | None, optional): Function to call when moving to the next section.
        eval_map (dict | None, optional): Evaluation map for scoring.
        next_label (str | None, optional): Label for the next button.
        answer_options (list[str] | None, optional): List of answer options.
        points_dict (dict[str, int] | None, optional): Mapping of answers to points.

    Returns:
        None
    """
    if answer_options is None:
        answer_options = ["Ja", "Teilweise", "Nein"]
    if points_dict is None:
        points_dict = {"Ja": 2, "Teilweise": 1, "Nein": 0}
    index_key = f"frage_index_{prefix}"
    if index_key not in st.session_state:
        st.session_state[index_key] = 0
    for i in range(len(questions)):
        key = f"{prefix}_{i}"
        if key not in st.session_state:
            st.session_state[key] = None

    def advance_index() -> None:
        """
        Advance the current question index if the current question is answered.
        """
        i = st.session_state.current_radio_key
        if st.session_state[f"{prefix}_{i}"] is not None:
            if st.session_state[index_key] < len(questions) - 1:
                st.session_state[index_key] += 1

    for i in range(st.session_state[index_key] + 1):
        col1, col2 = st.columns([5, 4])
        with col1:
            st.write(questions[i])
        with col2:
            st.session_state.current_radio_key = i
            st.radio(
                label="",
                options=answer_options,
                horizontal=True,
                key=f"{prefix}_{i}",
                label_visibility="collapsed",
                on_change=advance_index,
            )
    alle_beantwortet = all(
        st.session_state[f"{prefix}_{i}"] is not None for i in range(len(questions))
    )
    if alle_beantwortet:
        gesamtpunkte = sum(
            points_dict.get(st.session_state[f"{prefix}_{i}"], 0)
            for i in range(len(questions))
        )
        if eval_map:
            compute_and_show_results(gesamtpunkte, eval_map,)
        if toggle_next and next_label:
            if st.button(next_label):
                toggle_next()
                st.rerun()