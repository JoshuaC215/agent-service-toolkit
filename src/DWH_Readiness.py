import asyncio

import streamlit as st

from client.auth import Auth
from streamlit_utils.dwh_helpers import (
    challenge_eval_map,
    questions_for_dwh_readiness_challenge,
    questions_for_dwh_readiness_satisfaction,
    render_questionnaire,
    render_slider_section,
    satisfaction_eval_map,
)

APP_TITLE = "roosi DWH Readiness"
APP_ICON = "🤖"
HIDE_SIDEBAR = True

async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        # collapse sidebar on default
        initial_sidebar_state="collapsed" if HIDE_SIDEBAR else "auto",
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )

    # hide sidebar
    if HIDE_SIDEBAR:
        st.html(
            """
        <style>
            [data-testid="stSidebarCollapsedControl"] {
                display: none
            }
        </style>
        """
        )
    if "show_chat_input" not in st.session_state:
        st.session_state.show_chat_input = False
    if "show_slider" not in st.session_state:
        st.session_state.show_slider = False
    if "show_dwh_readiness_check_challenge" not in st.session_state:
        st.session_state.show_dwh_readiness_check_challenge = False
    if "show_dwh_readiness_check_satisfaction" not in st.session_state:
        st.session_state.show_dwh_readiness_check_satisfaction = False
    if "show_start_button" not in st.session_state:
        st.session_state.show_start_button = True
    if "show_additional_infos" not in st.session_state:
        st.session_state.show_additional_infos = False
    if "show_final_check" not in st.session_state:
        st.session_state.show_final_check = False
    if "show_title" not in st.session_state:
        st.session_state.show_title = True
    if "additional_infos" not in st.session_state:
        st.session_state.additional_infos = ""
    if "title" not in st.session_state:
        st.session_state.title= "roosi DWH Readiness Check"
    if "challenge_final_result" not in st.session_state:
        st.session_state.challenge_final_result = ""
    if "company_size" not in st.session_state:
        st.session_state.company_size =0
    if "tech_affinity" not in st.session_state:
        st.session_state.tech_affinity = 0
    if "satisfaction_final_result" not in st.session_state:
        st.session_state.satisfaction_final_result = ""
    if st.session_state.show_title:
        if st.session_state.show_dwh_readiness_check_challenge:
            st.session_state.title = "Herausforderungen & Schmerzpunkte identifizieren"
        if st.session_state.show_dwh_readiness_check_satisfaction:
            st.session_state.title = "Zufriedenheit mit bestehenden Lösungen"
        st.title(st.session_state.title)

    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()
    
    auth = Auth(default_login=True)
    if not auth.is_logged_in():
        return

    if "antwort" not in st.session_state:
        st.session_state.antwort = None
   
    if st.session_state.show_start_button:  
        st.write("Willkommen beim DWH Readiness-Check! Bevor Sie in die Welt des Datenmanagements eintauchen, möchten wir Ihnen helfen, den Reifegrad Ihres Unternehmens für Implementierungen von Data-Warehouse, Data-Lake oder Lakehouse Projekten zu bestimmen. Durch diesen Fragebogen erhalten Sie wertvolle Einblicke in Ihre aktuelle Dateninfrastruktur und können Maßnahmen zur Modernisierung planen. Ob Schwierigkeiten bei der Datenintegration oder beim Erzeugen konsistenter Analysen – dieser Test wird Ihnen dabei helfen, Herausforderungen zu identifizieren und zielgerichtet anzugehen. Viel Erfolg beim Entdecken und Optimieren Ihrer Datenstrategie!")
        if st.button("Starten Sie mit dem DWH Readiness-Check"):
            toggle_slider()
            st.rerun()
    # --- Main UI logic using helpers ---

    if st.session_state.show_slider:
        render_slider_section()
        # Add button to hide the slider section
        if st.button("Weiter"):
            toggle_additional_infos()
            st.rerun()
        # Wenn Slider geschlossen wurde, direkt die Fragen anzeigen
        if not st.session_state.show_slider:
            st.session_state.show_dwh_readiness_check_challenge = True
            st.session_state.show_title = True
            st.rerun()

    if st.session_state.show_additional_infos:
        st.session_state.additional_infos = st.text_area(label="Hier können Sie schreiben was Ihnen besonders wichtig ist für Ihr Unternehmen: ")

        if st.button("Weiter"):
            toggle_dwh_readiness_check_challenge()
            st.rerun()
        
    if st.session_state.show_final_check:
            st.write("Auswertung Ihrer Ergebnisse:")
            st.write("Relevante Themen für Ihre Organisation:")
            st.write(st.session_state.additional_infos)
            st.write("Unternehmensgröße: ")
            st.write(st.session_state.company_size)
            st.write("Einschätzung der technischen Affinität Ihres Unternehmens:")
            st.write(st.session_state.tech_affinity)
            st.write("Analyse Ihrer Bewertung der aktuellen Herausforderungen:")
            st.write(st.session_state.challenge_final_result)
            st.write("Analyse der Effizienz und Leistungsfähigkeit Ihrer Dateninfrastruktur:")
            st.write(st.session_state.satisfaction_final_result)

    if st.session_state.show_dwh_readiness_check_challenge:
        st.write("Bewerten Sie die aktuellen Herausforderungen in Ihrem Datenmanagement.")
        # Streamlit runs in an event loop, so we must use await instead of asyncio.run
        questions = questions_for_dwh_readiness_challenge
        render_questionnaire(
            questions,
            prefix="challenge",
            toggle_next=toggle_dwh_readiness_check_satisfaction,
            eval_map=challenge_eval_map,
            next_label="Bewerten Sie ihre bestehende Lösung",
        )

    if st.session_state.show_dwh_readiness_check_satisfaction:
        st.write("Bewerten Sie die Effizienz und Leistungsfähigkeit Ihrer aktuellen Dateninfrastruktur:")
        questions = questions_for_dwh_readiness_satisfaction
        render_questionnaire(
            questions,
            prefix="satisfaction",
            toggle_next=toggle_dwh_final_check,
            eval_map=satisfaction_eval_map,
            next_label="Zur finalen Auswertung",
        )

def toggle_slider():
    st.session_state.show_slider = True
    st.session_state.show_start_button = False
    st.session_state.show_title = False
    st.session_state.show_additional_infos = False
    st.session_state.show_final_check = False

def toggle_additional_infos():
    st.session_state.show_additional_infos = True
    st.session_state.show_slider = False
    st.session_state.show_final_check = False

def toggle_dwh_readiness_check_challenge():
    st.session_state.show_title = True
    st.session_state.show_additional_infos = False
    st.session_state.show_dwh_readiness_check_challenge = True
    st.session_state.show_final_check = False

def toggle_dwh_readiness_check_satisfaction():
    st.session_state.show_title = True
    st.session_state.show_slider = False
    st.session_state.show_dwh_readiness_check_challenge= False
    st.session_state.show_dwh_readiness_check_satisfaction = True
    st.session_state.show_final_check = False

def toggle_dwh_final_check():
    st.session_state.show_final_check = True
    st.session_state.show_additional_infos = False
    st.session_state.show_title = False
    st.session_state.show_slider = False
    st.session_state.show_dwh_readiness_check_challenge= False
    st.session_state.show_dwh_readiness_check_satisfaction = False

if __name__ == "__main__":
    asyncio.run(main())
