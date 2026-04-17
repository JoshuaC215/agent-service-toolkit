import asyncio
import os
import uuid
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from client import AgentClient
from client.auth import Auth
from schema.models import OpenwebuiModelName
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


def _init_agent_client() -> AgentClient:
    """
    Initialisiert und cached den AgentClient (analog zum Muster in Skill_Companion),
    ohne Lazy-Import und mit sauberer URL-Auflösung (ohne /api/v1-Fallback).
    """
    if "agent_client" in st.session_state:
        return st.session_state.agent_client  # type: ignore[return-value]

    load_dotenv()
    agent_url = os.getenv("BACKEND_URL")
    if not agent_url:
        host = os.getenv("HOST", "127.0.0.1")
        port = os.getenv("PORT", "8080")
        agent_url = f"http://{host}:{port}"

    api_key = st.session_state.get("owui-token", "")
    client = AgentClient(
        base_url=agent_url,
        api_key=str(api_key or ""),
        get_info=True,
    )
    # Agent verifizieren und setzen
    try:
        client.update_agent("dwh_readiness_summary", verify=True)
    except Exception as e:
        # Falls der Service während des Init noch nicht bereit ist, keine harte Exception werfen
        # Statt stumm zu ignorieren, protokollieren wir den Fehler, um Debugging zu erleichtern.
        st.info(f"Agent verification skipped during init: {e}")
    st.session_state.agent_client = client
    return client


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
        st.session_state.title = "roosi DWH Readiness Check"
    if "challenge_final_result" not in st.session_state:
        st.session_state.challenge_final_result = ""
    if "company_size" not in st.session_state:
        st.session_state.company_size = 0
    if "tech_affinity" not in st.session_state:
        st.session_state.tech_affinity = 0
    if "satisfaction_final_result" not in st.session_state:
        st.session_state.satisfaction_final_result = ""
        st.title(st.session_state.title)
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    auth = Auth(default_login=True)
    if not auth.is_logged_in():
        return
    # Session-scope IDs/Client initialisieren (wie im Skill_Companion-Muster)
    st.session_state.setdefault("run_id", str(uuid.uuid4()))
    _init_agent_client()

    if "antwort" not in st.session_state:
        st.session_state.antwort = None

    if st.session_state.show_start_button:
        st.write(
            "Willkommen beim DWH Readiness-Check! Bevor Sie in die Welt des Datenmanagements eintauchen, möchten wir Ihnen helfen, den Reifegrad Ihres Unternehmens für Implementierungen von Data-Warehouse, Data-Lake oder Lakehouse Projekten zu bestimmen. Durch diesen Fragebogen erhalten Sie wertvolle Einblicke in Ihre aktuelle Dateninfrastruktur und können Maßnahmen zur Modernisierung planen. Ob Schwierigkeiten bei der Datenintegration oder beim Erzeugen konsistenter Analysen – dieser Test wird Ihnen dabei helfen, Herausforderungen zu identifizieren und zielgerichtet anzugehen. Viel Erfolg beim Entdecken und Optimieren Ihrer Datenstrategie!"
        )
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
            st.session_state.show_title = False
            st.rerun()

    if st.session_state.show_additional_infos:
        st.session_state.additional_infos = st.text_area(
            label="Hier können Sie schreiben was Ihnen besonders wichtig ist für Ihr Unternehmen: "
        )

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
        # ---------------- Persistenz & LLM-Analyse ----------------
        # readiness_result Dict wird einmalig erzeugt oder aktualisiert
        readiness_dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "company_size": st.session_state.get("company_size"),
            "tech_affinity": st.session_state.get("tech_affinity"),
            "additional_infos": st.session_state.get("additional_infos", ""),
            "challenge_result": st.session_state.get("challenge_final_result", ""),
            "satisfaction_result": st.session_state.get("satisfaction_final_result", ""),
        }
        # Füge Antworten aus Session-State hinzu
        for key, val in st.session_state.items():
            if isinstance(key, str):
                key_str = key
                if isinstance(val, str) and key_str.startswith(("challenge_", "satisfaction_")):
                    readiness_dict[key_str] = val
        st.session_state["readiness_result"] = readiness_dict

        if st.button("LLM-Analyse erstellen"):
            client: AgentClient = _init_agent_client()
            with st.spinner("LLM generiert Zusammenfassung …"):
                try:
                    response = await client.ainvoke(
                        message="READINESS_INPUT",
                        model=OpenwebuiModelName.GPT_4O,  # erzwinge OWUI-Modell, um 401 durch fehlende OpenAI-Creds zu vermeiden
                        thread_id=st.session_state.get("thread_id") or str(uuid.uuid4()),
                        user_id=st.session_state.get("user", {}).get("id"),
                        run_id=st.session_state.get("run_id"),
                        agent_config={"input": {"readiness_result": readiness_dict}},
                    )
                    import json

                    parsed = json.loads(response.content)
                    st.session_state["llm_summary"] = (
                        parsed  # TODO continue below to use this parsed "nicely" in output ui
                    )
                except Exception as e:
                    st.error(f"Fehler bei LLM-Analyse: {e}")
        # Anzeige falls vorhanden
        if "llm_summary" in st.session_state:
            import json as _json

            _raw = st.session_state["llm_summary"]

            # Robust in ein Dict konvertieren
            if isinstance(_raw, str):
                try:
                    llm = _json.loads(_raw)
                except Exception:
                    llm = {"summary": _raw, "recommendations": []}
            elif isinstance(_raw, dict):
                llm = _raw
            else:
                try:
                    llm = _json.loads(str(_raw))
                except Exception:
                    llm = {"summary": str(_raw), "recommendations": []}

            summary = llm.get("summary", "")
            recs = llm.get("recommendations", []) or []

            st.markdown("---")
            st.subheader("LLM-Zusammenfassung")
            if isinstance(summary, (list, dict)):  # noqa: UP038
                # Fallback: falls Summary kein String ist, zeige JSON lesbar
                st.json(summary)
            elif isinstance(summary, str) and summary.strip():
                # Präsentationsfreundlich als Blockquote
                st.markdown(f"> {summary.strip()}")
            else:
                st.info("Keine Zusammenfassung generiert.")

            st.subheader("Empfehlungen")
            if isinstance(recs, list) and recs:
                for rec in recs:
                    if isinstance(rec, str):
                        st.markdown(f"- {rec}")
                    else:
                        st.markdown(f"- {_json.dumps(rec, ensure_ascii=False)}")
            else:
                st.info("Keine Empfehlungen generiert.")

    if st.session_state.show_dwh_readiness_check_challenge:
        # Streamlit runs in an event loop, so we must use await instead of asyncio.run
        questions = questions_for_dwh_readiness_challenge
        if questions is not None:
            render_questionnaire(
                questions,
                prefix="challenge",
                toggle_next=toggle_dwh_readiness_check_satisfaction,
                eval_map=challenge_eval_map,
                next_label="Bewerten Sie ihre bestehende Lösung",
            )
        else:
            st.warning("Fragen für den DWH Readiness Challenge-Check konnten nicht geladen werden.")

    if st.session_state.show_dwh_readiness_check_satisfaction:
        st.write(
            "Bewerten Sie die Effizienz und Leistungsfähigkeit Ihrer aktuellen Dateninfrastruktur:"
        )
        questions = questions_for_dwh_readiness_satisfaction
        if questions is not None:
            render_questionnaire(
                questions,
                prefix="satisfaction",
                toggle_next=toggle_dwh_final_check,
                eval_map=satisfaction_eval_map,
                next_label="Zur finalen Auswertung",
            )
        else:
            st.warning(
                "Fragen für den DWH Readiness Zufriedenheits-Check konnten nicht geladen werden."
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
    st.session_state.show_dwh_readiness_check_challenge = False
    st.session_state.show_dwh_readiness_check_satisfaction = True
    st.session_state.show_final_check = False


def toggle_dwh_final_check():
    st.session_state.show_final_check = True
    st.session_state.show_additional_infos = False
    st.session_state.show_title = False
    st.session_state.show_slider = False
    st.session_state.show_dwh_readiness_check_challenge = False
    st.session_state.show_dwh_readiness_check_satisfaction = False


if __name__ == "__main__":
    asyncio.run(main())
