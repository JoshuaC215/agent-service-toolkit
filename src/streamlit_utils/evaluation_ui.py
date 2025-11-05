import logging
import os
import uuid
from collections.abc import Callable

import streamlit as st

logger = logging.getLogger(__name__)


def _render_eval_login(
    *,
    username_key: str,
    password_key: str,
    login_button_key: str,
    close_button_key: str,
    is_admin_user: Callable[[dict], bool],
) -> None:
    """
    Render evaluation login UI and handle actions. Requires admin account.
    Dependency-injected is_admin_user to avoid circular imports.
    """
    st.caption("Bitte melden Sie sich an, um die Auswertung zu starten.")
    eval_username = st.text_input("Username", key=username_key)
    eval_password = st.text_input("Password", type="password", key=password_key)
    col_login_a, col_login_b = st.columns([1, 1])
    with col_login_a:
        if st.button("Login", key=login_button_key):
            try:
                from client.auth import Auth as _Auth

                user = _Auth(default_login=True).login(eval_username, eval_password)
                if user:
                    st.session_state["owui-token"] = user["token"]
                    st.session_state["user"] = user
                    st.session_state["run_id"] = str(uuid.uuid4())
                    if is_admin_user(user):
                        st.session_state.eval_require_login = False
                        st.rerun()
                    else:
                        st.error(
                            "Nur Admins dürfen die Auswertung verwenden. Bitte mit einem Admin-Account anmelden."
                        )
                else:
                    st.error("Login fehlgeschlagen")
            except Exception as e:
                st.error(f"Login Fehler: {e}")
    with col_login_b:
        if st.button("Schließen", key=close_button_key):
            st.session_state.evaluation_dialog_open = False
            st.session_state.eval_image_path = None
            st.session_state.eval_image_bytes = None
            st.session_state.eval_image_filename = None
            st.session_state.eval_error = None
            st.session_state.eval_require_login = False
            st.rerun()


def _ensure_eval_defaults() -> None:
    """
    Ensure default values for evaluation date/time inputs exist in session_state (UTC).
    """
    from datetime import datetime as _dt
    from datetime import time as _time

    if st.session_state.get("eval_from_date") is None:
        st.session_state.eval_from_date = _dt.utcnow().date()
    if st.session_state.get("eval_to_date") is None:
        st.session_state.eval_to_date = _dt.utcnow().date()
    if "eval_from_time" not in st.session_state or st.session_state.eval_from_time is None:
        st.session_state.eval_from_time = _time(hour=0, minute=0, second=0)
    if "eval_to_time" not in st.session_state or st.session_state.eval_to_time is None:
        st.session_state.eval_to_time = _time(hour=23, minute=59, second=59)


def _run_generation(from_dt, to_dt) -> None:
    """
    Generate evaluation image in-memory and persist bytes in session_state.
    """
    st.session_state.eval_running = True
    st.session_state.eval_error = None
    st.session_state.eval_image_path = None
    st.session_state.eval_image_bytes = None
    st.session_state.eval_image_filename = None

    from_ts = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_ts = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    agent_name = os.getenv("AGENT_NAME", "skillcompanion_interrupted")

    try:
        from langfuse_evaluation.skillcompanion_evaluation import (
            generate_skillcompanion_png_bytes as _gen_bytes,
        )

        with st.spinner("Erstelle Auswertung..."):
            png_bytes, filename = _gen_bytes(
                from_timestamp=from_ts,
                to_timestamp=to_ts,
                agent_name=agent_name,
            )
        st.session_state.eval_image_bytes = png_bytes
        st.session_state.eval_image_filename = filename
    except Exception as e:
        logger.exception("Evaluation generation failed")
        st.session_state.eval_error = f"Fehler beim Erstellen der Auswertung: {e}"
    finally:
        st.session_state.eval_running = False
        st.rerun()


def _render_results_and_download() -> None:
    """
    Render error/result display and optional download button.
    """
    if st.session_state.get("eval_error"):
        st.error(st.session_state.eval_error)

    image_bytes = st.session_state.get("eval_image_bytes")
    if image_bytes:
        st.image(image_bytes, caption="Skill Check Ergebnisse")
        st.download_button(
            "Download PNG",
            data=image_bytes,
            file_name=st.session_state.get("eval_image_filename") or "skill_levels_bar_chart.png",
            mime="image/png",
            key="eval_download_btn",
        )


def _render_evaluation_content(mode: str, *, is_admin_user: Callable[[dict], bool]) -> None:
    """
    Unified evaluation content renderer for both dialog and fallback container.

    mode:
      - "dialog": force separate date+time inputs
      - "fallback": same behavior (separate date+time), kept for future extension
    """
    # Gate: login/admin required?
    require_login = st.session_state.get("eval_require_login", False) or "owui-token" not in st.session_state
    if require_login:
        if mode == "dialog":
            _render_eval_login(
                username_key="eval_username_modal2",
                password_key="eval_password_modal2",
                login_button_key="eval_login_btn_modal2",
                close_button_key="eval_close_notlogged2",
                is_admin_user=is_admin_user,
            )
        else:
            _render_eval_login(
                username_key="eval_username_modal_fb2",
                password_key="eval_password_modal_fb2",
                login_button_key="eval_login_btn_modal_fb2",
                close_button_key="eval_close_notlogged_fb2",
                is_admin_user=is_admin_user,
            )
        return

    # Logged-in flow
    from datetime import datetime as _dt

    _ensure_eval_defaults()

    # Separate date and time inputs (consistent across dialog and fallback)
    col_from_date, col_from_time = st.columns([2, 1])
    with col_from_date:
        st.session_state.eval_from_date = st.date_input(
            "Von (Datum, UTC)",
            value=st.session_state.eval_from_date,
            key="eval_from_date_input",
        )
    with col_from_time:
        st.session_state.eval_from_time = st.time_input(
            "Von (Uhrzeit, UTC)",
            value=st.session_state.eval_from_time,
            key="eval_from_time_input",
            step=60,
        )

    col_to_date, col_to_time = st.columns([2, 1])
    with col_to_date:
        st.session_state.eval_to_date = st.date_input(
            "Bis (Datum, UTC)",
            value=st.session_state.eval_to_date,
            key="eval_to_date_input",
        )
    with col_to_time:
        st.session_state.eval_to_time = st.time_input(
            "Bis (Uhrzeit, UTC)",
            value=st.session_state.eval_to_time,
            key="eval_to_time_input",
            step=60,
        )

    # Actions
    col_gen, col_close = st.columns([1, 1])
    with col_gen:
        if st.button("Generieren", key="eval_generate_btn"):
            from_dt = _dt.combine(st.session_state.eval_from_date, st.session_state.eval_from_time)
            to_dt = _dt.combine(st.session_state.eval_to_date, st.session_state.eval_to_time)

            if from_dt > to_dt:
                st.error("Ungültiger Zeitraum: Von ist nach Bis.")
            else:
                _run_generation(from_dt, to_dt)

    with col_close:
        if st.button("Schließen", key="eval_close_btn"):
            st.session_state.evaluation_dialog_open = False
            st.session_state.eval_image_path = None
            st.session_state.eval_image_bytes = None
            st.session_state.eval_image_filename = None
            st.session_state.eval_error = None
            st.rerun()

    _render_results_and_download()