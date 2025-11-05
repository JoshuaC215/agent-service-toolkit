from datetime import datetime

import pytest
import streamlit as st

# Import the module under test
from streamlit_utils import evaluation_ui as ui


class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture(autouse=True)
def reset_session_state():
    # Ensure clean session_state for every test
    st.session_state.clear()
    yield
    st.session_state.clear()


@pytest.fixture
def patch_streamlit_minimal(monkeypatch):
    # Basic no-op renderers
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(st, "error", lambda *a, **k: None)
    monkeypatch.setattr(st, "image", lambda *a, **k: None)
    monkeypatch.setattr(st, "download_button", lambda *a, **k: None)

    # Context returns
    monkeypatch.setattr(st, "columns", lambda sizes: (_DummyCtx(), _DummyCtx()))
    monkeypatch.setattr(st, "spinner", lambda *a, **k: _DummyCtx())
    monkeypatch.setattr(st, "rerun", lambda: None)

    # Inputs default
    monkeypatch.setattr(
        st, "date_input", lambda label, value=None, key=None: value
    )
    monkeypatch.setattr(
        st, "time_input",
        lambda label, value=None, key=None, step=None: value,
    )

    # Defaults to avoid accidental clicks unless overridden inside test
    monkeypatch.setattr(
        st,
        "button",
        lambda label, key=None: False,
    )
    monkeypatch.setattr(
        st,
        "text_input",
        lambda label, key=None, type="default": "",
    )


def test_ensure_eval_defaults_sets_defaults(patch_streamlit_minimal):
    assert st.session_state.get("eval_from_date") is None
    assert st.session_state.get("eval_to_date") is None
    assert st.session_state.get("eval_from_time") is None
    assert st.session_state.get("eval_to_time") is None

    ui._ensure_eval_defaults()

    assert st.session_state.get("eval_from_date") is not None
    assert st.session_state.get("eval_to_date") is not None
    assert st.session_state.get("eval_from_time") is not None
    assert st.session_state.get("eval_to_time") is not None


def test_render_eval_login_success_admin(monkeypatch, patch_streamlit_minimal):
    # Simulate clicking the Login button only
    def _button(label, key=None):
        return key in {"eval_login_btn_modal2", "eval_login_btn_modal_fb2"}

    monkeypatch.setattr(st, "button", _button)

    # Provide username/password via text_input
    def _text_input(label, key=None, type="default"):
        return {"eval_username_modal2": "alice", "eval_password_modal2": "secret"}.get(key, "")

    monkeypatch.setattr(st, "text_input", _text_input)

    # Mock Auth.login to return an admin user
    class _User(dict):
        pass

    def _fake_login(self, username, password):
        return _User(token="T", id="U1", role="admin")

    from client.auth import Auth
    monkeypatch.setattr(Auth, "login", _fake_login, raising=False)

    # is_admin_user injected checker
    def _is_admin_user(u: dict) -> bool:
        return True

    # preconditions
    st.session_state.eval_require_login = True
    st.session_state.pop("owui-token", None)
    st.session_state.pop("user", None)
    st.session_state.pop("run_id", None)

    # Call UI
    ui._render_eval_login(
        username_key="eval_username_modal2",
        password_key="eval_password_modal2",
        login_button_key="eval_login_btn_modal2",
        close_button_key="eval_close_notlogged2",
        is_admin_user=_is_admin_user,
    )

    assert st.session_state.get("owui-token") == "T"
    assert isinstance(st.session_state.get("user"), dict)
    assert isinstance(st.session_state.get("run_id"), str)
    assert st.session_state.get("eval_require_login") is False


def test_render_evaluation_content_requires_login_invokes_login(monkeypatch, patch_streamlit_minimal):
    # Force login required
    st.session_state.eval_require_login = True
    st.session_state.pop("owui-token", None)

    called = {"login_called": False}

    # Patch internal login renderer to capture invocation
    monkeypatch.setattr(
        ui,
        "_render_eval_login",
        lambda **kwargs: called.__setitem__("login_called", True),
    )

    ui._render_evaluation_content("dialog", is_admin_user=lambda u: True)
    assert called["login_called"] is True


def test_run_generation_success_sets_image_path(monkeypatch, patch_streamlit_minimal):
    # Mock subprocess.run to simulate success
    class _Result:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(
        ui,
        "os",
        ui.os,  # keep real os for env copy
    )
    import subprocess as _real_subprocess

    monkeypatch.setattr(
        _real_subprocess, "run", lambda *a, **k: _Result()
    )

    # Ensure rerun doesn't break tests
    monkeypatch.setattr(st, "rerun", lambda: None)

    start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    end = datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=0)

    ui._run_generation(start, end)

    assert st.session_state.get("eval_running") is False
    assert st.session_state.get("eval_error") in (None, "")
    # We just check that a path string was set (file may not exist in test context)
    assert isinstance(st.session_state.get("eval_image_path"), str) and len(st.session_state.eval_image_path) > 0


def test_render_evaluation_content_generate_calls_run_generation(monkeypatch, patch_streamlit_minimal):
    # Logged-in state
    st.session_state.eval_require_login = False
    st.session_state["owui-token"] = "X"

    # Initialize date/time defaults
    ui._ensure_eval_defaults()

    # Make the "Generieren" button return True only for the generate key
    def _button(label, key=None):
        return key == "eval_generate_btn"

    monkeypatch.setattr(st, "button", _button)

    # Track whether _run_generation was called with expected datetimes
    called = {"args": None}

    def _fake_run_generation(from_dt, to_dt):
        called["args"] = (from_dt, to_dt)

    monkeypatch.setattr(ui, "_run_generation", _fake_run_generation)

    # Avoid rerun in test
    monkeypatch.setattr(st, "rerun", lambda: None)

    ui._render_evaluation_content("dialog", is_admin_user=lambda u: True)

    assert called["args"] is not None
    # Combined datetimes should be datetime objects
    assert isinstance(called["args"][0], datetime)
    assert isinstance(called["args"][1], datetime)