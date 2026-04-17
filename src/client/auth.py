import logging
import os
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
# Default HTTP timeout (seconds) for OWUI auth requests
REQUEST_TIMEOUT = float(os.getenv("OWUI_HTTP_TIMEOUT", "10"))


class Auth:
    load_dotenv()

    OWUI_BASE_URL = os.getenv("OWUI_BASE_URL")

    def __init__(self, default_login: bool):
        if not default_login:
            if self.is_logged_in():
                self.account_partials()
            else:
                self.login_page()
        else:
            if not self.is_logged_in():
                self.silent_default_login()

    def is_logged_in(self) -> bool:
        if "owui-token" in st.session_state:
            return True

        return False

    def login_page(self):
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = self.login(username, password)
            if user:
                st.session_state["owui-token"] = user["token"]
                st.session_state["user"] = user
                st.session_state["run_id"] = str(uuid4())
                st.rerun()

    def account_partials(self):
        st.sidebar.success(f"Willkommen, {st.session_state['user']['name']}!")
        if st.sidebar.button("Logout", key="logout"):
            del st.session_state["owui-token"]
            del st.session_state["user"]
            st.rerun()

    def login(self, username, password):
        url = f"{self.OWUI_BASE_URL}/api/v1/auths/signin"
        payload = {"email": username, "password": password}
        logger.info(url)
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Login failed")
            return None

    def silent_default_login(self):
        url = f"{self.OWUI_BASE_URL}/api/v1/auths/signin"
        username = os.getenv("OWUI_DEFAULT_USER_NAME")
        password = os.getenv("OWUI_DEFAULT_USER_PASSWORD")
        payload = {"email": username, "password": password}
        logger.info(url)
        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                user = response.json()
                if user:
                    st.session_state["owui-token"] = user["token"]
                    st.session_state["user"] = user
                    st.session_state["run_id"] = str(uuid4())
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

    def compare_login_user(self):
        """Ensures logged in user is actually the same as in owui."""
        url = f"{self.OWUI_BASE_URL}/api/v1/auths"
        headers = {"Authorization": f"Bearer {st.session_state['owui-token']}"}
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        logger.info(response.json())
        if response.status_code == 200:
            user = response.json()
            if (
                st.session_state["user"]["id"] == user["id"]
                and st.session_state["owui-token"] == user["token"]
            ):
                return True

        return False
