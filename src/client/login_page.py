import os

import httpx
import streamlit as st


def _logout():
    st.session_state.clear()
    st.query_params.clear()


def customer_login(base_url: str) -> dict | None:
    """A first version of account login using server-issued user access keys."""
    if identity := st.session_state.get("customer_identity"):
        st.sidebar.caption(f"当前用户：{identity['user_id']}")
        st.sidebar.button("退出用户登录", on_click=_logout)
        return identity
    st.title("用户登录")
    st.caption("使用管理员分配的用户访问密钥登录，订单归属由后端校验。")
    with st.form("customer_login"):
        key = st.text_input("用户访问密钥", type="password")
        login = st.form_submit_button("登录", type="primary")
    if login:
        headers = {"X-User-Key": key}
        if secret := os.getenv("AUTH_SECRET"):
            headers["Authorization"] = f"Bearer {secret}"
        try:
            response = httpx.get(
                f"{base_url.rstrip('/')}/auth/me", headers=headers, timeout=10, trust_env=False
            )
            if response.status_code != 200:
                st.error("登录失败，请检查用户密钥和服务配置。")
                return None
            identity = response.json()
            if identity.get("role") != "customer":
                st.error("请使用普通用户访问密钥。人工审批请切换工作区。")
                return None
            st.session_state.customer_key = key
            st.session_state.customer_identity = identity
            st.rerun()
        except (httpx.HTTPError, ValueError):
            st.error("无法连接登录服务，请稍后重试。")
    return None
