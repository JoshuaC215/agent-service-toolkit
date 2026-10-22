"""Streamlit human approval workspace using the application's native styling."""

import os

import streamlit as st

from client.approval_client import ApprovalClient, ApprovalClientError

STATUS_LABELS = {"pending": "待审批", "approved": "已批准", "rejected": "已拒绝"}


def _logout():
    for name in ("approval_key", "approval_lookup", "approval_flash"):
        st.session_state.pop(name, None)


def _show_result(record):
    st.write(
        {
            "审批编号": record["approval_id"],
            "工单号": record["ticket_id"],
            "审批状态": STATUS_LABELS.get(record["status"], record["status"]),
            "审批人": record.get("reviewer_id") or "—",
            "审批理由": record.get("reason") or "—",
            "提交时间": record.get("created_at"),
            "审批时间": record.get("reviewed_at") or "—",
        }
    )


def draw_approval_page(base_url: str):
    with st.sidebar:
        st.header(":material/fact_check: 人工审批")
        st.caption("售后申请 · 人工处理")
        key = st.text_input("审批密钥", type="password", key="approval_key")
        st.button("退出审批身份", use_container_width=True, on_click=_logout)
        st.caption("密钥仅用于当前会话，不会发送给模型。")

    st.title("人工审批")
    st.caption("核对订单与申请材料，记录人工审批决定。")
    st.info("批准仅表示申请获批，不会执行退款，也不会修改订单状态。")
    if not key:
        st.markdown("请先在侧边栏输入审批密钥。")
        return
    client = ApprovalClient(base_url, key, os.getenv("AUTH_SECRET"))
    try:
        with st.spinner("正在加载审批列表…"):
            pending = client.pending()
    except ApprovalClientError as exc:
        st.error(str(exc))
        return

    if message := st.session_state.pop("approval_flash", None):
        st.success(message)
    left, right = st.columns([3, 1])
    left.metric("待审批申请", len(pending))
    if right.button(":material/refresh: 刷新", use_container_width=True):
        st.rerun()

    queue_tab, submit_tab, lookup_tab, execute_tab, owner_tab = st.tabs(
        ["待审批", "提交审批", "查询结果", "模拟退款", "订单归属"]
    )
    with queue_tab:
        if not pending:
            st.success("暂无待审批申请。可在“提交审批”中送审已有工单。")
        else:
            st.dataframe(
                [{"工单号": r["ticket_id"], "提交时间": r["created_at"]} for r in pending],
                hide_index=True,
                use_container_width=True,
            )
            selected = st.selectbox(
                "选择要处理的工单", [r["ticket_id"] for r in pending], key="approval_selected"
            )
            try:
                ticket = client.ticket(selected)
                order = client.order(ticket["order_id"])
                st.subheader("申请详情")
                a, b, c = st.columns(3)
                a.metric("订单号", ticket["order_id"])
                b.metric("订单状态", order["status"])
                c.metric("订单金额", f"{order['amount']} {order['currency']}")
                st.caption(
                    f"工单：{selected} · 类型：{ticket['issue_type']} · 状态：{ticket['status']}"
                )
                st.text(ticket["description"])
                st.caption("材料完整性请结合工单描述人工核实；系统不会自动判断证据充分。")
                with st.form(f"approval_review_{selected}"):
                    choice = st.radio("审批决定", ["批准", "拒绝"], horizontal=True)
                    reason = st.text_area("审批理由（必填）", max_chars=4000)
                    confirmed = st.checkbox("我已核对该工单、订单及申请材料，确认提交此决定")
                    send = st.form_submit_button(
                        "提交审批决定",
                        type="primary",
                        disabled=ticket["status"] not in ("open", "in_progress"),
                    )
                if send:
                    if not reason.strip() or not confirmed:
                        st.warning("请填写审批理由，并确认已核对申请。")
                    else:
                        result = client.review(
                            selected, "approved" if choice == "批准" else "rejected", reason.strip()
                        )
                        st.session_state.approval_flash = (
                            f"工单 {selected} {STATUS_LABELS[result['status']]}，"
                            f"审批人：{result['reviewer_id']}。未执行退款。"
                        )
                        st.rerun()
            except ApprovalClientError as exc:
                st.error(str(exc))
    with owner_tab:
        st.caption("仅管理员可分配历史订单。已归属其他用户的订单不能在这里转移。")
        with st.form("assign_order_owner"):
            order_id = st.text_input("待分配订单号", max_chars=32).strip()
            owner_id = st.text_input("用户编号（需已配置访问密钥）", max_chars=64).strip()
            confirmed_owner = st.checkbox("确认该订单属于此用户，包括其关联工单和审批数据")
            assign = st.form_submit_button("分配订单归属")
        if assign:
            if not order_id or not owner_id or not confirmed_owner:
                st.warning("请填写订单号、用户编号，并确认归属。")
            else:
                try:
                    client.assign_owner(order_id, owner_id)
                    st.success(f"订单 {order_id} 已分配给 {owner_id}。")
                except ApprovalClientError as exc:
                    st.error(str(exc))
    with submit_tab:
        st.caption("将已有退款申请工单送入审批；不会创建新工单。")
        with st.form("approval_submit"):
            ticket_id = st.text_input("工单号", max_chars=32)
            confirm = st.checkbox("确认该工单为需要人工审批的退款申请")
            submit = st.form_submit_button("提交审批", type="primary")
        if submit:
            if not ticket_id.strip() or not confirm:
                st.warning("请填写工单号，并确认申请类型。")
            else:
                try:
                    result = client.submit(ticket_id.strip())
                    st.session_state.approval_flash = f"工单 {result['ticket_id']} 已进入待审批。"
                    st.rerun()
                except ApprovalClientError as exc:
                    st.error(str(exc))
    with lookup_tab:
        with st.form("approval_lookup_form"):
            lookup_id = st.text_input("查询工单号", max_chars=32)
            lookup = st.form_submit_button("查询审批结果")
        if lookup:
            st.session_state.approval_lookup = lookup_id.strip()
        if lookup_id := st.session_state.get("approval_lookup"):
            try:
                _show_result(client.get(lookup_id))
            except ApprovalClientError as exc:
                st.error(str(exc))
    with execute_tab:
        st.warning("仅为整单退款模拟：不转账、不修改真实订单状态，同一订单最多执行一次。")
        execution_ticket = st.text_input("待执行或查询的工单号", max_chars=32).strip()
        if execution_ticket:
            try:
                approval = client.get(execution_ticket)
                st.caption(f"审批状态：{STATUS_LABELS.get(approval['status'], approval['status'])}")
                ticket = client.ticket(execution_ticket)
                order = client.order(ticket["order_id"])
                st.write(
                    f"订单 {ticket['order_id']} · 模拟金额 {order['amount']} {order['currency']}"
                )
                with st.form(f"simulate_{execution_ticket}"):
                    confirmed = st.checkbox("确认对该申请执行整单模拟退款（不会真实转账）")
                    execute = st.form_submit_button(
                        "执行模拟退款", type="primary", disabled=approval["status"] != "approved"
                    )
                if execute:
                    if not confirmed:
                        st.warning("请先勾选模拟执行确认。")
                    else:
                        result = client.simulate_refund(execution_ticket)
                        st.success(
                            "模拟执行成功，未真实转账。"
                            if result["created"]
                            else "该申请已执行过，返回原记录，未重复执行。"
                        )
                        st.write(result["execution"])
                if st.button("查询已保存的执行结果"):
                    st.write(client.execution(execution_ticket))
            except ApprovalClientError as exc:
                st.error(str(exc))
