#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

WEB_UI_DIR = str(Path(__file__).resolve().parent)
if WEB_UI_DIR not in sys.path:
    sys.path.insert(0, WEB_UI_DIR)

from ros_bridge import RosBridge
from ui_state import EMPTY_STATUS, decode, person_label


st.set_page_config(page_title="Ranger Office RPG", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.stApp {background: #f7f8f4; color: #33413e}
.event {background:#fff; border-left:4px solid #8ba7d6; margin:5px 0; padding:7px 12px; border-radius:8px}
.truth-warning {background:#ffe8e8; color:#a40000; border:2px solid #d82929; padding:10px; border-radius:10px; font-weight:800}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def bridge_resource():
    return RosBridge()


bridge = bridge_resource()
st.title("Ranger Office RPG")
st.caption("动态语义拓扑、证据门控与在线重规划 · Phase 2 MVP")

left, center, right = st.columns([1.05, 2.15, 1.15], gap="large")

PRESETS = {
    "去904": "去904", "去906": "去906", "去908": "去908",
    "找张老师": "找张老师",
    "找张老师并提醒十点去906开会": "找到张老师，提醒他十点去906开会",
    "找李同学，未找到则巡检908后重试": "找到李同学提醒交报告；如果第一次没找到，巡检908后再找一次。",
    "立即执行一次办公室巡检": "立即执行一次办公室巡检",
    "执行8点每日巡检仿真": "每天8点巡检 lobby、904门口、906门口，发现访客就记录并通知张老师，结束后生成每日简报。",
}

with left:
    st.subheader("任务控制")
    provider = st.selectbox("LLM Provider", ["Offline", "Local Qwen",
                            "DeepSeek V4 Flash", "DeepSeek V4 Pro"])
    api_ready = bool(os.environ.get("LLM_API_KEY"))
    st.caption("DeepSeek API：" + ("已配置" if api_ready else "API未配置，将安全回退 Offline"))
    developer_truth = st.toggle("开发者真值模式", value=False,
                                help="只改变可视化，绝不改变任务规划。")
    bridge.set_developer_truth(developer_truth)
    if developer_truth:
        st.markdown('<div class="truth-warning">SIMULATION GROUND TRUTH — NOT AVAILABLE ON REAL ROBOT</div>',
                    unsafe_allow_html=True)
    mission = st.text_area("自然语言任务", key="mission_text", placeholder="例如：找张老师")
    if st.button("发送任务", type="primary", width="stretch"):
        if mission.strip():
            bridge.send_mission(mission.strip(), provider)
            st.toast("MissionSpec 已发送")
        else:
            st.warning("请输入任务")
    stop_col, reset_col = st.columns(2)
    if stop_col.button("停止任务", width="stretch"):
        ok, message = bridge.stop()
        (st.toast if ok else st.error)(message)
    if reset_col.button("重置场景", width="stretch"):
        ok, message = bridge.reset()
        (st.toast if ok else st.error)(message)
    st.markdown("##### 预设任务")
    for label in PRESETS:
        if st.button(label, key=f"preset_{label}", width="stretch"):
            bridge.send_mission(PRESETS[label], provider)


OBSERVATION_MARKERS = {
    "lobby": (1.2, 2.3), "room_904": (8.0, 4.15),
    "room_906": (13.35, 4.15), "room_908": (18.55, 4.15),
    "corridor_junction": (11.0, 3.75),
}


def office_figure(status, public_npcs, truth_payload, show_truth):
    fig = go.Figure()
    rooms = [("904", 5.8, 5.1, 4.4, 2.2, "#e8def1"),
             ("906", 10.8, 5.1, 4.4, 2.2, "#e4edf4"),
             ("908", 15.8, 5.1, 4.4, 2.2, "#e4efe7")]
    active = status.get("target_region")
    visited = set(status.get("visited_regions") or [])
    for name, x, y, w, h, color in rooms:
        region = f"room_{name}"
        border = "#74a2d7" if active == region else ("#79a886" if region in visited else "#bec8c1")
        fig.add_shape(type="rect", x0=x, y0=y, x1=x+w, y1=y+h,
                      fillcolor=color, line=dict(color=border, width=5 if active == region else 2))
        fig.add_annotation(x=x+w/2, y=y+h/2, text=f"Room {name}", showarrow=False)
    fig.add_shape(type="rect", x0=0.5, y0=0.3, x1=21.5, y1=5.0,
                  fillcolor="#f1eee5", line=dict(color="#d2d3cb", width=2), layer="below")
    fig.add_annotation(x=1.7, y=1.0, text="Lobby", showarrow=False)
    fig.add_annotation(x=11.0, y=3.75, text="交叉口", showarrow=False)
    path = status.get("planned_path") or []
    if path:
        fig.add_trace(go.Scatter(x=[p[0] for p in path], y=[p[1] for p in path],
                                 mode="lines", line=dict(color="#72a0d8", width=5),
                                 name="当前规划路径"))
    robot = status.get("robot_pose", {"x": 1.2, "y": 2.3})
    fig.add_trace(go.Scatter(x=[robot.get("x", 1.2)], y=[robot.get("y", 2.3)],
                             mode="markers+text", text=["Ranger"], textposition="top center",
                             marker=dict(size=18, color="#3977c3", symbol="diamond"), name="Ranger"))
    colors = {"teacher_zhang": "#8d63ad", "student_li": "#4e9a6b", "visitor": "#df8b45"}
    if show_truth:
        for npc in truth_payload.get("npcs", []):
            fig.add_trace(go.Scatter(x=[npc.get("x")], y=[npc.get("y")], mode="markers+text",
                                     text=[f"{npc.get('display_name')} · TRUTH"], textposition="top center",
                                     marker=dict(size=17, color=colors.get(npc.get("identity_id"), "#999")),
                                     name=f"TRUTH {npc.get('display_name')}"))
    else:
        for npc in public_npcs.get("npcs", []):
            last_seen = npc.get("last_seen")
            if not last_seen:
                continue
            x, y = OBSERVATION_MARKERS.get(last_seen.get("region"), (None, None))
            if x is None:
                continue
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text",
                                     text=[f"{npc.get('display_name')} · LAST_SEEN"],
                                     textposition="top center",
                                     marker=dict(size=15, color=colors.get(npc.get("identity_id"), "#999"),
                                                 symbol="circle-open"),
                                     name=f"LAST_SEEN {npc.get('display_name')}"))
    fig.update_xaxes(range=[0, 22], visible=False)
    fig.update_yaxes(range=[0, 7.6], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=475, margin=dict(l=5, r=5, t=25, b=5),
                      paper_bgcolor="#f7f8f4", plot_bgcolor="#f7f8f4",
                      legend=dict(orientation="h", y=-0.05), hovermode="closest")
    return fig


@st.fragment(run_every=0.75)
def live_panel(show_truth):
    snapshot = bridge.snapshot()
    status = decode(snapshot["mission_status"], EMPTY_STATUS)
    npcs = decode(snapshot["npc_states"], {"npcs": []})
    truth = decode(snapshot["developer_truth"], {"npcs": []}) if show_truth else {"npcs": []}
    report = decode(snapshot["patrol_report"], {}) or status.get("patrol_report")
    with center:
        st.subheader("办公室俯视图")
        st.plotly_chart(office_figure(status, npcs, truth, show_truth), width="stretch",
                        config={"displayModeBar": False})
        probabilities = status.get("candidate_probabilities") or {}
        if probabilities:
            st.caption("候选概率：" + " · ".join(
                f"{region} {probability:.0%}" for region, probability in probabilities.items()))
        queue = status.get("search_queue") or []
        if queue:
            st.info("搜索队列：" + " → ".join(queue))
        progress = status.get("search_progress") or {}
        if progress.get("total"):
            st.progress(min(1.0, progress.get("searched", 0) / progress["total"]),
                        text=progress.get("text"))
        st.markdown("##### 人员知识（不含未发现真值）")
        knowledge = status.get("person_knowledge") or {
            item.get("identity_id"): item for item in npcs.get("npcs", [])}
        cols = st.columns(3)
        for column, identity in zip(cols, ("teacher_zhang", "student_li", "visitor")):
            item = knowledge.get(identity, {})
            column.metric(person_label(identity), item.get("knowledge_state", "UNKNOWN"))
        if report:
            st.markdown("##### 结构化巡检报告")
            st.json(report, expanded=False)
    with right:
        st.subheader("当前任务")
        fields = [
            ("原始指令", status.get("input_text") or "—"),
            ("任务类型", status.get("mission_type") or "—"),
            ("目标人物", person_label(status.get("target_person"))),
            ("状态机阶段", status.get("state_machine_phase", "—")),
            ("当前区域", status.get("current_region", "—")),
            ("当前动作", status.get("current_action", "—")),
            ("搜索进度", (status.get("search_progress") or {}).get("text", "—")),
            ("重试", f"{status.get('retry_count',0)}/{status.get('retry_limit',0)}"),
            ("视觉触发", "是" if status.get("visual_triggered") else "否"),
            ("身份确认", "是" if status.get("identity_confirmed") else "否"),
            ("提醒内容", status.get("reminder_text") or "—"),
            ("LLM Provider", status.get("llm_provider", "offline")),
            ("最近错误", status.get("last_error") or "—"),
        ]
        for label, value in fields:
            st.markdown(f"**{label}**  \n{value}")
    st.divider()
    st.subheader("事件时间线")
    events = []
    for raw in snapshot["events"][-20:]:
        try:
            events.append(json.loads(raw))
        except ValueError:
            pass
    if not events:
        st.caption("等待 ROS 事件…")
    for event in reversed(events):
        stamp_value = event.get("timestamp", event.get("stamp", 0))
        stamp = datetime.fromtimestamp(stamp_value).strftime("%H:%M:%S")
        st.markdown(f'<div class="event"><b>{stamp} · {event.get("event_type", "EVENT")}</b><br>{event.get("message", "")}</div>',
                    unsafe_allow_html=True)
    st.divider()
    st.subheader("Phase 2 语义规划调试")
    graph = decode(snapshot["topology_graph"], {})
    llm_debug = decode(snapshot["llm_planner_debug"], {})
    gate = decode(snapshot["plan_gate_result"], {})
    graph_col, api_col, gate_col = st.columns(3)
    with graph_col:
        with st.expander("1. 当前动态拓扑图 JSON", expanded=False):
            st.json(graph or {"status": "等待图快照"})
    with api_col:
        with st.expander("2. API 输入与返回 JSON", expanded=False):
            st.json({
                "api_called": llm_debug.get("api_called", False),
                "api_configured": llm_debug.get("api_configured", False),
                "provider": llm_debug.get("provider_used",
                                          llm_debug.get("provider_requested")),
                "request": llm_debug.get("request"),
                "response": llm_debug.get("response"),
                "fallback_response": llm_debug.get("fallback_response"),
                "error": llm_debug.get("error", ""),
            })
    with gate_col:
        with st.expander("3. plan_gate 与 graph_revision", expanded=False):
            st.json(gate or {
                "accepted": False,
                "graph_revision": status.get("graph_revision", 0),
                "rejected_reason": "等待规划",
            })


live_panel(developer_truth)
