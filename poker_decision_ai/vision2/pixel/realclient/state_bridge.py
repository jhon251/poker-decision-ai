"""V3F-P11-A2 — Runtime Snapshot → State/Replay 纯数据桥（安全边界）。

- 纯数据转换（aggregation/read-only）：Snapshot dict → replay-safe state dict；
- UNKNOWN/missing 严格保留（bet/action/stack 不转 0/前值/默认/推断）；
- 零决策/策略/EV；零 Bet/Action 视觉实现；
- 可 JSON 序列化（确定性 replay 基础）。
"""
from __future__ import annotations


#: 快照字段 → replay-state 字段（原样透传——UNKNOWN 标记保留）
_SNAPSHOT_TO_STATE = (
    "position", "pot", "board", "street", "hero_cards", "status", "stack",
    "bet", "action",
)


def snapshot_to_replay_state(snapshot: dict) -> dict:
    """Snapshot（P9 感知快照）→ replay-safe state dict。

    - 纯透传感知字段（原样——零转换/推断/补值）；
    - bet/action 保留其显式 UNKNOWN 标注（EVIDENCE-LIMITED / NOT-READY）——
      不创建值、不置 0、不推断；
    - ground_truth_status 保留 runtime_state_non_gt（感知层——非 GT）。
    """
    state = {}
    for f in _SNAPSHOT_TO_STATE:
        state[f] = snapshot.get(f)          # 原样（含 None/UNKNOWN 语义）
    state["ground_truth_status"] = snapshot.get(
        "ground_truth_status", "runtime_state_non_gt")
    return state


def validate_replay_state(state: dict) -> list:
    """缺失/UNKNOWN 保留校验（不拒绝——仅返回语义注解列表）。

    返回：缺失字段语义提示（不抛异常——replay 接受不完整状态）。
    """
    notes = []
    if state.get("bet") is None or (
            isinstance(state.get("bet"), dict) and
            state["bet"].get("status") == "UNKNOWN"):
        notes.append("bet=EVIDENCE_LIMITED(UNKNOWN)")
    if state.get("action") is None or (
            isinstance(state.get("action"), dict) and
            state["action"].get("status") == "UNKNOWN"):
        notes.append("action=NOT_READY(UNKNOWN)")
    return notes
