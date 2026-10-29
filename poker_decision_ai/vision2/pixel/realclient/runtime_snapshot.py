"""V3F-P9 — Runtime Snapshot Aggregation（感知层报告级快照）。

- aggregation-only / read-only / current-frame-only：只聚合已有 Observer
  的 summary() 输出（原样透传——零重新识别/二次推断/字段互推）；
- UNKNOWN 原样传播（Stack UNKNOWN → Snapshot UNKNOWN；不转 0/前值/默认）；
- Bet=EVIDENCE-LIMITED、Action=NOT-READY → 显式 UNKNOWN 标注（不创建值）；
- ground_truth_status = runtime_state_non_gt（感知快照——非 Ground Truth /
  非决策层）。
"""
from __future__ import annotations


#: 快照承载的 observer 字段名（→ observer 注册名；其 summary() 原样透传）
_OBSERVER_FIELDS = {
    "position": "position_runtime",
    "pot": "pot_runtime",
    "board": "board_runtime",
    "street": "street_runtime",
    "hero_cards": "hero_cards_runtime",
    "status": "status_runtime",
    "stack": "stack_runtime",
}

#: 显式隔离标注（本阶段冻结能力——不创建值）
_BLOCKED_FIELDS = {
    "bet": {"status": "UNKNOWN", "reason": "evidence_limited"},
    "action": {"status": "UNKNOWN", "reason": "not_ready"},
}


def build_snapshot(observers: dict) -> dict:
    """observers: {name: observer实例}（其 summary() 提供当前帧产物）。

    纯只读聚合：observer.summary() → 快照字段（原样；缺失→None 显式）；
    零推断/转换/fallback；Bet/Action 显式 UNKNOWN 标注。
    """
    snap = {}
    for field, key in _OBSERVER_FIELDS.items():
        obs = observers.get(key)
        if obs is None or not hasattr(obs, "summary"):
            snap[field] = None
            continue
        # 原样透传整个 summary（当前帧产物——零子字段提取/转换/推断）
        snap[field] = obs.summary()
    # 显式隔离字段（EVIDENCE-LIMITED / NOT-READY——不创建值）
    snap.update({k: dict(v) for k, v in _BLOCKED_FIELDS.items()})
    # V3F-P13-C Bet Runtime 最小接线：提供 bet_runtime observer → 用其 summary
    # 替换 placeholder（无 observer 时保持 P9 证据隔离占位语义不变——
    # 非值级 fallback；Action 仍 NOT-READY）
    bet_obs = observers.get("bet_runtime")
    if bet_obs is not None and hasattr(bet_obs, "summary"):
        snap["bet"] = bet_obs.summary()
    # V3F-P13-ACTION Runtime 最小接线：提供 action_runtime observer → 用其
    # summary 替换 placeholder（无 observer 时 action 保持 P9 NOT-READY 占位）
    action_obs = observers.get("action_runtime")
    if action_obs is not None and hasattr(action_obs, "summary"):
        snap["action"] = action_obs.summary()
    snap["ground_truth_status"] = "runtime_state_non_gt"
    return snap
