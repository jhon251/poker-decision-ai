"""V3F-P30-7-C — ActionEvent Contract（事件来源白名单 + provenance）。

来源白名单（冻结；禁 source ∈ {FRAME_DIFF_GUESS, POT_DIFF, STACK_DIFF,
BET_DIFF}）：
  HAND_HISTORY  — 合法 hand-history 文本重放（行为事件）；
  EVENT_STREAM  — StateEngine 事件流（行为事件）；
  VISUAL_FACT   — V2M 视觉事实事件（ROI_CHANGED/BET_CHANGED/... NOT
                  行为归因——绝不由帧差推断 call/raise）；
  UNKNOWN       — 无合法来源（单帧视觉——不猜历史）。

ActionEvent 载体：event_id/timestamp/street/seat/action/amount/all_in/
source/status/provenance。
红线：视觉事实 ≠ 行为事件；不自动把 diff 归因为动作。
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: 事件来源白名单（行为事件仅 HAND_HISTORY / EVENT_STREAM）
ALLOWED_SOURCES = ("HAND_HISTORY", "EVENT_STREAM", "VISUAL_FACT", "UNKNOWN")

#: 行为动作（仅合法来源可产生）——视觉事实不能产生行为动作
BEHAVIOR_ACTIONS = ("fold", "check", "call", "bet", "raise", "allin")


@dataclass
class ActionEvent:
    event_id: str = ""
    timestamp: float = 0.0
    street: str = ""
    seat: str | None = None
    action: str = ""                # 行为动作（HAND_HISTORY/EVENT_STREAM only）
    amount: float | None = None
    all_in: bool = False
    source: str = "UNKNOWN"
    status: str = "UNKNOWN"         # RECOGNIZED | UNKNOWN | VISUAL_FACT_ONLY
    provenance: dict = field(default_factory=dict)

    def is_behavior(self) -> bool:
        """仅当来源合法且 action 属行为动作。"""
        return (self.source in ("HAND_HISTORY", "EVENT_STREAM")
                and self.action in BEHAVIOR_ACTIONS)

    def to_dict(self) -> dict:
        return {"event_id": self.event_id, "timestamp": self.timestamp,
                "street": self.street, "seat": self.seat,
                "action": self.action, "amount": self.amount,
                "all_in": self.all_in, "source": self.source,
                "status": self.status, "provenance": dict(self.provenance)}


def visual_fact_event(kind: str, frame_id: int,
                      detail: dict | None = None) -> ActionEvent:
    """V2M 视觉事实 → ActionEvent（status=VISUAL_FACT_ONLY——**不归因行为**；
    action=""。

    视觉事实（ROI/BET/STACK 变化）可以进入事件流，但绝不被自动解释为
    call/raise——调用方若需要行为历史必须走 HAND_HISTORY/EVENT_STREAM。
    """
    return ActionEvent(event_id=f"vf-{frame_id}", timestamp=0.0,
                       street="", seat=None, action="", amount=None,
                       all_in=False, source="VISUAL_FACT",
                       status="VISUAL_FACT_ONLY",
                       provenance={"kind": kind, "frame_id": frame_id,
                                   "detail": dict(detail or {})})


def unknown_source(reason: str = "single_frame_no_event_source") -> ActionEvent:
    """无合法来源（单帧）→ UNKNOWN（不猜历史）。"""
    return ActionEvent(source="UNKNOWN", status="UNKNOWN",
                       provenance={"reason": reason})
