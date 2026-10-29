"""V3F-P30-6-B — ActionEventSource Contract（合法事件源边界）。

允许：HAND_HISTORY（hand_history 文本重放）· EVENT_STREAM（StateEngine
事件流）。
禁止：SINGLE_FRAME→猜历史 · CURRENT_FRAME→补历史 · previous→冒充 ·
UNKNOWN→[]（均为红线）。

状态机（纯函数/无状态）：
- from_hand_history(text) → RECOGNIZED + event_count + events（合法重放）；
- no_source() → UNKNOWN（providence: no legal event source）；
- 绝不从帧推断（本模块无帧输入）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionEventSource:
    source: str                    # HAND_HISTORY | EVENT_STREAM | NONE
    status: str                    # RECOGNIZED | UNKNOWN
    event_count: int = 0
    events: list = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {"source": self.source, "status": self.status,
                "event_count": self.event_count, "events": list(self.events),
                "rationale": self.rationale}


def from_hand_history(parsed_actions: list | None) -> ActionEventSource:
    """合法源：hand_history parser 的 actions → RECOGNIZED。

    parsed_actions 必须来自 parse_hand_history(.actions)（真实文本重放）——
    不是帧推断。空列表 = 无动作记录 → UNKNOWN（≠[] 伪装）。
    """
    acts = list(parsed_actions or [])
    if not acts:
        return ActionEventSource(source="HAND_HISTORY", status="UNKNOWN",
                                 event_count=0, events=[],
                                 rationale="hand_history 无动作记录"
                                           "（非空历史伪造——UNKNOWN）")
    recs = []
    for i, a in enumerate(acts):
        if not isinstance(a, dict):
            continue
        act = str(a.get("action") or "").rstrip("s")
        if act in ("post", "kollect", ""):
            continue                          # 盲注不是动作（不伪造）
        recs.append({"street": str(a.get("street") or "preflop"),
                     "player_id": str(a.get("player") or a.get(
                         "player_id") or ""),
                     "action": act, "amount": float(a.get("amount") or 0.0),
                     "all_in": bool(a.get("all_in")), "seq": i + 1})
    if not recs:
        return ActionEventSource(source="HAND_HISTORY", status="UNKNOWN",
                                 event_count=0, events=[],
                                 rationale="hand_history 仅盲注无动作"
                                           "（UNKNOWN）")
    return ActionEventSource(source="HAND_HISTORY", status="RECOGNIZED",
                             event_count=len(recs), events=recs,
                             rationale="合法 hand_history 文本重放")


def from_event_stream(events: list | None) -> ActionEventSource:
    """合法源：StateEngine 事件流（非帧推断）。"""
    evs = list(events or [])
    if not evs:
        return ActionEventSource(source="EVENT_STREAM", status="UNKNOWN",
                                 event_count=0, events=[],
                                 rationale="无事件流（UNKNOWN——非 [] 伪装）")
    return ActionEventSource(source="EVENT_STREAM", status="RECOGNIZED",
                             event_count=len(evs), events=evs,
                             rationale="合法 StateEngine 事件流")


def no_source() -> ActionEventSource:
    """无合法事件源（单帧视觉）→ UNKNOWN（红线：不得从帧猜历史）。"""
    return ActionEventSource(source="NONE", status="UNKNOWN", event_count=0,
                             events=[],
                             rationale="单帧视觉无事件序列——不伪造历史")
