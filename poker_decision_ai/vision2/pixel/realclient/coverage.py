"""V2L — 时序变化事件 / 覆盖率矩阵 / 完整性（确定性，无视觉依赖）。

原则：
  - Temporal 只记录"发生了什么视觉变化"（STACK_CHANGED / BET_APPEARED /
    BUTTON_MOVED / ...），绝不产生 PLAYER_FOLDED / PLAYER_CALLED 等行为推断；
  - Coverage 只评估"视觉事实是否可观察"，不追求字段数；
  - Completeness 只判断"信息是否存在"，绝不产生下注/EV/策略建议。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ------------------------------------------------------------------ P8 时序
def temporal_changes(prev: dict, curr: dict) -> list:
    """两帧逐座视觉事实 diff → 纯视觉变化事件列表。

    prev/curr: {seat: {"stack": float|None, "bet": float|None,
                       "folded": bool|None, "all_in": bool|None,
                       "button": bool}} + board/hero 键。
    只输出视觉变化事实；不做行为归因。
    """
    events = []
    for seat in range(6):
        a = (prev.get(seat) or {})
        b = (curr.get(seat) or {})
        if a.get("stack") != b.get("stack"):
            events.append({"kind": "STACK_CHANGED", "seat": seat,
                           "from": a.get("stack"), "to": b.get("stack")})
        if not a.get("bet") and b.get("bet"):
            events.append({"kind": "BET_APPEARED", "seat": seat,
                           "to": b.get("bet")})
        if a.get("bet") and not b.get("bet"):
            events.append({"kind": "BET_DISAPPEARED", "seat": seat,
                           "from": a.get("bet")})
        if not a.get("folded") and b.get("folded"):
            events.append({"kind": "FOLD_MARKER_APPEARED", "seat": seat})
        if not a.get("all_in") and b.get("all_in"):
            events.append({"kind": "ALLIN_MARKER_APPEARED", "seat": seat})
        if a.get("button") != b.get("button"):
            events.append({"kind": "BUTTON_MOVED", "seat": seat,
                           "to": b.get("button")})
    if len(prev.get("board") or []) < len(curr.get("board") or []):
        events.append({"kind": "BOARD_CARD_APPEARED",
                       "count": len(curr.get("board") or [])})
    return events


# ------------------------------------------------------------------ P9 覆盖
COVERAGE = ("RELIABLE", "PARTIAL", "UNCERTAIN", "UNAVAILABLE")


@dataclass
class CoverageEntry:
    field: str
    status: str
    reason: str

    def to_dict(self) -> dict:
        return {"field": self.field, "status": self.status,
                "reason": self.reason}


def coverage_matrix() -> list:
    """当前 Vision Agent 视觉事实覆盖率（静态，基于既有真实证据）。"""
    return [
        CoverageEntry("table", "RELIABLE", "10 帧实测，几何稳定"),
        CoverageEntry("street", "PARTIAL", "由 board 牌数推导；board 识别 2 张→null"),
        CoverageEntry("board", "PARTIAL", "rank+suit 4/10；suit 判别力待独立验证"),
        CoverageEntry("hero_cards", "UNCERTAIN", "严格门控；判别余量不足"),
        CoverageEntry("pot", "UNAVAILABLE", "客户端无 pot 显示（10 帧零金文本）"),
        CoverageEntry("players", "RELIABLE", "6 座几何实测"),
        CoverageEntry("player.stack", "PARTIAL", "23/60 字段；p2/p4 百万缩写待 M 模板"),
        CoverageEntry("player.bet", "UNCERTAIN", "仅 p5 有实测 bet ROI"),
        CoverageEntry("player.folded", "UNCERTAIN", "p2 盖牌标记已测 ROI，模板未建"),
        CoverageEntry("player.all_in", "UNCERTAIN", "无可靠视觉标记"),
        CoverageEntry("button", "UNCERTAIN", "帧6 小区域检测过；紧凑过滤 0/10 待校准"),
        CoverageEntry("position", "UNAVAILABLE",
                      "由 positions_for 推导；视觉层不计算"),
    ]


# ------------------------------------------------------------------ P10 完整性
def completeness(state_players: list, board, hero_cards, street) -> dict:
    """信息存在性判断（COMPLETE_FOR_STATE_RECONSTRUCTION / INCOMPLETE）。"""
    missing = []
    if not street:
        missing.append("street")
    if not board:
        missing.append("board")
    if not hero_cards:
        missing.append("hero_cards")
    for p in state_players:
        if p.get("stack") is None:
            missing.append(f"player_{p.get('seat')}.stack")
        if p.get("bet") is None:
            missing.append(f"player_{p.get('seat')}.bet")
    status = ("COMPLETE_FOR_STATE_RECONSTRUCTION" if not missing
              else "INCOMPLETE")
    return {"status": status, "missing_fields": missing}
