"""P31-D/E — OpponentProfile + Range Integration（sample-aware）。

profile 来源（禁止伪造/禁止小样本冒充）：
  RECOGNIZED — 有合法历史（>= LOW 最小样本）；
  UNKNOWN    — 无可靠数据（不得转 random/固定类型）。

Range mode 新增三级（P31-E）：
  PLAYER_BASED — 足够样本（>=25 = MEDIUM 起）——历史驱动；
  POPULATION   — 仅群体统计（无玩家数据）；
  UNKNOWN      — 无可靠数据（不得转 random）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .player_history import (PlayerHistoryStats, reliability_for,
                             RELIABILITY_LEVELS)

#: PLAYER_BASED 最小样本（MEDIUM 起——25 hands；<25 不升级）
PLAYER_BASED_MIN_HANDS = 25


@dataclass
class OpponentProfile:
    player_id: str = ""
    status: str = "UNKNOWN"               # RECOGNIZED | UNKNOWN
    sample_size: int = 0
    reliability: str = "INSUFFICIENT"
    preflop_stats: dict = field(default_factory=dict)
    postflop_stats: dict = field(default_factory=dict)
    position_stats: dict = field(default_factory=dict)
    # P43-C4：条件行为统计（真实历史加工；空 = 无条件数据，诚实缺失）
    conditional_stats: dict = field(default_factory=dict)
    source: str = ""                      # HAND_HISTORY | EVENT_STREAM
    provenance: str = ""

    @classmethod
    def from_history(cls, s: PlayerHistoryStats) -> "OpponentProfile":
        """合法历史 → profile（RECOGNIZED——仅当有样本）。"""
        recognized = s is not None and s.hands > 0
        return cls(
            player_id=(s.player_id if s else ""),
            status="RECOGNIZED" if recognized else "UNKNOWN",
            sample_size=(s.hands if s else 0),
            reliability=(s.reliability if s else "INSUFFICIENT"),
            preflop_stats={"vpip": (s.vpip if s else None),
                           "pfr": (s.pfr if s else None),
                           "three_bet": (s.three_bet if s else None),
                           "fold_to_three_bet": (s.fold_to_three_bet
                                                 if s else None)},
            postflop_stats={"cbet": (s.cbet if s else None),
                            "fold_to_cbet": (s.fold_to_cbet if s else None),
                            "aggression": (s.aggression if s else None),
                            "wtsd": (s.wtsd if s else None)},
            position_stats=dict((s.position_stats if s else {})),
            conditional_stats=dict(getattr(s, "conditional_stats", None)
                                   or {}),
            source=(s.source if s else ""),
            provenance=(s.provenance if s else "no_history"))

    @classmethod
    def unknown(cls, reason: str = "no_reliable_history") -> "OpponentProfile":
        return cls(player_id="", status="UNKNOWN", sample_size=0,
                   reliability="INSUFFICIENT", source="", provenance=reason)

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "status": self.status,
                "sample_size": self.sample_size,
                "reliability": self.reliability,
                "preflop_stats": dict(self.preflop_stats),
                "postflop_stats": dict(self.postflop_stats),
                "position_stats": dict(self.position_stats),
                "conditional_stats": dict(self.conditional_stats),
                "source": self.source, "provenance": self.provenance}


def range_mode_for(profile: OpponentProfile | None,
                   has_population: bool = False) -> str:
    """profile → range mode（P31-E 严格分级）。

    PLAYER_BASED 仅当：profile RECOGNIZED + sample >= 25（MEDIUM 级）；
    POPULATION 仅当：无 player 数据但群体数据存在；
    UNKNOWN：无可靠数据（绝不 random）。
    """
    if profile is not None and profile.status == "RECOGNIZED" \
            and profile.sample_size >= PLAYER_BASED_MIN_HANDS:
        return "PLAYER_BASED"
    if has_population:
        return "POPULATION"
    return "UNKNOWN"
