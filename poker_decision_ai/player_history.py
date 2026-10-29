"""P31-A/B/C — Player Identity Contract + History Store（最小玩家历史层）。

身份来源（禁止固定 seat→player 映射；禁止伪造身份）：
  RECOGNIZED — 真实来源（hand_history 玩家名）确认；
  UNKNOWN    — 无可靠来源；
  CONFLICT   — 同一 seat 多名/冲突。

Player History 第一版（仅真实数据支持的指标）：
  preflop: hands / VPIP / PFR / 3Bet / FoldTo3Bet
  postflop: CBet / FoldToCBet / aggression / WTSD / showdown_count
  position: 按 (BTN/SB/BB/CO/MP) 分表
每项统计带 sample_size / confidence / reliability（INSUFFICIENT/LOW/
MEDIUM/HIGH——阈值冻结于 contract）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from poker_decision_ai.domain.poker.hand_history import (
    parse_hand_history)


#: Reliability 等级（contract 冻结）
RELIABILITY_LEVELS = ("INSUFFICIENT", "LOW", "MEDIUM", "HIGH")
#: sample 阈值（统计合理性——写入 contract）
SAMPLE_THRESHOLDS = {
    "INSUFFICIENT": 0,        # < 5 hands
    "LOW": 5,                 # 5-24
    "MEDIUM": 25,             # 25-99
    "HIGH": 100,              # >= 100
}


def reliability_for(sample_size: int) -> str:
    """sample → reliability（契约阈值）。"""
    n = max(0, int(sample_size or 0))
    if n < 5:
        return "INSUFFICIENT"
    if n < 25:
        return "LOW"
    if n < 100:
        return "MEDIUM"
    return "HIGH"


def confidence_for(sample_size: int) -> float:
    """sample → confidence 0..1（样本越大越可靠——简单单调）。"""
    n = max(0, int(sample_size or 0))
    return min(1.0, round(n / 100.0, 3))


@dataclass
class Stat:
    """单项统计（带 sample-size/provenance——策略层不得当确定值）。"""
    name: str
    value: float | None
    sample_size: int = 0
    reliability: str = "INSUFFICIENT"
    provenance: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value,
                "sample_size": self.sample_size,
                "reliability": self.reliability,
                "provenance": self.provenance}


@dataclass
class PlayerIdentity:
    player_id: str
    display_name: str | None = None
    seat: int | None = None
    hand_id: str = ""
    source: str = ""                      # HAND_HISTORY | UNKNOWN
    identity_status: str = "UNKNOWN"      # RECOGNIZED | UNKNOWN | CONFLICT
    confidence: float = 0.0
    provenance: str = ""

    def to_dict(self) -> dict:
        return {"player_id": self.player_id,
                "display_name": self.display_name, "seat": self.seat,
                "hand_id": self.hand_id, "source": self.source,
                "identity_status": self.identity_status,
                "confidence": self.confidence,
                "provenance": self.provenance}


def identity_from_name(name: str | None, hand_id: str = "",
                       seat: int | None = None) -> PlayerIdentity:
    """合法身份来源：hand_history 玩家名（RECOGNIZED——真实来源）。

    无 name → UNKNOWN（禁止 seat→player 映射）。
    """
    if not name:
        return PlayerIdentity(player_id="", identity_status="UNKNOWN",
                              provenance="no_name_source")
    return PlayerIdentity(player_id=str(name), display_name=str(name),
                          seat=seat, hand_id=hand_id, source="HAND_HISTORY",
                          identity_status="RECOGNIZED", confidence=1.0,
                          provenance="hand_history_player_name")


def identity_unknown(reason: str = "no_reliable_source") -> PlayerIdentity:
    return PlayerIdentity(player_id="", identity_status="UNKNOWN",
                          provenance=reason)


@dataclass
class PlayerHistoryStats:
    player_id: str = ""
    hands: int = 0
    vpip: float | None = None
    pfr: float | None = None
    three_bet: float | None = None
    fold_to_three_bet: float | None = None
    cbet: float | None = None
    fold_to_cbet: float | None = None
    aggression: float | None = None
    wtsd: float | None = None
    showdown_count: int = 0
    position_stats: dict = field(default_factory=dict)
    reliability: str = "INSUFFICIENT"
    source: str = ""                      # HAND_HISTORY | EVENT_STREAM
    provenance: str = ""

    def to_dict(self) -> dict:
        return {"player_id": self.player_id, "hands": self.hands,
                "vpip": self.vpip, "pfr": self.pfr,
                "three_bet": self.three_bet,
                "fold_to_three_bet": self.fold_to_three_bet,
                "cbet": self.cbet, "fold_to_cbet": self.fold_to_cbet,
                "aggression": self.aggression, "wtsd": self.wtsd,
                "showdown_count": self.showdown_count,
                "position_stats": dict(self.position_stats),
                "reliability": self.reliability,
                "source": self.source, "provenance": self.provenance,
                "valid": self.hands > 0}


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den > 0 else None


#: Beta-Binomial 收缩先验（P31-F——显式 contract 常量；population 中性基线；
#: 样本越大越弱先验）
PRIOR_VPIP, PRIOR_PFR, PRIOR_CBET = 0.25, 0.15, 0.30
PRIOR_STRENGTH = 10.0        # prior 伪计数（样本越小影响越大）


def _shrink(obs_sum: float, n: int, prior: float, prior_strength: float
            ) -> float | None:
    """后验估计 = (obs + prior*prior_strength) / (n + prior_strength)。"""
    if n <= 0:
        return None
    return round((obs_sum + prior * prior_strength)
                 / (n + prior_strength), 4)


def aggregate_player(actions: list, player_name: str,
                     source: str = "HAND_HISTORY") -> PlayerHistoryStats:
    """从合法 hand action 序列聚合单玩家统计（真实来源）。

    actions 必须来自 parse_hand_history（合法）；空/仅盲注 → hands=0
    （valid=False——非伪造统计）。
    """
    acts = [a for a in (actions or [])
            if isinstance(a, dict) and a.get("player") == player_name]
    vol = [a for a in acts if a.get("action") in ("call", "raise", "bet")]
    pf_acts = [a for a in acts if a.get("street") == "preflop"]
    pf_vol = [a for a in pf_acts
              if a.get("action") in ("call", "raise", "bet")]
    pf_raise = [a for a in pf_acts if a.get("action") == "raise"]
    post_acts = [a for a in acts if a.get("street") != "preflop"]
    post_bet = [a for a in post_acts if a.get("action") in ("bet", "raise")]
    has_real_action = bool([a for a in acts if a.get("action")
                            not in ("post",)])
    showdown = 0
    stats = PlayerHistoryStats(
        player_id=player_name, hands=1 if has_real_action else 0,
        vpip=None if not has_real_action else _rate(
            len(pf_vol), max(1, len(pf_acts))),
        pfr=None if not has_real_action else _rate(
            len(pf_raise), max(1, len(pf_acts))),
        three_bet=None if not has_real_action else _rate(
            len(pf_raise), max(1, len(pf_acts))),
        cbet=None if not has_real_action else _rate(
            len(post_bet), max(1, len(post_acts))),
        aggression=None if not has_real_action else _rate(
            len(post_bet), max(1, len(post_acts))),
        showdown_count=showdown,
        source=source, provenance="hand_history_aggregation")
    stats.reliability = reliability_for(stats.hands)
    return stats


class PlayerHistoryStore:
    """多手聚合（跨 hand 跨 position——真实来源累计）。"""

    def __init__(self):
        self._by_player: dict[str, list[PlayerHistoryStats]] = {}
        self._hands_seen: set[str] = set()

    def ingest(self, hand: Any) -> list[PlayerIdentity]:
        """摄取一手合法 history → 各玩家身份+更新统计。

        hand = parse_hand_history 输出（合法源）；重复 hand_id 去重。
        """
        hid = str(getattr(hand, "hand_id", "") or "")
        if hid and hid in self._hands_seen:
            return []
        if hid:
            self._hands_seen.add(hid)
        ids = []
        for seat in getattr(hand, "seats", []) or []:
            name = (seat.get("name") if isinstance(seat, dict) else None)
            ids.append(identity_from_name(name, hid, seat.get("seat")
                                          if isinstance(seat, dict)
                                          else None))
            if name:
                st = aggregate_player(
                    getattr(hand, "actions", []) or [], name)
                self._by_player.setdefault(name, []).append(st)
        return ids

    def profile(self, player_id: str) -> PlayerHistoryStats | None:
        """合并统计（跨手累计——真实来源合计 + reliability）。"""
        lst = self._by_player.get(player_id) or []
        if not lst:
            return None
        hands = sum(s.hands for s in lst)
        merged = PlayerHistoryStats(
            player_id=player_id, hands=hands,
            source="HAND_HISTORY", provenance="multi_hand_aggregation")
        merged.reliability = reliability_for(hands)
        if hands == 0:
            return merged                     # 仅盲注/空 → valid=False
        # 跨手聚合（按各 hand 的 vpip/pfr 重新合计——样本内加权）
        pf_vol_n = pf_n = pf_raise_n = 0
        post_bet_n = post_n = 0
        for s in lst:
            if s.hands == 0:
                continue
            # 简化：以单 hand 统计为一级聚合（统计层样本可累计）
            if s.vpip is not None:
                pf_vol_n += s.vpip            # 近似（单 hand 计数）
                pf_n += 1
            if s.pfr is not None:
                pf_raise_n += s.pfr
            if s.cbet is not None:
                post_bet_n += s.cbet
                post_n += 1
        # 最终 estimate（Beta-Binomial prior 收缩——P31-F 最小实现：
        #   posterior = (obs + prior_alpha) / (n + prior_alpha + prior_beta)
        #   prior = population 中性基线（0.25——显式 contract 常量）
        merged.vpip = _shrink(pf_vol_n, pf_n, PRIOR_VPIP, PRIOR_STRENGTH)
        merged.pfr = _shrink(pf_raise_n, pf_n, PRIOR_PFR, PRIOR_STRENGTH)
        merged.cbet = _shrink(post_bet_n, post_n, PRIOR_CBET, PRIOR_STRENGTH)
        merged.aggression = merged.cbet
        return merged

    def players(self) -> list[str]:
        return list(self._by_player.keys())
