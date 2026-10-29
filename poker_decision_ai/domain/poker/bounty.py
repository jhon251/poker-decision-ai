"""确定性 Bounty 数学（V1.1 PKO/KO/Mystery）—— 禁止 LLM 猜测。

PKO 核心公式（均有注释推导）：

  EV(call, 可淘汰对手) = eq*(pot + to_call + bounty_chips) - (1-eq)*to_call
  盈亏平衡 → effective_required_equity = to_call / (pot + 2*to_call + bounty_chips)

即：对手头上的赏金把"所需胜率"向下拉（赏金越大越值得追），这与普通
底池赔率所需胜率（to_call / (pot + 2*to_call)）是两个不同概念，必须在
Bounty Equity Agent 中分别输出 normal 与 bounty_adjusted。
"""
from __future__ import annotations

from dataclasses import dataclass, field


def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def normal_required_equity(pot: float, to_call: float) -> float:
    """普通所需胜率 = to_call / (pot + 2*to_call)（pot 含对手当前下注）。"""
    if to_call <= 0:
        return 0.0
    return to_call / (pot + 2 * to_call)


def bounty_to_chips(bounty_usd: float, chip_dollar_value: float) -> float:
    """赏金换算为筹码：bounty_usd / 每筹码美元价值。"""
    if chip_dollar_value <= 0 or bounty_usd <= 0:
        return 0.0
    return bounty_usd / chip_dollar_value


def chip_dollar_value(total_prize_pool: float, total_chips: float) -> float:
    """每筹码的美元价值（近似：剩余奖池/总筹码；PKO 常把赏金池折算进奖池）。"""
    if total_chips <= 0:
        return 0.0
    return total_prize_pool / total_chips


def effective_required_equity(pot: float, to_call: float, bounty_chips: float) -> float:
    """加入赏金后的有效所需胜率（对手可被淘汰时）。

    推导：EV = eq*(pot+to_call+B) - (1-eq)*to_call = 0
         => eq* = to_call / (pot + 2*to_call + B)
    """
    if to_call <= 0:
        return 0.0
    denom = pot + 2 * to_call + max(0.0, bounty_chips)
    if denom <= 0:
        return 1.0
    return to_call / denom


def knockout_probability(hero_equity: float, villain_covered: bool) -> float:
    """淘汰对手的概率 ≈ 英雄胜率（对手全下且英雄码量能覆盖时）。

    近似假设（注释）：摊牌胜率即淘汰概率；多人底池需三方胜率，V1.1 取
    双方胜率近似并在 Bounty Agent 中标注。
    """
    if not villain_covered:
        return 0.0
    return clip(float(hero_equity))


def bounty_ev_chips(hero_equity: float, villain_covered: bool, bounty_chips: float) -> float:
    """赏金 EV（筹码）= P(淘汰) * 赏金筹码。"""
    return knockout_probability(hero_equity, villain_covered) * max(0.0, bounty_chips)


def call_ev_with_bounty(equity: float, pot_bb: float, to_call_bb: float,
                        bounty_chips_bb: float) -> float:
    """含赏金的跟注 EV（bb）：eq*(pot+to_call+B) - (1-eq)*to_call。"""
    return equity * (pot_bb + to_call_bb + max(0.0, bounty_chips_bb)) - (1 - equity) * to_call_bb


def progressive_bounty_growth(hero_bounty: float, victim_bounty: float,
                              rate: float = 0.5) -> float:
    """PKO 递增赏金：淘汰对手后 hero 赏金 += rate * victim_bounty（默认 50%）。"""
    return float(hero_bounty) + float(rate) * float(victim_bounty)


def bounty_pressure(bounty_chips_bb: float, effective_stack_bb: float) -> float:
    """赏金压力：赏金筹码相对有效码量的强度，0..1。"""
    if effective_stack_bb <= 0:
        return 0.0
    return clip(bounty_chips_bb / max(1.0, effective_stack_bb))


def bounty_share_of_buyin(bounty_usd: float, buyin_usd: float) -> float:
    """赏金占买入比例（K.O. 固定赏金、PKO 起始赏金的归一化参考）。"""
    if buyin_usd <= 0:
        return 0.0
    return clip(bounty_usd / buyin_usd, 0.0, 2.0)


@dataclass
class BountyInfo:
    """统一的赏金上下文（State 里的 bounty 块）。"""

    bounty_type: str = ""               # pko | ko | mystery | ""（无赏金）
    is_bounty: bool = False
    progressive: bool = False           # PKO 递增
    growth_rate: float = 0.5            # PKO 递增比例
    total_bounty_pool: float = 0.0      # 剩余赏金池（$）
    hero_bounty: float = 0.0
    villain_bounty: float = 0.0         # 最大对手赏金
    chip_value: float = 0.0             # $/chip（折算用）
    mystery_phase_active: bool = False  # Mystery Bounty 揭晓阶段是否开始
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bounty_type": self.bounty_type, "is_bounty": self.is_bounty,
            "progressive": self.progressive, "growth_rate": self.growth_rate,
            "total_bounty_pool": self.total_bounty_pool,
            "hero_bounty": self.hero_bounty, "villain_bounty": self.villain_bounty,
            "chip_value": self.chip_value,
            "mystery_phase_active": self.mystery_phase_active,
            "notes": list(self.notes),
        }

    @classmethod
    def from_state(cls, state: dict) -> "BountyInfo":
        t = state.get("tournament") if isinstance(state.get("tournament"), dict) else {}
        players = state.get("players") or []
        hero_b = 0.0
        villain_b = 0.0
        # villain_bounty 只统计仍在争夺本手底池的对手（active/allin），
        # 已弃牌的玩家不构成当前决策的淘汰对象（契约 §4b bounty 语义）。
        for p in players:
            if not isinstance(p, dict):
                continue
            b = float(p.get("bounty") or 0.0)
            if p.get("is_hero"):
                hero_b = b
            elif b > villain_b and p.get("status") in ("active", "allin"):
                villain_b = b
        btype = str(t.get("bounty_type") or "")
        total_pool = float(t.get("total_bounty_pool") or 0.0)
        total_chips = float(t.get("total_chips") or 0.0)
        prize_pool = float(t.get("prize_pool") or 0.0)
        chip_val = chip_dollar_value(prize_pool + total_pool, total_chips)
        return cls(
            bounty_type=btype,
            is_bounty=bool(t.get("is_bounty") or btype),
            progressive=btype == "pko" or bool(t.get("progressive")),
            growth_rate=float(t.get("bounty_rate") or 0.5),
            total_bounty_pool=total_pool,
            hero_bounty=hero_b, villain_bounty=villain_b,
            chip_value=chip_val,
            mystery_phase_active=bool(t.get("mystery_phase_active")),
        )
