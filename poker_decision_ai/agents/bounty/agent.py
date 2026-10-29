"""Bounty Agent (赏金基础): deterministic bounty context extracted from state.

V1.1 §4b 契约。全部确定性计算，公式在 domain/poker/bounty.py。

Input : {state: dict, tournament_stage: dict|None, icm: dict|None}
Output: BountyOutput

注意：`bounty` 模块尚未在 domain/poker/__init__.py 导出，这里直接导入子模块
（不改 __init__.py，交付报告说明需要的导出）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker.bounty import (
    BountyInfo,
    bounty_pressure,
    bounty_to_chips,
    knockout_probability,
    progressive_bounty_growth,
)


@dataclass
class BountyOutput:
    bounty_type: str = ""                       # pko | ko | mystery | ""
    is_bounty: bool = False
    progressive: bool = False
    hero_bounty: float = 0.0                    # hero 当前赏金（$）
    villain_bounty: float = 0.0                 # 最大 active 对手赏金（$）
    villain_bounties: dict = field(default_factory=dict)   # {pid: $}
    current_bounty: float = 0.0                 # 别名：hero 当前赏金（$）
    bounty_value: float = 0.0                   # 最大对手赏金价值（$）
    knockout_value: float = 0.0                 # 淘汰该对手可获赏金（$）
    progressive_growth: float = 0.0             # 淘汰后 hero 赏金增长值（$）
    bounty_chips: float = 0.0                   # 最大赏金换算筹码（chip_value 折算）
    chip_value: float = 0.0                     # $/chip
    bounty_pressure: float = 0.0                # 0..1 赏金/有效筹码压力
    bounty_ev: float = 0.0                      # 追赏金 EV（bb，确定性估算）
    bounty_confidence: float = 0.0              # 0..1 估算质量
    covered: bool = False                       # hero 码量是否覆盖最大赏金对手
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _bb(state: dict) -> float:
    return float(state.get("bb") or 0.0) or 1.0


def _hero_of(players: list) -> dict | None:
    for p in players:
        if isinstance(p, dict) and p.get("is_hero"):
            return p
    return None


def _active_villains(players: list) -> list:
    return [p for p in players
            if isinstance(p, dict) and not p.get("is_hero")
            and p.get("status") in ("active", "allin")]


def _eff_stack_bb(state: dict) -> float:
    """有效码量（bb）：hero 与所有 active/allin 对手的最小码量。"""
    bb = _bb(state)
    stacks: list[float] = []
    for p in state.get("players") or []:
        if not isinstance(p, dict):
            continue
        if p.get("is_hero") or p.get("status") in ("active", "allin"):
            s = float(p.get("stack") or 0.0)
            if s > 0:
                stacks.append(s)
    eff = min(stacks) if stacks else 0.0
    return eff / bb


class BountyAgent(BaseAgent):
    name: ClassVar[str] = "bounty"
    version: ClassVar[str] = "1.1.0"
    zh_name: ClassVar[str] = "赏金基础 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state", "tournament_stage", "icm"]
    supported_modes: ClassVar[list[str]] = ["bounty"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = BountyOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for bounty analysis"
        # 无赏金（cash）也 applicable=True：输出 is_bounty=False 全 0，由模式过滤跳过。
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> BountyOutput:
        state = inp.get("state") or {}
        out = BountyOutput()
        warnings: list[dict] = []

        # ---- 确定性 bounty 上下文（domain BountyInfo.from_state） ----
        info = BountyInfo.from_state(state)
        out.bounty_type = info.bounty_type
        out.is_bounty = info.is_bounty
        out.progressive = info.progressive
        out.hero_bounty = info.hero_bounty
        out.current_bounty = info.hero_bounty
        out.chip_value = info.chip_value

        if not info.is_bounty:
            out.confidence = 1.0
            out.bounty_confidence = 0.0
            warnings.append({"code": "NO_BOUNTY",
                             "message": "state 无赏金块（cash / 普通 MTT）；输出全 0，由模式过滤跳过"})
            out.warnings = warnings
            out.evidence = [{"kind": "bounty", "source": "state",
                             "text": "no bounty context; all-zero output", "weight": 1.0}]
            return out

        # ---- chip_value 无数据（total_chips=0） ----
        if info.chip_value <= 0:
            warnings.append({"code": "NO_CHIP_VALUE",
                             "message": "total_chips=0 导致 chip_value 无法计算，赏金筹码折算为 0"})

        players = state.get("players") or []
        hero_p = _hero_of(players)
        active_villains = _active_villains(players)

        # ---- 每个 active villain 的赏金 ----
        villain_bounties: dict[str, float] = {}
        for p in active_villains:
            pid = str(p.get("player_id") or "")
            if pid:
                villain_bounties[pid] = float(p.get("bounty") or 0.0)
        out.villain_bounties = villain_bounties

        # ---- 最大赏金对手（active） ----
        target: dict | None = None
        for p in active_villains:
            if target is None or float(p.get("bounty") or 0.0) > float(target.get("bounty") or 0.0):
                target = p

        bb = _bb(state)
        if target is None:
            out.villain_bounty = 0.0
            out.bounty_value = 0.0
            out.covered = False
            warnings.append({"code": "NO_ACTIVE_VILLAIN",
                             "message": "无 active villain，无法计算最大赏金对手相关字段"})
        else:
            villain_bounty = float(target.get("bounty") or 0.0)
            hero_stack = float(hero_p.get("stack") or 0.0) if hero_p else 0.0
            villain_stack = float(target.get("stack") or 0.0)
            covered = villain_stack > 0 and hero_stack >= villain_stack
            out.villain_bounty = villain_bounty
            out.bounty_value = villain_bounty
            out.covered = covered

        # ---- knockout_value：PKO=growth_rate*对手赏金；KO/mystery=对手赏金 ----
        if info.progressive:
            out.knockout_value = info.growth_rate * out.villain_bounty
        else:
            out.knockout_value = out.villain_bounty

        # ---- progressive_growth：淘汰后 hero 赏金增长值（$） ----
        # 用 domain progressive_bounty_growth(hero, victim, rate) = hero + rate*victim，
        # 减去 hero 当前赏金即"增长值"。非 PKO 时为 0。
        if info.progressive:
            new_hero_bounty = progressive_bounty_growth(info.hero_bounty,
                                                        out.villain_bounty,
                                                        info.growth_rate)
            out.progressive_growth = new_hero_bounty - info.hero_bounty
        else:
            out.progressive_growth = 0.0

        # ---- bounty_chips（最大赏金换算筹码）+ 赏金压力 ----
        out.bounty_chips = bounty_to_chips(out.villain_bounty, info.chip_value)
        bounty_chips_bb = out.bounty_chips / bb if bb > 0 else 0.0
        out.bounty_pressure = bounty_pressure(bounty_chips_bb, _eff_stack_bb(state))

        # ---- bounty_ev：KO 概率(中性 0.5) × 赏金筹码（bb） ----
        # bounty agent 无 equity 输入，KO 概率用 0.5 中性近似（注释）。
        ko_prob = knockout_probability(0.5, out.covered)
        out.bounty_ev = ko_prob * bounty_chips_bb
        warnings.append({"code": "NEUTRAL_EQUITY",
                         "message": "bounty agent 无 equity 输入，KO 概率用 0.5 中性近似"})
        # 无 all-in 对手：bounty_ev 为潜在 KO 价值（假设可淘汰时的期望）
        if not any(p.get("status") == "allin" for p in active_villains):
            warnings.append({"code": "NO_ALLIN",
                             "message": "无 all-in 对手，bounty_ev 为潜在 KO 价值（非即时兑现）"})

        # ---- 估算质量 ----
        out.bounty_confidence = 0.8 if (out.covered and info.chip_value > 0) else 0.5
        out.confidence = round(out.bounty_confidence, 4)

        out.evidence = [{
            "kind": "bounty", "source": "state",
            "text": (f"type={out.bounty_type} progressive={out.progressive} "
                     f"hero_bounty={out.hero_bounty} villain_bounty={out.villain_bounty} "
                     f"covered={out.covered} pressure={out.bounty_pressure}"),
            "weight": 1.0,
        }]
        out.warnings = warnings
        return out
