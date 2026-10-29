"""Bounty Equity Agent (赏金胜率): 赏金如何拉低跟注所需胜率。

V1.1 §4b 契约。全确定性，公式在 domain/poker/bounty.py 与 pot_odds.py。

核心公式（见 bounty.py 头注释）：
    EV(call, 可淘汰对手) = eq*(pot + to_call + bounty_chips) - (1-eq)*to_call
    normal_required_equity  = to_call / (pot + 2*to_call)
    effective_required_equity = to_call / (pot + 2*to_call + bounty_chips)

Input : {state: dict, equity: dict|None, bounty: dict|None}
Output: BountyEquityOutput
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker.bounty import (
    call_ev_with_bounty,
    effective_required_equity,
    normal_required_equity,
)
from poker_decision_ai.domain.poker.pot_odds import ev_of_call


@dataclass
class BountyEquityOutput:
    normal_required_equity: float = 0.0
    bounty_adjusted_required_equity: float = 0.0
    required_equity_delta: float = 0.0          # normal - adjusted（赏金拉低的幅度）
    bounty_chips_used: float = 0.0              # bb
    villain_covered: bool = False
    ev_call_normal_bb: float = 0.0
    ev_call_with_bounty_bb: float = 0.0
    effective_decision_equity: float = 0.0      # Judge 应使用的所需胜率
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _bb(state: dict) -> float:
    return float(state.get("bb") or 0.0) or 1.0


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class BountyEquityAgent(BaseAgent):
    name: ClassVar[str] = "bounty_equity"
    version: ClassVar[str] = "1.1.0"
    zh_name: ClassVar[str] = "赏金胜率 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state", "equity", "bounty"]
    supported_modes: ClassVar[list[str]] = ["bounty"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = BountyEquityOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for bounty equity analysis"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> BountyEquityOutput:
        state = inp.get("state") or {}
        equity_in = inp.get("equity") or {}
        bounty_in = inp.get("bounty") or {}
        out = BountyEquityOutput()
        warnings: list[dict] = []

        bb = _bb(state)
        pot = _num(state.get("pot"))
        to_call = _num(state.get("to_call"))
        pot_bb = pot / bb
        to_call_bb = to_call / bb

        # ---- 普通所需胜率 ----
        normal = normal_required_equity(pot_bb, to_call_bb)
        out.normal_required_equity = round(normal, 6)

        # ---- 赏金输入 ----
        covered = bool(bounty_in.get("covered")) if isinstance(bounty_in, dict) else False
        out.villain_covered = covered
        # bounty 输出的 bounty_chips 是绝对筹码；折算 bb
        bounty_chips_bb = _num(bounty_in.get("bounty_chips")) / bb if isinstance(bounty_in, dict) else 0.0
        has_bounty = isinstance(bounty_in, dict) and bool(bounty_in.get("is_bounty"))

        if has_bounty and covered and bounty_chips_bb > 0:
            out.bounty_chips_used = bounty_chips_bb
            adjusted = effective_required_equity(pot_bb, to_call_bb, bounty_chips_bb)
            out.bounty_adjusted_required_equity = round(adjusted, 6)
        else:
            # 无赏金 / 未覆盖 / 赏金筹码为 0 → 按 normal + warning
            out.bounty_chips_used = 0.0
            out.bounty_adjusted_required_equity = round(normal, 6)
            if not has_bounty:
                warnings.append({"code": "NO_BOUNTY",
                                 "message": "bounty 输入缺失/无赏金，按普通所需胜率（adjusted=normal）"})
            elif not covered:
                warnings.append({"code": "NOT_COVERED",
                                 "message": "hero 码量未覆盖对手（covered=False），无法兑现赏金，按 normal"})
            else:
                warnings.append({"code": "ZERO_BOUNTY_CHIPS",
                                 "message": "赏金筹码为 0（chip_value 缺失），按 normal"})

        out.required_equity_delta = round(out.normal_required_equity
                                          - out.bounty_adjusted_required_equity, 6)

        # ---- hero equity（缺失 → 0.5 中性 + warning） ----
        hero_equity = equity_in.get("hero_equity") if isinstance(equity_in, dict) else None
        if hero_equity is None:
            hero_equity = 0.5
            warnings.append({"code": "NO_EQUITY",
                             "message": "equity 输入缺失，hero_equity 用 0.5 中性近似"})
        else:
            hero_equity = _num(hero_equity, 0.5)

        # ---- EV(call)：普通 vs 含赏金（bb） ----
        out.ev_call_normal_bb = round(ev_of_call(hero_equity, pot_bb, to_call_bb), 6)
        out.ev_call_with_bounty_bb = round(
            call_ev_with_bounty(hero_equity, pot_bb, to_call_bb,
                                out.bounty_chips_used), 6)

        # ---- Judge 应使用的有效所需胜率 ----
        out.effective_decision_equity = round(
            out.bounty_adjusted_required_equity
            if (has_bounty and covered and bounty_chips_bb > 0)
            else out.normal_required_equity, 6)

        # ---- 置信度 ----
        conf = 0.9
        if hero_equity is None or not (has_bounty and covered):
            conf = 0.7
        out.confidence = round(conf, 4)

        out.evidence = [{
            "kind": "bounty_equity", "source": "bounty",
            "text": (f"normal={out.normal_required_equity} adjusted="
                     f"{out.bounty_adjusted_required_equity} delta="
                     f"{out.required_equity_delta} covered={covered}"),
            "weight": 1.0,
        }]
        out.warnings = warnings
        return out
