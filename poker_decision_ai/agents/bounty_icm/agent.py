"""Bounty ICM Agent (赏金 ICM): KO 概率 × 赏金价值的 $EV 与 ICM 线性合成。

V1.1 §4b 契约。全确定性。

近似说明（注释）：真实 PKO 的 $EV 是更复杂的状态转移（淘汰对手同时改变
chip 分布与赏金再分配），这里 combined_ev = icm_ev + bounty_ev 是**线性下界
估计**，仅供 Judge 参考，不是最终答案。

Input : {state, icm, equity, bounty, tournament_stage}（dict|None）
Output: BountyIcmOutput
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker.bounty import BountyInfo, knockout_probability


@dataclass
class BountyIcmOutput:
    knockout_probability: dict = field(default_factory=dict)   # {pid: 0..1}
    bounty_ev_by_villain: dict = field(default_factory=dict)   # {pid: $EV}
    icm_ev: dict | None = None                                  # 透传 icm
    risk_premium: dict | None = None                            # 透传 icm
    combined_ev: dict = field(default_factory=dict)             # {pid: icm+bounty}
    total_bounty_ev: float = 0.0
    bubble_plus_bounty: bool = False
    note: str = ""
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _active_villains(players: list) -> list:
    return [p for p in players
            if isinstance(p, dict) and not p.get("is_hero")
            and p.get("status") in ("active", "allin")]


class BountyIcmAgent(BaseAgent):
    name: ClassVar[str] = "bounty_icm"
    version: ClassVar[str] = "1.1.0"
    zh_name: ClassVar[str] = "赏金 ICM Agent"
    layer: ClassVar[str] = "tournament"
    depends: ClassVar[list[str]] = ["state", "icm", "equity", "bounty", "tournament_stage"]
    supported_modes: ClassVar[list[str]] = ["bounty"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = BountyIcmOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for bounty ICM analysis"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> BountyIcmOutput:
        state = inp.get("state") or {}
        icm_in = inp.get("icm") or {}
        equity_in = inp.get("equity") or {}
        bounty_in = inp.get("bounty") or {}
        tournament_stage = inp.get("tournament_stage") or {}
        out = BountyIcmOutput()
        warnings: list[dict] = []

        out.note = ("合成值仅供 Judge 参考：combined_ev = icm_ev + bounty_ev 是线性下界近似，"
                    "真实 PKO $EV 涉及更复杂的状态转移，不是最终答案。")

        # ---- 透传 icm ----
        if isinstance(icm_in, dict) and icm_in.get("icm_ev"):
            out.icm_ev = dict(icm_in.get("icm_ev") or {})
        else:
            warnings.append({"code": "NO_ICM", "message": "icm 输入缺失/无 icm_ev"})
        if isinstance(icm_in, dict) and icm_in.get("risk_premium"):
            out.risk_premium = dict(icm_in.get("risk_premium") or {})

        info = BountyInfo.from_state(state)
        has_bounty = info.is_bounty
        if not has_bounty:
            warnings.append({"code": "NO_BOUNTY", "message": "state 无赏金块，输出空结果"})
            out.confidence = 0.0
            out.warnings = warnings
            return out

        # ---- equity vs 各 villain（hero_equity_vs_each 或 hero_equity 均值） ----
        hero_equity_vs_each = (equity_in.get("hero_equity_vs_each")
                               if isinstance(equity_in, dict) else None) or {}
        hero_equity_mean = _num((equity_in or {}).get("hero_equity"), 0.5)

        players = state.get("players") or []
        hero_p = next((p for p in players if isinstance(p, dict) and p.get("is_hero")), None)
        hero_stack = _num(hero_p.get("stack")) if hero_p else 0.0

        # growth_rate：PKO 递增比例（state 未提供时默认 0.5）
        growth_rate = info.growth_rate
        progressive = info.progressive

        for v in _active_villains(players):
            pid = str(v.get("player_id") or "")
            if not pid:
                continue
            villain_bounty = _num(v.get("bounty"))
            villain_stack = _num(v.get("stack"))
            covered = villain_stack > 0 and hero_stack >= villain_stack

            # KO 概率：优先 hero_equity_vs_each，否则 hero_equity 均值
            eq = hero_equity_vs_each.get(pid)
            eq = _num(eq, None) if eq is not None else None
            if eq is None:
                eq = hero_equity_mean
            out.knockout_probability[pid] = round(knockout_probability(eq, covered), 6)

            # knockout_value：PKO = growth_rate*对手赏金；否则 = 对手赏金
            ko_value = (growth_rate * villain_bounty) if progressive else villain_bounty
            out.bounty_ev_by_villain[pid] = round(
                out.knockout_probability[pid] * ko_value, 6)

        # ---- combined_ev = icm_ev + bounty_ev（线性近似） ----
        for pid, b_ev in out.bounty_ev_by_villain.items():
            icm_val = _num((out.icm_ev or {}).get(pid))
            out.combined_ev[pid] = round(icm_val + b_ev, 6)

        out.total_bounty_ev = round(sum(out.bounty_ev_by_villain.values()), 6)

        # ---- bubble + 大赏金 ----
        stage = tournament_stage.get("stage") or (state.get("tournament") or {}).get("stage")
        is_bubble = str(stage or "") == "bubble"
        big_bounty = info.villain_bounty > 0
        out.bubble_plus_bounty = bool(is_bubble and big_bounty
                                      and out.total_bounty_ev > 0)

        out.confidence = round(0.8 if out.icm_ev else 0.5, 4)

        out.evidence = [{
            "kind": "bounty_icm", "source": "bounty_icm",
            "text": (f"{len(out.bounty_ev_by_villain)} villain(s), total_bounty_ev="
                     f"{out.total_bounty_ev}, bubble_plus_bounty={out.bubble_plus_bounty}"),
            "weight": 1.0,
        }]
        out.warnings = warnings
        return out
