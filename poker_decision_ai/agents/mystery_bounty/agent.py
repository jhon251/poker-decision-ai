"""Mystery Bounty Agent (神秘赏金): 确定性期望值 + 策略提示（基础版）。

V1.1 §4b 契约。Provider 接口预留（core/providers/mystery_bounty.py），
本 Agent 使用 DeterministicMysteryProvider：
  pending 阶段期望值 = 剩余赏金池 / 剩余信封数（信封数缺失时用剩余玩家数近似）。
  bounty_ev = expected_value × knockout_probability(0.5 中性, covered)。

Input : {state, bounty, icm, tournament_stage}（dict|None）
Output: MysteryBountyOutput
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.providers.mystery_bounty import DeterministicMysteryProvider
from poker_decision_ai.domain.poker.bounty import knockout_probability


@dataclass
class MysteryBountyOutput:
    phase: str = "pending"                     # pending | active
    mystery_pool_remaining: float = 0.0
    expected_bounty_value: float = 0.0         # 揭晓前期望信封价值（$）
    bounty_ev: float = 0.0                     # expected_value × KO 概率
    strategy_note: str = ""
    provider: str = "deterministic"
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _covered(state: dict, bounty_in: dict | None) -> bool:
    """hero 码量是否覆盖最大 active 赏金对手；优先 bounty 输出，否则从 state 计算。"""
    if isinstance(bounty_in, dict) and "covered" in bounty_in:
        return bool(bounty_in.get("covered"))
    players = state.get("players") or []
    hero_p = next((p for p in players if isinstance(p, dict) and p.get("is_hero")), None)
    hero_stack = _num(hero_p.get("stack")) if hero_p else 0.0
    max_v_stack = 0.0
    for p in players:
        if not isinstance(p, dict) or p.get("is_hero"):
            continue
        if p.get("status") not in ("active", "allin"):
            continue
        max_v_stack = max(max_v_stack, _num(p.get("stack")))
    return max_v_stack > 0 and hero_stack >= max_v_stack


class MysteryBountyAgent(BaseAgent):
    name: ClassVar[str] = "mystery_bounty"
    version: ClassVar[str] = "1.1.0"
    zh_name: ClassVar[str] = "神秘赏金 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state", "bounty", "icm", "tournament_stage"]
    supported_modes: ClassVar[list[str]] = ["bounty:mystery"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = MysteryBountyOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for mystery bounty analysis"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> MysteryBountyOutput:
        state = inp.get("state") or {}
        bounty_in = inp.get("bounty") or {}
        out = MysteryBountyOutput()
        warnings: list[dict] = []

        tournament = state.get("tournament")
        if not isinstance(tournament, dict):
            out.provider = "deterministic"
            out.phase = "pending"
            out.strategy_note = "无锦标赛上下文，无法计算神秘赏金期望值。"
            warnings.append({"code": "NO_TOURNAMENT",
                             "message": "state.tournament 缺失，输出全 0"})
            out.warnings = warnings
            out.evidence = [{"kind": "mystery_bounty", "source": "state",
                             "text": "no tournament; all-zero output", "weight": 1.0}]
            return out

        # ---- 确定性 Provider 取揭晓信息 ----
        info = await DeterministicMysteryProvider().prize_info(state)
        out.provider = "deterministic"
        out.phase = info.phase
        out.mystery_pool_remaining = round(float(info.total_pool), 4)
        out.expected_bounty_value = round(float(info.expected_value), 4)

        covered = _covered(state, bounty_in if isinstance(bounty_in, dict) else None)
        # KO 概率用 0.5 中性（神秘赏金信封价值揭晓前与胜率无关的近似）
        out.bounty_ev = round(out.expected_bounty_value * knockout_probability(0.5, covered), 4)

        if info.phase == "active":
            out.strategy_note = ("神秘赏金已揭晓：按对手实际信封价值决策，"
                                 "大信封对手值得更激进地追淘汰。")
        else:
            out.strategy_note = ("神秘赏金尚未揭晓：按期望值（剩余池/剩余信封数）决策，"
                                 f"期望信封价值=${out.expected_bounty_value}。")

        for n in info.notes:
            warnings.append({"code": "ENVELOPE_APPROX", "message": str(n)})
        if info.expected_value <= 0:
            warnings.append({"code": "ZERO_EXPECTED",
                             "message": "剩余赏金池或信封数为 0，期望值为 0"})

        out.confidence = round(0.8 if info.expected_value > 0 else 0.5, 4)

        out.evidence = [{
            "kind": "mystery_bounty", "source": "mystery_bounty",
            "text": (f"phase={out.phase} expected={out.expected_bounty_value} "
                     f"bounty_ev={out.bounty_ev} covered={covered}"),
            "weight": 1.0,
        }]
        out.warnings = warnings
        return out
