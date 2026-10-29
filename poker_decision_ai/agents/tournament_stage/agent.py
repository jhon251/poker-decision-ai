"""Tournament Stage Agent (阶段识别): deterministic stage classification.

Rules (documented):
  - cash -> not applicable.
  - players_remaining <= 2           -> heads_up
  - players_remaining <= 9           -> final_table
  - itm_distance <= 0                -> itm  (already in the money)
  - ratio = remaining/total:
        > 0.6   -> early
        > 0.35  -> mid
        else    -> bubble when 0 < itm_distance <= 5, otherwise late
  - bb_avg = avg_stack / bb  (avg_stack = tournament.avg_stack or
    total_chips / players_remaining).

Input : {state}
Output: TournamentStageOutput
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext

_STAGES = ("early", "mid", "late", "bubble", "itm", "final_table", "heads_up")


@dataclass
class TournamentStageOutput:
    stage: str = "unknown"
    bb_avg: float = 0.0
    players_remaining: int = 0
    total_players: int = 0
    itm_distance: int = 0
    pay_jump_context: str = ""
    stage_context: str = ""
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class TournamentStageAgent(BaseAgent):
    name: ClassVar[str] = "tournament_stage"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "阶段识别 Agent"
    layer: ClassVar[str] = "tournament"
    depends: ClassVar[list[str]] = ["state"]
    supported_modes: ClassVar[list[str]] = ["tournament", "bounty"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = TournamentStageOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if not state:
            return False, "no state"
        if state.get("game_type") == "cash":
            return False, "cash game has no tournament stage"
        return True, ""

    # ------------------------------------------------------------------ handle
    async def handle(self, inp: dict, ctx: AgentContext) -> TournamentStageOutput:
        state = inp.get("state") or {}
        out = TournamentStageOutput()

        tournament = state.get("tournament") if isinstance(state.get("tournament"), dict) else {}
        if not tournament:
            out.warnings.append({"code": "NO_TOURNAMENT", "message": "no tournament info; cannot classify stage"})
            out.confidence = 0.2
            return out

        remaining = int(tournament.get("players_remaining") or 0)
        total = int(tournament.get("total_players") or 0)
        itm_players = int(tournament.get("itm_players") or 0)
        out.players_remaining = remaining
        out.total_players = total or remaining
        out.itm_distance = remaining - itm_players if itm_players > 0 else remaining

        # ---- stage classification ----
        if remaining <= 2:
            stage = "heads_up"
        elif remaining <= 9:
            stage = "final_table"
        elif itm_players > 0 and out.itm_distance <= 0:
            stage = "itm"
        else:
            ratio = remaining / total if total > 0 else 1.0
            if ratio > 0.6:
                stage = "early"
            elif ratio > 0.35:
                stage = "mid"
            elif itm_players > 0 and 0 < out.itm_distance <= 5:
                stage = "bubble"
            else:
                stage = "late"
        out.stage = stage

        # ---- bb_avg ----
        avg_stack = tournament.get("avg_stack")
        if avg_stack is None:
            total_chips = float(tournament.get("total_chips") or 0)
            avg_stack = total_chips / remaining if remaining > 0 else 0.0
        bb = float(state.get("bb") or 0) or 1.0
        out.bb_avg = round(float(avg_stack or 0) / bb, 2)

        # ---- pay_jump_context ----
        next_payout = tournament.get("next_payout")
        if isinstance(next_payout, (int, float)) and itm_players > 0:
            out.pay_jump_context = (f"下一奖金跳变 ${float(next_payout):.2f}，"
                                    f"距进圈 {max(out.itm_distance, 0)} 人")
        else:
            out.pay_jump_context = f"距进圈 {max(out.itm_distance, 0)} 人"

        # ---- stage_context ----
        out.stage_context = (f"阶段={stage}，剩余 {remaining}/{out.total_players} 人，"
                             f"平均码量 {out.bb_avg}bb；{out.pay_jump_context}")

        out.evidence.append({"kind": "stage", "text": out.stage_context,
                             "source": "tournament", "weight": 1.0})
        return out
