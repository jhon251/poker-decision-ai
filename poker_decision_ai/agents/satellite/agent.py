"""Satellite Agent (卫星赛): seat-equity analysis for satellite tournaments.

A satellite's objective is to *win a seat* (maximize seat probability), NOT to
maximize chip EV.  This agent computes `satellite_seat_ev` (ICM with payouts
[1]*seats) over the active players and derives a conservative/aggressive
guidance note from the hero's seat probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker import satellite_seat_ev


def _active_players(state: dict) -> list:
    players = state.get("players") or []
    return [p for p in players if p.get("status") in ("active", "allin")]


@dataclass
class SatelliteOutput:
    applicable: bool = False
    seat_ev: dict = field(default_factory=dict)     # pid -> seat probability 0..1
    hero_seat_ev: float = 0.0
    seats_remaining: int = 0
    players_remaining: int = 0
    satellite_mode: str = "seats"
    notes: list = field(default_factory=list)
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class SatelliteAgent(BaseAgent):
    name: ClassVar[str] = "satellite"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "卫星赛 Agent"
    layer: ClassVar[str] = "tournament"
    depends: ClassVar[list[str]] = ["state", "icm", "tournament_stage"]
    supported_modes: ClassVar[list[str]] = ["tournament:satellite"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = SatelliteOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if state.get("game_type") != "satellite":
            return False, "not a satellite game (game_type != 'satellite')"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> SatelliteOutput:
        state = inp.get("state") or {}
        out = SatelliteOutput(applicable=True)
        tournament = state.get("tournament") or {}

        # ---- seats = tournament.satellite_seats；缺失 -> 用 itm_players 估算 ----
        # 估算假设：晋级名额 ≈ 奖励圈人数（卫星赛通常 seats == itm 边界，近似）。
        seats = tournament.get("satellite_seats")
        if seats is None or int(seats) <= 0:
            itm = tournament.get("itm_players") or 1
            seats = max(1, int(itm))
            out.warnings.append({
                "code": "SEATS_ESTIMATED",
                "message": "tournament.satellite_seats 缺失，用 itm_players 估算晋级席位数",
            })

        # ---- stacks = {pid: stack} over active players ----
        active = _active_players(state)
        stacks = {p["player_id"]: float(p.get("stack") or 0.0) for p in active}
        hero_id = str(state.get("hero_id") or "")

        out.seats_remaining = int(seats)
        out.players_remaining = len(stacks)

        if not stacks:
            out.warnings.append({"code": "NO_ACTIVE", "message": "no active players to compute seat EV"})
            out.confidence = 0.0
            return out

        # satellite_seat_ev: ICM with payouts [1]*seats -> ev = P(win a seat), 0..1
        result = satellite_seat_ev(stacks, int(seats))
        out.seat_ev = {pid: round(float(ev), 6) for pid, ev in result.ev.items()}
        out.hero_seat_ev = round(out.seat_ev.get(hero_id, 0.0), 4)

        # ---- notes ----
        # 固定说明：卫星赛目标 = 最大化晋级概率(Seat EV)，不是筹码 EV。
        out.notes.append("卫星赛目标=最大化晋级概率(Seat EV)，不是筹码 EV")
        if out.hero_seat_ev > 0.9:
            out.notes.append("晋级几乎锁定：保守策略，避免大底池对抗")
        elif out.hero_seat_ev < 0.15:
            out.notes.append("晋级希望渺茫：可考虑激进")
        else:
            out.notes.append("晋级概率中等：平衡打法，关注短码玩家淘汰")

        out.evidence = [{
            "kind": "satellite",
            "text": f"hero seat EV={out.hero_seat_ev}, seats={int(seats)}, "
                    f"players={len(stacks)}, method={result.method}",
            "source": "satellite_seat_ev",
            "weight": 1.0,
        }]
        return out
