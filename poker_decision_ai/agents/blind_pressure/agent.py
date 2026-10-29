"""Blind Pressure Agent (盲注压力): deterministic stack-pressure analysis.

Measures how much the hero's stack depth forces action before the blinds eat
it.  Works for BOTH cash and tournament states (deep cash -> ~0 pressure),
so `applicable()` only skips when the state itself is missing.

All formulas are deterministic (spec §十一: never LLM-computed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext


def _clip01(x: float) -> float:
    """Clamp a float into [0, 1]."""
    return max(0.0, min(1.0, x))


def _hero_player(state: dict) -> dict | None:
    """Find the hero player dict (is_hero flag wins, else hero_id match)."""
    players = state.get("players") or []
    hero_id = state.get("hero_id")
    for p in players:
        if p.get("is_hero"):
            return p
    for p in players:
        if hero_id and p.get("player_id") == hero_id:
            return p
    return None


def _active_players(state: dict) -> list:
    players = state.get("players") or []
    return [p for p in players if p.get("status") in ("active", "allin")]


def _position_of(state: dict, player_id: str | None) -> str | None:
    if not player_id:
        return None
    for p in state.get("players") or []:
        if p.get("player_id") == player_id:
            return (p.get("position") or "").upper()
    return None


# Preflop action order (earlier index = acts earlier).  Used to detect whether
# everyone in front of the hero has folded (a steal opportunity).
_POS_ORDER = ["UTG", "UTG+1", "UTG+2", "MP", "MP+1", "HJ", "CO", "BTN", "SB", "BB"]


def _positions_before(pos: str) -> list:
    p = (pos or "").upper()
    if p not in _POS_ORDER:
        return []
    return _POS_ORDER[: _POS_ORDER.index(p)]


@dataclass
class BlindPressureOutput:
    stack_bb: float = 0.0          # hero stack in big blinds
    m_score: float = 0.0           # M = stack / (sb + bb + ante * active_players)
    blind_pressure: float = 0.0    # 0..1
    steal_pressure: float = 0.0    # 0..1
    resteal_pressure: float = 0.0  # 0..1
    open_shove_zone: bool = False  # True when stack_bb <= 15
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class BlindPressureAgent(BaseAgent):
    name: ClassVar[str] = "blind_pressure"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "盲注压力 Agent"
    layer: ClassVar[str] = "tournament"
    depends: ClassVar[list[str]] = ["state", "tournament_stage"]
    supported_modes: ClassVar[list[str]] = ["tournament", "bounty"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = BlindPressureOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if not state:
            return False, "state missing"
        if not state.get("players"):
            return False, "state.players missing"
        # cash 也适用（深码 -> 压力≈0），不因 game_type 跳过。
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> BlindPressureOutput:
        state = inp.get("state") or {}
        out = BlindPressureOutput()
        hero_p = _hero_player(state)
        if hero_p is None:
            out.warnings.append({"code": "NO_HERO", "message": "hero player not found in state"})
            out.confidence = 0.0
            return out

        bb = float(state.get("bb") or 0.0)
        if bb <= 0:
            bb = 1.0                                   # guard: avoid div-by-zero
        sb = float(state.get("sb") or 0.0)
        ante = float(state.get("ante") or 0.0)
        stack = float(hero_p.get("stack") or 0.0)
        hero_id = str(state.get("hero_id") or "")
        hero_pos = (hero_p.get("position") or "").upper()

        # ---- stack in big blinds ----
        stack_bb = stack / bb
        out.stack_bb = round(stack_bb, 2)

        # ---- M = stack / (sb + bb + ante * active_players) ----
        # 每次完整一圈的盲注成本（含 ante），M 是更精确的生存压力度量。
        n_active = len(_active_players(state))
        orbit_cost = sb + bb + ante * n_active
        if orbit_cost <= 0:
            orbit_cost = bb
        m_score = stack / orbit_cost
        out.m_score = round(m_score, 2)

        # ---- blind_pressure: 线性分段，40bb 以下开始有压力 ----
        # pressure = 1 - stack_bb/40  (0bb -> 1.0, >=40bb -> 0.0)
        pressure = _clip01(1.0 - stack_bb / 40.0)
        # M < 20 -> 额外 +0.15（M 低代表盲注侵蚀更快，压力叠加）
        if m_score < 20:
            pressure = _clip01(pressure + 0.15)
        out.blind_pressure = round(pressure, 3)

        # ---- steal_pressure ----
        # BTN/CO/SB 且前位全弃 -> 0.6 + (1 - stack_bb/40) * 0.4；否则 0.3 基线。
        ahead = [
            p for p in _active_players(state)
            if (p.get("player_id") != hero_id
                and (p.get("position") or "").upper() in _positions_before(hero_pos))
        ]
        steal_opportunity = hero_pos in ("BTN", "CO", "SB") and not ahead
        if steal_opportunity:
            steal = _clip01(0.6 + (1.0 - stack_bb / 40.0) * 0.4)
        else:
            steal = 0.3
        out.steal_pressure = round(steal, 3)

        # ---- resteal_pressure ----
        # to_call > 0 且 last_aggressor 位置在 CO/BTN/SB -> 0.7 基线；
        # BB 面对 steal 且 stack_bb < 30 -> 更高。
        to_call = float(state.get("to_call") or 0.0)
        la_pos = _position_of(state, state.get("last_aggressor"))
        facing_steal = to_call > 0 and la_pos in ("CO", "BTN", "SB")
        if facing_steal:
            resteal = 0.7
            if hero_pos == "BB" and stack_bb < 30:
                resteal = _clip01(0.7 + 0.2)
        else:
            resteal = 0.2
        out.resteal_pressure = round(resteal, 3)

        # ---- open_shove_zone ----
        out.open_shove_zone = stack_bb <= 15

        out.evidence = [{
            "kind": "blind_pressure",
            "text": f"stack={stack_bb}bb, M={m_score:.1f}, pressure={out.blind_pressure}, "
                    f"open_shove={out.open_shove_zone}",
            "source": "state",
            "weight": 1.0,
        }]
        return out
