"""Board Agent (公共牌分析): deterministic board texture analysis.

This file is the REFERENCE IMPLEMENTATION for all V1 agents - new agents
follow exactly this structure:
  - module `agents/<name>/agent.py` with ONE agent class
  - Input / Output dataclasses (contracts are derived from type hints)
  - optional meta fields confidence / evidence / warnings on Output
  - `applicable()` dynamic skip hook
  - pure-deterministic logic (no LLM in this agent)
  - unit tests in tests/unit/test_<name>.py

Input : {state: dict}           (from State Agent, validated at runtime)
Output: BoardFeatures (below)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.domain.poker.cards import RANK_VALUES, parse_card


@dataclass
class BoardFeatures:
    street: str = "preflop"
    board: list = field(default_factory=list)
    texture: str = "unknown"                     # dry | wet | static | dynamic
    paired: bool = False
    monotone: bool = False
    two_tone: bool = False
    rainbow: bool = False
    flush_draw_present: bool = False
    straight_draw_present: bool = False
    straight_possible: bool = False
    nut_flush_possible: bool = False
    high_card_rank: str = ""
    overcard_to_board: str = ""                  # lowest rank NOT on board ("" when unknown)
    connectedness: float = 0.0                   # 0..1
    wetness_score: float = 0.0                   # 0..1
    range_interaction: dict = field(default_factory=dict)   # {board_favors, rationale}
    draw_structure: list = field(default_factory=list)      # e.g. ["flush_draw","gutshot"]
    confidence: float = 1.0                      # deterministic -> 1.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class BoardAgent(BaseAgent):
    name: ClassVar[str] = "board"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "公共牌分析 Agent"
    layer: ClassVar[str] = "analysis"
    depends: ClassVar[list[str]] = ["state"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = BoardFeatures

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        state = raw_input.get("state") or {}
        if not state.get("board"):
            return False, "no board yet (preflop) - board analysis only applies postflop"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> BoardFeatures:
        state = inp.get("state") or {}
        board = [str(c) for c in state.get("board", [])]
        street = str(state.get("street", "preflop"))
        out = BoardFeatures(street=street, board=board)

        if not board:
            out.warnings.append({"code": "NO_BOARD", "message": "empty board"})
            out.confidence = 0.0
            return out

        cards = [parse_card(c) for c in board]
        ranks = sorted(c.rank for c in cards)
        suits = [c.suit for c in cards]

        # ---- paired / flush structure ----
        out.paired = len(set(ranks)) < len(ranks)
        suit_counts = {s: suits.count(s) for s in set(suits)}
        max_suit = max(suit_counts.values())
        out.monotone = max_suit >= 3 and len(board) >= 3
        out.two_tone = len(suit_counts) == 2 and not out.monotone
        out.rainbow = len(suit_counts) >= 3 and max_suit <= 2
        # a flop with 2+ of a suit has a flush draw; 3+ = made flush (turn/river)
        out.flush_draw_present = (len(board) == 3 and max_suit >= 2)
        out.nut_flush_possible = out.flush_draw_present or (max_suit >= 3 and len(board) >= 3)

        # ---- straight analysis ----
        rset = set(ranks)
        windows = 0
        for high in range(5, 15):
            needed = {r for r in range(high - 4, high + 1)}
            have = len(rset & needed)
            if have >= 3 and len(board) == 3:
                windows += 1
            if have >= 4 and len(board) >= 4:
                windows += 1
        out.straight_draw_present = windows > 0
        out.straight_possible = windows > 0
        # connectedness: share of adjacent rank gaps <= 2
        gaps = [b - a for a, b in zip(ranks, ranks[1:])]
        out.connectedness = round(sum(1 for g in gaps if g <= 2) / max(1, len(gaps)), 3)

        # ---- high card ----
        out.high_card_rank = {v: k for k, v in RANK_VALUES.items()}[max(ranks)] if ranks else ""
        for r in (14, 13, 12, 11, 10):
            if r not in rset:
                out.overcard_to_board = {v: k for k, v in RANK_VALUES.items()}[r]
                break

        # ---- draw structure ----
        if out.flush_draw_present:
            out.draw_structure.append("flush_draw")
        if windows:
            out.draw_structure.append("straight_draw")
        if out.paired:
            out.draw_structure.append("trips_draw")

        # ---- texture + wetness ----
        wet = 0.0
        if out.flush_draw_present:
            wet += 0.4
        if out.straight_draw_present:
            wet += 0.35
        if out.monotone:
            wet += 0.5
        if out.paired:
            wet -= 0.1
        out.wetness_score = round(max(0.0, min(1.0, wet)), 3)
        if out.wetness_score >= 0.5:
            out.texture = "wet"
        elif out.paired or out.connectedness <= 0.2:
            out.texture = "static"
        else:
            out.texture = "dry"
        if out.texture in ("static",) and out.wetness_score >= 0.25:
            out.texture = "dynamic" if out.wetness_score < 0.5 else "wet"

        # ---- range interaction (board favors which archetype) ----
        high_cards = sum(1 for r in ranks if r >= 10)
        top_rank = max(ranks)
        if high_cards >= 2 or (high_cards >= 1 and top_rank >= 12):
            out.range_interaction = {"board_favors": "aggressor",
                                     "rationale": "broadway-heavy board favours the preflop raiser's range"}
        elif high_cards == 1 and not out.monotone:
            out.range_interaction = {"board_favors": "neutral",
                                     "rationale": "single low broadway card; both ranges connect"}
        else:
            out.range_interaction = {"board_favors": "defender",
                                     "rationale": "low/mid board favours the caller's range"}

        out.evidence = [{"kind": "board", "text": f"{street} board {board}: texture={out.texture}, "
                                                  f"wetness={out.wetness_score}, paired={out.paired}",
                         "source": "board", "weight": 1.0}]
        return out
