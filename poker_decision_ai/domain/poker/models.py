"""Unified poker domain models: state, actions, tournaments (typed dataclasses).

The State Agent is the ONLY producer of these; every downstream agent
consumes the state as a dict validated against STATE_SCHEMA and may coerce it
back to typed objects via `coerce_state()`.

Cash and Tournament states are separate dataclasses (spec §八): a tournament
hand carries TournamentInfo, a cash hand carries CashInfo, never both.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from ...core.errors import SchemaError


class Street(str, enum.Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class GameType(str, enum.Enum):
    CASH = "cash"
    MTT = "mtt"
    SATELLITE = "satellite"


class ActionType(str, enum.Enum):
    POST_SB = "post_sb"
    POST_BB = "post_bb"
    POST_ANTE = "post_ante"
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALLIN = "allin"
    SHOW = "show"


class TournamentStage(str, enum.Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    BUBBLE = "bubble"
    ITM = "itm"
    FINAL_TABLE = "final_table"
    HEADS_UP = "heads_up"


@dataclass
class PlayerState:
    player_id: str
    name: str
    seat: int
    stack: float
    status: str = "active"                    # active | folded | allin | out
    hole_cards: list = field(default_factory=list)
    position: str | None = None               # BTN/SB/BB/UTG/MP/CO...
    is_hero: bool = False
    table_image_notes: str = ""
    bounty: float = 0.0                       # V1.1 赏金（货币单位；0 = 无赏金）


@dataclass
class ActionRecord:
    street: str
    player_id: str
    action: str
    amount: float = 0.0
    all_in: bool = False
    seq: int = 0


@dataclass
class CashInfo:
    stakes: str = ""                          # e.g. NL10
    sb: float = 0.0
    bb: float = 0.0
    currency: str = "USD"
    rake_percent: float = 0.0
    cap_bb: float | None = None


@dataclass
class TournamentInfo:
    tournament_id: str = ""
    name: str = ""
    total_players: int = 0
    players_remaining: int = 0
    total_chips: float = 0.0
    payouts: list = field(default_factory=list)      # descending prize list
    next_payout: float | None = None                 # next pay jump value
    itm_players: int = 0                             # players that finish in the money
    satellite_seats: int | None = None               # set for satellites
    stage: str | None = None
    avg_stack: float | None = None
    # ---- V1.1 Bounty 字段 ----
    bounty_type: str | None = None                   # pko | ko | mystery
    is_bounty: bool = False
    progressive: bool = False                        # PKO 递增赏金
    bounty_rate: float = 0.5                         # PKO 递增比例（淘汰后 += rate*对手赏金）
    total_bounty_pool: float = 0.0                   # 剩余赏金池（$）
    prize_pool: float = 0.0                          # 常规奖池（$）
    mystery_phase_active: bool = False               # Mystery Bounty 揭晓阶段


@dataclass
class PokerStateBase:
    """Common core of every hand state."""

    hand_id: str
    game_type: str                              # cash | mtt | satellite
    street: str                                 # preflop|flop|turn|river|showdown
    board: list = field(default_factory=list)
    pot: float = 0.0
    current_bet: float = 0.0                    # largest bet hero faces this street
    to_call: float = 0.0                        # amount hero must add to continue
    min_raise_to: float = 0.0
    sb: float = 0.0
    bb: float = 0.0
    ante: float = 0.0
    players: list = field(default_factory=list)     # list[PlayerState] as dicts
    button_seat: int | None = None
    hero_id: str = ""
    hero_seat: int | None = None
    action_history: list = field(default_factory=list)   # list[ActionRecord]
    street_actions: dict = field(default_factory=dict)    # street -> list[ActionRecord]
    available_actions: list = field(default_factory=list)
    last_aggressor: str | None = None
    rake: float = 0.0
    source: str = ""                            # vision | hand_history | manual


@dataclass
class CashState(PokerStateBase):
    cash: CashInfo | None = None


@dataclass
class TournamentState(PokerStateBase):
    tournament: TournamentInfo | None = None


def coerce_state(data: dict) -> CashState | TournamentState:
    """Validate a state dict and return the typed variant.

    State dicts produced by the State Agent always carry BOTH `cash` and
    `tournament` keys (one of them null).  The typed dataclasses are split, so
    the irrelevant key is stripped before strict validation (STATE_SCHEMA
    allows it, the Cash/Tournament dataclasses do not).
    """
    from ...core.schemas import from_dict

    if not isinstance(data, dict):
        raise SchemaError("state must be a JSON object")
    gt = data.get("game_type")
    if gt == GameType.CASH:
        filtered = {k: v for k, v in data.items() if k != "tournament"}
        return from_dict(CashState, filtered)
    filtered = {k: v for k, v in data.items() if k != "cash"}
    return from_dict(TournamentState, filtered)


# ---------------------------------------------------------------- helpers
def player_by_id(state: PokerStateBase, player_id: str) -> PlayerState | None:
    for p in state.players:
        if isinstance(p, dict) and p.get("player_id") == player_id:
            return PlayerState(**{k: v for k, v in p.items() if k in PlayerState.__dataclass_fields__})
        if isinstance(p, PlayerState) and p.player_id == player_id:
            return p
    return None


def hero(state: PokerStateBase) -> PlayerState | None:
    return player_by_id(state, state.hero_id) if state.hero_id else None


def active_players(state: PokerStateBase) -> list:
    return [p for p in state.players
            if (p.get("status") if isinstance(p, dict) else p.status) in ("active", "allin")]


def effective_stack(state: PokerStateBase) -> float:
    """Smallest stack among players still contesting the pot (chips)."""
    h = hero(state)
    if h is None:
        return 0.0
    stacks = [h.stack] + [p.stack if isinstance(p, PlayerState) else p.get("stack", 0.0)
                          for p in active_players(state)
                          if (p.player_id if isinstance(p, PlayerState) else p.get("player_id")) != state.hero_id]
    return min(stacks) if stacks else h.stack


def bb_amount(state: PokerStateBase) -> float:
    return state.bb if state.bb > 0 else 1.0


def in_bb(value: float, state: PokerStateBase) -> float:
    return value / bb_amount(state)


def pot_bb(state: PokerStateBase) -> float:
    return in_bb(state.pot, state)


def positions_for(players: list, button_seat: int) -> dict[int, str]:
    """Position names for n-handed table (seats -> label).

    座位约定（docs/HAND_HISTORY.md）：座位顺时针编号，SB 是按钮顺位的
    下一座（座位号 +1，最大号回绕到 1）。标签按按钮轮转分配——按钮座
    永远得到 BTN、其后依次 SB、BB，再往前推 UTG/MP/CO。
    """
    n = len(players)
    seats = sorted(p.seat if isinstance(p, PlayerState) else p["seat"] for p in players)
    if n <= 2:
        return _hu_positions(players, button_seat)
    labels = _position_labels(n)
    if button_seat is None or button_seat not in seats:
        # 无按钮信息：回退固定顺序（仅用于数据不完整时的保守标注）
        return {seat: labels[i] for i, seat in enumerate(seats)}
    b = seats.index(button_seat)
    out = {}
    for i, seat in enumerate(seats):
        k = (i - b - 3) % n      # b -> BTN(n-3), b+1 -> SB(n-2), b+2 -> BB(n-1)
        out[seat] = labels[k]
    return out


def _hu_positions(players: list, button_seat: int) -> dict[int, str]:
    out = {}
    for p in players:
        seat = p.seat if isinstance(p, PlayerState) else p["seat"]
        out[seat] = "BTN/SB" if seat == button_seat else "BB"
    return out


def _position_labels(n: int) -> list[str]:
    if n == 3:
        return ["BTN", "SB", "BB"]
    if n == 4:
        return ["UTG", "BTN", "SB", "BB"]
    if n == 5:
        return ["UTG", "CO", "BTN", "SB", "BB"]
    if n == 6:
        return ["UTG", "MP", "CO", "BTN", "SB", "BB"]
    if n == 7:
        return ["UTG", "UTG+1", "MP", "CO", "BTN", "SB", "BB"]
    if n == 8:
        return ["UTG", "UTG+1", "MP", "MP+1", "CO", "BTN", "SB", "BB"]
    if n == 9:
        return ["UTG", "UTG+1", "UTG+2", "MP", "MP+1", "CO", "BTN", "SB", "BB"]
    labels = ["UTG"] + [f"UTG+{i}" for i in range(1, n - 5)] + ["MP", "HJ", "CO", "BTN", "SB", "BB"]
    return labels


def position_of(state: PokerStateBase, player_id: str) -> str | None:
    if state.button_seat is None:
        return None
    pos = positions_for(state.players, state.button_seat)
    p = player_by_id(state, player_id)
    if p is None:
        return None
    seat = p.seat if isinstance(p, PlayerState) else p.get("seat")
    return pos.get(seat)


def street_order() -> list[str]:
    return [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER, Street.SHOWDOWN]


#: canonical JSON schema for the state consumed by every downstream agent
STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hand_id": {"type": "string"},
        "game_type": {"type": "string", "enum": ["cash", "mtt", "satellite"]},
        "street": {"type": "string", "enum": ["preflop", "flop", "turn", "river", "showdown"]},
        "board": {"type": "array", "items": {"type": "string"}},
        "pot": {"type": "number"},
        "current_bet": {"type": "number"},
        "to_call": {"type": "number"},
        "min_raise_to": {"type": "number"},
        "sb": {"type": "number"},
        "bb": {"type": "number"},
        "ante": {"type": "number"},
        "players": {"type": "array", "items": {"type": "object"}},
        "button_seat": {"type": ["integer", "null"]},
        "hero_id": {"type": "string"},
        "hero_seat": {"type": ["integer", "null"]},
        "action_history": {"type": "array", "items": {"type": "object"}},
        "street_actions": {"type": "object"},
        "available_actions": {"type": "array", "items": {"type": "string"}},
        "last_aggressor": {"type": ["string", "null"]},
        "rake": {"type": "number"},
        "source": {"type": "string"},
        "cash": {"type": ["object", "null"]},
        "tournament": {"type": ["object", "null"]},
    },
    "required": ["hand_id", "game_type", "street"],
}


def state_to_dict(state: PokerStateBase) -> dict:
    from ..schemas import to_dict

    return to_dict(state)
