"""Canonical Hand History Schema（V2 Phase 4 标准化层）。

Canonical Hand 是全系统数据标准格式：任何来源（手牌文本/JSON/CSV/tracker 导出）
都先规范化为本结构，再进入 Validated / Replay / Experiment 层。

原则：原始数据缺少的字段 → null，绝不猜测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.schemas import json_schema_for, to_dict


@dataclass
class CanonicalPlayer:
    player_id: str
    seat: int | None = None
    name: str = ""
    stack_start: float | None = None      # 开局筹码（货币/记分牌单位）
    stack_end: float | None = None        # 终局筹码（可能 null）
    hole_cards: list = field(default_factory=list)
    position: str | None = None
    bounty: float | None = None           # PKO/KO 赏金；null=无该信息
    is_hero: bool = False
    status_end: str | None = None         # active|folded|allin|out（终局状态）


@dataclass
class CanonicalAction:
    street: str
    seq: int
    player_id: str
    action: str                           # post|fold|check|call|bet|raise
    amount: float | None = None           # 增量金额（下注/加注增量或跟注额）
    to: float | None = None               # 该玩家本街总注额（raises $X to $Y 的 Y）
    all_in: bool = False


@dataclass
class CanonicalHand:
    hand_id: str
    source: str = ""                      # 文件/来源标识（如 "data/hand_history/x.txt"）
    source_format: str = ""               # hand_history_text|canonical_json|records_csv
    game_mode: str | None = None          # cash|tournament|bounty
    game_type: str | None = None          # cash|mtt|satellite
    tournament_type: str | None = None    # mtt|satellite|freezeout
    bounty_type: str | None = None        # pko|ko|mystery
    stakes: str | None = None             # NL10 / Tournament
    sb: float | None = None
    bb: float | None = None
    ante: float | None = None
    currency: str | None = None
    table_size: int | None = None
    button_seat: int | None = None
    started_at: str | None = None         # ISO8601（原始数据无时间 → null）
    players: list = field(default_factory=list)      # list[CanonicalPlayer]
    hero_id: str | None = None
    hole_cards: list = field(default_factory=list)   # hero 手牌（未知 → []）
    preflop_actions: list = field(default_factory=list)
    flop: list = field(default_factory=list)
    flop_actions: list = field(default_factory=list)
    turn: list = field(default_factory=list)
    turn_actions: list = field(default_factory=list)
    river: list = field(default_factory=list)
    river_actions: list = field(default_factory=list)
    pot_sizes: dict = field(default_factory=dict)     # {street: pot}（原始缺失 → 缺键）
    bet_sizes: list = field(default_factory=list)     # [{street, player_id, size}]
    all_in: list = field(default_factory=list)        # [player_id]
    folds: list = field(default_factory=list)         # [player_id]
    showdown: dict | None = None          # {happened, shows: [{player_id, cards, desc}]}
    winners: list = field(default_factory=list)       # [{player_id, amount}]
    rake: float | None = None
    total_pot: float | None = None
    meta: dict = field(default_factory=dict)          # {license, tags, ...}

    def to_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalHand":
        from ...core.schemas import from_dict

        return from_dict(cls, data)


CANONICAL_HAND_SCHEMA: dict[str, Any] = json_schema_for(CanonicalHand)


def parsed_hand_to_canonical(parsed, *, source: str = "", source_format: str = "hand_history_text",
                             license_info: str = "", hero_id: str | None = None) -> CanonicalHand:
    """现有 ParsedHand → CanonicalHand。缺失字段一律 None/空，绝不猜测。"""
    from .hand_history import ParsedHand

    hand: ParsedHand = parsed
    players = []
    for s in hand.seats:
        is_hero = bool(s.get("is_hero")) or (s.get("name") == (hero_id or hand.hero_id))
        players.append(CanonicalPlayer(
            player_id=str(s.get("name")),
            seat=int(s.get("seat")) if s.get("seat") is not None else None,
            name=str(s.get("name")),
            stack_start=float(s.get("stack")) if s.get("stack") is not None else None,
            hole_cards=list(hand.hole_cards) if is_hero else [],
            position=None,                       # 由 State 层推导（canonical 不猜）
            bounty=float(s.get("bounty")) if s.get("bounty") else None,
            is_hero=is_hero,
        ))
    actions_by_street: dict[str, list[CanonicalAction]] = {
        "preflop": [], "flop": [], "turn": [], "river": [],
    }
    seq = 0
    for a in hand.actions:
        street = str(a.get("street", "preflop"))
        if street not in actions_by_street:
            actions_by_street[street] = []
        seq += 1
        actions_by_street[street].append(CanonicalAction(
            street=street, seq=seq, player_id=str(a.get("player")),
            action=str(a.get("action")),
            amount=float(a["amount"]) if a.get("amount") is not None else None,
            to=float(a["to"]) if a.get("to") is not None else None,
            all_in=bool(a.get("all_in")),
        ))
    game = str(hand.game or "cash")
    game_mode = None
    game_type = None
    tournament_type = None
    bounty_type = None
    if game == "cash":
        game_mode, game_type = "cash", "cash"
    elif game == "satellite":
        game_mode, game_type, tournament_type = "tournament", "satellite", "satellite"
    else:  # mtt
        t = hand.tournament or {}
        bt = t.get("bounty_type")
        if bt or t.get("is_bounty"):
            game_mode, game_type = "bounty", "mtt"
            bounty_type = bt or "pko"
        else:
            game_mode, game_type, tournament_type = "tournament", "mtt", "mtt"
    flop_cards = list(hand.street_boards.get("flop") or [])
    turn_cards = list(hand.street_boards.get("turn") or [])
    river_cards = list(hand.street_boards.get("river") or [])
    showdown = None
    if hand.shows:
        showdown = {"happened": True, "shows": [{"player_id": s.get("player"),
                                                 "cards": s.get("cards") or [],
                                                 "desc": s.get("desc") or ""}
                                                for s in hand.shows]}
    return CanonicalHand(
        hand_id=str(hand.hand_id or ""),
        source=source, source_format=source_format,
        game_mode=game_mode, game_type=game_type,
        tournament_type=tournament_type, bounty_type=bounty_type,
        sb=float(hand.sb) if hand.sb else None,
        bb=float(hand.bb) if hand.bb else None,
        ante=None,
        currency=str(hand.currency) if hand.currency else None,
        table_size=int(hand.max_players) if hand.max_players else None,
        button_seat=int(hand.button_seat) if hand.button_seat is not None else None,
        started_at=hand.started_at,            # 解析器已提取头部时间（缺失 → None）
        players=players,
        hero_id=hero_id or hand.hero_id or None,
        hole_cards=list(hand.hole_cards),
        preflop_actions=actions_by_street["preflop"],
        flop=flop_cards, flop_actions=actions_by_street["flop"],
        turn=turn_cards, turn_actions=actions_by_street["turn"],
        river=river_cards, river_actions=actions_by_street["river"],
        pot_sizes={},
        bet_sizes=[],
        all_in=[a.player_id for st in actions_by_street.values() for a in st if a.all_in],
        folds=[a.player_id for st in actions_by_street.values() for a in st if a.action == "fold"],
        showdown=showdown,
        winners=[{"player_id": w.get("player"), "amount": float(w.get("amount") or 0.0)}
                 for w in hand.winners],
        rake=float(hand.rake) if hand.rake else None,
        total_pot=float(hand.pot) if hand.pot else None,
        meta={"license": license_info},
    )


def canonical_to_vision_observation(hand: CanonicalHand, street: str) -> dict | None:
    """Canonical → 决策点 vision_observation（供 Replay/分析复用）。

    street 必须有公共牌与行动历史；无法构造（缺 hero 手牌/座位等）→ None（不猜）。
    """
    from .state_engine import PokerStateEngine

    if not hand.hero_id or not hand.hole_cards:
        return None
    eng = PokerStateEngine(hero_id=hand.hero_id)
    seats = [{"seat": p.seat, "name": p.player_id, "stack": p.stack_start,
              "is_hero": p.is_hero, "bounty": p.bounty or 0.0}
             for p in hand.players if p.seat is not None]
    if not seats:
        return None
    try:
        eng.apply({"type": "HAND_START", "hand_id": hand.hand_id,
                   "game_type": hand.game_type or "cash", "sb": hand.sb or 0.0,
                   "bb": hand.bb or 0.0, "ante": hand.ante or 0.0,
                   "button_seat": hand.button_seat, "seats": seats})
        eng.apply({"type": "HOLE_CARDS", "player_id": hand.hero_id,
                   "cards": hand.hole_cards})
        for a in hand.preflop_actions:
            eng.apply(_action_event(a))
        for st, cards, acts in (("flop", hand.flop, hand.flop_actions),
                                ("turn", hand.turn, hand.turn_actions),
                                ("river", hand.river, hand.river_actions)):
            if cards:
                eng.apply({"type": st.upper(), "street": st, "board": cards})
            for a in acts:
                eng.apply(_action_event(a))
            if st == street:
                break
    except Exception:
        return None
    state = eng.snapshot()
    return {
        "source": "canonical", "hero_seat": state.get("hero_seat"),
        "button_seat": state.get("button_seat"), "street": state.get("street"),
        "hole_cards": list(hand.hole_cards), "board": list(state.get("board") or []),
        "seats": [{"seat": p["seat"], "name": p["name"], "stack": p["stack"],
                   "in_hand": p["status"] in ("active", "allin")}
                  for p in state["players"]],
        "pot": state.get("pot"), "current_bet": state.get("current_bet"),
        "to_call": state.get("to_call"),
        "blinds": {"sb": state.get("sb"), "bb": state.get("bb")},
        "ante": state.get("ante"),
        "available_actions": list(state.get("available_actions") or []),
        "confidence": 1.0, "extra": {"game_type": state.get("game_type")},
    }


def _action_event(a: CanonicalAction) -> dict:
    kind = a.action
    if kind in ("check", "post"):
        # 引擎事件类型体系里 check/post 用 PLAYER_ACTION + action 字段表达
        return {"type": "PLAYER_ACTION", "street": a.street, "player_id": a.player_id,
                "action": kind, "amount": a.amount or 0.0, "to": a.to, "all_in": a.all_in}
    return {"type": kind.upper(), "street": a.street, "player_id": a.player_id,
            "action": kind, "amount": a.amount or 0.0, "to": a.to,
            "all_in": a.all_in}
