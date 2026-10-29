"""Event Driven State Engine（V2 Phase 6）。

用 12 类事件增量更新牌局状态，而不是每次行动重建整个牌局：
    HAND_START, HOLE_CARDS, FLOP, TURN, RIVER,
    PLAYER_ACTION(BET/CALL/RAISE/FOLD/ALL_IN), SHOWDOWN, HAND_END

引擎内部维护 V1 标准 PokerState dict + 每街投入增量表；apply(event) 只更新
受影响的字段（pot/current_bet/to_call/status/street），返回最新快照。
事件顺序错误/非法事件 → StateEngineError（调用方决定降级）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.errors import FrameworkError

STREETS = ("preflop", "flop", "turn", "river", "showdown")


class StateEngineError(FrameworkError):
    code = "STATE_ENGINE_ERROR"


@dataclass
class StateEngineEvent:
    type: str
    street: str = ""
    player_id: str = ""
    action: str = ""          # fold|check|call|bet|raise
    amount: float = 0.0       # 增量（raise 时为加注增量）
    to: float | None = None   # 该玩家本街总注额（raises $X to $Y 的 Y）
    all_in: bool = False
    board: list = field(default_factory=list)
    cards: list = field(default_factory=list)
    seats: list = field(default_factory=list)
    button_seat: int | None = None
    sb: float = 0.0
    bb: float = 0.0
    ante: float = 0.0
    game_type: str = "cash"
    hand_id: str = ""
    winners: list = field(default_factory=list)
    shows: list = field(default_factory=list)
    seq: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type, "street": self.street, "player_id": self.player_id,
            "action": self.action, "amount": self.amount, "to": self.to,
            "all_in": self.all_in, "board": list(self.board), "cards": list(self.cards),
            "seats": self.seats, "button_seat": self.button_seat,
            "sb": self.sb, "bb": self.bb, "ante": self.ante, "game_type": self.game_type,
            "hand_id": self.hand_id, "winners": self.winners, "shows": self.shows,
            "seq": self.seq, "extra": dict(self.extra),
        }


def _player_state(seat: dict, is_hero: bool) -> dict:
    return {
        "player_id": str(seat.get("name")), "name": str(seat.get("name")),
        "seat": int(seat.get("seat")), "stack": float(seat.get("stack") or 0.0),
        "status": "active", "hole_cards": [], "position": None, "is_hero": is_hero,
        "bounty": float(seat.get("bounty") or 0.0),
    }


class PokerStateEngine:
    """增量牌局状态机（线程不安全：单请求内使用）。"""

    def __init__(self, hero_id: str = "") -> None:
        self.hero_id = hero_id
        self.state: dict[str, Any] = {
            "hand_id": "", "game_type": "cash", "street": "preflop",
            "board": [], "pot": 0.0, "current_bet": 0.0, "to_call": 0.0,
            "min_raise_to": 0.0, "sb": 0.0, "bb": 0.0, "ante": 0.0,
            "players": [], "button_seat": None, "hero_id": hero_id, "hero_seat": None,
            "action_history": [], "street_actions": {}, "available_actions": [],
            "last_aggressor": None, "rake": 0.0, "source": "event_stream",
            "cash": None, "tournament": None,
        }
        self._street_invested: dict[tuple[str, str], float] = {}
        self._status: dict[str, str] = {}
        self._seq = 0
        self.warnings: list[str] = []
        self.events: list[dict] = []

    # ---------------------------------------------------------------- 接口
    def apply(self, event: dict) -> dict:
        """应用一个事件，返回最新状态快照。非法/乱序事件抛 StateEngineError。"""
        ev = StateEngineEvent(**{k: v for k, v in event.items()
                                 if k in StateEngineEvent.__dataclass_fields__})
        self._apply(ev)
        self.events.append(ev.to_dict())
        return self.snapshot()

    def snapshot(self) -> dict:
        import copy

        return copy.deepcopy(self.state)

    def reset(self) -> None:
        self.__init__(self.hero_id)  # type: ignore[misc]

    # ---------------------------------------------------------------- 事件处理
    def _apply(self, ev: StateEngineEvent) -> None:
        t = ev.type.upper()
        if t == "HAND_START":
            self._on_hand_start(ev)
        elif t == "HOLE_CARDS":
            self._on_hole_cards(ev)
        elif t in ("FLOP", "TURN", "RIVER"):
            self._on_street_card(ev, t.lower())
        elif t in ("PLAYER_ACTION", "BET", "CALL", "RAISE", "FOLD", "ALL_IN"):
            self._on_action(ev)
        elif t == "SHOWDOWN":
            self.state["street"] = "showdown"
            for s in ev.shows:
                self._log("show", ev.street, s.get("player_id") or s.get("player", ""), 0.0)
        elif t == "HAND_END":
            self.state["pot"] = float(ev.amount or self.state["pot"])
            for w in ev.winners:
                self._log("win", ev.street, w.get("player_id") or w.get("player", ""), 0.0)
        else:
            raise StateEngineError(f"unknown event type '{ev.type}'")

    # ---------------------------------------------------------------- 具体事件
    def _on_hand_start(self, ev: StateEngineEvent) -> None:
        s = self.state
        s["hand_id"] = ev.hand_id or s["hand_id"]
        s["game_type"] = ev.game_type or "cash"
        s["sb"], s["bb"], s["ante"] = float(ev.sb), float(ev.bb), float(ev.ante)
        s["button_seat"] = ev.button_seat
        s["players"] = [_player_state(seat, str(seat.get("name")) == self.hero_id or bool(seat.get("is_hero")))
                        for seat in ev.seats]
        s["hero_id"] = self.hero_id or s["hero_id"]
        for p in s["players"]:
            if p["is_hero"]:
                s["hero_seat"] = p["seat"]
                s["hero_id"] = p["player_id"]
        s["street"] = "preflop"
        s["current_bet"] = s["bb"]
        self._refresh_positions()

    def _on_hole_cards(self, ev: StateEngineEvent) -> None:
        for p in self.state["players"]:
            if p["player_id"] == ev.player_id or (ev.player_id == "" and p["is_hero"]):
                p["hole_cards"] = list(ev.cards)

    def _on_street_card(self, ev: StateEngineEvent, street: str) -> None:
        if self.state["street"] == "preflop" and street != "flop":
            raise StateEngineError(f"street order violation: {street} before flop")
        self.state["street"] = street
        for card in ev.board:
            if card not in self.state["board"]:
                self.state["board"].append(str(card))
        # 新街：重置本街下注基准
        self.state["current_bet"] = 0.0
        self.state["to_call"] = 0.0
        self.state["last_aggressor"] = None
        self.state["min_raise_to"] = self.state["bb"]
        self._refresh_actions()

    def _on_action(self, ev: StateEngineEvent) -> None:
        s = self.state
        street = ev.street or s["street"]
        pid = ev.player_id
        kind = (ev.action or ev.type).lower()
        contrib = 0.0
        if kind == "post":
            contrib = float(ev.amount)
            s["pot"] += contrib
            self._add_invested(street, pid, contrib)
            if street == "preflop" and contrib >= s["bb"] - 1e-9:
                s["current_bet"] = max(s["current_bet"], s["bb"])
            self._log("post", street, pid, contrib)
            self._refresh_actions()
            return
        if kind in ("bet", "raise"):
            if kind == "raise" and ev.to is not None:
                prior = self._invested(street, pid)
                contrib = max(0.0, float(ev.to) - prior)
            else:
                contrib = float(ev.amount)
            s["pot"] += contrib
            self._add_invested(street, pid, contrib)
            s["current_bet"] = max(s["current_bet"], self._invested(street, pid))
            s["last_aggressor"] = pid
        elif kind == "call":
            contrib = float(ev.amount)
            s["pot"] += contrib
            self._add_invested(street, pid, contrib)
        elif kind == "fold":
            self._status[pid] = "folded"
        elif kind == "check":
            pass
        else:
            raise StateEngineError(f"unknown action kind '{kind}'")

        if ev.all_in or kind == "all_in":
            self._status[pid] = "allin"
        self._sync_player_statuses()
        if ev.player_id == s["hero_id"]:
            self._log(kind, street, pid, contrib)
        else:
            self._log(kind, street, pid, contrib)
        # 增量更新 to_call/min_raise/available_actions
        hero_street = self._invested(street, s["hero_id"]) if s["hero_id"] else 0.0
        s["to_call"] = max(0.0, round(s["current_bet"] - hero_street, 4))
        base = s["current_bet"] if s["current_bet"] > 0 else s["bb"]
        s["min_raise_to"] = max(0.0, round(2.0 * base - hero_street, 4))
        self._refresh_actions()

    # ---------------------------------------------------------------- helpers
    def _invested(self, street: str, pid: str) -> float:
        return self._street_invested.get((street, pid), 0.0)

    def _add_invested(self, street: str, pid: str, amount: float) -> None:
        if amount > 0:
            self._street_invested[(street, pid)] = self._invested(street, pid) + amount

    def _sync_player_statuses(self) -> None:
        for p in self.state["players"]:
            if p["player_id"] in self._status:
                p["status"] = self._status[p["player_id"]]

    def _log(self, action: str, street: str, pid: str, amount: float) -> None:
        self._seq += 1
        rec = {"street": street, "player_id": pid, "action": action,
               "amount": round(self._invested(street, pid), 4) if action in ("bet", "raise", "call") else round(amount, 4),
               "all_in": self._status.get(pid) == "allin", "seq": self._seq}
        self.state["action_history"].append(rec)
        self.state["street_actions"].setdefault(street, []).append(rec)

    def _refresh_positions(self) -> None:
        from .models import positions_for

        s = self.state
        pos = positions_for(s["players"], s["button_seat"]) if s["button_seat"] is not None else {}
        for p in s["players"]:
            p["position"] = pos.get(p["seat"])

    def _refresh_actions(self) -> None:
        s = self.state
        to_call = s["to_call"]
        hero_p = next((p for p in s["players"] if p["is_hero"]), None)
        hero_stack = float(hero_p["stack"]) if hero_p else 0.0
        if to_call <= 0:
            actions = ["check", "bet"]
            if hero_stack <= s["pot"]:
                actions.append("allin")
        elif hero_stack <= to_call:
            actions = ["fold", "call"]
        else:
            actions = ["fold", "call", "raise"]
        s["available_actions"] = actions


# ---------------------------------------------------------------- 从手牌历史构造事件流
def events_from_parsed_hand(hand) -> list[dict]:
    """把 ParsedHand 转成事件流（HAND_START → HOLE_CARDS → 行动 → FLOP...）。"""
    events: list[dict] = []
    seats = [{"seat": s["seat"], "name": s["name"], "stack": s["stack"],
              "is_hero": s.get("is_hero", False), "bounty": s.get("bounty", 0.0)}
             for s in hand.seats]
    events.append({"type": "HAND_START", "hand_id": hand.hand_id, "game_type": hand.game,
                   "sb": hand.sb, "bb": hand.bb, "button_seat": hand.button_seat,
                   "seats": seats})
    hero = hand.hero_id
    if hand.hole_cards:
        events.append({"type": "HOLE_CARDS", "player_id": hero, "cards": hand.hole_cards})
    # 街牌事件必须与行动按时间顺序交错：遇到某街第一个行动前先发该街牌事件
    emitted_streets: set[str] = set()
    board_so_far: list[str] = []
    for a in hand.actions:
        street = a.get("street", "preflop")
        if street in ("flop", "turn", "river") and street not in emitted_streets:
            emitted_streets.add(street)
            board_so_far = []
            for prev in ("flop", "turn", "river"):
                board_so_far.extend(hand.street_boards.get(prev, []))
                if prev == street:
                    break
            events.append({"type": street.upper(), "street": street,
                           "board": list(board_so_far)})
        kind = a.get("action", "")
        if kind == "post":
            events.append({"type": "PLAYER_ACTION", "street": street,
                           "player_id": a["player"], "action": "post",
                           "amount": a.get("amount", 0.0)})
        elif kind == "fold":
            events.append({"type": "FOLD", "street": street, "player_id": a["player"]})
        elif kind == "check":
            events.append({"type": "PLAYER_ACTION", "street": street,
                           "player_id": a["player"], "action": "check"})
        elif kind == "call":
            events.append({"type": "CALL", "street": street, "player_id": a["player"],
                           "amount": a.get("amount", 0.0), "all_in": a.get("all_in", False)})
        elif kind == "bet":
            events.append({"type": "BET", "street": street, "player_id": a["player"],
                           "amount": a.get("amount", 0.0), "to": a.get("to"),
                           "all_in": a.get("all_in", False)})
        elif kind == "raise":
            events.append({"type": "RAISE", "street": street, "player_id": a["player"],
                           "amount": a.get("amount", 0.0), "to": a.get("to"),
                           "all_in": a.get("all_in", False)})
    return events
