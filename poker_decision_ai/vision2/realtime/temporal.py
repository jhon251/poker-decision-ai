"""Temporal State Tracking：上一观测 vs 当前观测 → 事实性 delta/事件。

硬规则（禁止过度推断）：
  - pot 10→14→18 只能报告 "pot increased +4.0 (10.0 -> 14.0)"；
  - 绝不输出 "对手下注 4bb" 之类的行动者归因，除非视觉证据（action 区域
    文字/行动记录）直接支持，且支持时也只按匿名座位事实陈述；
  - 所有 delta/事件携带 evidence_frame_ids，可回放追问。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..schemas import STREET_RANK, RawVisionState



class EventKind(str, Enum):
    NEW_HAND = "NEW_HAND"
    FLOP_APPEARED = "FLOP_APPEARED"
    TURN_APPEARED = "TURN_APPEARED"
    RIVER_APPEARED = "RIVER_APPEARED"
    POT_CHANGED = "POT_CHANGED"
    STACK_CHANGED = "STACK_CHANGED"
    BET_CHANGED = "BET_CHANGED"
    ACTION_AREA_CHANGED = "ACTION_AREA_CHANGED"
    HERO_CARDS_CHANGED = "HERO_CARDS_CHANGED"


@dataclass(frozen=True)
class ObservationSnapshot:
    """单帧观测快照（temporal 层消费的最小接口）。"""
    frame_id: int
    timestamp: float | None
    raw: RawVisionState | None = None
    normalized: Any = None
    status: str = "RECOGNITION_OK"
    blocked: bool = False


@dataclass(frozen=True)
class FieldDelta:
    field: str
    before: Any
    after: Any
    kind: str                       # increased | decreased | changed |
                                    # appeared | disappeared
    delta: float | None = None      # 数值字段的差值
    evidence_frame_ids: tuple = ()

    def description(self) -> str:
        """事实性描述（无行动者归因）。"""
        if self.kind == "increased":
            return f"{self.field} increased +{self.delta} " \
                   f"({self.before} -> {self.after})"
        if self.kind == "decreased":
            return f"{self.field} decreased {self.delta} " \
                   f"({self.before} -> {self.after})"
        if self.kind == "appeared":
            return f"{self.field} appeared ({self.after})"
        if self.kind == "disappeared":
            return f"{self.field} disappeared (was {self.before})"
        return f"{self.field} changed ({self.before} -> {self.after})"

    def to_dict(self) -> dict:
        return {"field": self.field, "before": self.before,
                "after": self.after, "kind": self.kind, "delta": self.delta,
                "evidence_frame_ids": list(self.evidence_frame_ids)}


@dataclass(frozen=True)
class TemporalEvent:
    kind: EventKind
    frame_id: int
    timestamp: float | None
    description: str
    evidence_frame_ids: tuple = ()
    fields: tuple = ()

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "frame_id": self.frame_id,
                "timestamp": self.timestamp, "description": self.description,
                "evidence_frame_ids": list(self.evidence_frame_ids),
                "fields": list(self.fields)}


@dataclass(frozen=True)
class StateTransition:
    previous_frame_id: int
    new_frame_id: int
    timestamp: float | None
    changed_fields: tuple = ()
    deltas: tuple = ()              # tuple[FieldDelta, ...]
    events: tuple = ()              # tuple[TemporalEvent, ...]
    evidence_frame_ids: tuple = ()

    def to_dict(self) -> dict:
        return {
            "previous_frame_id": self.previous_frame_id,
            "new_frame_id": self.new_frame_id,
            "timestamp": self.timestamp,
            "changed_fields": list(self.changed_fields),
            "deltas": [d.to_dict() for d in self.deltas],
            "events": [e.to_dict() for e in self.events],
            "evidence_frame_ids": list(self.evidence_frame_ids),
        }


def _board_cards(raw: RawVisionState) -> list | None:
    """返回可见公共牌列表。board=None → 未知；board dict 存在但全 None → []（确认无牌）。"""
    if raw.board is None:
        return None
    flop = raw.board.get("flop")
    turn = raw.board.get("turn")
    river = raw.board.get("river")
    out = list(flop or []) + list(turn or []) + list(river or [])
    return out


def _num_changed(name: str, before, after, prev_id, new_id) -> FieldDelta | None:
    if before == after:
        return None
    if before is None:
        return FieldDelta(name, before, after, "appeared",
                          evidence_frame_ids=(prev_id, new_id))
    if after is None:
        return FieldDelta(name, before, after, "disappeared",
                          evidence_frame_ids=(prev_id, new_id))
    try:
        d = float(after) - float(before)
        kind = "increased" if d > 0 else ("decreased" if d < 0 else "changed")
        return FieldDelta(name, before, after, kind, delta=d,
                          evidence_frame_ids=(prev_id, new_id))
    except (TypeError, ValueError):
        return FieldDelta(name, before, after, "changed",
                          evidence_frame_ids=(prev_id, new_id))


def _list_changed(name: str, before, after, prev_id, new_id) -> FieldDelta | None:
    if before == after:
        return None
    if before is None:
        return FieldDelta(name, before, after, "appeared",
                          evidence_frame_ids=(prev_id, new_id))
    if after is None:
        return FieldDelta(name, before, after, "disappeared",
                          evidence_frame_ids=(prev_id, new_id))
    return FieldDelta(name, before, after, "changed",
                      evidence_frame_ids=(prev_id, new_id))


def _is_new_hand(prev: RawVisionState, curr: RawVisionState) -> bool:
    """保守的新手牌判定（仅依据可确认的视觉事实，不猜测）。

    要求上一帧存在"进行中/已进行手牌"的证据（街道 ≥ flop 或有公共牌），
    且当前帧已重置（street=preflop 且确认无公共牌）。preflop→preflop
    的 pot 变化不构成新手牌。
    """
    prev_board = _board_cards(prev)
    curr_board = _board_cards(curr)
    prev_had_hand = bool(
        (prev.street is not None and prev.street != "preflop")
        or (prev_board)
    )
    curr_reset = (curr.street == "preflop"
                  and curr_board is not None and len(curr_board) == 0)
    return prev_had_hand and curr_reset


def compute_transition(prev: ObservationSnapshot, curr: ObservationSnapshot
                       ) -> StateTransition:
    """比较两帧 raw 状态 → 事实性 deltas + 事件（无推断）。"""
    p, c = prev.raw, curr.raw
    prev_id, new_id = prev.frame_id, curr.frame_id
    deltas: list = []
    events: list = []

    if p is None or c is None:
        return StateTransition(prev_id, new_id, curr.timestamp,
                               evidence_frame_ids=(prev_id, new_id))

    # ---- 数值字段：pot / hero bet / hero stack / to_call ----
    pot_d = _num_changed("pot", p.pot, c.pot, prev_id, new_id)
    if pot_d:
        deltas.append(pot_d)
        events.append(TemporalEvent(
            EventKind.POT_CHANGED, new_id, curr.timestamp, pot_d.description(),
            evidence_frame_ids=(prev_id, new_id), fields=("pot",)))
    bet_d = _num_changed("hero.current_bet", p.hero_current_bet,
                         c.hero_current_bet, prev_id, new_id)
    if bet_d:
        deltas.append(bet_d)
        events.append(TemporalEvent(
            EventKind.BET_CHANGED, new_id, curr.timestamp, bet_d.description(),
            evidence_frame_ids=(prev_id, new_id), fields=("hero.current_bet",)))
    for d in (_num_changed("hero.stack", p.hero_stack, c.hero_stack,
                           prev_id, new_id),
              _num_changed("hero.amount_to_call", p.hero_amount_to_call,
                           c.hero_amount_to_call, prev_id, new_id)):
        if d:
            deltas.append(d)

    # ---- 每座位 stack / current_bet ----
    p_seats = {s.seat: s for s in (p.seats or []) if s.seat is not None}
    c_seats = {s.seat: s for s in (c.seats or []) if s.seat is not None}
    for seat in sorted(set(p_seats) | set(c_seats)):
        ps, cs = p_seats.get(seat), c_seats.get(seat)
        for fname, getter, ev in (
                ("stack", lambda s: s.stack, EventKind.STACK_CHANGED),
                ("current_bet", lambda s: s.current_bet, EventKind.BET_CHANGED)):
            d = _num_changed(f"players[{seat}].{fname}",
                             getter(ps) if ps else None,
                             getter(cs) if cs else None, prev_id, new_id)
            if d:
                deltas.append(d)
                events.append(TemporalEvent(
                    ev, new_id, curr.timestamp, d.description(),
                    evidence_frame_ids=(prev_id, new_id),
                    fields=(f"players[{seat}].{fname}",)))

    # ---- hero 手牌 ----
    hc_d = _list_changed("hero.hole_cards", p.hero_hole_cards,
                         c.hero_hole_cards, prev_id, new_id)
    if hc_d:
        deltas.append(hc_d)
        events.append(TemporalEvent(
            EventKind.HERO_CARDS_CHANGED, new_id, curr.timestamp,
            hc_d.description(), evidence_frame_ids=(prev_id, new_id),
            fields=("hero.hole_cards",)))

    # ---- board / 街道推进 ----
    pb, cb = _board_cards(p), _board_cards(c)
    plen = None if pb is None else len(pb)
    clen = None if cb is None else len(cb)
    if plen != clen:
        if plen is None:
            deltas.append(FieldDelta("board", pb, cb, "appeared",
                                     evidence_frame_ids=(prev_id, new_id)))
        elif clen is None:
            deltas.append(FieldDelta("board", pb, cb, "disappeared",
                                     evidence_frame_ids=(prev_id, new_id)))
        elif clen > plen and clen == 3 and plen == 0:
            deltas.append(FieldDelta("board.flop", pb, cb, "appeared",
                                     evidence_frame_ids=(prev_id, new_id)))
            events.append(TemporalEvent(
                EventKind.FLOP_APPEARED, new_id, curr.timestamp,
                f"flop appeared ({cb})", evidence_frame_ids=(prev_id, new_id),
                fields=("board.flop",)))
        elif clen > plen and clen == 4 and plen == 3:
            deltas.append(FieldDelta("board.turn", pb, cb, "appeared",
                                     evidence_frame_ids=(prev_id, new_id)))
            events.append(TemporalEvent(
                EventKind.TURN_APPEARED, new_id, curr.timestamp,
                f"turn appeared ({cb})", evidence_frame_ids=(prev_id, new_id),
                fields=("board.turn",)))
        elif clen > plen and clen == 5 and plen == 4:
            deltas.append(FieldDelta("board.river", pb, cb, "appeared",
                                     evidence_frame_ids=(prev_id, new_id)))
            events.append(TemporalEvent(
                EventKind.RIVER_APPEARED, new_id, curr.timestamp,
                f"river appeared ({cb})", evidence_frame_ids=(prev_id, new_id),
                fields=("board.river",)))
        else:
            deltas.append(FieldDelta("board", pb, cb, "changed",
                                     evidence_frame_ids=(prev_id, new_id)))
    elif pb != cb and pb is not None:
        deltas.append(FieldDelta("board", pb, cb, "changed",
                                 evidence_frame_ids=(prev_id, new_id)))

    # ---- action_history（视觉证据支持的记录变化）----
    ah_d = _list_changed("action_history", _ah_sig(p), _ah_sig(c),
                         prev_id, new_id)
    if ah_d:
        deltas.append(ah_d)
        events.append(TemporalEvent(
            EventKind.ACTION_AREA_CHANGED, new_id, curr.timestamp,
            f"action_history changed ({_ah_len(p)} -> {_ah_len(c)} entries)",
            evidence_frame_ids=(prev_id, new_id),
            fields=("action_history",)))

    # ---- 新手牌 ----
    if _is_new_hand(p, c):
        events.append(TemporalEvent(
            EventKind.NEW_HAND, new_id, curr.timestamp,
            "new hand detected (board reset, street=preflop)",
            evidence_frame_ids=(prev_id, new_id), fields=("street", "board")))

    changed = tuple(sorted({d.field for d in deltas}))
    ev_ids = tuple(sorted({fid for d in deltas for fid in d.evidence_frame_ids}
                          | {fid for e in events
                             for fid in e.evidence_frame_ids}))
    return StateTransition(prev_id, new_id, curr.timestamp,
                           changed_fields=changed, deltas=tuple(deltas),
                           events=tuple(events), evidence_frame_ids=ev_ids)


def _ah_sig(raw: RawVisionState) -> list:
    return [(a.player, a.action, a.amount, a.street, a.order)
            for a in (raw.action_history or [])]


def _ah_len(raw: RawVisionState) -> int:
    return len(raw.action_history or [])


def explainability_issues(prev_raw: RawVisionState, curr_raw: RawVisionState,
                          transition: StateTransition) -> list:
    """保守不变量：哪些变化无法由视觉事实解释（→ 触发兜底复核）。

    只做结构检查（pot 缩水 / 同街换牌 / 手牌变化但无新手牌 / 街道倒退），
    不推断任何行动者行为。
    """
    issues: list = []
    new_hand = any(e.kind is EventKind.NEW_HAND for e in transition.events)
    for d in transition.deltas:
        if d.field == "pot" and d.kind == "decreased" and not new_hand:
            issues.append("pot decreased without new hand")
        if d.field == "hero.hole_cards" and d.kind == "changed" \
                and not new_hand:
            issues.append("hero cards changed without new hand")
        if d.field == "board" and d.kind == "changed" and not new_hand:
            issues.append("board changed in place without street progression")
        if d.field == "board" and d.kind == "disappeared" and not new_hand:
            issues.append("board disappeared without new hand")
    if not new_hand and prev_raw.street is not None \
            and curr_raw.street is not None:
        rp, rc = STREET_RANK.get(prev_raw.street, -1), \
            STREET_RANK.get(curr_raw.street, -1)
        if rc < rp:
            issues.append("street regressed without new hand")
    return issues


class TemporalTracker:
    """保持上一/当前观测；每次 update 产出可追溯 StateTransition。"""

    def __init__(self):
        self.previous: ObservationSnapshot | None = None
        self.current: ObservationSnapshot | None = None
        self.last_transition: StateTransition | None = None

    def compute_prelim(self, prev: ObservationSnapshot,
                       curr: ObservationSnapshot) -> StateTransition | None:
        """不改变内部状态的预计算（供兜底触发判断用）。"""
        transition = compute_transition(prev, curr)
        if prev.raw is not None and curr.raw is not None:
            issues = explainability_issues(prev.raw, curr.raw, transition)
            if issues:
                transition = self._attach_issues(curr, transition, issues)
        return transition

    def update(self, snapshot: ObservationSnapshot) -> StateTransition | None:
        transition = None
        if self.current is not None:
            transition = self.compute_prelim(self.current, snapshot)
        self.previous = self.current
        self.current = snapshot
        self.last_transition = transition
        return transition

    @staticmethod
    def _attach_issues(snapshot: ObservationSnapshot,
                       transition: StateTransition,
                       issues: list) -> StateTransition:
        events = list(transition.events)
        for issue in issues:
            events.append(TemporalEvent(
                EventKind.ACTION_AREA_CHANGED, snapshot.frame_id,
                snapshot.timestamp, f"unexplained: {issue}",
                evidence_frame_ids=transition.evidence_frame_ids,
                fields=("temporal.unexplained",)))
        return StateTransition(transition.previous_frame_id,
                               transition.new_frame_id, transition.timestamp,
                               transition.changed_fields, transition.deltas,
                               tuple(events), transition.evidence_frame_ids)

