"""Vision Agent V2 — 15 条确定性 Poker State Validator。

原则（硬规则，禁止静默修复）：
  - null = unknown：进入 uncertainty（status=uncertain/missing），不判 invalid；
  - 非 null 值之间的正向矛盾 = validation_error（valid=false）；
  - 任何关键字段冲突显式暴露，绝不静默纠正。

位置（Rule 14）由冻结 `positions_for` 在 assembly 层推导；validator 只校验
button/seat 事实是否可用于推导（视觉层不得猜 position）。
"""
from __future__ import annotations

from .schemas import (
    ACTIONS, STREET_RANK, STREETS, RawVisionState, UncertaintyEntry)

#: 短牌记法 [2-9TJQKA][cdhs]
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _card(c: str | None) -> str | None:
    """标准化单张牌；None→None；非法记法→'malformed' 标记（视作不确定）。"""
    if c is None:
        return None
    s = str(c).strip()
    if len(s) != 2:
        return "malformed"
    r, su = s[0].upper(), s[1].lower()
    if r not in _RANKS or su not in _SUITS:
        return "malformed"
    return r + su


def _cards(cs) -> list:
    if cs is None:
        return []
    return [c for c in cs]           # 保留 None 与原始元素，逐项 _card 处理


def _err(errors, code, field, message):
    errors.append({"code": code, "field": field, "message": message})


def _uncertain(uncertainty, field, status, confidence=None):
    uncertainty.append(UncertaintyEntry(field=field, status=status,
                                        confidence=confidence))


def _num_ge(v, code, field, label, errors, uncertainty, warn_when_none=True):
    if v is None:
        if warn_when_none:
            _uncertain(uncertainty, field, "missing")
        return
    if v < 0:
        _err(errors, code, field, f"{label} 不得为负（{v}）")


def validate_raw(raw: RawVisionState) -> tuple[list, list, list]:
    """返回 (errors, uncertainty, warnings)。绝不修改 raw。"""
    errors: list = []
    uncertainty: list = []
    warnings: list = []

    hero_cards = _cards(raw.hero_hole_cards) if raw.hero_hole_cards is not None else None
    board = raw.board or {}
    board_flop = _cards(board.get("flop")) if board.get("flop") is not None else None
    board_turn = _cards(board.get("turn")) if board.get("turn") is not None else None
    board_river = _cards(board.get("river")) if board.get("river") is not None else None
    board_cards = (board_flop or []) + (board_turn or []) + (board_river or [])

    # ---- Rule 6/7/8：pot/stack/bet ≥ 0 ----
    _num_ge(raw.pot, "pot_negative", "pot", "pot", errors, uncertainty)
    for sp in (raw.side_pots or []):
        _num_ge(sp, "side_pot_negative", "pot.side_pots", "side pot", errors, uncertainty)
    _num_ge(raw.hero_stack, "hero_stack_negative", "hero.stack", "hero stack", errors, uncertainty)
    _num_ge(raw.hero_current_bet, "hero_bet_negative", "hero.current_bet", "hero bet", errors, uncertainty)
    _num_ge(raw.hero_amount_to_call, "to_call_negative", "hero.amount_to_call",
            "amount_to_call", errors, uncertainty)
    blinds = raw.blinds or {}
    for k in ("sb", "bb", "ante"):
        _num_ge(blinds.get(k), "blind_negative", f"game.blinds.{k}", k, errors,
                uncertainty, warn_when_none=(k in ("sb", "bb")))

    # ---- Rule 13 / seat 事实 ----
    seats = raw.seats if raw.seats is not None else None
    seat_nums = []
    hero_count = 0
    if seats is not None:
        for i, s in enumerate(seats):
            if s.seat is None:
                _uncertain(uncertainty, f"players[{i}].seat", "missing")
                continue
            if s.seat in seat_nums:
                _err(errors, "duplicate_seat", "players", f"座位号重复: {s.seat}")
            seat_nums.append(s.seat)
            if raw.hero_seat is not None and s.seat == raw.hero_seat:
                hero_count += 1
            # Rule 7/8/9 每玩家
            _num_ge(s.stack, "stack_negative", f"players[{i}].stack", "stack",
                    errors, uncertainty)
            _num_ge(s.current_bet, "bet_negative", f"players[{i}].current_bet",
                    "current_bet", errors, uncertainty)
            if s.stack is not None and s.current_bet is not None \
                    and s.current_bet > s.stack and not s.all_in:
                _err(errors, "bet_exceeds_stack", f"players[{i}]",
                     f"非 all-in 时下注 {s.current_bet} 超过筹码 {s.stack}")
            if s.folded is True and s.visible_action not in (None, "fold"):
                _err(errors, "folded_with_action", f"players[{i}]",
                     f"folded 玩家出现行动 {s.visible_action}")
    if raw.hero_seat is None:
        _uncertain(uncertainty, "hero.seat", "missing")
    elif seats is not None:
        if hero_count == 0:
            _err(errors, "hero_seat_not_found", "hero.seat",
                 f"hero_seat={raw.hero_seat} 未对应任何 seat")
        elif hero_count > 1:
            _err(errors, "multiple_hero", "hero.seat",
                 f"hero_seat 对应 {hero_count} 个 seat（应恰为 1）")

    # ---- Rule 9（hero）：非 all-in 时 hero bet ≤ stack ----
    if (raw.hero_stack is not None and raw.hero_current_bet is not None
            and raw.hero_current_bet > raw.hero_stack and not raw.hero_all_in):
        _err(errors, "hero_bet_exceeds_stack", "hero",
             f"非 all-in 时 hero 下注 {raw.hero_current_bet} 超过筹码 {raw.hero_stack}")

    # ---- Rule 1/2：重复牌 ----
    norm_hero = [_card(c) for c in (hero_cards or [])]
    norm_board = [_card(c) for c in board_cards]
    all_cards = [c for c in norm_hero + norm_board if c is not None]
    seen = {}
    for c in all_cards:
        if c == "malformed":
            _uncertain(uncertainty, "cards", "uncertain")
            continue
        if c in seen:
            _err(errors, "duplicate_card", "cards", f"可见牌重复: {c}")
        seen[c] = True

    # ---- Rule 3/4/5：街牌张数（形状） ----
    street = raw.street
    if street is not None and street not in STREETS:
        _err(errors, "invalid_street", "game.street", f"非法街道 {street}")
    if board_flop is not None and len(board_flop) != 3:
        _err(errors, "flop_length", "board.flop", f"flop 应为 3 张，实际 {len(board_flop)}")
    if board_turn is not None and len(board_turn) != 1:
        _err(errors, "turn_length", "board.turn", f"turn 应为 1 张，实际 {len(board_turn)}")
    if board_river is not None and len(board_river) != 1:
        _err(errors, "river_length", "board.river", f"river 应为 1 张，实际 {len(board_river)}")
    # 街与牌面存在性：street 前进但对应牌缺 → uncertain（不判 invalid）
    if street is not None and STREET_RANK.get(street, -1) >= 1 and board_flop is None:
        _uncertain(uncertainty, "board.flop", "missing")
    if street is not None and STREET_RANK.get(street, -1) >= 2 and board_turn is None:
        _uncertain(uncertainty, "board.turn", "missing")
    if street is not None and STREET_RANK.get(street, -1) >= 3 and board_river is None:
        _uncertain(uncertainty, "board.river", "missing")

    # ---- Rule 15：关键字段冲突 ----
    if street == "preflop" and any(c for c in board_cards if c is not None):
        _err(errors, "board_on_preflop", "board",
             "street=preflop 但存在公共牌（冲突）")
    if hero_cards is not None and len(hero_cards) != 2:
        _err(errors, "hero_cards_length", "hero.hole_cards",
             f"hero 手牌应恰 2 张，实际 {len(hero_cards)}")
    if raw.poker_variant not in (None, "nlhe"):
        _err(errors, "variant_unsupported", "game.poker_variant",
             f"仅支持 nlhe，实际 {raw.poker_variant}")
    if raw.game_type not in (None, "cash", "mtt", "satellite"):
        _err(errors, "game_type_invalid", "game.game_type",
             f"非法 game_type {raw.game_type}")

    # ---- Rule 12：action order 合法 ----
    ah = raw.action_history
    if ah is not None:
        orders = []
        for i, a in enumerate(ah):
            if a.order is not None:
                orders.append((a.order, i))
            if a.action is not None and a.action not in ACTIONS:
                _err(errors, "invalid_action", f"action_history[{i}].action",
                     f"非法行动 {a.action}")
            if a.street is not None and a.street not in STREETS:
                _err(errors, "invalid_action_street",
                     f"action_history[{i}].street", f"非法街道 {a.street}")
        for (o1, i1), (o2, i2) in zip(orders, orders[1:]):
            if o1 == o2:
                _err(errors, "duplicate_action_order", "action_history",
                     f"行动 order 重复: {o1}")
            elif o1 > o2:
                _err(errors, "action_order_regression", "action_history",
                     f"行动 order 倒退: {o1} -> {o2}")

    # ---- Rule 11：street 不倒退（action_history 内 + game.street）----
    last_rank = -1
    for i, a in enumerate(ah or []):
        if a.street is None:
            _uncertain(uncertainty, f"action_history[{i}].street", "missing")
            continue
        r = STREET_RANK[a.street]
        if r < last_rank:
            _err(errors, "street_regression", "action_history",
                 f"街道倒退（action_history[{i}]）")
        last_rank = max(last_rank, r)
    if street is not None and last_rank >= 0 and STREET_RANK.get(street, -1) < last_rank:
        _err(errors, "game_street_behind_action", "game.street",
             f"game.street={street} 早于最后行动街道")

    # ---- Rule 10：folded 玩家不得有后续行动 ----
    if seats is not None and ah is not None:
        folded_seats = {s.seat for s in seats if s.folded is True and s.seat is not None}
        pid_of = {s.seat: s.anonymous_player_id for s in seats
                  if s.seat is not None and s.anonymous_player_id}
        # 用 anonymous_player_id 关联行动；找不到 id 用 seat 字符串
        def pid(s):
            return pid_of.get(s.seat) or f"seat{s.seat}"
        fold_order = {}
        for a in ah:
            if a.action == "fold" and a.player is not None:
                fold_order.setdefault(a.player, a.order)
        for i, a in enumerate(ah):
            if a.player is None or a.action is None or a.order is None:
                continue
            if a.player in fold_order and fold_order[a.player] is not None \
                    and a.order > fold_order[a.player] and a.action != "fold":
                _err(errors, "folded_later_action", f"action_history[{i}]",
                     f"folded 玩家 {a.player} 出现后续行动 {a.action}")

    # ---- Rule 14（结构）：button 事实可用于推导 ----
    if raw.button_seat is None:
        _uncertain(uncertainty, "button_seat", "missing")
    elif seats is not None and raw.button_seat not in seat_nums:
        _err(errors, "button_seat_not_found", "button_seat",
             f"button_seat={raw.button_seat} 未对应任何 seat")

    return errors, uncertainty, warnings
