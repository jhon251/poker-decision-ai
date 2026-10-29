"""Vision Agent V2 — Raw → Normalize 装配器。

职责：RawVisionState → validate_raw → NormalizedVisionState。
position 只由冻结 `positions_for` 从 (seat, button_seat) 确定性推导；button 缺失
时 position 一律 None（不猜）。视觉模型绝不参与 position。
"""
from __future__ import annotations

from .schemas import (
    NormalizedVisionState, RawVisionState, UncertaintyEntry, ValidationResult,
    to_dict)
from .validator import validate_raw


def _positions_for(seats: list, button_seat) -> dict:
    """确定性位置推导（复用冻结 models.positions_for；不改冻结文件）。

    M8 契约守卫（不改冻结 `positions_for`）：冻结函数在
    `button_seat not in seats` 时走"回退固定顺序"分支，会把第一个座位
    标成 BTN —— 这与本层契约「dealer 冲突 → position 一律 null，不猜」
    直接冲突。因此本层**绝不把无效 button_seat 传给冻结函数**：
    button_seat 为 None、或不属于任何已知座位时，直接返回全部 None
    （与 validator 的 `button_seat_not_found` 判定一致）。
    """
    valid = {s["seat"] for s in seats if s.get("seat") is not None}
    from poker_decision_ai.domain.poker.models import positions_for
    players = [{"seat": s["seat"], "name": s.get("anonymous_player_id")
                or f"player_{s['seat']}", "stack": s.get("stack") or 0.0}
               for s in seats if s.get("seat") is not None]
    if button_seat is None or button_seat not in valid:
        # 无有效 dealer 证据：不推导、不 fallback（UNKNOWN → null）
        return {s: None for s in valid}
    try:
        return positions_for(players, button_seat)
    except Exception:
        return {s: None for s in valid}


def assemble(raw: RawVisionState) -> NormalizedVisionState:
    """Raw → 校验 → Normalized。绝不修改 raw；缺失/矛盾显式暴露。"""
    errors, uncertainty, warnings = validate_raw(raw)

    # 合并 validator 产出的 uncertainty 与 raw 自带 uncertainty
    merged_unc: list = [u for u in uncertainty]
    if raw.uncertainty:
        for u in raw.uncertainty:
            merged_unc.append(u)

    # players
    raw_seats = raw.seats if raw.seats is not None else []
    norm_players = []
    for i, s in enumerate(raw_seats):
        norm_players.append({
            "seat": s.seat, "occupied": s.occupied,
            "anonymous_player_id": s.anonymous_player_id,
            "stack": s.stack, "current_bet": s.current_bet,
            "folded": s.folded, "all_in": s.all_in,
            "in_hand": s.in_hand, "visible_action": s.visible_action,
            "position": None,          # 下方确定性推导填充
        })
    pos_map = _positions_for(norm_players, raw.button_seat)
    for p in norm_players:
        if p["seat"] is not None:
            p["position"] = pos_map.get(p["seat"])

    hero = {
        "seat": raw.hero_seat,
        "hole_cards": list(raw.hero_hole_cards) if raw.hero_hole_cards is not None else None,
        "stack": raw.hero_stack,
        "current_bet": raw.hero_current_bet,
        "amount_to_call": raw.hero_amount_to_call,
        "all_in": raw.hero_all_in,
        "acted": raw.hero_acted,
        "position": pos_map.get(raw.hero_seat) if raw.hero_seat is not None else None,
    }

    board = {
        "flop": list(raw.board["flop"]) if (raw.board and raw.board.get("flop") is not None) else None,
        "turn": list(raw.board["turn"]) if (raw.board and raw.board.get("turn") is not None) else None,
        "river": list(raw.board["river"]) if (raw.board and raw.board.get("river") is not None) else None,
    }
    pot = {"pot": raw.pot,
           "side_pots": list(raw.side_pots) if raw.side_pots is not None else None}
    game = {
        "poker_variant": raw.poker_variant, "game_type": raw.game_type,
        "table_size": raw.table_size,
        "blinds": raw.blinds,
        "street": raw.street,
    }
    action_history = [
        {"player": a.player, "action": a.action, "amount": a.amount,
         "street": a.street, "order": a.order, "confidence": a.confidence}
        for a in (raw.action_history or [])
    ]

    return NormalizedVisionState(
        source=raw.source,
        synthetic_test_only=raw.synthetic_test_only,
        validation=ValidationResult(valid=len(errors) == 0,
                                    errors=errors, warnings=warnings),
        uncertainty=[u if isinstance(u, dict) else to_dict(u)
                     for u in merged_unc],
        game=game, hero=hero, players=norm_players, board=board, pot=pot,
        action_history=action_history,
    )
