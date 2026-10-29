"""V3F-P56 — Strategy Core v3.1 SACA（street-aware continuation adjustments）测试。

覆盖数学设计声明的不变量：
  I1  river/preflop 无 CV（逐位保持 P41-P45 语义）；
  I2  fold EV 恒 0；
  I5  CASH utility==ev 位等（含 CV）；
  I10 位置/手牌缺失 → CV 中性（缺信息不猜测）；
  A8  realization 方向（IP>1>OOP）；
  A9  implied odds（听牌确定性数牌）；
  A10 initiative（OOP=1.0 不惩罚下注）；
  competitive_actions 边际语义；MTT/PKO objective 保留；
  determinism；UNKNOWN 传播；monotonicity。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai.domain.poker.strategy_core import (
    FUTURE_POT_FRACTION,
    KAPPA_IMPLIED,
    MIXED_MARGIN_FRAC_POT,
    REALIZATION,
    STREETS_LEFT,
    compute_ev,
    _draw_improve_prob,
)


def _state(street="flop", **over):
    """6-max 完整状态：hero BTN (seat2), villain BB (seat3), SB (seat1)。"""
    st = {"game_type": "cash", "street": street,
          "board": ["Ah", "Kd", "7c"] if street != "preflop" else [],
          "pot": 10.0, "to_call": 0.0, "bb": 1.0, "sb": 0.5,
          "hero_id": "hero", "button_seat": 2,
          "players": [
              {"player_id": "v1", "seat": 1, "position": "SB",
               "status": "active", "hole_cards": [], "stack": 100},
              {"player_id": "hero", "seat": 2, "position": "BTN",
               "is_hero": True, "status": "active",
               "hole_cards": ["9c", "8c"], "stack": 100},
              {"player_id": "v2", "seat": 3, "position": "BB",
               "status": "folded", "hole_cards": [], "stack": 100}],
          "action_history": [{"street": "preflop", "player_id": "hero",
                              "action": "raise", "amount": 2.5}],
          "available_actions": ["fold", "check", "bet"]}
    st.update(over)
    return st


def _eq(v=0.45):
    return {"hero_equity": v,
            "outcome": {"win": v, "tie": 0.0, "lose": round(1 - v, 6)}}


# ------------------------------------------------------------ I1 river/preflop
def test_i1_river_bit_frozen():
    """river：R=1、implied=0、F=0 → cv 全 None + 公式逐位闭式。"""
    st = _state(street="river", board=["Ah", "Kd", "7c", "2s", "9d"],
                to_call=3.0, available_actions=["fold", "call", "raise"])
    r = compute_ev(st, _eq(0.5))
    assert r.cv_model == "none"
    assert all(c.cv_bb is None for c in r.candidates)
    call = next(c for c in r.candidates if c.action == "call")
    # pot=10bb, to_call=3bb, eq=0.5 → 5 − 1.5 = 3.5（P41 闭式逐位）
    assert call.ev_bb == pytest.approx(3.5, abs=1e-9)


def test_i1_preflop_no_cv():
    st = _state(street="preflop", board=[], to_call=0.0)
    r = compute_ev(st, _eq(0.5))
    assert r.cv_model == "none"
    assert all(c.cv_bb is None for c in r.candidates)


# ------------------------------------------------------------ I10 缺信息中性
def test_i10_missing_position_neutral():
    """无位置标签/无 villain → CV 中性，frozen 公式逐位（P41 fixture 形态）。"""
    st = {"game_type": "cash", "street": "flop", "board": ["Ah", "Kd", "7c"],
          "pot": 10.0, "to_call": 0.0, "bb": 1.0, "hero_id": "hero",
          "players": [{"player_id": "hero", "is_hero": True, "stack": 100,
                       "hole_cards": ["9c", "8c"]}]}
    r = compute_ev(st, _eq(0.6))
    assert r.cv_model == "none"
    check = next(c for c in r.candidates if c.action == "check")
    assert check.ev_bb == pytest.approx(6.0, abs=1e-9)   # 0.6×10 逐位
    assert check.cv_bb is None


def test_i10_missing_hole_cards_no_implied():
    """hole cards 缺失 → implied=0（realization 仍可激活——位置已知）。"""
    st = _state()
    st["players"][1]["hole_cards"] = []
    r = compute_ev(st, _eq(0.45))
    check = next(c for c in r.candidates if c.action == "check")
    F = FUTURE_POT_FRACTION * 10.0 * STREETS_LEFT["flop"]
    # R(BTN=IP, flop)=1.10 → cv=(0.1)×0.45×F
    assert check.cv_bb == pytest.approx(0.10 * 0.45 * F, abs=1e-9)


# ------------------------------------------------------------ A8/A9/A10 方向
def test_a8_realization_direction():
    """IP check 获得正 CV；OOP check 获得负 CV（同手同街）。"""
    eq = _eq(0.5)
    ip = compute_ev(_state(), eq)
    oop = compute_ev(_state(players=[
        {"player_id": "hero", "seat": 1, "position": "SB", "is_hero": True,
         "status": "active", "hole_cards": ["9c", "8c"], "stack": 100},
        {"player_id": "v1", "seat": 2, "position": "BTN", "status": "folded",
         "hole_cards": [], "stack": 100},
        {"player_id": "v2", "seat": 3, "position": "BB", "status": "active",
         "hole_cards": [], "stack": 100}]), eq)
    check_ip = next(c for c in ip.candidates if c.action == "check")
    check_oop = next(c for c in oop.candidates if c.action == "check")
    assert check_ip.cv_bb > 0 > check_oop.cv_bb
    # 公式闭式：(R−1)×eq×F；SB 相对 BB villain → OOP
    F = FUTURE_POT_FRACTION * 10.0 * 2
    assert check_ip.cv_bb == pytest.approx((1.10 - 1) * 0.5 * F, abs=1e-9)
    assert check_oop.cv_bb == pytest.approx((0.90 - 1) * 0.5 * F, abs=1e-9)


def test_a9_implied_odds_flush_draw():
    """同花听牌（4 同花色）→ 9 outs → p_improve ≈ 0.345；call/check 获得
    κ×p×F 的隐含赔率分量。"""
    st = _state(board=["Ah", "7c", "2c"])
    st["players"][1]["hole_cards"] = ["9c", "4c"]      # +7c,2c = 4 clubs
    p, desc = _draw_improve_prob(["9c", "4c"], ["Ah", "7c", "2c"])
    assert p == pytest.approx(1 - (38 * 37) / (47 * 46), abs=1e-9)
    assert desc == "flush_draw"
    r = compute_ev(st, _eq(0.35))
    check = next(c for c in r.candidates if c.action == "check")
    F = FUTURE_POT_FRACTION * 10.0 * 2
    expected_cv = (1.10 - 1) * 0.35 * F + KAPPA_IMPLIED * p * F
    assert check.cv_bb == pytest.approx(expected_cv, abs=1e-9)
    # call 生存性：面对下注，听牌 call 的 CV 使其有生存空间
    st2 = _state(board=["Ah", "7c", "2c"], to_call=3.0,
                 available_actions=["fold", "call", "raise"])
    st2["players"][1]["hole_cards"] = ["9c", "4c"]
    r2 = compute_ev(st2, _eq(0.30))
    call = next(c for c in r2.candidates if c.action == "call")
    assert call.cv_bb is not None and call.cv_bb > 0
    cv_call_expected = ((1.10 - 1) * 0.30 * F
                        + KAPPA_IMPLIED * p * F)
    assert call.cv_bb == pytest.approx(cv_call_expected, abs=1e-9)


def test_a9_straight_draw_oesd():
    """OESD：8 outs（确定性数牌）。"""
    # hole Tc Jc, board Qd Kd 2h → 9 或 A 补成 9/A+T-J-Q-K straight → 8 outs
    p, desc = _draw_improve_prob(["Tc", "Jc"], ["Qd", "Kd", "2h"])
    assert p == pytest.approx(1 - (39 * 38) / (47 * 46), abs=1e-9)
    assert "straight_draw(8)" in desc


def test_a10_initiative_oop_neutral():
    """OOP bet 的 initiative CV = 1.0 → 不惩罚下注；IP bet 小幅加成。"""
    st_bet = _state(to_call=0.0)
    st_bet["players"][1]["hole_cards"] = ["Ah", "Kd"]  # 无听牌
    r_ip = compute_ev(st_bet, _eq(0.6))
    bet_ip = next(c for c in r_ip.candidates
                  if c.action == "bet" and c.amount_bb == 5.0)
    F = FUTURE_POT_FRACTION * 10.0 * 2
    assert bet_ip.cv_bb == pytest.approx((1.06 - 1) * 0.6 * F, abs=1e-9)
    st_oop = _state(players=[
        {"player_id": "hero", "seat": 1, "position": "SB", "is_hero": True,
         "status": "active", "hole_cards": ["Ah", "Kd"], "stack": 100},
        {"player_id": "v1", "seat": 2, "position": "BTN", "status": "folded",
         "hole_cards": [], "stack": 100},
        {"player_id": "v2", "seat": 3, "position": "BB", "status": "active",
         "hole_cards": [], "stack": 100}])
    r_oop = compute_ev(st_oop, _eq(0.6))
    bet_oop = next(c for c in r_oop.candidates
                   if c.action == "bet" and c.amount_bb == 5.0)
    assert not bet_oop.cv_bb                          # OOP INIT=1.0 → 零 CV


# ------------------------------------------------------------ I5 / objective
def test_i5_cash_utility_bit_equal_with_cv():
    st = _state()
    st["players"][1]["hole_cards"] = ["9c", "4c"]
    r = compute_ev(st, _eq(0.35))
    for c in r.candidates:
        assert c.utility_bb == pytest.approx(c.ev_bb, abs=1e-12)


def test_p44_mtt_objective_scales_cv():
    """MTT：k± 缩放作用于含 CV 的 gain/loss——P44 框架保留。"""
    st = _state()
    st["game_type"] = "mtt"
    st["tournament"] = {"payouts": [50, 30, 20]}
    st["players"][1]["hole_cards"] = ["9c", "4c"]
    r = compute_ev(st, _eq(0.35))
    call = next((c for c in r.candidates if c.action == "call"), None)
    bet = next((c for c in r.candidates if c.action == "bet"
                and c.amount_bb == 5.0), None)
    check = next(c for c in r.candidates if c.action == "check")
    # CV 仍在（objective 框架不因 CV 改变）且 utility 排序键生效
    assert r.objective == "chip_ev+icm"
    assert check.cv_bb is not None
    if call is not None:
        assert call.utility_bb == pytest.approx(call.ev_bb, abs=1e-9) or \
            call.utility_bb < call.ev_bb


def test_competitive_actions_margin():
    """mixed strategy 表达：utility 距最优 ≤ 0.10×pot 的动作 ∈ 集合。"""
    st = _state()
    r = compute_ev(st, _eq(0.45))
    margin = MIXED_MARGIN_FRAC_POT * 10.0
    best_u = r.best_utility_bb
    expected = sorted({c.action for c in r.candidates
                       if (c.utility_bb if c.utility_bb is not None
                           else c.ev_bb) >= best_u - margin})
    assert r.competitive_actions == expected
    assert r.best_action in r.competitive_actions


# ------------------------------------------------------------ 不变量回归
def test_deterministic_and_unknown_propagation():
    st = _state()
    a = compute_ev(st, _eq(0.45))
    b = compute_ev(st, _eq(0.45))
    assert a.to_summary() == b.to_summary()
    # equity 缺失 → available=False（UNKNOWN 语义不变）
    r = compute_ev(_state(), {})
    assert r.available is False and r.unavailable_reason


def test_monotonicity_preserved_with_cv():
    """同街内大注 EV 单调性保持（eq>0.5 → 大注更优；eq<0.5 → 更差）。"""
    st = _state()
    st["players"][1]["hole_cards"] = ["Ah", "Kd"]
    r = compute_ev(st, _eq(0.6))
    bets = [c.ev_bb for c in r.candidates if c.action == "bet"]
    check = next(c.ev_bb for c in r.candidates if c.action == "check")
    assert max(bets) > check
    r2 = compute_ev(st, _eq(0.45))
    bets2 = [c.ev_bb for c in r2.candidates if c.action == "bet"]
    check2 = next(c.ev_bb for c in r2.candidates if c.action == "check")
    assert all(b < check2 for b in bets2 if b is not None)
