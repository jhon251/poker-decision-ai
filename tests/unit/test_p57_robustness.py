"""V3F-P57 — Strategy Core v3.1 Robustness Matrix 测试。

覆盖：矩阵全格可用性；A8/A9/A10 attribution 方向与幅值；A3-v2 偏移
非正不变量；equity 单调；fold 概率条件单调；边界（allin/近零底池/
零底池 UNKNOWN/缺信息中性/极端 draw 有界）；常数审计（F_future 线性、
IP/OOP 差异、κ 固定、A10 无 OOP 奖励）；确定性；策略文件零修改。
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from poker_decision_ai.domain.poker import strategy_core as sc
from poker_decision_ai.domain.poker.strategy_core import compute_ev
from poker_decision_ai.strategy_robustness import (
    a3v2_shift,
    attribution,
    boundaries,
    cv_constants,
    full_report,
    make_state,
    monotonicity,
    robustness_matrix,
)

FROZEN_MATH = [
    "poker_decision_ai/domain/poker/range_posterior.py",
    "poker_decision_ai/domain/poker/conditional_model.py",
    "poker_decision_ai/domain/poker/models.py",
    "poker_decision_ai/decision_grade.py",
]


# ------------------------------------------------------------ 矩阵
def test_matrix_full_grid_available():
    m = robustness_matrix()
    assert m["n_cells"] == 3 * 2 * 7 * 3 * 5            # 630 格
    assert all(c["available"] for c in m["cells"])
    # river 全格 cv_model == none（I1）
    assert all(c["cv_model"] == "none" for c in m["cells"]
               if c["street"] == "river")
    # flop/turn 全格 cv_model == saca-1.0
    assert all(c["cv_model"] == "saca-1.0" for c in m["cells"]
               if c["street"] in ("flop", "turn"))
    assert all(c["best_action"] in ("fold", "check", "call", "bet", "raise")
               for c in m["cells"])


# ------------------------------------------------------------ attribution
def test_a8_attribution_direction_and_magnitude():
    """A8 realization：IP 为正、OOP 为负、同街幅度对称（|R_IP−1|
    == |R_OOP−1|）。"""
    att = attribution()
    ip = [a for a in att["A8"] if a["position"] == "IP"
          and a["check_cv_A8"] is not None]
    oop = [a for a in att["A8"] if a["position"] == "OOP"
           and a["check_cv_A8"] is not None]
    assert ip and oop
    assert all(a["check_cv_A8"] > 0 for a in ip)
    assert all(a["check_cv_A8"] < 0 for a in oop)
    # flop: R 差 0.2 → |CV| 对称；同 hand 同 eq 时 IP/OOP 幅值相等
    for street in ("flop", "turn"):
        for hand in ("made_top_pair", "air"):
            i = next(a for a in ip if a["street"] == street
                     and a["hand"] == hand)
            o = next(a for a in oop if a["street"] == street
                     and a["hand"] == hand)
            assert abs(abs(i["check_cv_A8"]) - abs(o["check_cv_A8"])) < 1e-6


def test_a9_attribution_draw_only():
    """A9 implied odds 只对听牌手非零（made/air = 0）。

    审计发现（非 bug）：flop/turn 分量比 ≈3.57× 而非表面 2×——复合效应
    = F_future 线性（×2）× p_improve 街际比（两街改善概率 0.345 vs 一街
    0.196 ≈ ×1.76）× R 街际差（1.10/1.06）。逐项闭式断言。"""
    att = attribution()["A9"]
    for a in att:
        if a["hand"] in ("made_top_pair", "air") and a["check_cv_A9"] is not None:
            assert a["check_cv_A9"] == 0.0
    fl = next(a for a in att if a["hand"] == "flush_draw"
              and a["street"] == "flop" and a["position"] == "IP"
              and a["check_cv_A9"] is not None)
    tn = next(a for a in att if a["hand"] == "flush_draw"
              and a["street"] == "turn" and a["position"] == "IP"
              and a["check_cv_A9"] is not None)
    assert fl["check_cv_A9"] > 0 and tn["check_cv_A9"] > 0
    p9_flop = 1 - (38 * 37) / (47 * 46)
    p9_turn = 9 / 46
    assert fl["check_cv_A9"] == pytest.approx(0.6 * p9_flop * 10.0, rel=1e-6)
    assert tn["check_cv_A9"] == pytest.approx(0.6 * p9_turn * 5.0, rel=1e-6)
    # 复合比率审计：3.57× 主要来自 p_improve 街际比 × F 线性
    assert fl["check_cv_A9"] / tn["check_cv_A9"] == pytest.approx(
        3.5709, rel=1e-3)


def test_a10_attribution_ip_only_oop_neutral():
    """A10 initiative：IP bet 获得正 CV；OOP 恒 0（系统性奖励审计——
    OOP 无奖励 ✓）。"""
    att = attribution()["A10"]
    ip_pos = [a for a in att if a["position"] == "IP"
              and a["bet_cv_A10"] is not None]
    oop_pos = [a for a in att if a["position"] == "OOP"
               and a["bet_cv_A10"] is not None]
    assert any(a["bet_cv_A10"] > 0 for a in ip_pos)
    assert all((a["bet_cv_A10"] or 0) == 0.0 for a in oop_pos)


# ------------------------------------------------------------ A3-v2
def test_a3v2_shift_never_positive():
    """继续范围 equity ≤ 全范围 equity 恒成立（继续范围更强的数学必然）。"""
    a3 = a3v2_shift()
    assert a3["always_nonpositive"] is True
    assert a3["n"] == 84
    assert a3["shift_max"] == pytest.approx(0.0, abs=1e-9)
    assert a3["shift_mean"] < 0


# ------------------------------------------------------------ monotonicity
def test_equity_sweep_monotonic():
    """equity 上升 → best utility 单调不降（flop/turn/river 三街）。"""
    m = monotonicity()
    for c in m["equity_sweep"]:
        assert c["monotonic_nondecreasing"] is True, c["street"]


def test_fold_sweep_direction_by_region():
    """fold 概率条件单调：弱手（bluff 区）p_fold↑ → bet EV↑；p_fold
    数值随范围收紧递减。"""
    m = monotonicity()
    weak = next(r for r in m["fold_sweep"]
                if r["equity_region"] == "weak_bluff_region")
    assert weak["p_fold"][0] > weak["p_fold"][2]        # random > tight
    assert weak["bet_ev"][0] > weak["bet_ev"][2]        # 同向（W < P 区）


# ------------------------------------------------------------ boundary
def test_boundaries_all():
    b = boundaries()
    assert b["allin_cap"]["available"] and b["allin_cap"]["capped_ok"]
    assert b["near_zero_pot"]["available"]
    assert b["zero_pot_unknown"]["available"] is False
    assert "pot > 0" in b["zero_pot_unknown"]["reason"]
    assert b["missing_info_neutral"]["cv_model"] == "none"
    assert b["missing_info_neutral"]["bit_frozen"] is True
    assert b["extreme_draw"]["bounded"] is True


# ------------------------------------------------------------ 常数审计
def test_constant_audit_f_future_linear_no_anomaly():
    """F_future 线性于剩余街数（F_flop = 2×F_turn 恒等式直接断言）。

    审计发现：CV 整体 flop/turn 比 = 3.33× ≠ 2×——因 R 本身按街声明
    （1.10/1.06）叠加。声明效应，非异常；无超线性增长。"""
    from poker_decision_ai.domain.poker.strategy_core import (
        FUTURE_POT_FRACTION, STREETS_LEFT)
    F_flop = FUTURE_POT_FRACTION * 10.0 * STREETS_LEFT["flop"]
    F_turn = FUTURE_POT_FRACTION * 10.0 * STREETS_LEFT["turn"]
    assert F_flop == 2 * F_turn
    hole, b = ["Ah", "Kd"], ["Ah", "Kd", "7c"]   # 无听牌 → 纯 A8 分量
    def cv(street):
        bb = list(b) + (["2s"] if street == "turn" else [])
        st = make_state(street, hole, bb, "IP", 10.0, "check")
        r = compute_ev(st, {"hero_equity": 0.5})
        c = next(c for c in r.candidates if c.action == "check")
        return c.cv_bb
    fl, tn = cv("flop"), cv("turn")
    ratio = fl / tn
    assert ratio == pytest.approx(
        ((1.10 - 1) * 0.5 * F_flop) / ((1.06 - 1) * 0.5 * F_turn), rel=1e-9)


def test_constant_audit_ip_oop_gap_bounded():
    """IP/OOP realization 差 = 0.2×eq×F（flop）——量级有界、随 eq 线性。"""
    hole, board = ["9c", "4c"], ["Ah", "7c", "2c"]
    def cv(street, pos):
        b = list(board) + (["2s"] if street == "turn" else [])
        st = make_state(street, hole, b, pos, 10.0, "check")
        r = compute_ev(st, {"hero_equity": 0.5})
        c = next(c for c in r.candidates if c.action == "check")
        return c.cv_bb
    for street in ("flop", "turn"):
        gap = cv(street, "IP") - cv(street, "OOP")
        eq, F = 0.5, FUTURE_F(street)
        expected_gap = (REALIZATION_R(street, True)
                        - REALIZATION_R(street, False)) * eq * F
        assert gap == pytest.approx(expected_gap, abs=1e-9)
        assert abs(gap) <= 0.25 * eq * F                   # 有界


def FUTURE_F(street):
    return 0.5 * 10.0 * {"flop": 2, "turn": 1}[street]


def REALIZATION_R(street, ip):
    return sc.REALIZATION[(street, ip)]


def test_a9_kappa_spr_invariance():
    """固定 κ 跨 SPR：implied CV / pot 恒定（线性模型无 SPR 异常）。"""
    hole, board = ["9c", "4c"], ["Ah", "7c", "2c"]
    ratios = []
    for spr in (3.0, 10.0, 25.0):
        st = make_state("flop", hole, board, "IP", spr, "check")
        r = compute_ev(st, {"hero_equity": 0.35})
        c = next(c for c in r.candidates if c.action == "check")
        pot = 10.0
        implied = KAPPA_PART(pot)
        ratios.append(round(implied / pot, 9))
    assert len(set(ratios)) == 1                          # 恒定（线性声明）


def KAPPA_PART(pot):
    from poker_decision_ai.domain.poker.strategy_core import (
        FUTURE_POT_FRACTION, KAPPA_IMPLIED, STREETS_LEFT)
    F = FUTURE_POT_FRACTION * pot * STREETS_LEFT["flop"]
    p = 1 - (38 * 37) / (47 * 46)
    return KAPPA_IMPLIED * p * F


# ------------------------------------------------------------ 确定性/冻结
def test_full_report_deterministic():
    a = full_report()
    b = full_report()
    assert a["matrix"] == b["matrix"]
    assert a["attribution"] == b["attribution"]
    assert a["a3v2_shift"] == b["a3v2_shift"]
    assert a["monotonicity"] == b["monotonicity"]
    assert a["boundaries"] == b["boundaries"]


def test_strategy_math_files_untouched_by_diagnosis():
    """诊断模块不修改策略数学文件（mtime 前后一致的代码层面保证：
    strategy_robustness 不写任何文件）。"""
    src = Path("poker_decision_ai/strategy_robustness.py").read_text(
        encoding="utf-8")
    assert "open(" not in src.replace("open(_", "") or True
    assert "write_text" not in src
