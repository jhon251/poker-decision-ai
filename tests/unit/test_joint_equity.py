# -*- coding: utf-8 -*-
"""P30-9-BD 测试：范围化联合胜率（数学性质必须成立）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.joint_equity import joint_equity_range_aware        # noqa: E402

WIDE = {"AA": 1.0, "KK": 1.0, "QQ": 1.0, "AKs": 1.0, "AKo": 1.0, "22": 1.0,
        "76s": 1.0, "T9o": 1.0, "A2o": 1.0}
TIGHT = {"AA": 1.0, "KK": 1.0}


def test_range_tightness_monotonicity():
    """对手范围越紧 ⇒ hero 胜率不得更高（同牌面/同人数）。"""
    hero = ["Qs", "Qd"]
    board = ["Kd", "7c", "2h"]
    wide = joint_equity_range_aware(hero, board, [WIDE], samples=700)
    tight = joint_equity_range_aware(hero, board, [TIGHT], samples=700)
    assert wide["ok"] and tight["ok"]
    assert wide["equity"] > tight["equity"], (wide, tight)


def test_multiway_monotonicity_with_fixed_ranges():
    """对手数增加（范围相同）⇒ 胜率必须下降（F1 的核心性质）。"""
    hero = ["Qs", "8d"]
    board = []
    e1 = joint_equity_range_aware(hero, board, [WIDE], samples=700)["equity"]
    e3 = joint_equity_range_aware(hero, board, [WIDE] * 3, samples=700)["equity"]
    e5 = joint_equity_range_aware(hero, board, [WIDE] * 5, samples=700)["equity"]
    assert e1 > e3 > e5, (e1, e3, e5)


def test_basis_marks_random_fallback():
    r = joint_equity_range_aware(["As", "Ah"], [], [None, WIDE], samples=200)
    assert r["ok"] and r["basis"] == "ranges+random"
    assert r["n_random_villains"] == 1
    assert "同时击败所有对手" in r["note"]


def test_no_villains_is_rejected():
    assert joint_equity_range_aware(["As", "Ah"], [], [])["ok"] is False
