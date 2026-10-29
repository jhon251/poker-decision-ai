# -*- coding: utf-8 -*-
"""P30-9-T：to_call 未知 ⇒ 用对手投入下界（消除"所需胜率=0 ⇒ 总是激进"）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import facing_evidence as fe                        # noqa: E402

# 用户 22:50 真实帧的识别结果（识别链实测输出）
USER_OBS = {"street": "preflop", "pot": 0.24, "to_call": None,
             "hero_seat": 5, "button_seat": 6, "blinds": None,
             "hole_cards": ["Qh", "6h"],
             "seats": [{"seat": i} for i in range(1, 7)],
             "extra": {"v3f_bets": {"S1": 0.02, "S2": 0.05, "S6": None,
                                     "S5": None},
                        "v3f_hero_position": "CO"}}


def test_lower_bound_from_user_frame():
    lb = fe.to_call_lower_bound(USER_OBS)
    assert lb["to_call"] == 0.05           # 对手最高投入 0.05 − hero 0
    assert lb["is_lower_bound"] is True
    assert lb["source"] == "lower_bound_from_seat_bets"
    assert lb["detail"]["max_opponent_bet"] == 0.05


def test_lower_bound_subtracts_hero_bet():
    """1876 几何：hero 的投注区是 "S4"（底部中间）。"""
    obs = {"hero_seat": 5, "street": "flop", "pot": 0.4,
           "extra": {"v3f_bets": {"S1": None, "S2": 0.12, "S3": None,
                                   "S4": 0.05, "S5": None, "S6": None}}}
    lb = fe.to_call_lower_bound(obs)
    assert lb["to_call"] == 0.07           # 0.12 − 0.05
    assert lb["detail"]["hero_bet"] == 0.05


def test_lower_bound_none_when_hero_matched():
    obs = {"hero_seat": 5, "street": "flop", "pot": 0.4,
           "extra": {"v3f_bets": {"S1": None, "S2": 0.05, "S3": None,
                                   "S4": 0.05, "S5": None, "S6": None}}}
    lb = fe.to_call_lower_bound(obs)
    assert lb["to_call"] is None
    assert lb["source"] == "hero_already_matched"


def test_lower_bound_none_without_evidence():
    assert fe.to_call_lower_bound({})["to_call"] is None
    assert fe.to_call_lower_bound({"extra": {"v3f_bets": {}}})["to_call"] is None
    assert fe.to_call_lower_bound(
        {"hero_seat": 5, "extra": {"v3f_bets": {"S1": None}}})["to_call"] is None


def test_zero_odds_bias_is_gone_in_math():
    """对照：to_call=0 ⇒ 所需胜率 0；用下界 0.05 ⇒ 所需胜率 ~0.17（不再是白送）。"""
    pot, tc = 0.24, 0.05
    req = tc / (pot + tc)
    assert 0.15 < req < 0.20
    assert 0.0 == 0.0 / (pot + 0.0)        # 旧口径 = 0（任何牌都过线）
