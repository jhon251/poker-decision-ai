# -*- coding: utf-8 -*-
"""P30-9-W：面对下注不可确认 ⇒ 不给 bet/check（避免不合法动作）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import action_semantics as asem                     # noqa: E402

# 用户反馈现场之一：翻前有人下注，但我们看不到对手投入、hero 座位已有投入
HERO_INVESTED_NO_OPP = {"street": "preflop", "pot": 0.4, "to_call": None,
                         "hero_seat": 5, "blinds": {"sb": 0.02, "bb": 0.05},
                         "seats": [{"seat": i} for i in range(1, 7)],
                         "extra": {"v3f_bets": {"S5": 0.15}}}


def test_bet_is_suppressed_when_facing_unverifiable():
    fd = {"action": "bet", "recommended_size": 8.0, "reasoning": "底池 8.0bb。"}
    out = asem.align_final_decision(final_decision=fd,
                                    obs=HERO_INVESTED_NO_OPP)
    assert out["action"] is None                      # 不给（可能非法的）动作
    assert out["recommended_size"] is None
    assert out["advice_quality"] == "unverifiable_facing"
    assert out["state_incomplete"]["facing"] == "unverifiable"
    assert "无法确认你是面对下注还是无人下注" in out["reasoning"]


def test_check_is_suppressed_too():
    fd = {"action": "check", "recommended_size": None, "reasoning": "x"}
    out = asem.align_final_decision(final_decision=fd,
                                    obs=HERO_INVESTED_NO_OPP)
    assert out["action"] is None


def test_facing_call_or_raise_is_kept_when_evidence_exists():
    """有对手投入证据 ⇒ 正常对齐（bet→raise / check→call），不抑制。"""
    obs = {"to_call": None, "hero_seat": 5, "street": "preflop",
           "extra": {"v3f_bets": {"S2": 0.05, "S5": 0.02}}}
    out = asem.align_final_decision(
        final_decision={"action": "bet", "recommended_size": 5.0,
                         "reasoning": "x"}, obs=obs)
    assert out["action"] == "raise"                   # 面对下注世界里的同族动作
    assert "unverifiable_facing" != out.get("advice_quality")


def test_hero_not_invested_keeps_action_with_warning():
    """hero 未投入且无对手投入 ⇒ 保留动作但标注"未确认"（不抑制）。

    注意：用**翻后**场景——翻前的 `bet` 会被 P30-9-Y 的街道路词法规则改为 raise。
    """
    obs = {"to_call": None, "hero_seat": 5, "street": "flop",
           "board": ["Kh", "7d", "2c"],
           "extra": {"v3f_bets": {"S5": None}}}
    out = asem.align_final_decision(
        final_decision={"action": "bet", "recommended_size": 4.0,
                         "reasoning": "x"}, obs=obs)
    assert out["action"] == "bet"
    assert "未能确认是否面对下注" in out["reasoning"]
