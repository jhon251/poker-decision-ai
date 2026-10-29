# -*- coding: utf-8 -*-
"""P30-9-Y：翻前不存在 bet；翻前 check 仅 BB 且无人加注时合法。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import action_semantics as asem                     # noqa: E402

# 用户实测现场：翻前输出 BET（客户端无法执行）
PF = {"street": "preflop", "board": [], "pot": 0.19, "to_call": None,
      "hero_seat": 5, "button_seat": 3, "blinds": {"sb": 0.02, "bb": 0.05},
      "hole_cards": ["Ah", "Kd"],
      "seats": [{"seat": i} for i in range(1, 7)],
      "extra": {"v3f_bets": {"S1": None, "S2": None, "S6": None, "S5": None}}}


def test_preflop_bet_is_rewritten_to_raise():
    out = asem.street_vocabulary_fix(
        {"action": "bet", "recommended_size": 1.9, "reasoning": "x"}, PF)
    assert out["action"] == "raise"
    assert any("bet→raise" in n for n in out["action_semantics"]["rewrites"])
    assert out["recommended_size"] == 1.9        # 词法校正不动尺寸


def test_preflop_check_becomes_call_when_raise_pending():
    obs = {**PF, "extra": {"v3f_bets": {"S2": 0.15, "S5": None}}}
    out = asem.street_vocabulary_fix({"action": "check", "reasoning": "x"}, obs)
    assert out["action"] == "call"


def test_preflop_check_kept_for_bb_without_raise():
    # BB 座 = button+2；令 button=3、hero=5 ⇒ BB=5 ⇒ hero 是 BB，且无人加注
    obs = {**PF, "button_seat": 3, "hero_seat": 5,
           "to_call": 0.0, "extra": {"v3f_bets": {"S1": 0.02, "S2": 0.05}}}
    out = asem.street_vocabulary_fix({"action": "check", "reasoning": "x"}, obs)
    assert out["action"] == "check"


def test_preflop_check_not_kept_when_hero_is_not_bb():
    obs = {**PF, "hero_seat": 4, "button_seat": 3, "to_call": 0.0,
           "extra": {"v3f_bets": {"S1": 0.02, "S2": 0.05}}}
    out = asem.street_vocabulary_fix({"action": "check", "reasoning": "x"}, obs)
    assert out["action"] == "call"


def test_postflop_untouched():
    obs = {"street": "flop", "board": ["Kh", "7d", "2c"], "to_call": 0.0}
    for act in ("bet", "check", "raise", "call", "fold"):
        out = asem.street_vocabulary_fix({"action": act, "reasoning": "x"}, obs)
        assert out["action"] == act, act


def test_full_pipeline_turns_preflop_bet_into_raise():
    """主流程（含合法集对齐）也要把翻前 bet 变成 raise。"""
    out = asem.align_final_decision(
        final_decision={"action": "bet", "recommended_size": 1.9,
                         "reasoning": "底池 3.8bb。"}, obs=PF)
    assert out["action"] == "raise"
    assert "bet→raise" in str(out["action_semantics"].get("rewrites"))
