# -*- coding: utf-8 -*-
"""P30-9-V：无法确认是否面对下注 ⇒ 撤尺寸 + 告警（实测第二张真实帧）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import action_semantics as asem                     # noqa: E402
from poker_decision_ai import facing_evidence as fe                        # noqa: E402

# 第二张真实帧的识别结果：hero 已加注（桌上有 0.15 属于 hero），对手 0.25 漏读
USER_OBS2 = {"street": "preflop", "pot": 0.4, "to_call": None, "hero_seat": 5,
              "button_seat": 3, "hole_cards": ["Kh", "Jh"], "blinds": None,
              "seats": [{"seat": i} for i in range(1, 7)],
              # P30-9-AB：1876 bet 环按几何归属，hero 的投注区键为 "S4"（底部中间）
              "extra": {"v3f_bets": {"S1": None, "S2": None, "S3": None,
                                      "S4": 0.15, "S5": None, "S6": None},
                         "v3f_hero_position": "BB"}}


def test_facing_unverifiable_detected_on_user_frame():
    fv = fe.facing_unverifiable(USER_OBS2)
    assert fv["unverifiable"] is True
    assert fv["reason"] == "hero_bet_present_no_opponent_bet_visible"
    assert fv["detail"]["hero_bet"] == 0.15
    assert fv["detail"]["max_opponent_bet"] == 0.0


def test_not_unverifiable_when_opponent_bet_visible():
    obs = {"to_call": None, "hero_seat": 5,
           "extra": {"v3f_bets": {"S1": None, "S2": 0.25, "S3": None,
                                   "S4": 0.15, "S5": None, "S6": None}}}
    assert fe.facing_unverifiable(obs)["unverifiable"] is False


def test_not_unverifiable_when_to_call_known():
    obs = {"to_call": 0.07, "hero_seat": 5,
           "extra": {"v3f_bets": {"S1": None, "S2": None, "S3": None,
                                   "S4": 0.15, "S5": None, "S6": None}}}
    assert fe.facing_unverifiable(obs)["unverifiable"] is False


def test_no_bet_visible_is_unverifiable_but_soft():
    """完全看不到投入 ⇒ 仍属不可确认，但原因是"无投入可见"（温和，不抑制动作）。"""
    obs = {"to_call": None, "hero_seat": 5,
           "extra": {"v3f_bets": {"S5": None}}}
    fv = fe.facing_unverifiable(obs)
    assert fv["unverifiable"] is True
    assert fv["reason"] == "no_bet_visible_to_call_unknown"


def test_size_withheld_and_action_suppressed():
    """P30-9-W 起：hero 已有投入且看不到对手投入 ⇒ 尺寸撤下**且不给动作**。"""
    fd = {"action": "bet", "recommended_size": 8.0, "reasoning": "底池 8.0bb。"}
    out = asem.align_final_decision(final_decision=fd, obs=USER_OBS2)
    assert out["recommended_size"] is None
    assert out["action"] is None
    assert out["advice_quality"] == "unverifiable_facing"
    assert "无法确认你是面对下注还是无人下注" in out["reasoning"]
    assert out["state_incomplete"]["facing"] == "unverifiable"
    assert out.get("rule_size_suggestion") in (None, {})


def test_known_facing_keeps_behaviour():
    obs = {"to_call": 0.07, "hero_seat": 5, "street": "preflop",
           "blinds": {"sb": 0.02, "bb": 0.05},
           "extra": {"v3f_bets": {"S2": 0.12, "S5": 0.05}}}
    fd = {"action": "call", "recommended_size": None, "reasoning": "x"}
    out = asem.align_final_decision(final_decision=fd, obs=obs)
    assert "state_incomplete" not in out
    assert out["action"] == "call"
