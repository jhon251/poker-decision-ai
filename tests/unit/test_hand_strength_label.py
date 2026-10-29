# -*- coding: utf-8 -*-
"""P30-9-Z：成手类别标注 + 诈唬风险提示（只标注，不改策略）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import action_semantics as asem                     # noqa: E402
from poker_decision_ai import hand_strength_label as hsl                   # noqa: E402


def _obs(hole, board):
    return {"street": "turn", "hole_cards": hole, "board": board,
            "pot": 0.84, "to_call": 0.0, "hero_seat": 5, "button_seat": 3,
            "blinds": {"sb": 0.02, "bb": 0.05},
            "seats": [{"seat": i} for i in range(1, 7)],
            "extra": {"v3f_hero_position": "CO"}}


TURN_AIR = _obs(["As", "Qd"], ["Kd", "7c", "2h", "5s"])          # 高牌（用户场景形状）
TURN_PAIR = _obs(["As", "Kd"], ["Kd", "7c", "2h", "5s"])         # 一对
TURN_FLUSH_DRAW = _obs(["As", "Qs"], ["Ks", "7s", "2h", "5s"])   # 同花（已成）
TURN_MADE_FLUSH = _obs(["As", "9s"], ["Ks", "Qs", "7s", "2s"])   # 同花


def test_hand_class_high_card_and_pair():
    assert hsl.hero_hand_class(TURN_AIR)["category"] == "high_card"
    assert hsl.hero_hand_class(TURN_AIR)["zh"] == "高牌"
    assert hsl.hero_hand_class(TURN_AIR)["showdown_value"] is False
    assert hsl.hero_hand_class(TURN_PAIR)["category"] == "one_pair"
    assert hsl.hero_hand_class(TURN_PAIR)["showdown_value"] is True


def test_hand_class_flush_is_a_flush_not_high_card():
    """用户强调"我当时是高牌，不是同花"——八张同花必须判为同花。"""
    assert hsl.hero_hand_class(TURN_MADE_FLUSH)["category"] == "flush"
    assert hsl.hero_hand_class(TURN_FLUSH_DRAW)["category"] == "flush"


def test_hand_class_preflop_unknown():
    hc = hsl.hero_hand_class({"hole_cards": ["As", "Kd"], "board": []})
    assert hc["category"] is None and hc["reason"] == "insufficient_cards"


def test_bluff_risk_flagged_on_high_card_raise():
    out = hsl.bluff_risk_note(
        {"action": "raise", "recommended_size": 24.0, "reasoning": "底池 16.8bb。"},
        TURN_AIR)
    assert out["advice_risk"]["kind"] == "bluff_no_showdown_value"
    assert out["advice_risk"]["hand_class_zh"] == "高牌"
    assert "诈唬" in out["reasoning"]
    assert "没有建模对手弃牌" in out["reasoning"]
    assert out["recommended_size"] == 24.0        # 标注不改尺寸
    assert out["action_semantics"]["hand_class"]["category"] == "high_card"


def test_no_bluff_flag_with_pair_or_non_aggressive():
    assert "advice_risk" not in hsl.bluff_risk_note(
        {"action": "raise", "reasoning": "x"}, TURN_PAIR)
    assert "advice_risk" not in hsl.bluff_risk_note(
        {"action": "check", "reasoning": "x"}, TURN_AIR)


def test_pipeline_attaches_label_end_to_end():
    obs = dict(TURN_AIR)
    obs["to_call"] = 0.2
    obs["extra"] = {"v3f_hero_position": "CO", "v3f_bets": {"S2": 0.2}}
    out = asem.align_final_decision(
        final_decision={"action": "raise", "recommended_size": 1.2,
                         "reasoning": "底池 16.8bb。"}, obs=obs)
    assert out["action"] == "raise"
    assert out["advice_risk"]["hand_class_zh"] == "高牌"
    assert "无摊牌价值" in out["reasoning"]
