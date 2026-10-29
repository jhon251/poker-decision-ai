"""P30-9-AV 测试：自有成手判定 + 跟注诚实提示。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import action_semantics as asem                     # noqa: E402
from poker_decision_ai import hand_strength_label as hsl                   # noqa: E402


def _hc(hole, board):
    return hsl.hero_hand_class({"hole_cards": hole, "board": board})


def test_paired_board_without_hole_contribution_is_playing_board():
    """用户局面：Q♠8♦ 于 K♦7♣7♥ —— 那一对来自公共牌，**不是**你的成手。"""
    hc = _hc(["Qs", "8d"], ["Kd", "7c", "7h"])
    assert hc["hero_contributes"] is False
    assert hc["plays_board"] is True
    assert hc["showdown_value"] is False
    assert hc["board_pair_ranks"] == ["7"]


def test_unpaired_miss_is_high_card_without_showdown():
    hc = _hc(["Qs", "8d"], ["Kd", "7c", "2h"])
    assert hc["category"] == "high_card"
    assert hc["hero_contributes"] is False
    assert hc["showdown_value"] is False


def test_pocket_pair_and_hole_hitting_board_have_showdown_value():
    assert _hc(["Qs", "Qd"], ["Kd", "7c", "7h"])["showdown_value"] is True
    a7 = _hc(["Ah", "7s"], ["Kd", "7c", "7h"])
    assert a7["category"] == "three_of_a_kind"
    assert a7["hero_contributes"] is True


def test_draw_without_made_hand_has_no_showdown_value():
    hc = _hc(["Qs", "Js"], ["Ts", "9s", "2h"])
    assert hc["category"] == "high_card"
    assert hc["showdown_value"] is False
    assert hc["draws"]["flush_draw"] is True      # 听牌事实仍然记录


def _obs(action_facing=True):
    return {"street": "flop", "hole_cards": ["Qs", "8d"],
            "board": ["Kd", "7c", "7h"], "pot": 1.0,
            "to_call": 0.33 if action_facing else 0.0, "hero_seat": 5,
            "button_seat": 3, "blinds": {"sb": 0.05, "bb": 0.15},
            "seats": [{"seat": i, "in_hand": True} for i in range(1, 7)],
            "extra": {"v3f_hero_position": "BB",
                       "v3f_bets": {"S1": 0.33} if action_facing else {}}}


def test_call_without_showdown_value_gets_honest_note():
    out = asem.align_final_decision(
        final_decision={"action": "call", "reasoning": "x"}, obs=_obs())
    assert out["advice_risk"]["kind"] == "call_without_showdown_value"
    assert "没有自有成手" in out["reasoning"]
    assert "依赖对手下注范围足够宽" in out["reasoning"]


def test_call_with_made_hand_no_note():
    obs = _obs()
    obs["hole_cards"] = ["Qs", "Qd"]          # 口袋对 ⇒ 自有成手
    out = asem.align_final_decision(
        final_decision={"action": "call", "reasoning": "x"}, obs=obs)
    kind = (out.get("advice_risk") or {}).get("kind")
    assert kind != "call_without_showdown_value"
    assert "没有自有成手" not in str(out.get("reasoning"))


def test_raise_on_paired_board_still_flagged_as_bluff():
    out = asem.align_final_decision(
        final_decision={"action": "raise", "recommended_size": 1.0,
                         "reasoning": "x"},
        obs={**_obs(), "to_call": 0.33})
    assert out["advice_risk"]["kind"] in ("bluff_no_showdown_value",
                                           "call_without_showdown_value")
