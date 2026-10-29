# -*- coding: utf-8 -*-
"""P30-9-AI：多人胜率独立核验 + 残留盲注筹码守卫。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import action_semantics as asem                     # noqa: E402
from poker_decision_ai import multiway_equity as mw                        # noqa: E402
from poker_decision_ai import facing_evidence as fe                        # noqa: E402


def test_multiway_equity_matches_known_values():
    """独立核验必须接近公认值（AA≈0.49 / Q8o≈0.17 对 5 个随机对手）。"""
    aa = mw.multiway_equity(["As", "Ah"], [], 5, samples=1500)["equity"]
    q8 = mw.multiway_equity(["Qs", "8d"], [], 5, samples=1500)["equity"]
    assert 0.42 <= aa <= 0.55, aa
    assert 0.12 <= q8 <= 0.23, q8
    # 单挑对照
    q8_hu = mw.multiway_equity(["Qs", "8d"], [], 1, samples=1500)["equity"]
    assert 0.48 <= q8_hu <= 0.60, q8_hu


def test_engine_mean_of_headsup_is_detected_as_inflated():
    obs = {"hole_cards": ["Qs", "8d"], "board": [],
           "seats": [{"seat": i, "in_hand": True} for i in range(1, 7)]}
    # 引擎实测输出 0.4386（= 逐对手单挑平均）
    out = mw.equity_consistency(obs, 0.4386, samples=1500)
    assert out["n_villains"] == 5
    assert out["inflated"] is True
    assert out["ratio"] and out["ratio"] > 1.5
    assert "同时击败所有对手" in out["note"]


def test_single_villain_not_flagged():
    obs = {"hole_cards": ["Qs", "8d"], "board": [],
           "seats": [{"seat": 1, "in_hand": True}, {"seat": 5, "in_hand": True}]}
    out = mw.equity_consistency(obs, 0.54, samples=800)
    assert out["n_villains"] == 1 and out["inflated"] is False


def test_pipeline_labels_inflated_equity():
    obs = {"street": "preflop", "hole_cards": ["Qs", "8d"], "board": [],
           "pot": 0.4, "to_call": 0.35, "hero_seat": 5, "button_seat": 3,
           "blinds": {"sb": 0.05, "bb": 0.15},
           "seats": [{"seat": i, "in_hand": True} for i in range(1, 7)],
           "extra": {"v3f_hero_position": "BB",
                      "v3f_bets": {"S1": 0.5, "S4": 0.15, "S2": None,
                                    "S3": None, "S5": None, "S6": None}}}
    out = asem.align_final_decision(
        final_decision={"action": "call", "reasoning": "x",
                         "key_evidence": ["hero ['Qs','8d'] vs 5 villain(s) on []: "
                                           "hero_equity=0.4386"]}, obs=obs)
    assert out["advice_risk"]["kind"] == "multiway_equity_inflated"
    mw_info = out["advice_risk"]["multiway_equity"]
    assert mw_info["engine"] == 0.4386 and mw_info["n_villains"] == 5
    assert mw_info["multiway_estimate"] < 0.25
    assert "多人底池胜率被高估" in out["reasoning"]


# ── 残留盲注筹码守卫（"没注可跟却 fold"这一类） ─────────────────────
FLOP_STALE = {"street": "flop", "board": ["Kd", "7c", "7h"], "pot": 0.4,
               "to_call": None, "hero_seat": 5, "button_seat": 3,
               "blinds": {"sb": 0.05, "bb": 0.15},
               "seats": [{"seat": i} for i in range(1, 7)],
               "extra": {"v3f_hero_position": "BB",
                          "v3f_bets": {"S1": 0.15, "S4": 0.15, "S2": None,
                                        "S3": None, "S5": None, "S6": None}}}


def test_stale_blind_bet_detected_postflop():
    st = fe.stale_blind_bet(FLOP_STALE)
    assert st["stale"] is True
    assert st["reason"] == "opponent_bet_equals_big_blind"


def test_stale_bet_is_not_treated_as_facing():
    fv = fe.facing_unverifiable(FLOP_STALE)
    assert fv["unverifiable"] is True
    assert fv["reason"] == "postflop_bet_looks_like_leftover"


def test_preflop_blinds_are_not_stale():
    pre = {**FLOP_STALE, "street": "preflop", "board": []}
    assert fe.stale_blind_bet(pre)["stale"] is False


def test_fold_with_nothing_to_call_becomes_check():
    """没注可跟时 fold 非法 ⇒ 必须变 check（锁定这一类"让我 fold"）。"""
    obs = {"street": "flop", "board": ["Kd", "7c", "7h"], "pot": 0.4,
           "to_call": 0.0, "hero_seat": 5, "button_seat": 3,
           "blinds": {"sb": 0.05, "bb": 0.15},
           "seats": [{"seat": i} for i in range(1, 7)],
           "extra": {"v3f_hero_position": "BB", "v3f_bets": {}}}
    out = asem.align_final_decision(
        final_decision={"action": "fold", "reasoning": "x"}, obs=obs)
    assert out["action"] == "check", out
