# -*- coding: utf-8 -*-
"""P30-9-E 证据推导测试：不知道前面有人下注（用逐座位投入证据修复）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import action_semantics as asem                    # noqa: E402
from poker_decision_ai import facing_evidence as fe                       # noqa: E402


# ── 证据优先级 ───────────────────────────────────────────────────────
def test_explicit_available_actions_wins():
    d = fe.derived_facing({"available_actions": ["fold", "call", "raise"],
                           "to_call": None})
    assert d["source"] == "available_actions" and d["facing"] == "BET_PENDING"


def test_to_call_known_decides_both_ways():
    assert fe.derived_facing({"to_call": 2000.0})["facing"] == "BET_PENDING"
    assert fe.derived_facing({"to_call": 0.0})["facing"] == "NO_BET"


def test_v3f_bets_evidence_facing_bet():
    """真实 1876 帧的证据形态：hero_seat=5，S5 未投入，对手 S1/S2/S6 有投入。"""
    obs = {"to_call": None, "hero_seat": 5,
           "extra": {"v3f_bets": {"S1": 1000, "S2": 2000, "S6": 1920, "S5": None}}}
    d = fe.derived_facing(obs)
    assert d["source"] == "v3f_bets" and d["facing"] == "BET_PENDING"
    assert d["legal"] == ["fold", "call", "raise"]
    assert d["detail"]["max_opponent_bet"] == 2000.0


def test_v3f_bets_evidence_no_bet():
    obs = {"to_call": None, "hero_seat": 5,
           "extra": {"v3f_bets": {"S1": 0, "S2": 0, "S5": 0}}}
    d = fe.derived_facing(obs)
    assert d["facing"] == "NO_BET" and d["legal"] == ["check", "bet"]


def test_hero_seat_unknown_is_not_guessed():
    """hero 座未知 ⇒ 无法归属投入 ⇒ UNKNOWN（不猜；不发明合法集）。"""
    obs = {"to_call": None, "hero_seat": None,
           "extra": {"v3f_bets": {"S1": 1000, "S2": 0}}}
    d = fe.derived_facing(obs)
    assert d["facing"] == "UNKNOWN" and d["legal"] == []
    assert d["source"] == "v3f_bets_hero_seat_unknown"


def test_unreadable_bets_is_unknown():
    obs = {"to_call": None, "hero_seat": 5,
           "extra": {"v3f_bets": {"S1": None, "S2": None}}}
    d = fe.derived_facing(obs)
    assert d["facing"] == "UNKNOWN" and d["source"] == "v3f_bets_unreadable"


def test_hero_already_matched_is_no_bet():
    """hero 已投入 ≥ 对手最高 ⇒ 无待跟注（check/bet）。"""
    obs = {"to_call": None, "hero_seat": 5,
           "extra": {"v3f_bets": {"S1": 1000, "S5": 2000}}}
    d = fe.derived_facing(obs)
    assert d["facing"] == "NO_BET" and d["legal"] == ["check", "bet"]


def test_no_evidence_stays_unknown():
    d = fe.derived_facing({})
    assert d["facing"] == "UNKNOWN" and d["legal"] == []
    assert d["source"] == "no_evidence"


# ── 与动作对齐联动（修复前这是"不知道有人下注"的现场） ────────────────
def test_alignment_uses_seat_bet_evidence_to_fix_action():
    fd = {"action": "bet", "alternative_action": "check", "recommended_size": 16.15,
          "reasoning": "底池 40.4bb，需跟注0.00bb。solver 建议过牌。"}
    obs = {"to_call": None, "hero_seat": 5,
           "extra": {"v3f_bets": {"S1": 1000, "S2": 2000, "S6": 1920, "S5": None}}}
    out = asem.align_final_decision(final_decision=fd, obs=obs)
    assert out["action"] == "raise"
    assert out["alternative_action"] == "call"
    assert out["recommended_size"] is None
    assert out["action_semantics"]["legal_source"] == "v3f_bets"
    assert out["action_semantics"]["facing_from_evidence"] == "BET_PENDING"
    assert "需跟注0.00bb" not in out["reasoning"]


def test_alignment_respects_no_bet_evidence():
    fd = {"action": "call", "alternative_action": None, "recommended_size": None}
    obs = {"to_call": None, "hero_seat": 5,
           "extra": {"v3f_bets": {"S1": 0, "S2": 0, "S5": 0}}}
    out = asem.align_final_decision(final_decision=fd, obs=obs)
    assert out["action"] == "check"       # call 在无人下注世界 ⇒ check
    assert out["action_semantics"]["facing_from_evidence"] == "NO_BET"


# ── hero 牌诊断 ──────────────────────────────────────────────────────
def test_hero_cards_diagnosis_reasons():
    ok = fe.hero_cards_diagnosis({"hole_cards": ["5c", "3c"], "hero_seat": 5,
                                   "button_seat": 6})
    assert ok["ok"] is True and ok["reason"] is None
    missing_seat = fe.hero_cards_diagnosis({"hole_cards": [], "hero_seat": None})
    assert missing_seat["reason"] == "hero_seat_unrecognized"
    missing_dealer = fe.hero_cards_diagnosis({"hole_cards": [], "hero_seat": 5,
                                               "button_seat": None})
    assert missing_dealer["reason"] == "dealer_button_unrecognized"
    no_match = fe.hero_cards_diagnosis({"hole_cards": [], "hero_seat": 5,
                                         "button_seat": 6})
    assert no_match["reason"] == "hero_cards_region_no_match"
    partial = fe.hero_cards_diagnosis({"hole_cards": ["5c"], "hero_seat": 5,
                                        "button_seat": 6})
    assert partial["reason"] == "hero_cards_partial_match"


def test_evidence_contract_declares_no_fabrication():
    c = fe.evidence_contract()
    assert c["fabricates"] is False
    assert c["frozen_layers_modified"] is False
    assert c["priority"][-1] == "UNKNOWN"
