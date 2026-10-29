# -*- coding: utf-8 -*-
"""P30-9-O：盲注一致性守卫（header 误读 100× 的实测案例）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import facing_evidence as fe                        # noqa: E402


def _obs(blinds, pot=None, bets=None):
    o = {"blinds": blinds, "pot": pot, "street": "flop", "hero_seat": 5,
         "seats": [{"seat": i} for i in range(1, 7)], "extra": {}}
    if bets is not None:
        o["extra"]["v3f_bets"] = bets
    return o


# ── 用户实测案例：header 读 2/5，但底池 0.41、桌上投入 0.05 ───────────
def test_legal_blind_post_below_bb_is_NOT_flagged():
    """用户 22:23 实测：blinds 0.02/0.05（手动输入）+ 桌上投入 {S5: 0.02}（SB）
    + pot 0.2 ⇒ **合法**，不得判矛盾（旧判据 bet<bb/2 会误杀）。"""
    c = fe.blinds_consistent(_obs({"sb": 0.02, "bb": 0.05}, pot=0.2,
                                   bets={"S1": None, "S2": None, "S6": None,
                                          "S5": 0.02}))
    assert c["ok"] is True, c
    assert c["reason"] == "consistent"


def test_slightly_undercounted_pot_is_tolerated():
    """pot 略小于 sb+bb（识别误差）⇒ 不算矛盾（阈值 = 半个盲注轮）。"""
    c = fe.blinds_consistent(_obs({"sb": 2.0, "bb": 5.0}, pot=6.0))
    assert c["ok"] is True


def test_user_case_header_misread_is_flagged_by_pot():
    c = fe.blinds_consistent(_obs({"sb": 2.0, "bb": 5.0}, pot=0.41,
                                   bets={"S2": None, "S6": 0.05}))
    assert c["ok"] is False
    assert c["reason"] == "pot_less_than_half_blind_round"
    assert c["detail"]["pot_minus_blinds"] < 0


def test_user_case_header_misread_is_flagged_by_observed_bet():
    c = fe.blinds_consistent(_obs({"sb": 2.0, "bb": 5.0}, pot=None,
                                   bets={"S5": 0.05}))
    assert c["ok"] is False and c["reason"] == "observed_amounts_far_below_blinds"
    # 比值口径 = min(sb,bb)/桌上金额 = 2/0.05 = 40（≥20 ⇒ 判矛盾）
    assert c["detail"]["ratio_blind_over_bet"] == 40.0


def test_correct_micro_blinds_are_consistent():
    # 16:50 实测帧：pot 0.12 = 0.02 + 0.05 + 0.05 ⇒ 自洽
    c = fe.blinds_consistent(_obs({"sb": 0.02, "bb": 0.05}, pot=0.12,
                                   bets={"S1": 0.02, "S2": 0.05, "S5": 0.05}))
    assert c["ok"] is True and c["reason"] == "consistent"


def test_2_5_blinds_with_consistent_bets_are_ok():
    c = fe.blinds_consistent(_obs({"sb": 2.0, "bb": 5.0}, pot=21.0,
                                   bets={"S2": 5.0, "S6": 10.0}))
    assert c["ok"] is True


def test_no_blinds_or_no_evidence_is_not_a_counterproof():
    assert fe.blinds_consistent(_obs(None, pot=10.0))["ok"] is True
    assert fe.blinds_consistent(_obs({"sb": 2.0, "bb": 5.0}))["ok"] is True
    assert fe.blinds_consistent({})["ok"] is True


def test_contract_declares_guard():
    assert fe.evidence_contract()["blinds_consistency_guard"] is True
