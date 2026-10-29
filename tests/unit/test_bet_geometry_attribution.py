# -*- coding: utf-8 -*-
"""P30-9-AB：bet 环几何归属（修正 hero 自身投入；实测三帧）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import facing_evidence as fe                        # noqa: E402

# 三帧实测的 v3f_bets（观察者输出，键为 bet 环编号）
U2250 = {"hero_seat": 5, "street": "preflop", "pot": 0.24, "to_call": None,
         "extra": {"v3f_bets": {"S1": 0.02, "S2": 0.05, "S6": None,
                                 "S3": 0.12, "S4": 0.05, "S5": None}}}
U2300 = {"hero_seat": 5, "street": "preflop", "pot": 0.4, "to_call": None,
         "extra": {"v3f_bets": {"S1": None, "S2": None, "S6": None,
                                 "S3": None, "S4": 0.25, "S5": 0.15}}}
CANON = {"hero_seat": 5, "street": "preflop", "pot": 8.92, "to_call": None,
         "extra": {"v3f_bets": {"S1": 1000, "S2": 2000, "S6": 1920,
                                 "S3": 2000, "S4": 2000, "S5": None}}}


def test_geometry_attribution_user_2250_matches_client_call_amount():
    """客户端显示「跟注 0.07」；几何归属必须得到同一个数。"""
    hb = fe.hero_opp_bets(U2250)
    assert hb["mapping"] == "geometry_1876"
    assert hb["hero_bet"] == 0.05          # 底中（hero）自己的 0.05
    assert hb["opp_max"] == 0.12           # 右上 S3 的加注
    assert fe.to_call_lower_bound(U2250)["to_call"] == 0.07


def test_geometry_attribution_user_2300_hero_is_aggressor():
    hb = fe.hero_opp_bets(U2300)
    assert hb["hero_bet"] == 0.25 and hb["opp_max"] == 0.15
    assert fe.to_call_lower_bound(U2300)["to_call"] is None   # hero 无需跟注


def test_geometry_attribution_canonical():
    hb = fe.hero_opp_bets(CANON)
    assert hb["hero_bet"] == 2000.0 and hb["opp_max"] == 2000.0
    assert fe.to_call_lower_bound(CANON)["to_call"] is None


def test_label_mismatch_is_why_geometry_needed():
    """证明：按 hero_seat=5 查 bet 环会拿到错的值（这就是原缺陷）。"""
    bets = U2250["extra"]["v3f_bets"]
    assert bets["S5"] is None              # 按编号查 ⇒ 得到 None（错）
    assert bets["S4"] == 0.05              # 真正属于 hero 的是 "S4" 位置


def test_other_layouts_fall_back_to_hero_seat_matching():
    obs = {"hero_seat": 3, "street": "flop",
           "extra": {"v3f_bets": {"S_TL": 5.0, "S_TR": 2.0, "seat3": 1.0}}}
    hb = fe.hero_opp_bets(obs)
    assert hb["mapping"] == "hero_seat"
    assert hb["hero_bet"] == 1.0 and hb["opp_max"] == 5.0


def test_facing_unverifiable_uses_geometry():
    # hero 有投入、看不到对手投入 ⇒ 不可确认（原来会把 hero 的 0.05 当成"对手"或漏掉）
    obs = {"hero_seat": 5, "street": "preflop", "pot": 0.19,
           "extra": {"v3f_bets": {"S1": None, "S2": None, "S6": None,
                                   "S3": None, "S4": 0.05, "S5": None}}}
    fv = fe.facing_unverifiable(obs)
    assert fv["unverifiable"] is True
    assert fv["detail"]["hero_bet"] == 0.05
    assert fv["reason"] == "hero_bet_present_no_opponent_bet_visible"
