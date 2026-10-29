# -*- coding: utf-8 -*-
"""P30-9-BB/BC 测试：SB 锚定方位推导 + 引擎交叉校验 + 弃牌排除。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.position_derive import (derive_positions_by_zone,   # noqa: E402
                                                zone_to_seat)
from poker_decision_ai.line_tracker import LineTracker, CLOCKWISE_ZONES   # noqa: E402


def _obs(street="preflop", bets=None, blinds=None, hero_seat=5, button_seat=6,
         status=None, pos="CO"):
    return {"street": street, "hero_seat": hero_seat, "button_seat": button_seat,
            "hole_cards": ["Qs", "8d"], "board": [], "pot": 0.24,
            "blinds": blinds or {"sb": 0.02, "bb": 0.05},
            "extra": {"v3f_bets": bets or {}, "v3f_hero_position": pos,
                       **({"v3f_status": status} if status else {})}}


def test_sb_anchor_derives_full_ring():
    """只有 SB 位能恰好投入 sb ⇒ 整圈方位唯一确定。"""
    bets = {"S1": 0.02, "S2": 0.05, "S3": 0.12, "S4": 0.05}
    d = derive_positions_by_zone(_obs(bets=bets), CLOCKWISE_ZONES)
    assert d and d["by_zone"]["S1"] == "SB" and d["by_zone"]["S2"] == "BB"
    assert d["by_zone"]["S4"] == "CO" and d["hero_position_derived"] == "CO"


def test_ambiguous_sb_returns_none():
    """两个区都等于 sb（异常读数）⇒ 拒绝推断。"""
    bets = {"S1": 0.02, "S6": 0.02}
    assert derive_positions_by_zone(_obs(bets=bets), CLOCKWISE_ZONES) is None


def test_postflop_not_applicable():
    bets = {"S1": 0.02}
    assert derive_positions_by_zone(_obs(street="flop", bets=bets),
                                     CLOCKWISE_ZONES) is None


def test_zone_to_seat_crosschecks_button_seat():
    """推导出的 BTN 座位号必须等于引擎 button_seat，否则拒绝输出。"""
    bets = {"S1": 0.02, "S2": 0.05}
    d = derive_positions_by_zone(_obs(bets=bets, button_seat=6), CLOCKWISE_ZONES)
    assert zone_to_seat(_obs(bets=bets, button_seat=6), d)["by_zone"]["S1"] == 1
    # 引擎 button_seat 与几何不符 ⇒ 拒绝（宁可不给）
    assert zone_to_seat(_obs(bets=bets, button_seat=2), d) is None


def test_line_events_carry_position_seat_and_folded_flag():
    tr = LineTracker()
    tr.observe(_obs(bets={"S1": 0.02, "S2": 0.05}), frame_id="f1")
    out = tr.observe(_obs(bets={"S1": 0.02, "S2": 0.05, "S3": 0.15},
                           status={"1": "folded"}), frame_id="f2")
    ev = [e for e in out["action_history"] if e.get("zone") == "S3"]
    assert ev and ev[0].get("position") in ("UTG", "MP", "CO", "BTN")
    assert "seat" in ev[0]
    assert out["provenance"]["zone_positions"]
    assert out["provenance"]["zone_seats"]


def test_folded_seat_is_skipped_for_current_player():
    """已弃牌座位不得被当作"下一位行动者"。"""
    tr = LineTracker()
    tr.observe(_obs(bets={"S1": 0.02, "S2": 0.05}), frame_id="f1")
    # S2(BB) 加注，其顺时针下一位 S3 已弃牌 ⇒ 应跳到 S5
    out = tr.observe(_obs(bets={"S1": 0.02, "S2": 0.20, "S3": 0.0, "S5": 0.0},
                           status={"3": "folded"}), frame_id="f2")
    assert out["current_player_zone"] != "S3"
    assert out["provenance"]["folded_seats"] == [3]
