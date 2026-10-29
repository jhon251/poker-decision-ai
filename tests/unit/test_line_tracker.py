# -*- coding: utf-8 -*-
"""P30-9-AZ 测试：跨帧线路追踪（只输出可证事件；冷启动不推断）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.line_tracker import (LineTracker,                  # noqa: E402
                                             attach_to_observation)


def _obs(street, bets):
    return {"street": street, "hero_seat": 5, "button_seat": 3,
            "hole_cards": ["Qs", "2s"], "board": [], "pot": 0.05,
            "blinds": {"sb": 0.01, "bb": 0.02},
            "extra": {"v3f_bets": bets}}


def test_cold_start_never_invents_events():
    tr = LineTracker()
    out = tr.observe(_obs("preflop", {"S1": None, "S2": None, "S3": None,
                                       "S4": None, "S5": None, "S6": None}))
    assert out["action_history"] == []
    assert "no_prior_frame" in out["unknown"]
    assert out["current_player_zone"] is None


def test_raise_when_prior_unreadable_reports_commitment_only():
    """上一帧读不到 ⇒ 只能报"该街累计投入"，并标注 prior_unreadable（不假装知道增量）。"""
    tr = LineTracker()
    tr.observe(_obs("preflop", {"S4": 0.01, "S3": 0.02, "S1": None,
                                 "S2": None, "S5": None, "S6": None}), frame_id="f1")
    out = tr.observe(_obs("preflop", {"S4": 0.01, "S3": 0.02, "S1": 0.10,
                                       "S2": None, "S5": None, "S6": None}),
                     frame_id="f2")
    ev = [e for e in out["action_history"] if e.get("zone") == "S1"]
    assert ev and ev[0]["kind"] == "raise"
    assert ev[0]["amount"] == 0.10            # 累计投入（非增量）
    assert ev[0]["prior_unreadable"] is True
    assert ev[0]["source"] == "derived_from_commitment_delta"
    # 下一位(S2)读不到 ⇒ 拒绝断言（不跳到更远的人去猜）
    assert out["current_player_zone"] is None
    assert "current_player_next_zone_unreadable" in out["unknown"]


def test_exact_delta_when_prior_readable():
    """上一帧**可读**（显式 0）⇒ 给出精确增量。"""
    tr = LineTracker()
    tr.observe(_obs("preflop", {"S1": 0.0, "S2": 0.0, "S3": 0.02, "S4": 0.01,
                                 "S5": 0.0, "S6": 0.0}), frame_id="f1")
    out = tr.observe(_obs("preflop", {"S1": 0.10, "S2": 0.0, "S3": 0.02,
                                       "S4": 0.01, "S5": 0.0, "S6": 0.0}),
                     frame_id="f2")
    ev = [e for e in out["action_history"] if e.get("zone") == "S1"]
    assert ev and ev[0]["kind"] == "raise"
    assert ev[0]["amount"] == 0.10            # 0.0 → 0.10 的精确增量
    assert ev[0]["prior_unreadable"] is False


def test_call_when_matching_current_max():
    tr = LineTracker()
    tr.observe(_obs("preflop", {"S1": 0.10, "S3": 0.02}), frame_id="f1")
    out = tr.observe(_obs("preflop", {"S1": 0.10, "S3": 0.10}), frame_id="f2")
    ev = [e for e in out["action_history"] if e.get("zone") == "S3"]
    assert ev and ev[0]["kind"] == "call" and ev[0]["amount"] == 0.08


def test_street_change_resets_baseline():
    tr = LineTracker()
    tr.observe(_obs("preflop", {"S1": 0.10, "S4": 0.02}), frame_id="f1")
    out = tr.observe(_obs("flop", {"S1": 0.10, "S4": 0.02}), frame_id="f2")
    kinds = [e["kind"] for e in out["action_history"]]
    assert "street_change" in kinds
    ev = [e for e in out["action_history"] if "amount" in e]
    # 跨街后的投入属于**新街**（不得被记成翻前事件）
    assert ev and all(e["street"] == "flop" for e in ev)


def test_to_call_derived_when_readable():
    tr = LineTracker()
    tr.observe(_obs("preflop", {"S1": 0.10, "S3": 0.10, "S4": 0.02}),
               frame_id="f1")
    out = tr.observe(_obs("preflop", {"S1": 0.10, "S3": 0.10, "S4": 0.02}),
                     frame_id="f2")
    # hero 在 S4（底中），当前注额 0.10 ⇒ hero 需补 0.08
    assert out["to_call_derived"] == 0.08


def test_unreadable_commitments_mark_unknown_and_no_to_call():
    tr = LineTracker()
    tr.observe(_obs("preflop", {"S3": None, "S4": None}), frame_id="f1")
    out = tr.observe(_obs("preflop", {"S3": None, "S4": None}), frame_id="f2")
    assert out["to_call_derived"] is None
    assert "commitments_unreadable" in out["unknown"]


def test_attach_only_adds_fields():
    obs = _obs("preflop", {"S1": 0.1})
    before = dict(obs["extra"])
    out = attach_to_observation(obs, {"action_history": [{"kind": "raise"}],
                                       "provenance": {"v": 1},
                                       "current_player_zone": "S2",
                                       "to_call_derived": 0.05,
                                       "to_call_exact": True})
    assert out["extra"]["v3f_action_history_status"] == "DERIVED"
    assert out["extra"]["v3f_current_player_derived"] == "S2"
    assert out["extra"]["v3f_to_call_derived"] == 0.05
    # 原有字段一个都不能变
    for k, v in before.items():
        assert out["extra"][k] == v
    assert out["street"] == obs["street"] and out["pot"] == obs["pot"]


def test_current_player_derived_when_all_readable():
    """全部可读时：加注者之后第一个未跟平者即为当前行动者。"""
    tr = LineTracker()
    tr.observe(_obs("preflop", {"S1": 0.0, "S2": 0.0, "S3": 0.02, "S4": 0.01,
                                 "S5": 0.0, "S6": 0.0}), frame_id="f1")
    out = tr.observe(_obs("preflop", {"S1": 0.10, "S2": 0.0, "S3": 0.02,
                                       "S4": 0.01, "S5": 0.0, "S6": 0.0}),
                     frame_id="f2")
    assert out["current_player_zone"] == "S2"
    assert out["current_player_role"] == "top_center"
