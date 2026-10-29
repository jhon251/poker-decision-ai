"""Vision V2ZW — Board State Machine 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.board_state import (  # noqa: E402
    BoardStateMachine)


def _m():
    return BoardStateMachine(session_id="t")


def test_new_hand_to_preflop():
    m = _m()
    m.observe(1, [])
    assert m.state == "PREFLOP"


def test_flop_confirmed_after_two_frames():
    m = _m()
    m.observe(1, [])
    m.observe(2, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "PREFLOP"
    m.observe(3, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "FLOP"
    rec = m.records[-1].to_dict()
    assert rec["previous_state"] == "PREFLOP"
    assert rec["board_cards"] == ["3c", "6h", "Th"]


def test_3_to_5_rejected_turn_not_observed():
    m = _m()
    m.observe(1, [])
    m.observe(2, [(0, "3c"), (1, "6h"), (2, "Th")])
    m.observe(3, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "FLOP"
    m.observe(4, [(0, "3c"), (1, "6h"), (2, "Th"), (3, "Ac"), (4, "5h")])
    m.observe(5, [(0, "3c"), (1, "6h"), (2, "Th"), (3, "Ac"), (4, "5h")])
    assert m.state == "UNKNOWN"
    assert "illegal_transition" in m.records[-1].transition_reason
    assert "turn_not_observed" in m.records[-1].transition_reason


def test_turn_then_river():
    m = _m()
    m.observe(1, [])
    for _ in range(2):
        m.observe(2, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "FLOP"
    for _ in range(2):
        m.observe(3, [(0, "3c"), (1, "6h"), (2, "Th"), (3, "Ac")])
    assert m.state == "TURN"
    for _ in range(2):
        m.observe(4, [(0, "3c"), (1, "6h"), (2, "Th"), (3, "Ac"),
                      (4, "5h")])
    assert m.state == "RIVER"


def test_hand_end_reset():
    m = _m()
    m.observe(1, [])
    for _ in range(2):
        m.observe(2, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "FLOP"
    m.observe(9, [])
    assert m.state == "NEW_HAND"
    assert m.board_cards == []
    # 旧板不泄漏：新 hand 重新确认（3 卡需 2 帧）
    m.observe(10, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "NEW_HAND"
    m.observe(11, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "FLOP"


def test_confirmed_card_leak_blocked():
    m = _m()
    m.observe(1, [])
    for _ in range(2):
        m.observe(2, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "FLOP"
    # 已确认卡无故改变（3c → Ks）
    m.observe(3, [(0, "Ks"), (1, "6h"), (2, "Th")])
    assert m.state == "UNKNOWN"


def test_5_to_3_rejected():
    m = _m()
    m.observe(1, [])
    for _ in range(2):
        m.observe(2, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "FLOP"
    for _ in range(2):
        m.observe(3, [(0, "3c"), (1, "6h"), (2, "Th"), (3, "Ac")])
    assert m.state == "TURN"
    for _ in range(2):
        m.observe(4, [(0, "3c"), (1, "6h"), (2, "Th"), (3, "Ac"),
                      (4, "5h")])
    assert m.state == "RIVER"
    # 5→3 无 reset → 拒绝
    m.observe(5, [(0, "3c"), (1, "6h"), (2, "Th")])
    m.observe(6, [(0, "3c"), (1, "6h"), (2, "Th")])
    assert m.state == "UNKNOWN"


def test_0_to_5_jump_rejected():
    """0→5 直接跳转（无 FLOP/TURN 证据）→ UNKNOWN。"""
    m = _m()
    m.observe(1, [])
    for _ in range(2):
        m.observe(2, [(0, "3c"), (1, "6h"), (2, "Th"), (3, "Ac"),
                      (4, "5h")])
    assert m.state == "UNKNOWN"


def test_state_record_fields():
    m = _m()
    m.observe(1, [])
    for _ in range(2):
        m.observe(2, [(0, "3c"), (1, "6h"), (2, "Th")])
    rec = m.records[-1].to_dict()
    for k in ("state", "cardinality", "board_cards", "previous_state",
              "transition_reason", "frame_id", "timestamp", "fingerprint",
              "evidence_frame_ids", "confidence", "ground_truth_status",
              "layout_version"):
        assert k in rec, k
    assert rec["ground_truth_status"] == "runtime_state_non_gt"
    assert rec["confidence"] >= 0.5


def test_source_redlines():
    src = (Path(__file__).parent.parent / "poker_decision_ai" / "vision2"
           / "pixel" / "realclient"
           / "board_state.py").read_text(encoding="utf-8")
    for bad in ("expected_value", "recommend", "exploit", "gto", "bluff",
                "pyautogui", "pynput", "click", "keypress"):
        assert bad not in src, f"含禁止 token {bad}"
