"""Vision V2L — 覆盖率 / 时序事件 / 完整性测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coverage import (  # noqa: E402
    COVERAGE, completeness, coverage_matrix, temporal_changes)


def test_coverage_matrix_complete_and_valid():
    m = coverage_matrix()
    fields = [c.field for c in m]
    assert len(m) == 12
    for c in m:
        assert c.status in COVERAGE, c.field


def test_coverage_pot_unavailable_and_position():
    by_field = {c.field: c for c in coverage_matrix()}
    assert by_field["pot"].status == "UNAVAILABLE"
    assert by_field["position"].status == "UNAVAILABLE"


def test_temporal_only_visual_events():
    prev = {0: {"stack": 100, "bet": None, "folded": None, "all_in": None,
                "button": False}}
    curr = {0: {"stack": 90, "bet": 10, "folded": True, "all_in": None,
                "button": False}}
    events = temporal_changes(prev, curr)
    kinds = {e["kind"] for e in events}
    assert "STACK_CHANGED" in kinds
    assert "BET_APPEARED" in kinds
    assert "FOLD_MARKER_APPEARED" in kinds
    # 禁止行为归因事件
    for e in events:
        assert not e["kind"].startswith("PLAYER_")


def test_temporal_button_and_board():
    prev = {"board": ["A"], 2: {"button": True}}
    curr = {"board": ["A", "K", "Q"], 2: {"button": False}}
    events = temporal_changes(prev, curr)
    kinds = {e["kind"] for e in events}
    assert "BUTTON_MOVED" in kinds
    assert "BOARD_CARD_APPEARED" in kinds


def test_temporal_no_change_no_events():
    assert temporal_changes({0: {"stack": 5}}, {0: {"stack": 5}}) == []


def test_completeness_incomplete_lists_missing():
    res = completeness([{"seat": 0, "stack": 100, "bet": None},
                        {"seat": 1, "stack": None, "bet": None}],
                       ["As", "Kh", "2c"], None, "flop")
    assert res["status"] == "INCOMPLETE"
    assert "hero_cards" in res["missing_fields"]
    assert "player_1.stack" in res["missing_fields"]


def test_completeness_complete():
    players = [{"seat": i, "stack": 100, "bet": 0} for i in range(2)]
    res = completeness(players, ["a", "b", "c"], ["As", "Kh"], "flop")
    assert res["status"] == "COMPLETE_FOR_STATE_RECONSTRUCTION"
    assert res["missing_fields"] == []


def test_coverage_module_redlines():
    src = (Path(__file__).parent.parent
           / "poker_decision_ai" / "vision2" / "pixel" / "realclient"
           / "coverage.py").read_text(encoding="utf-8")
    banned = ("expected_value", "recommend", "exploit", "gto", "bluff",
              "villain", "ev_", "bet_size", "cv2", "torch")
    for bad in banned:
        assert bad not in src, f"含禁止 token {bad}"
