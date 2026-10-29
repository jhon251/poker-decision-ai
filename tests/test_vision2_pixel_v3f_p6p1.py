"""Vision V3F-P6-P1 — Bet Recognition Foundation 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    BET_ROIS_1419, BetRecognizer, parse_bet_amount)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


def _frame(path: str):
    w, h, rgba = decode_png(Path(path).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def test_no_bet_glyphs_unknown_first():
    """无 bet 尺度字形 → 恒 UNKNOWN（精度优先——防跨尺度错值）。"""
    r = BetRecognizer(rois=BET_ROIS_1419)
    f = _frame("data/vision2/coinpoker_evidence/"
               "SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    for _ in range(3):
        res = r.recognize(f)
        for v in res.values():
            assert v["status"] == "UNKNOWN"
            assert v["bet"] is None
            assert v["reason"] == "no_bet_scale_glyphs"


def test_parse_contract():
    assert parse_bet_amount("1,000") == 1000
    assert parse_bet_amount("2,000") == 2000
    assert parse_bet_amount("1,600") == 1600
    assert parse_bet_amount("960") == 960
    assert parse_bet_amount(None) is None
    assert parse_bet_amount("ABC") is None


def test_bet_roi_distinct_from_stack():
    """Bet ROI ≠ Stack ROI（隔离——数值不得来自座牌区）。"""
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (
        SEAT_ANCHORS_1419)
    for seat, (sr) in BET_ROIS_1419.items():
        sx, sy = SEAT_ANCHORS_1419.get(seat, (None, None))
        if sx is None:
            continue
        # bet 区与座锚保持距离（不同视觉区域）
        assert abs(sr[0] - sx) > 100 or abs(sr[1] - sy) > 100, seat


def test_no_pot_consumer():
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/coinpoker_bet.py"
    ).read_text(encoding="utf-8")
    assert "PotRecognizer" not in src
    assert "PotObserver" not in src


def test_unknown_not_zero_no_previous():
    """UNKNOWN ≠ 0/前值（重复识别均为 None——无缓存）。"""
    r = BetRecognizer(rois=BET_ROIS_1419)
    f = _frame("data/vision2/coinpoker_evidence/"
               "SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    prev = None
    for _ in range(3):
        res = r.recognize(f)
        for v in res.values():
            assert v["bet"] is None
        prev = res
    assert prev is not None


def test_stack_regression_guard_still():
    """434,504 Stack guard 不回归（模块独立）。"""
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (
        StackRecognizer)
    r = StackRecognizer(window="1419")
    f = _frame("data/vision2/coinpoker_evidence/"
               "SCENE_CHANGED_60_1787489266339_full.png")
    outs = [r.recognize_in_region(f, (150, 880, 500, 1060))
            for _ in range(3)]
    assert outs[0]["value"] == 434504
    assert "134,504" not in {o["raw_text"] for o in outs}
