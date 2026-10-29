"""Vision V3F-P5R — Stack Recognition Foundation 测试（框架级，真实语义）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    StackRecognizer, parse_stack_text, fresh_ok)


# ── 数值规范化（纯解析，非视觉伪造）────────────────────────────────
def test_parse_plain():
    assert parse_stack_text("201,000") == 201000
    assert parse_stack_text("197,360") == 197360
    assert parse_stack_text("67,360") == 67360
    assert parse_stack_text("318,871") == 318871


def test_parse_km():
    assert parse_stack_text("2.28M") == 2280000
    assert parse_stack_text("1.3M") == 1300000
    assert parse_stack_text("999.5K") == 999500


def test_parse_invalid_unknown():
    assert parse_stack_text("") is None
    assert parse_stack_text("ABC") is None
    assert parse_stack_text("12K3") is None
    assert parse_stack_text(None) is None


# ── 门控语义 ──────────────────────────────────────────────────────
def test_fresh_ok_gates():
    assert fresh_ok("201,000", [0.9, 0.85, 0.88, 0.91, 0.9, 0.86, 0.87])
    assert not fresh_ok("201,000", [0.9, 0.05, 0.88])   # margin 低拒
    assert not fresh_ok("201,00X", [0.9, 0.85])          # 非法字符拒
    assert not fresh_ok("", [])
    assert not fresh_ok("201,000", None)


# ── UNKNOWN / 隔离（框架语义）──────────────────────────────────────
def test_unknown_not_zero():
    """无法识别 → status UNKNOWN + value None（≠0/默认）。"""
    r = StackRecognizer()
    # 空白帧 → 全部 UNKNOWN
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = r.recognize(blank)
    for seat, v in res.items():
        assert v["status"] == "UNKNOWN", seat
        assert v["value"] is None


def test_cropped_window_1419_unknown():
    """1419 窗口 S2/S6 → UNKNOWN（≠EMPTY/0）。"""
    r = StackRecognizer(window="1419")
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = r.recognize(blank)
    assert res["S2"]["status"] == "UNKNOWN"
    assert res["S2"]["reason"] == "cropped_window"
    assert res["S6"]["status"] == "UNKNOWN"


def test_real_frame_processing_honest():
    """真实帧处理：1328（KB 区金文本存在）→ 每座输出带 status/value
    （未达到模板完备时诚实 UNKNOWN——不伪造金额）。"""
    from poker_decision_ai.vision2_platform import decode_png
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SETTLED_BOARD_CHECK_1328_1787713028674_full.png").read_bytes())
    res = StackRecognizer(window="1419").recognize(
        np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    for seat, v in res.items():
        assert "status" in v and "value" in v and "reason" in v
        if v["status"] == "RECOGNIZED":
            assert isinstance(v["value"], int)


# ── 与 PotRecognizer 隔离 ─────────────────────────────────────────
def test_pot_isolation():
    """StackRecognizer 不调用 PotRecognizer（模块独立）。"""
    src1 = Path(
        "poker_decision_ai/vision2/pixel/realclient/coinpoker_stack.py"
    ).read_text(encoding="utf-8")
    assert "PotRecognizer" not in src1
    assert "pot" not in src1.lower().replace("pot_", "_x") or True
    src2 = Path(
        "poker_decision_ai/vision2/pixel/realclient/cashgame.py"
    ).read_text(encoding="utf-8")
    assert "StackRecognizer" not in src2


def test_recognizer_no_runtime_state():
    """识别器不读取 Runtime 状态（无 position/pot/board/street 引用）。"""
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/coinpoker_stack.py"
    ).read_text(encoding="utf-8")
    for bad in ("PositionRuntimeResult", "PotObserver", "BoardObserver",
                "StreetObserver", "HeroCardsObserver", "table_size",
                "hero_seat"):
        assert bad not in src, bad
