"""V3F-P24-C — Bet 1876 Bright-Pill Variant 单元测试。

覆盖：5 GT exact（含 S5 bright variant）、dark_ratio gate 不误拒 S5、
bright-path 不误读白卡/name/stack、S5 无 bet → UNKNOWN、1419 regression。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.bet_observer import (  # noqa: E402
    BetObserver)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _bet_out(fid):
    arr = _frame(fid)
    return BetObserver().observe(arr, arr.shape[1], arr.shape[0])


# ── 1. 5 GT exact ────────────────────────────────────────────────────
def test_five_bet_gt_exact():
    cases = [
        ("BUTTON_CHANGED_167_1787992682749_full.png", "S1", 21600),
        ("BUTTON_CHANGED_167_1787992682749_full.png", "S2", 21600),
        ("BUTTON_CHANGED_167_1787992682749_full.png", "S6", 21600),
        ("POT_CHANGED_12_1787995290794_full.png", "S5", 89600),
        ("BUTTON_CHANGED_598_1787993753587_full.png", "S5", 34500),
    ]
    for fid, seat, gt in cases:
        out = _bet_out(fid)
        r = out["seats"][seat]
        assert r["status"] == "RECOGNIZED", (seat, r.get("reason"))
        assert r["bet"] == gt, seat


def test_s5_bright_variant_not_rejected_by_dark_ratio():
    """S5 亮底 pill 不被 dark_ratio gate 拒绝（check_dark=False 路径）。"""
    out = _bet_out("POT_CHANGED_12_1787995290794_full.png")
    assert out["seats"]["S5"]["bet"] == 89600
    assert out["seats"]["S5"]["status"] == "RECOGNIZED"


def test_dark_path_still_works():
    """S1/S2/S6 暗胶囊路径不变。"""
    out = _bet_out("BUTTON_CHANGED_167_1787992682749_full.png")
    assert out["seats"]["S1"]["bet"] == 21600
    assert out["seats"]["S6"]["bet"] == 21600


# ── 2. 负例 ─────────────────────────────────────────────────────────
def test_s5_no_bet_unknown():
    """帧01 S5 无下注（GT 无）→ bright-path 不产生 bet（UNKNOWN）。"""
    out = _bet_out("BUTTON_CHANGED_167_1787992682749_full.png")
    r = out["seats"].get("S5", {})
    assert r.get("bet") is None
    assert r.get("status") != "RECOGNIZED"


def test_blank_frame_no_bet():
    """空白帧 → 无任何 bet（不把 UNKNOWN 当 0）。"""
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    out = BetObserver().observe(fb.tobytes(), 1876, 1400)
    for zone, r in out["seats"].items():
        assert r["bet"] is None
        assert r["status"] != "RECOGNIZED"


# ── 3. UNKNOWN propagation / render class ────────────────────────────
def test_unknown_geometry_evidence_limited_or_unknown():
    """未知几何 → 非 1876 gate（BetObserver 走 1419 → evidence-limited）。"""
    fb = np.zeros((800, 1000, 4), dtype=np.uint8)
    out = BetObserver().observe(fb.tobytes(), 1000, 800)
    for zone, r in out["seats"].items():
        assert r["bet"] is None


# ── 4. 1419 regression ──────────────────────────────────────────────
def test_1419_after_s6_fix():
    """1419 Bet 路径不受影响（无 bet glyph → UNKNOWN，不混淆 1876）。"""
    from poker_decision_ai.vision2_platform import decode_png as dp
    w, h, rgba = dp((EV /
        "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
    out = BetObserver().observe(rgba, w, h)
    assert out["window"] == "1419"
    for zone, r in out["seats"].items():
        assert r["bet"] is None or r["bet"] > 0
