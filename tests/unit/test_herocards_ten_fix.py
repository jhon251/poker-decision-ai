# -*- coding: utf-8 -*-
"""P30-9-AE：10↔J 误读修复（完整正立 T 模板入库 + rank 窗口加宽）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                                        # noqa: E402
from poker_decision_ai.vision2.pixel.realclient import coinpoker_herocards1876 as hc
from poker_decision_ai.vision2_platform.png_decode import decode_png       # noqa: E402


def _frame(rel: str):
    w, h, rgba = decode_png((ROOT / rel).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def test_user_frame_reads_true_ten_of_hearts():
    """用户 GT 帧（Kh Th）：修复前读 Jh，修复后必须读 Th。"""
    out = hc.HeroCardsRecognizer1876().recognize(
        _frame("data/vision2/coinpoker_evidence/user_2300_preflop_kts.png"))
    assert out["status"] == "OK", out
    assert out["value"] == ["Kh", "Th"], out


def test_other_frames_unchanged():
    assert hc.HeroCardsRecognizer1876().recognize(
        _frame("data/vision2/coinpoker_evidence/user_2250_preflop_q6s.png")
    )["value"] == ["Qh", "6h"]
    assert hc.HeroCardsRecognizer1876().recognize(
        _frame("data/vision2/coinpoker_evidence/"
               "8bf82f134651db0add626be0304a2ff1.png"))["value"] == ["5c", "3c"]


def test_full_ten_template_present_and_complete():
    """rank_T 必须存在"完整正立"实例（宽 >= 30），否则双字形仍会被切。"""
    prep = hc._load_templates()
    shapes = [t.shape for (t, _t0, _tn) in prep.get("T", [])]
    assert shapes, "缺少 rank_T 模板"
    assert any(w >= 30 for (_th, w) in shapes), f"rank_T 无非残缺实例: {shapes}"


def test_gates_and_thresholds_untouched():
    """红线：阈值一分未降。"""
    assert hc.MIN_SCORE == 0.7
    assert hc.MIN_MARGIN == 0.08
    assert hc.CARD_PRESENCE_MIN == 8000
    assert hc.RED_INK_FRAC_MIN == 0.15


def test_rank_windows_wide_enough_for_two_glyph_rank():
    for key, (x0, y0, x1, y1) in hc.RANK_WINDOWS.items():
        assert (x1 - x0) >= 40, f"{key} rank 窗过窄（双字形会被切）: {x1 - x0}"
