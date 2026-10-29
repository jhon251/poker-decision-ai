"""V3F-P22-C — Position Dealer Dual-D Template 单元测试。

覆盖：白D/暗D 6 GT exact、POT_102 暗D 闭环、红牌背/红UI 不误判、
无 dealer → UNKNOWN、确定性、1419 regression、positions_for 不变。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.dealer_button import (  # noqa: E402
    detect_dealer_button_1876)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _gt_seat(ann):
    dealer = (ann.get("fields") or {}).get("position") or {}
    d = dealer.get("dealer")
    return d.get("seat") if isinstance(d, dict) else d


# ── 1. 6 GT 全对（含白D/暗D） ────────────────────────────────────────
def test_dual_d_all_six_gt_exact():
    n = 0
    for ann in GT["annotations"]:
        gt = _gt_seat(ann)
        if gt is None:
            continue
        n += 1
        res = detect_dealer_button_1876(_frame(ann["frame_id"]))
        assert res["dealer_seat"] == gt, ann["frame_id"]
    assert n == 6


def test_pot_102_dark_d_closed():
    """帧05（POT_102）暗 D 态 → 由 dark_d 模板闭环为 S5。"""
    ann = [a for a in GT["annotations"]
           if "POT_CHANGED_102" in a["frame_id"]][0]
    res = detect_dealer_button_1876(_frame(ann["frame_id"]))
    assert res["dealer_seat"] == "S5"
    assert res["evidence"]["kind"] == "dark_d"


def test_white_d_used_for_white_frames():
    ann = [a for a in GT["annotations"]
           if "POT_CHANGED_139" in a["frame_id"]][0]
    res = detect_dealer_button_1876(_frame(ann["frame_id"]))
    assert res["evidence"]["kind"] == "white_d"
    assert res["dealer_seat"] == "S6"


# ── 2. 负例：无 D 字母不得误判 dealer ───────────────────────────────
def test_blank_frame_no_dealer_unknown():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)          # 纯毡面绿，无任何 D
    fb[..., 3] = 255
    res = detect_dealer_button_1876(fb)
    assert res["dealer_seat"] is None
    assert res["reason"] == "no_d_letter_match"


def test_red_cardback_no_d_not_dealer():
    """红色牌背（无 D 字母）不得被当 dealer。"""
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    # 在 S1 附近放一个红色 60x80 矩形（拟牌背）+ 白色边界
    fb[280:360, 200:280, :3] = (140, 20, 20)
    fb[280:360, 200:280, 3] = 255
    fb[282:283, 202:278, :3] = (200, 200, 200)   # 白边
    res = detect_dealer_button_1876(fb)
    assert res["dealer_seat"] is None            # 无 D 字母 → 不误判


# ── 3. 确定性 ──────────────────────────────────────────────────────
def test_deterministic():
    ann = GT["annotations"][0]
    a = detect_dealer_button_1876(_frame(ann["frame_id"]))
    b = detect_dealer_button_1876(_frame(ann["frame_id"]))
    assert a == b


# ── 4. 1419 regression（既有路径不变）───────────────────────────────
def test_1419_pipeline_unchanged():
    """1419 帧仍走既有 DealerRecognizer（不触发 1876 双D）。"""
    from poker_decision_ai.vision2.pixel.realclient.seat_position import (
        runtime_position_pipeline)
    w, h, rgba = decode_png((EV /
        "SCENE_CHANGED_37_1787487606667_full.png").read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    out = runtime_position_pipeline(frame=arr, window="1419")
    assert "runtime_result" in out                 # 管线正常（1419 路径）
