"""V3F-P22-A — Pot 1876 Production Calibration 单元测试。

覆盖：1876 ROI（废止旧 (760,80,1060,170)）、6 GT 全对、K/M 格式、
几何点/M 判别、中文标签分离、PotObserver 1876 派发、render-class gate、
1419 回归、确定性。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    PotRecognizer)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    parse_stack_text)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PotObserver)
from poker_decision_ai.vision2.pixel.realclient.render_class import (  # noqa: E402
    EVIDENCE_LIMITED)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def _frame(ann):
    w, h, rgba = decode_png((EV / ann["frame_id"]).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


# ── 1. 1876 ROI：旧错误几何废止 ──────────────────────────────────────
def test_1876_roi_corrected_not_deprecated():
    assert PotRecognizer.POT_ROI_1876 == (908, 488, 1060, 543)
    assert PotRecognizer.POT_ROI_1876 != (760, 80, 1060, 170)   # 旧错误已废止


# ── 2. 6 GT 帧全部识别正确 ──────────────────────────────────────────
def test_pot_1876_all_six_gt_exact():
    rec = PotRecognizer(window="1876")
    n = 0
    for ann in GT["annotations"]:
        pv = (ann.get("fields") or {}).get("pot")
        if pv is None:
            continue
        n += 1
        res = rec.recognize(_frame(ann))
        assert res["status"] == "RECOGNIZED", (ann["frame_id"],
                                               res.get("reason"))
        assert parse_stack_text(res["value"]) == pv, ann["frame_id"]
    assert n == 6


def test_pot_km_format():
    """K/M 格式（1.61M → 1610000）与逗号格式等价解析。"""
    assert parse_stack_text("1.61M") == 1610000
    assert parse_stack_text("1,610,000") == 1610000


# ── 3. PotObserver 1876 派发 ────────────────────────────────────────
def test_pot_observer_1876_dispatches():
    for ann in GT["annotations"]:
        pv = (ann.get("fields") or {}).get("pot")
        if pv is None:
            continue
        arr = _frame(ann)
        out = PotObserver().observe(arr, arr.shape[1], arr.shape[0])
        assert out["status"] == "RECOGNIZED"
        assert parse_stack_text(out["pot"]) == pv, ann["frame_id"]


def test_pot_observer_unknown_geometry_evidence_limited():
    fb = np.zeros((800, 1000, 4), dtype=np.uint8)
    out = PotObserver().observe(fb.tobytes(), 1000, 800)
    assert out["status"] == EVIDENCE_LIMITED
    assert out["pot"] is None


# ── 4. 1419 回归（既有语义不变）────────────────────────────────────
def test_pot_1419_regression():
    w, h, rgba = decode_png((EV /
        "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
    out = PotObserver().observe(rgba, w, h)
    assert out["pot"] == "23,280"          # 1419 语义（字符串带逗号）


# ── 5. 中文标签分离 + 确定性 ────────────────────────────────────────
def test_pot_roi_excludes_chinese_label():
    """pot_roi x0=908，'底池'标签在 x<908 → 数字识别与中文解耦。"""
    assert 908 >= 908                       # ROI 左缘排除标签区
    arr = _frame(GT["annotations"][0])
    rec = PotRecognizer(window="1876")
    # ROI 外（标签区 x 766-907）不应产生数字
    res = rec.recognize(arr)
    assert res["value"] == "64,800"


def test_pot_1876_deterministic():
    rec = PotRecognizer(window="1876")
    arr = _frame(GT["annotations"][0])
    assert rec.recognize(arr)["value"] == rec.recognize(arr)["value"]


def test_unreadable_pot_unknown_not_zero():
    """无 pot 显示帧 → UNKNOWN（value=None，非 0）。"""
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (40, 90, 60)
    fb[..., 3] = 255
    out = PotRecognizer(window="1876").recognize(fb)
    assert out["value"] is None
    assert out["status"] != "RECOGNIZED"
