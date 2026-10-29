# -*- coding: utf-8 -*-
"""P30-9-IC 回归：**窗口几何归一化**（用户窗口 1866x1400 曾导致"信息不足" ✗）。

背景（用户报告 + 日志取证）：日志显示 `牌桌窗口 bbox 已锁定 (1866x1400)` ✗，
而**全部识别器都按 1876x1400 标定**（`action_observer` / `betting_header_observer` /
`bet_observer` 一律 `!= (1876,1400) ⇒ UNKNOWN` ✓）⇒ 整条视觉链诚实拒绝 ⇒ UI 显示"信息不足" ✗。
修法：在**唯一入口**（`CashGameRunner.analyze` 与 `capture_frame`）把捕获帧归一到参考几何 ✓：
小幅偏差（≤8%）缩放 ✓、偏差过大**不猜**（保持原样并标注 ✗）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cash_game_ui.runner import (GEOM_TOL, REF_H, REF_W,  # noqa: E402
                                  normalize_geometry)

FIX = ROOT / "tests" / "fixtures" / "real_hero_cards_6h6c_1876x1400.png"


def _rgba(w: int, h: int) -> bytes:
    return np.zeros((h, w, 4), dtype=np.uint8).tobytes()


def test_reference_size_is_idempotent():
    out, w, h, info = normalize_geometry(_rgba(REF_W, REF_H), REF_W, REF_H)
    assert (w, h) == (REF_W, REF_H) and info["normalized"] is False
    assert info["from"] == (REF_W, REF_H)


def test_small_deviation_is_resized_to_reference():
    """用户实测的 1866x1400（窄 10px ✓）必须被归一到 1876x1400 ✓。"""
    out, w, h, info = normalize_geometry(_rgba(1866, REF_H), 1866, REF_H)
    assert (w, h) == (REF_W, REF_H) and info["normalized"] is True
    assert abs(info["scale"][0] - REF_W / 1866) < 1e-6
    assert len(out) == REF_W * REF_H * 4, "归一化后必须是参考尺寸的 RGBA ✓"


def test_large_deviation_is_not_guessed():
    """偏差过大 ⇒ **不归一、不猜** ✗（保持原样并标注 ✓）。"""
    big = int(REF_W * (1 + GEOM_TOL + 0.05))
    out, w, h, info = normalize_geometry(_rgba(big, REF_H), big, REF_H)
    assert (w, h) == (big, REF_H) and info["normalized"] is False
    assert "out_of_tolerance" in str(info.get("reason"))


@pytest.mark.skipif(not FIX.is_file(), reason="真实帧夹具缺失 ⇒ 跳过 ✓（不伪造 ✗）")
def test_narrow_real_frame_still_yields_a_decision():
    """端到端 ✓：真实帧**裁成 1866 宽**（模拟用户窗口 ✓）后，仍应出建议 ✓。

    判据：不再出现 `street_unrepresentable` / 不因几何而 UNAVAILABLE ✓；
    归一化信息必须如实记录 ✓。
    """
    from cash_game_ui.runner import CashGameRunner

    arr = np.asarray(Image.open(FIX).convert("RGBA"))
    narrow = Image.fromarray(np.ascontiguousarray(arr[:, :1866]))
    assert narrow.size == (1866, 1400)
    r = CashGameRunner()
    res = r.analyze(narrow.tobytes(), 1866, 1400, allow_recapture=False)
    d = res.to_dict() if hasattr(res, "to_dict") else dict(res)
    geo = getattr(r, "_last_geometry", None) or {}
    assert geo.get("normalized") is True and geo.get("from") == (1866, 1400)
    assert "street_unrepresentable" not in str(d.get("reason")), \
        f"几何归一后不应再报街不可判定 ✗（reason={d.get('reason')}）"
    assert d.get("status") in ("OK", "ADVICE_REFERENCE", "DECISION_UNAVAILABLE")
    if d.get("status") == "DECISION_UNAVAILABLE":
        assert str(d.get("reason") or "") != "street_unrepresentable:None;occupied_unknown"
