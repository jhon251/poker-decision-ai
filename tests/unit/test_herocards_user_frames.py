# -*- coding: utf-8 -*-
"""P30-9-AM/AN：用户保存帧的底牌必须读对（旋转渲染 + 贴合的牌面）。

真值由用户截图人眼确认：112328 = 10s 7c；113101 = 3h 2d。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                                        # noqa: E402
from poker_decision_ai.vision2.pixel.realclient import coinpoker_herocards1876 as hc
from poker_decision_ai.vision2_platform.png_decode import decode_png       # noqa: E402

FRAMES_DIR = ROOT / "dist" / "logs" / "frames"
CASES = [
    ("20261010-112328-DECISION_UNAVAILABLE.png", ["Ts", "7c"]),
    ("20261010-113101-DECISION_UNAVAILABLE.png", ["3h", "2d"]),
    ("20261010-114807-DECISION_UNAVAILABLE.png", ["Qs", "2s"]),
    ("20261010-114822-DECISION_UNAVAILABLE.png", ["Qs", "2s"]),
]


def _frame(name: str):
    fp = FRAMES_DIR / name
    if not fp.exists():
        import pytest
        pytest.skip(f"缺帧 {name}")
    w, h, rgba = decode_png(fp.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def test_user_frames_read_true_hands():
    rec = hc.HeroCardsRecognizer1876()
    for name, truth in CASES:
        out = rec.recognize(_frame(name))
        assert out["status"] == "OK", (name, out)
        assert out["value"] == truth, (name, out["value"], truth)


def test_rotated_rank_templates_present():
    """这些帧是旋转角标 ⇒ 模板集必须含对应旋转体（否则得分 < 0.7）。"""
    prep = hc._load_templates()
    for rank in ("7", "3", "2", "T"):
        shapes = [t.shape for (t, _a, _b) in prep.get(rank, [])]
        assert shapes, f"缺 rank_{rank}"
        assert any(h_ > w_ for (h_, w_) in shapes), f"rank_{rank} 无旋转(高>宽)实例: {shapes}"


def test_two_glyph_structure_tiebreak_present():
    """结构化消歧必须存在：仅 margin 不足时用"墨迹分量数"区分 10 与单字形 rank。"""
    src = (ROOT / "poker_decision_ai/vision2/pixel/realclient/"
           "coinpoker_herocards1876.py").read_text(encoding="utf-8")
    assert "_glyph_component_count" in src
    assert "two_glyph_structure" in src and "single_glyph_structure" in src
    # 分数门未降
    assert hc.MIN_SCORE == 0.7 and hc.MIN_MARGIN == 0.08
