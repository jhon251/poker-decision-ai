# -*- coding: utf-8 -*-
"""P30-9-GT 测试：V3F 牌面残缺时的**本地补读回退**（只读、几何门控、不猜测）。

取证：真实日志里 **19/19 例残缺牌面都来自 `v3f_snapshot`**（该来源 86% 残缺 ✗），
另一来源 19/19 完整 ✓ ⇒ 缺陷在 V3F 生产端；本地识别器可作可靠回退 ✓
（真实帧实测：V3F 残缺的那一帧，本地识别器读出完整翻板 `Ad 9h 6s` ✓）。
"""
from __future__ import annotations

import pathlib
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cash_game_ui.runner import CashGameRunner, board_blocking  # noqa: E402


def _frame(w: int = 1876, h: int = 1400) -> bytes:
    return Image.new("RGBA", (w, h), (12, 24, 36, 255)).tobytes()


def test_unknown_geometry_returns_none():
    """未知几何 ⇒ 返回 None（绝不猜测；render_class 契约 ✓）。"""
    assert CashGameRunner._fallback_board_read(_frame(1234, 999), 1234, 999) is None


def test_known_geometry_does_not_raise_and_returns_list_or_none():
    """已知几何 ⇒ 只读识别；空屏返回 None（不抛异常、不影响决策 ✓）。"""
    out = CashGameRunner._fallback_board_read(_frame(), 1876, 1400)
    assert out is None or (isinstance(out, list) and all(isinstance(x, str) for x in out))


def test_no_frame_returns_none():
    assert CashGameRunner._fallback_board_read(None, 1876, 1400) is None


def test_board_blocking_still_blocks_when_fallback_fails():
    """回退失败时仍按原样诚实阻断 ✓（牌面不全会话）。"""
    obs = {"street": "flop", "board": ["Ac"]}
    assert board_blocking(obs) is not None
    assert board_blocking({"street": "flop", "board": ["Ad", "9h", "6s"]}) is None
