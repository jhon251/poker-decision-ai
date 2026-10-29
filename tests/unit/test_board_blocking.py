# -*- coding: utf-8 -*-
"""P30-9-GA 测试：公共牌不足 ⇒ **诚实阻断**（不产出牌面相关建议）。

取证背景：真实会话里 19/25 个翻后局面的 state.board 只有 1 张牌（如 ['Ac']），
引擎仍照常给建议 ⇒ 牌面相关计算不可信。修复：不足则阻断（并在阻断前有限重拍）。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cash_game_ui.runner import board_blocking  # noqa: E402


def test_incomplete_boards_block():
    assert board_blocking({"street": "flop", "board": ["Ac"]})
    assert board_blocking({"street": "flop", "board": ["Ac", "Kd"]})
    assert board_blocking({"street": "turn", "board": ["Ac", "Kd", "2h"]})
    assert board_blocking({"street": "river", "board": ["Ac", "Kd", "2h", "7s"]})


def test_complete_boards_pass():
    assert board_blocking({"street": "flop",
                                            "board": ["Ac", "Kd", "2h"]}) is None
    assert board_blocking({"street": "turn",
                                            "board": ["Ac", "Kd", "2h", "7s"]}) is None
    assert board_blocking({"street": "river",
                                            "board": ["Ac", "Kd", "2h", "7s", "9c"]}) is None
    # 翻前本就没有公共牌 ⇒ 不阻断
    assert board_blocking({"street": "preflop", "board": []}) is None


def test_detail_string_has_counts():
    s = board_blocking({"street": "turn", "board": ["Ac"]})
    assert s is not None and "1/4" in s and "turn" in s
