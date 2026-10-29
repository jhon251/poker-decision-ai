# -*- coding: utf-8 -*-
"""P30-9-P：盲注输入框小数点问题（全角/逗号归一 + 编辑后不被覆盖）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai import facing_evidence as fe                        # noqa: E402
from cash_game_ui.runner import CashGameRunner                             # noqa: E402


# ── 归一（覆盖中文输入法全角句点/逗号） ─────────────────────────────
def test_normalize_fullwidth_and_comma():
    assert fe.normalize_decimal_text("0。05") == "0.05"      # 中文句号
    assert fe.normalize_decimal_text("0，05") == "0.05"      # 中文逗号
    assert fe.normalize_decimal_text("0．05") == "0.05"      # 全角句点
    assert fe.normalize_decimal_text("0,05") == "0.05"       # 半角逗号
    assert fe.normalize_decimal_text(" 0.02 ") == "0.02"     # 空格
    assert fe.normalize_decimal_text("０.０５") == "0.05"     # 全角数字
    assert fe.normalize_decimal_text(".05") == "0.05"        # 前导点
    assert fe.normalize_decimal_text("0.0.5") == "0.05"      # 多点 ⇒ 只留一个
    assert fe.normalize_decimal_text("0.05bb") == "0.05"     # 尾随字母
    assert fe.normalize_decimal_text("") == ""
    assert fe.normalize_decimal_text(None) == ""


def test_parse_decimal_text():
    assert fe.parse_decimal_text("0。05") == 0.05
    assert fe.parse_decimal_text("5") == 5.0
    assert fe.parse_decimal_text("") is None
    assert fe.parse_decimal_text("abc") is None
    # "." 归一为 "0." ⇒ 0.0（下游 blinds 校验会拒绝非正值，故此返回值安全）
    assert fe.parse_decimal_text(".") == 0.0


# ── 兜底输入接受归一后的文本 ────────────────────────────────────────
def test_override_accepts_fullwidth_input():
    ev = CashGameRunner._blinds_override_ev({"sb": "0。02", "bb": "0。05"})
    assert ev["sb"] == 0.02 and ev["bb"] == 0.05
    ev2 = CashGameRunner._blinds_override_ev({"sb": "0,02", "bb": "0,05"})
    assert ev2["sb"] == 0.02 and ev2["bb"] == 0.05
    ev3 = CashGameRunner._blinds_override_ev({"bb": "0。05"})   # BB-only
    assert ev3["bb"] == 0.05 and ev3["sb"] is None


def test_override_still_rejects_garbage():
    for bad in ({"bb": "abc"}, {"bb": "0"}, {"bb": "."}, {"sb": "x", "bb": "5"}):
        assert CashGameRunner._blinds_override_ev(bad) == {}, bad


# ── 用户编辑后不被自动回写覆盖（纯函数） ────────────────────────────
def test_writeback_decision_cases():
    assert fe.writeback_decision("2", "5", "2", "5", False) == "noop"
    assert fe.writeback_decision("0.02", "0.05", "2", "5", True) == "skip_user_edited"
    assert fe.writeback_decision("", "", "2", "5", True) == "write"
    assert fe.writeback_decision("", "", "2", "5", False) == "write"
    assert fe.writeback_decision("2", "", "2", "5", True) == "skip_user_edited"
