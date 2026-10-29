"""Vision V3F-P5T — Stack Recognition Evidence Closure 测试（5 稳定源）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    StackRecognizer, parse_stack_text)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402

EV = "data/vision2/coinpoker_evidence/"


def _frame(path: str):
    w, h, rgba = decode_png(Path(path).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _stable3(outs):
    return len({(o.get("raw_text"), o.get("value")) for o in outs}) == 1 \
        and outs[0].get("status") == "RECOGNIZED"


# ── 5 个稳定源（×3 一致）───────────────────────────────────────
def test_stable_953909():
    r = StackRecognizer(window="1419")
    f = _frame(EV + "SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    outs = [r.recognize_in_region(f, (150, 300, 520, 500))
            for _ in range(3)]
    assert _stable3(outs), outs
    assert outs[0]["value"] == 953909


def test_stable_318871():
    r = StackRecognizer(window="1419")
    f = _frame(EV + "SETTLED_BOARD_CHECK_1328_1787713028674_full.png")
    outs = [r.recognize_in_region(f, (150, 900, 520, 1060))
            for _ in range(3)]
    assert _stable3(outs), outs
    assert outs[0]["value"] == 318871


def test_stable_718902():
    r = StackRecognizer(window="1419")
    f = _frame(EV + "SCENE_CHANGED_37_1787487606667_full.png")
    outs = [r.recognize_in_window(f, "37_BL") for _ in range(3)]
    assert _stable3(outs), outs
    assert outs[0]["value"] == 718902


def test_stable_388080():
    r = StackRecognizer(window="1419")
    f = _frame(EV + "STATUS_UI_CHANGED_246_1787488634077_full.png")
    outs = [r.recognize_in_region(f, (150, 330, 480, 520))
            for _ in range(3)]
    assert _stable3(outs), outs
    assert outs[0]["value"] == 388080


def test_stable_434504_no_134504():
    """434,504 回归 guard：=434504 且绝不出现 134504。"""
    r = StackRecognizer(window="1419")
    f = _frame(EV + "SCENE_CHANGED_60_1787489266339_full.png")
    outs = [r.recognize_in_region(f, (150, 880, 500, 1060))
            for _ in range(3)]
    assert _stable3(outs), outs
    assert outs[0]["value"] == 434504
    assert "134,504" not in {o["raw_text"] for o in outs}


def test_region_too_short_fragment_rejected():
    """1-3 字符片段（'5' 类）→ UNKNOWN（合理性门）。"""
    r = StackRecognizer(window="1419")
    f = _frame(EV + "SCENE_CHANGED_37_1787487606667_full.png")
    v = r.recognize_in_region(f, (150, 880, 500, 1060))
    # 该区域满值识别（窗路径已证 718,902）——区域路径至少不得给 '5' 值
    assert v["status"] != "RECOGNIZED" or (v.get("value") or 0) != 5


def test_parse_contract_still():
    assert parse_stack_text("2.57M") == 2570000
    assert parse_stack_text("192,720") == 192720
    assert parse_stack_text("201,000") == 201000
    assert parse_stack_text("") is None
