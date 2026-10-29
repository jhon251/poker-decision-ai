# -*- coding: utf-8 -*-
"""P30-9-DW 测试：6-max 尺寸近似的**有界倍数**（≤2×）与诚实 miss。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.providers.spot_table_6max import (   # noqa: E402
    SIZE_APPROX_MAX_FACTOR, get_spot_table_6max)


def _row():
    st = get_spot_table_6max()
    assert st.count() > 100
    return st, next(r for r in st._rows
                     if r["node"] == "facing_bet" and r.get("bet_ratio"))


def test_exact_within_strict_band():
    st, r = _row()
    h = st.lookup({"street": r["street"], "board": r["board"], "pot_bb": r["pot_bb"],
                    "node": "facing_bet",
                    "positions": {"oop": r["oop"], "ip": r["ip"]},
                    "bet_ratio": r["bet_ratio"], "to_call_bb": r.get("to_call_bb") or 1.0,
                    "stack_bb": 97.5})
    assert h is not None and h["is_approximate"] is False


def test_approx_within_factor_is_labeled():
    st, r = _row()
    h = st.lookup({"street": r["street"], "board": r["board"], "pot_bb": r["pot_bb"],
                    "node": "facing_bet",
                    "positions": {"oop": r["oop"], "ip": r["ip"]},
                    "bet_ratio": (r["bet_ratio"] or 0.3) * 1.8,
                    "to_call_bb": r.get("to_call_bb") or 1.0, "stack_bb": 97.5})
    assert h is not None, "1.8x 在 2x 界内应命中"
    # 同桶内其它行可能严格命中 ⇒ 只校验**标注契约**
    if h["is_approximate"]:
        assert h["size_approximate"] is True


def test_beyond_factor_is_honest_miss():
    st, r = _row()
    h = st.lookup({"street": r["street"], "board": r["board"], "pot_bb": r["pot_bb"],
                    "node": "facing_bet",
                    "positions": {"oop": r["oop"], "ip": r["ip"]},
                    "bet_ratio": (r["bet_ratio"] or 0.3) * 4.0,
                    "to_call_bb": r.get("to_call_bb") or 1.0, "stack_bb": 97.5})
    assert h is None, "超出 2× 倍数 ⇒ 必须诚实 miss（策略不可迁移）"


def test_factor_constant_documented():
    assert SIZE_APPROX_MAX_FACTOR == 2.0
