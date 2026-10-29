# -*- coding: utf-8 -*-
"""P30-9-EK 测试：6-max 查表的**行动者维度**（区分 OOP 过牌后面向 / IP 下注后面向）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.providers.spot_table_6max import (   # noqa: E402
    get_spot_table_6max, norm_pos)


def _rows():
    st = get_spot_table_6max()
    assert st.count() > 200
    return st


def test_actor_recorded():
    st = _rows()
    with_actor = [r for r in st._rows if r.get("actor")]
    assert len(with_actor) == len(st._rows), "每个节点都应记录 actor"
    assert all(r["actor"] in ("BB", "BTN", "SB", "CO", "LJ", "HJ") for r in with_actor)


def test_query_actor_filters():
    """同一 (位置/pot/node/尺寸) 下，OOP 面向与 IP 面向必须是**不同**节点。"""
    st = _rows()
    base = None
    for r in st._rows:
        if r["node"] == "facing_bet" and r.get("bet_ratio"):
            base = r
            break
    assert base is not None
    q = {"street": base["street"], "board": base["board"], "pot_bb": base["pot_bb"],
          "node": "facing_bet", "positions": {"oop": base["oop"], "ip": base["ip"]},
          "bet_ratio": base["bet_ratio"], "to_call_bb": base.get("to_call_bb") or 1.0,
          "stack_bb": 97.5}
    h_actor = st.lookup({**q, "actor": base["actor"]})
    assert h_actor is not None and h_actor["raw"].get("actor") == base["actor"]
    other = "BTN" if base["actor"] != "BTN" else "BB"
    h_other = st.lookup({**q, "actor": other})
    if h_other is not None:
        assert h_other["raw"].get("actor") == other, "actor 过滤失效"


def test_depth2_nodes_present():
    """深度 2（xb：OOP 过牌 → IP 下注 → OOP 面对）节点应已导入。"""
    st = _rows()
    xb = [r for r in st._rows if str(r.get("reached_by") or "").startswith("XB")]
    assert xb, "缺少深度 2 的 xb 节点"


def test_norm_pos_alias():
    assert norm_pos("UTG") == "LJ" and norm_pos("MP") == "HJ"
