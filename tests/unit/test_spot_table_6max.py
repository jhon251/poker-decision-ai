# -*- coding: utf-8 -*-
"""P30-9-CG 测试：6-max 位置键查表（匹配语义 / 位置归一化 / 诚实 miss / 锚定契约字段）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.providers.spot_table_6max import (   # noqa: E402
    SpotTable6max, get_spot_table_6max, norm_pos)


def test_position_normalization():
    assert norm_pos("UTG") == "LJ" and norm_pos("MP") == "HJ"
    assert norm_pos("btn") == "BTN" and norm_pos(None) == ""


def test_library_loads_and_has_both_nodes():
    st = get_spot_table_6max()
    assert st.count() > 100, "6-max 库应有大量节点"
    r = st.lookup({"street": "flop", "board": ["7s", "5h", "4h"], "pot_bb": 6.5,
                    "node": "first_in", "positions": {"oop": "LJ", "ip": "BTN"}})
    assert r is not None and r["solver_name"] == "postflop"
    assert r["node"] == "first_in" and r["actions"]
    # 引擎位置名（UTG）应能匹配库里的 LJ
    r2 = st.lookup({"street": "flop", "board": ["7s", "5h", "4h"], "pot_bb": 6.5,
                     "node": "first_in", "positions": {"oop": "UTG", "ip": "BTN"}})
    assert r2 is not None, "UTG 应归一化为 LJ 后命中"


def test_facing_node_requires_bet_ratio_bucket():
    st = get_spot_table_6max()
    ok = st.lookup({"street": "flop", "board": ["7s", "5h", "4h"], "pot_bb": 8.6,
                     "node": "facing_bet", "positions": {"oop": "LJ", "ip": "BTN"},
                     "bet_ratio": 0.33})
    assert ok is not None and ok["node"] == "facing_bet"
    assert ok["is_approximate"] is False, "严格命中才允许锚定"
    # 尺寸桶不匹配（满池下注不在解树内）⇒ 允许"尺寸近似"，但**必须标注且不锚定**
    # P30-9-DW：4x 级别的尺寸错配（满池 1.0 vs 解算 ≈0.24）⇒ **诚实 miss**
    far = st.lookup({"street": "flop", "board": ["7s", "5h", "4h"], "pot_bb": 8.6,
                      "node": "facing_bet", "positions": {"oop": "LJ", "ip": "BTN"},
                      "bet_ratio": 1.0, "to_call_bb": 2.0, "stack_bb": 97.5})
    assert far is None, "超 2x 倍数 ⇒ 不迁移策略（诚实 miss）"
    # 界内（≈1.8x）⇒ 近似且标注
    near = st.lookup({"street": "flop", "board": ["7s", "5h", "4h"], "pot_bb": 8.6,
                       "node": "facing_bet", "positions": {"oop": "LJ", "ip": "BTN"},
                       "bet_ratio": 0.45, "to_call_bb": 2.0, "stack_bb": 97.5})
    assert near is not None and near["is_approximate"] is True


def test_pot_approximation_is_labeled_and_bounded():
    """底池近似：≤60% 允许（标注且不锚定）；>60% 诚实 miss。"""
    st = get_spot_table_6max()
    near = st.lookup({"street": "flop", "board": ["Ad", "9h", "6s"], "pot_bb": 5.0,
                       "node": "facing_bet", "positions": {"oop": "BB", "ip": "HJ"},
                       "bet_ratio": 1.0, "to_call_bb": 2.5, "stack_bb": 99.0})
    if near is not None:                      # 命中时必须标注
        assert near["is_approximate"] is True
        assert near["pot_approximate"] is True
        assert near["raw"]["solved_pot_bb"] is not None
    far = st.lookup({"street": "flop", "board": ["Ad", "9h", "6s"], "pot_bb": 400.0,
                      "node": "facing_bet", "positions": {"oop": "BB", "ip": "HJ"},
                      "bet_ratio": 0.25})
    assert far is None, "底池差远超 60% 必须诚实 miss"


def test_never_crosses_position_pair_or_node():
    st = get_spot_table_6max()
    assert st.lookup({"street": "flop", "board": ["7s", "5h", "4h"], "pot_bb": 6.5,
                       "node": "first_in",
                       "positions": {"oop": "SB", "ip": "BTN"}}) is None
    assert st.lookup({"street": "river", "board": ["7s", "5h", "4h"], "pot_bb": 6.5,
                       "node": "first_in",
                       "positions": {"oop": "LJ", "ip": "BTN"}}) is None


def test_result_declares_anchor_contract_fields():
    st = SpotTable6max("data/solver/spots_6max")
    r = st.lookup({"street": "flop", "board": ["7s", "5h", "4h"], "pot_bb": 6.5,
                    "node": "first_in", "positions": {"oop": "LJ", "ip": "BTN"}})
    assert r["solver_name"] == "postflop"          # anchor_from_solver 契约
    assert r["is_approximate"] is False
    assert r["raw"]["node"] in ("first_in", "facing_bet")
    assert "external" in r["raw"]["source"] or r["raw"]["source"] == "external_gto_json"
