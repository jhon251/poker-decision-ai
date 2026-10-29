# -*- coding: utf-8 -*-
"""P30-9-ES 测试：HU spot 表的 **actor 维度**（禁止把 OOP 策略套给 IP 决策）。"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core.providers.spot_table import SpotTable   # noqa: E402

BASE = {"street": "flop", "board": ["Td", "9d", "6h"], "pot_bb": 3.5,
         "ranges": {"oop": "QQ-22,AQs-AJs,KQs,QJs,JTs,T9s,98s,87s,76s,AQo",
                     "ip": "22+,A2s+,K2s+,Q4s+,J6s+,T7s+,97s+,86s+,75s+,64s+,53s+,"
                           "A2o+,K8o+,Q9o+,J9o+,T9o"}}


def test_oop_actor_hits():
    st = SpotTable()
    h = st.lookup({**BASE, "actor": "oop"})
    assert h is not None and h["is_approximate"] is False


def test_ip_actor_honest_miss():
    """库内只有 OOP 首入解 ⇒ IP 首入必须**诚实 miss**（不得套用 OOP 策略）。"""
    st = SpotTable()
    assert st.lookup({**BASE, "actor": "ip"}) is None


def test_actor_optional_backward_compatible():
    st = SpotTable()
    h = st.lookup(BASE)
    assert h is not None, "未提供 actor 时应保持向后兼容（含近似路径）"


def test_library_actor_field_present():
    st = SpotTable()
    assert st._ensure_loaded()
    acts = {str(r.get("actor") or "oop").lower() for r in st._index.values()}
    assert acts <= {"oop", "ip"}
