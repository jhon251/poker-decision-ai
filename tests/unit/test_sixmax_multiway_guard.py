# -*- coding: utf-8 -*-
"""P30-9-EM：6-max 单挑解**多人守卫**的契约与库语义测试。

守卫本身已在**真实帧**（20261010-114822：MP/CO/BB 三人翻牌）上端到端验证
⇒ 产生 `SPOT6_MULTIWAY_SKIPPED` 警告且不再套用单挑解。此处锁定契约与库语义。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

AGENT = ROOT / "poker_decision_ai" / "agents" / "solver" / "agent.py"


def test_guard_and_flag_present():
    src = AGENT.read_text(encoding="utf-8")
    assert "SPOT6_MULTIWAY_SKIPPED" in src, "缺少多人守卫警告码"
    assert "SPOT6_ALLOW_MULTIWAY" in src, "缺少多人放宽开关"
    assert "len(_pl) >= 3 and not _allow_mw6" in src, "缺少多人守卫条件"


def test_library_is_heads_up_postflop_only():
    """库内节点均为两人（oop/ip）⇒ 多人底池无参考数据（守卫的前提）。"""
    from poker_decision_ai.core.providers.spot_table_6max import get_spot_table_6max
    st = get_spot_table_6max()
    assert st.count() > 200
    for r in st._rows:
        assert r.get("oop") and r.get("ip") and r["oop"] != r["ip"], r.get("spot_id")
        assert r.get("actor") in (r.get("oop"), r.get("ip")), r.get("spot_id")
