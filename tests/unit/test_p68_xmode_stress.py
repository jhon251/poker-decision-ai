"""V3F-P68 — 三模式最终交叉压力测试。

覆盖：120 格交叉矩阵（3 模式 × 4 街 × 5 bounty 画像 × 2 exploit）；
七组不变量（隔离/chip 恒等式/exploit 恒等式/cap/双计/UNKNOWN/无伪
GTO）；违禁上下文防线（CASH 拒 ICM+bounty、MTT 拒 bounty）；PKO
bounty 通道激活（P44 契约位置）；确定性；快照一致性。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_decision_ai.strategy_stress_xmode import (
    build_scenarios,
    determinism_sample,
    run_cell,
    run_cross_stress,
)

SNAPSHOT = Path("docs/V3F_P68_XMODE_STRESS.json")


def test_matrix_coverage():
    scns = build_scenarios()
    assert len(scns) == 120
    ids = [s["id"] for s in scns]
    assert len(set(ids)) == 120
    assert {s["mode"] for s in scns} == {"cash", "mtt", "pko"}
    assert {s["bounty_profile"] for s in scns} == {
        "none", "standard", "high", "mtt_stray_bounty",
        "cash_stray_context"}


def test_cross_stress_zero_violations():
    rep = run_cross_stress()
    assert rep["n_cells"] == 120
    assert rep["n_violation_cells"] == 0
    assert rep["violation_types"] == {}


def test_cross_stress_deterministic():
    d = determinism_sample(n=12)
    assert d["all_equal"] is True


def test_cash_isolation_full():
    """CASH：任何 bounty 画像下 icm/bounty 都不激活（sanitize 防线）。"""
    rep = run_cross_stress()
    for c in rep["cells"]:
        if c["mode"] != "cash":
            continue
        assert c["objective"] == "chip_ev"
        assert c["violations"] == []


def test_mtt_bounty_stray_rejected():
    """MTT + stray bounty 字段 → 无 bounty_util（sanitize 剥离）。"""
    rep = run_cross_stress()
    for c in rep["cells"]:
        if c["mode"] == "mtt" and \
                c["bounty_profile"] == "mtt_stray_bounty":
            assert c["violations"] == []


def test_pko_bounty_channel_active():
    """PKO + standard/high 画像 → objective 含 +bounty（P44 契约位置）。"""
    rep = run_cross_stress()
    for c in rep["cells"]:
        if c["mode"] == "pko" and c["bounty_profile"] in ("standard",
                                                          "high"):
            assert c["objective"] == "chip_ev+icm+bounty"


def test_snapshot_matches_live():
    if not SNAPSHOT.exists():
        pytest.skip("snapshot not generated")
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    live = run_cross_stress()
    assert live["n_cells"] == snap["n_cells"]
    assert live["n_violation_cells"] == snap["n_violation_cells"] == 0
    assert live["violation_types"] == snap["violation_types"]


def test_no_fake_gto_in_all_cells():
    rep = run_cross_stress()
    assert all(not c.get("violations")
               or all("gto" not in v.lower()
                      for v in c["violations"])
               for c in rep["cells"])
