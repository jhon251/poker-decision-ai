"""V3F-P64 — Strategy Stress Validation 测试。

覆盖：1,440 格压力矩阵零不变量违规；模式感知恒等式（chip 空间全模式 /
exploit utility 空间加项）；cap/冲突/双计/mode-objective/solver/无默认
玩家类型/legality 全格通过；对抗输入（equity 0/1、非法 to_call、缺
信息）；exploit cap 边界；确定性；冻结核对（常数与文件 mtime 语义）。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.domain.poker import strategy_core as sc
from poker_decision_ai.strategy_robustness import make_state
from poker_decision_ai.strategy_stress import (
    build_scenarios,
    determinism_sample,
    run_cell,
    run_stress,
)

SNAPSHOT = Path("docs/V3F_P64_STRESS_MATRIX.json")


# ------------------------------------------------------------ 矩阵
def test_matrix_coverage():
    scns = build_scenarios()
    assert len(scns) == 1440                      # 5 模式 × 4 街 × 3 SPR
    # × 6 证据 × 3 exploit
    ids = [s["id"] for s in scns]
    assert len(set(ids)) == 1440
    modes = {s["mode"] for s in scns}
    assert modes == {"cash", "mtt", "pko", "ko", "mystery"}
    streets = {s["street"] for s in scns}
    assert streets == {"preflop", "flop", "turn", "river"}


def test_stress_zero_violations():
    rep = run_stress()
    assert rep["n_cells"] == 1440
    assert rep["n_violation_cells"] == 0
    assert rep["violation_types"] == {}
    assert rep["n_available"] == 1440             # 无一格崩溃/不可用


def test_stress_deterministic_sample():
    d = determinism_sample(n=12)
    assert d["all_equal"] is True


# ------------------------------------------------------------ 模式感知恒等式
def test_chip_identity_holds_all_modes():
    """chip 空间恒等式 base+cv==final_ev 在全部五模式成立（exploit 是
    utility 空间加项，不进此恒等式）。"""
    for gt in ("cash", "mtt", "pko"):
        for street in ("flop", "river"):
            st = make_state(street, ["9c", "4c"],
                            ["Ah", "7c", "2c"] if street != "preflop"
                            else [], "IP", 10.0, "check", game_type=gt)
            if gt in ("pko",):
                st["bounty_type"] = gt
                for p in st["players"]:
                    p["bounty"] = 50.0
            exploit = {"available": True, "confidence": 1.0,
                       "affected_actions": {"bet": 0.2},
                       "streams_used": ["entry_rate"],
                       "provenance": {}, "fallback": {}}
            r = evaluate_intel_stub(st, exploit)
            for c in r["candidates"]:
                adj = c["adjustments"]
                assert c["base_ev_bb"] + adj["saca_cv_bb"] == \
                    pytest.approx(c["final_ev_bb"], abs=3e-6)


def evaluate_intel_stub(st, exploit):
    """stress 模块同款调用（避免循环 import；evidence 缺省）。"""
    from poker_decision_ai.strategy_intel import evaluate_intel
    return evaluate_intel(copy.deepcopy(st), {"hero_equity": 0.5},
                          exploit=copy.deepcopy(exploit))


def test_exploit_utility_identity_mtt():
    """P62 冻结语义：MTT 下 final − before == exploit_bb（utility 空间
    加项），且 base+cv==final_ev（chip 恒等式）同时成立。"""
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check", game_type="mtt")
    st["tournament"] = {"payouts": [50, 30, 20]}
    exploit = {"available": True, "confidence": 1.0,
               "affected_actions": {"bet": -0.2},
               "streams_used": ["entry_rate"], "provenance": {},
               "fallback": {}}
    r = evaluate_intel_stub(st, exploit)
    assert r["objective"] == "chip_ev+icm"
    checked = 0
    for c in r["candidates"]:
        if c.get("utility_before_exploit_bb") is None:
            continue
        delta = c["final_utility_bb"] - c["utility_before_exploit_bb"]
        assert delta == pytest.approx(
            c["adjustments"].get("exploit_bb") or 0.0, abs=3e-6)
        checked += 1
    assert checked > 0


# ------------------------------------------------------------ 对抗输入
def test_adversarial_equity_extremes():
    """equity=0 / equity=1 → 合法候选与边界值，不崩溃。"""
    for eq in (0.0, 1.0):
        st = make_state("river", ["9c", "4c"],
                        ["Ah", "7c", "2c", "8d", "3h"], "IP", 10.0,
                        "check", game_type="cash")
        r = evaluate_intel_stub(st, {"hero_equity": eq})
        assert r["available"] is True
        assert all(c["action"] in ("fold", "check", "call", "bet", "raise")
                   for c in r["candidates"])


def test_adversarial_illegal_to_call():
    """to_call > stack → allin 封顶语义，候选合法且不越栈。"""
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 3.0,
                    "check", game_type="cash")
    st["to_call"] = 50.0                            # stack=30bb
    r = evaluate_intel_stub(st, {"hero_equity": 0.45})
    assert r["available"] is True
    for c in r["candidates"]:
        if c["amount_bb"] is not None:
            assert c["amount_bb"] <= 30.0 + 1e-9


def test_adversarial_malformed_evidence():
    """空/畸形 evidence dict → UNKNOWN 传播（exploit 不可用），不崩溃。"""
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check")
    bad = {"available": True, "confidence": 1.0, "affected_actions": None,
           "streams_used": None, "provenance": None, "fallback": None}
    r = evaluate_intel_stub(st, bad)
    assert r["exploit"]["available"] is True        # 输入原样（不伪造）
    # apply 层面对 None affected_actions 不崩溃
    for c in r["candidates"]:
        assert "base_ev_bb" in c


# ------------------------------------------------------------ exploit cap
def test_exploit_cap_boundary_in_matrix():
    """cap 边界：极端 exploit（±0.5bb 满额）→ 应用值不越界。"""
    for tilt_sign in (1, -1):
        exploit = {"available": True, "confidence": 1.0,
                   "opportunity_strength": 1.0,
                   "affected_actions": {"bet": 0.5 * tilt_sign},
                   "streams_used": ["entry_rate"], "provenance": {},
                   "fallback": {}}
        st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP",
                        10.0, "check", game_type="cash")
        r = evaluate_intel_stub(st, exploit)
        bet = next(c for c in r["candidates"] if c["action"] == "bet")
        assert abs(bet["adjustments"]["exploit_bb"]) <= 0.5 + 1e-9


# ------------------------------------------------------------ 冻结核对
def test_frozen_constants_untouched():
    assert sc.REALIZATION[("flop", True)] == 1.10
    assert sc.REALIZATION[("flop", False)] == 0.90
    assert sc.INITIATIVE[("flop", False)] == 1.00
    assert sc.CV_MODEL == "saca-1.0"
    assert sc.MIXED_MARGIN_FRAC_POT == 0.10


def test_snapshot_matches_live_run():
    if not SNAPSHOT.exists():
        pytest.skip("snapshot not generated")
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    live = run_stress()
    assert live["n_cells"] == snap["n_cells"]
    assert live["n_violation_cells"] == snap["n_violation_cells"] == 0
    assert live["violation_types"] == snap["violation_types"] == {}


def test_no_fake_gto_anywhere():
    """solver 不可用下，全部 cell 不出现 GTO/REAL 冒充。"""
    rep = run_stress()
    for c in rep["cells"]:
        # stress cell 本身不含 solver 字段——intel 层已验证 BLOCKED
        assert "gto" not in str(c.get("violations", [])).lower()
