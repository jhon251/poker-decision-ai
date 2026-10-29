"""V3F-P65 — Mode Isolation Integration 测试。

覆盖：能力矩阵 contract；sanitize 剥离（cash/mtt 跨 mode 上下文）；
缺失上下文诚实 unavailable；同 state 三模式隔离（只有当前 mode 因素
改变 utility）；exploit/fusion 不绕过 mode；UNKNOWN；确定性；P60–P62
回归（intel 加性块不改变既有语义）。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai.strategy_intel import evaluate_intel
from poker_decision_ai.strategy_mode_switch import (
    MODE_CAPABILITIES,
    mode_capability_report,
    resolve_mode,
    sanitize_state_for_mode,
    verify_mode_isolation,
)
from poker_decision_ai.strategy_robustness import make_state

EQ = {"hero_equity": 0.45,
      "outcome": {"win": 0.45, "tie": 0.0, "lose": 0.55}}
RANGE = {"villains": {"v2": {"distribution": {
    "AA": 0.1, "KK": 0.1, "AKs": 0.1, "QQ": 0.1, "AQo": 0.2, "KQo": 0.4}}}}
EXPLOIT = {"available": True, "confidence": 1.0,
           "affected_actions": {"bet": 0.2}, "streams_used": ["entry_rate"],
           "provenance": {"module": "exploit_strategy"}, "fallback": {}}


def _base_state():
    """同一基础 state（poll/stack/board/action 完全一致）。"""
    return make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                      "check")


def adj_saca(c):
    return c["adjustments"]["saca_cv_bb"]


def _run(mode: str, with_context: bool = True) -> dict:
    st = _base_state()
    st["game_type"] = mode
    if with_context and mode in ("mtt", "pko", "ko", "mystery"):
        st["tournament"] = {"payouts": [50, 30, 20]}
    if with_context and mode in ("pko", "ko", "mystery"):
        # P65 契约：bounty 语义在 tournament.bounty_type（BountyInfo 口径）
        st["tournament"]["bounty_type"] = mode
        for p in st["players"]:
            p["bounty"] = 50.0
    return evaluate_intel(st, copy.deepcopy(EQ),
                          range_src=copy.deepcopy(RANGE),
                          exploit=copy.deepcopy(EXPLOIT))


# ------------------------------------------------------------ 能力矩阵
def test_capability_contract():
    assert MODE_CAPABILITIES["cash"] == {"icm": "forbidden",
                                         "payouts": "forbidden",
                                         "bounty": "forbidden"}
    assert MODE_CAPABILITIES["mtt"]["bounty"] == "forbidden"
    assert MODE_CAPABILITIES["mtt"]["icm"] == "permitted"
    assert MODE_CAPABILITIES["pko"]["bounty"] == "permitted"
    assert MODE_CAPABILITIES["mystery"]["bounty"] == "permitted"


def test_capability_report_enabled_vs_missing():
    st_rich = _base_state()
    st_rich["game_type"] = "pko"
    st_rich["tournament"] = {"payouts": [50, 30, 20], "bounty_type": "pko"}
    st_rich["players"][2]["bounty"] = 50.0
    rep = mode_capability_report(st_rich, "pko")
    assert rep["capabilities"]["icm"]["status"] == "enabled"
    assert rep["capabilities"]["bounty"]["status"] == "enabled"

    st_poor = _base_state()
    st_poor["game_type"] = "pko"
    rep2 = mode_capability_report(st_poor, "pko")
    assert rep2["capabilities"]["icm"]["status"] == \
        "unavailable:missing_payouts"
    assert rep2["capabilities"]["bounty"]["status"] == \
        "unavailable:missing_bounty_context"


def test_resolve_mode_unknown():
    mode, reason = resolve_mode({"game_type": "badmode"})
    assert mode == "unknown" and "badmode" in reason
    mode2, reason2 = resolve_mode({})
    assert mode2 == "unknown" and reason2 == "game_type missing"


# ------------------------------------------------------------ sanitize
def test_sanitize_cash_strips_tournament_and_bounty():
    st = _base_state()
    st["game_type"] = "cash"
    st["tournament"] = {"payouts": [50, 30, 20]}      # 跨 mode 污染源
    st["bounty_type"] = "pko"
    for p in st["players"]:
        p["bounty"] = 50.0
    clean, stripped = sanitize_state_for_mode(st, "cash")
    assert "tournament" not in clean
    assert "bounty_type" not in clean
    assert all("bounty" not in p for p in clean["players"])
    assert any("tournament" in x for x in stripped)
    assert any("bounty" in x for x in stripped)


def test_sanitize_mtt_strips_bounty_keeps_payouts():
    st = _base_state()
    st["game_type"] = "mtt"
    st["tournament"] = {"payouts": [50, 30, 20]}
    st["bounty_type"] = "pko"
    for p in st["players"]:
        p["bounty"] = 50.0
    clean, stripped = sanitize_state_for_mode(st, "mtt")
    assert "bounty_type" not in clean
    assert all("bounty" not in p for p in clean["players"])
    assert clean["tournament"]["payouts"] == [50, 30, 20]   # MTT 内合法
    assert any("bounty" in x for x in stripped)
    assert not any("tournament" in x for x in stripped)


def test_sanitize_pko_keeps_context():
    st = _base_state()
    st["game_type"] = "pko"
    st["tournament"] = {"payouts": [50, 30, 20]}
    st["bounty_type"] = "pko"
    for p in st["players"]:
        p["bounty"] = 50.0
    clean, stripped = sanitize_state_for_mode(st, "pko")
    assert stripped == []
    assert clean["bounty_type"] == "pko"
    assert all(p.get("bounty") for p in clean["players"])


# ------------------------------------------------------------ 三模式隔离
def test_same_state_three_modes_isolated():
    """同一 state 三次运行：只有当前 mode 的因素改变 utility/provenance。"""
    results = {}
    for mode in ("cash", "mtt", "pko"):
        r = _run(mode)
        results[mode] = r
        assert r["mode_isolation"]["violations"] == []
    # CASH：chip_ev、无 ICM/bounty
    rc = results["cash"]
    assert rc["objective"] == "chip_ev"
    assert rc["provenance"]["icm_available"] is False
    assert rc["provenance"]["bounty_available"] is False
    assert all(c["adjustments"].get("bounty_util_bb") in (None, 0.0)
               for c in rc["candidates"])
    # MTT：ICM 激活、bounty 禁用
    rm = results["mtt"]
    assert rm["objective"] == "chip_ev+icm"
    assert rm["provenance"]["icm_available"] is True
    assert all(c["adjustments"].get("bounty_util_bb") in (None, 0.0)
               for c in rm["candidates"])
    assert rm["mode_isolation"]["forbidden_context_stripped"] == []
    # PKO：ICM+bounty 通道齐备（bounty 激活 → objective 含 +bounty）
    rp = results["pko"]
    assert rp["objective"] == "chip_ev+icm+bounty"
    assert rp["provenance"]["icm_available"] is True
    assert rp["provenance"]["bounty_available"] is True
    # 隔离：三模式 best_action/competitive 相同来源但 utility 空间不同
    assert rc["candidates"][0]["final_utility_bb"] != \
        rp["candidates"][0]["final_utility_bb"]


def test_cross_mode_context_borrowing_forbidden():
    """CASH state 带锦标赛上下文 → 仍 chip_ev（不借用）；MTT state 带
    PKO bounty 字段 → 无 bounty 调整（sanitize 剥离）。"""
    st = _base_state()
    st["game_type"] = "cash"
    st["tournament"] = {"payouts": [50, 30, 20]}       # 违禁上下文
    r = evaluate_intel(st, copy.deepcopy(EQ), range_src=copy.deepcopy(RANGE))
    assert r["objective"] == "chip_ev"
    assert r["mode_isolation"]["forbidden_context_stripped"] != []
    assert r["mode_isolation"]["violations"] == []

    st2 = _base_state()
    st2["game_type"] = "mtt"
    st2["tournament"] = {"payouts": [50, 30, 20], "bounty_type": "pko"}
    for p in st2["players"]:
        p["bounty"] = 50.0                              # 违禁上下文
    r2 = evaluate_intel(st2, copy.deepcopy(EQ), range_src=copy.deepcopy(RANGE))
    assert r2["objective"] == "chip_ev+icm"
    assert all(c["adjustments"].get("bounty_util_bb") in (None, 0.0)
               for c in r2["candidates"])
    assert any("bounty" in x
               for x in r2["mode_isolation"]["forbidden_context_stripped"])


def test_missing_context_honest_fallback():
    """MTT/PKO 缺上下文 → 诚实 unavailable（不借用其他 mode 补全）。"""
    st = _base_state()
    st["game_type"] = "mtt"                             # 无 payouts
    r = evaluate_intel(st, copy.deepcopy(EQ), range_src=copy.deepcopy(RANGE))
    assert r["objective"] == "chip_ev"                  # 诚实降级
    assert r["provenance"]["icm_available"] is False
    cap = r["mode_isolation"]["capabilities"]["icm"]
    assert cap["status"] == "unavailable:missing_payouts"

    st2 = _base_state()
    st2["game_type"] = "pko"                            # 无 payouts/bounty
    r2 = evaluate_intel(st2, copy.deepcopy(EQ), range_src=copy.deepcopy(RANGE))
    assert r2["provenance"]["icm_available"] is False
    assert r2["provenance"]["bounty_available"] is False


# ------------------------------------------------------------ exploit/fusion
def test_exploit_fusion_do_not_bypass_mode():
    """exploit/fusion 在三模式下应用，但 mode/objective/capabilities 不被
    绕过（P62/P63 语义经 mode switch 保持）。"""
    for mode in ("cash", "mtt", "pko"):
        r = _run(mode)
        assert r["exploit"]["available"] is True
        assert r["fusion"] is not None
        assert r["mode_isolation"]["violations"] == []
        for c in r["candidates"]:
            assert c["provenance"]["mode"] == r["mode"]
            adj = c["adjustments"]
            # 模式感知恒等式（P64 冻结语义）：chip 空间 base+cv==final_ev
            # 全模式成立；CASH 下 base+cv+exploit==final_utility；
            # MTT/PKO 下 final−before==exploit_bb（utility 空间加项）。
            assert c["base_ev_bb"] + adj["saca_cv_bb"] == \
                pytest.approx(c["final_ev_bb"], abs=3e-6)
            if c.get("utility_before_exploit_bb") is not None:
                delta = c["final_utility_bb"] - \
                    c["utility_before_exploit_bb"]
                assert delta == pytest.approx(
                    adj.get("exploit_bb") or 0.0, abs=3e-6)
            elif mode == "cash":
                total = adj["saca_cv_bb"] + (adj.get("exploit_bb") or 0.0)
                assert c["base_ev_bb"] + total == \
                    pytest.approx(c["final_utility_bb"], abs=3e-6)
        if mode == "cash":
            assert all(c["adjustments"].get("bounty_util_bb") in (None, 0.0)
                       for c in r["candidates"])
        if mode in ("mtt", "pko"):
            # 模式感知恒等式：MTT/PKO utility 经 k± 缩放——chip 恒等式与
            # exploit delta 恒等式分别成立（P64 冻结语义）
            for c in r["candidates"]:
                if c.get("utility_before_exploit_bb") is None:
                    continue
                assert c["base_ev_bb"] + adj_saca(c) ==                     pytest.approx(c["final_ev_bb"], abs=3e-6)
                delta = c["final_utility_bb"] -                     c["utility_before_exploit_bb"]
                assert delta == pytest.approx(
                    c["adjustments"].get("exploit_bb") or 0.0, abs=3e-6)


def test_fusion_conflict_isolated_per_mode():
    """冲突 exploit 在任何模式下都不产生方向调整。"""
    conflict = {"available": False, "conflict": True,
                "reason": "conflicting evidence directions",
                "streams_used": ["entry_rate", "conditional_fold"]}
    for mode in ("cash", "mtt", "pko"):
        st = _base_state()
        st["game_type"] = mode
        if mode != "cash":
            st["tournament"] = {"payouts": [50, 30, 20]}
        if mode in ("pko", "ko", "mystery"):
            st["bounty_type"] = mode
        r = evaluate_intel(st, copy.deepcopy(EQ),
                           range_src=copy.deepcopy(RANGE),
                           exploit=copy.deepcopy(conflict))
        assert r["exploit"]["available"] is False
        for c in r["candidates"]:
            assert not c["adjustments"].get("exploit_bb")


# ------------------------------------------------------------ UNKNOWN/solver
def test_unknown_game_type_propagates():
    st = _base_state()
    st["game_type"] = "badmode"
    r = evaluate_intel(st, copy.deepcopy(EQ))
    assert r["mode_isolation"]["resolved_mode"] == "unknown"
    assert r["mode_isolation"]["capabilities"] is None


def test_solver_unavailable_across_modes():
    for mode in ("cash", "mtt", "pko"):
        r = _run(mode)
        assert r["solver_source"]["available"] is False
        assert "BLOCKED" in r["solver_source"]["status"]


def test_deterministic_across_modes():
    for mode in ("cash", "mtt", "pko"):
        a = _run(mode)
        b = _run(mode)
        assert a == b


# ------------------------------------------------------------ P60–P62 回归
def test_p60_regression_identity_unchanged():
    """P60 恒等式在 mode switch 下不变（additive 块不改变候选语义）。"""
    r = _run("cash")
    for c in r["candidates"]:
        adj = c["adjustments"]
        assert c["base_ev_bb"] + adj["saca_cv_bb"] == \
            pytest.approx(c["final_ev_bb"], abs=3e-6)


def test_p62_regression_exploit_identity_unchanged():
    r = _run("cash")
    for c in r["candidates"]:
        if c.get("utility_before_exploit_bb") is None:
            continue
        delta = c["final_utility_bb"] - c["utility_before_exploit_bb"]
        assert delta == pytest.approx(
            c["adjustments"].get("exploit_bb") or 0.0, abs=3e-6)
