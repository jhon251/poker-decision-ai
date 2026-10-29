"""V3F-P69 — Poker AI 最终系统级验收测试。

覆盖：CASH 视觉→State→Decision→Judge 闭环；MTT/PKO hand-history→
State→Decision 闭环；三模式隔离与能力矩阵；UNKNOWN/缺失上下文诚实
降级；双计终审（bounty_util / pko_context_bb / icm 分账）；安全边界
（无自动执行字段、无默认视觉伪造、solver 不冒充）；确定性。
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker_decision_ai.strategy_intel import evaluate_intel
from poker_decision_ai.strategy_robustness import make_state

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "data2" / "8bf82f134651db0add626be0304a2ff1.png"


# ------------------------------------------------------------ CASH 闭环
def _test_safe_cfg():
    """serial 环境安全的 realtime 时钟预算（P52-3 同款；语义零修改）。"""
    import types
    from poker_decision_ai.core.config import load_config
    cfg = load_config()
    rt = dict(cfg.realtime_config() or {})
    rt["agent_timeout_ms"] = max(int(rt.get("agent_timeout_ms", 300)), 5000)
    rt["deadline_ms"] = max(int(rt.get("deadline_ms", 3000)), 20000)
    cfg.realtime_config = types.MethodType(lambda self, _rt=rt: dict(_rt),
                                           cfg)
    return cfg


def test_cash_vision_to_judge_closure():
    """CASH 闭环：视觉帧 → frame_to_decision → READY + 合法动作 +
    provenance 非 REAL/GTO；无执行字段（输出-only）。"""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "tests" / "unit"))
    from poker_decision_ai.core.orchestration.runtime import build_runtime
    from poker_decision_ai.frame_decision import frame_to_decision
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    w, h, rgba = decode_png(FRAME.read_bytes())
    r = frame_to_decision(build_runtime(_test_safe_cfg()), rgba, w, h)
    assert r.status == "OK"
    fd = r.report["final_decision"]
    assert fd["action"] in ("fold", "check", "call", "bet", "raise", "allin")
    assert fd["confidence"] is not None
    assert r.provenance["decision_engine"] == "realtime"
    # 安全：输出无执行字段（输出-only 契约）
    blob = str(r.report).lower()
    for forbidden in ("click(", "sendinput", "postmessage", "pyautogui",
                      "execute_bet"):
        assert forbidden not in blob


# ------------------------------------------------------------ MTT/PKO 闭环
def _mtt_state():
    st = make_state("flop", ["9c", "4c"], ["Ah", "Kd", "7c"], "IP", 10.0,
                    "check", game_type="mtt")
    st["tournament"] = {"payouts": [50, 30, 20], "stage": "middle",
                        "players_remaining": 45, "itm_players": 10}
    return st


def _pko_state():
    st = _mtt_state()
    st["game_type"] = "pko"
    st["tournament"]["bounty_type"] = "pko"
    st["tournament"]["bounty_rate"] = 0.5
    for p in st["players"]:
        if not p.get("is_hero"):
            p["bounty"] = 3000.0
    return st


def test_mtt_hand_history_closure():
    """MTT：hand-history 形态 state → intel → chip_ev+icm 闭环。"""
    r = evaluate_intel(_mtt_state(), {"hero_equity": 0.45},
                       range_src={"villains": {"v2": {"distribution": {
                           "AA": 0.1, "QQ": 0.2, "AQo": 0.3, "KQo": 0.4}}}})
    assert r["available"] and r["mode"] == "mtt"
    assert r["objective"] == "chip_ev+icm"
    assert r["provenance"]["icm_available"] is True
    assert r["provenance"]["bounty_available"] is False
    for c in r["candidates"]:
        assert c["provenance"]["mode"] == "mtt"
        assert c["base_ev_bb"] + c["adjustments"]["saca_cv_bb"] == \
            pytest.approx(c["final_ev_bb"], abs=3e-6)


def test_pko_closure_bounty_channel_active():
    """PKO：ICM+bounty 通道齐备；knockout（P44）与 pko_context（P67）
    分账共存不混写。"""
    st = _pko_state()
    r = evaluate_intel(st, {"hero_equity": 0.45},
                       range_src={"villains": {"v2": {"distribution": {
                           "AA": 0.1, "QQ": 0.2, "AQo": 0.3, "KQo": 0.4}}}},
                       pko_context={"module": "pko_context",
                                    "bounty_type": "pko",
                                    "growth_rate": 0.5,
                                    "is_bounty": True,
                                    "factors": {
                                        "effective_bounty": {
                                            "status": "usable",
                                            "max_bounty_bb": 15.0},
                                        "bounty_pressure": {
                                            "status": "usable",
                                            "pressure": 0.15},
                                        "cover_relation": {
                                            "status": "usable",
                                            "relation": "covered"},
                                        "icm_x_bounty": {
                                            "status": "context_only"},
                                        "late_stage_bounty_pressure": {
                                            "status": "UNKNOWN"}}})
    assert r["objective"] == "chip_ev+icm+bounty"
    assert r["pko_context_gate"]["applied"] is True
    # 分账：淘汰型候选可同时携带 bounty_util（P44）与 pko_context_bb
    # （P67）——字段独立，不混写
    for c in r["candidates"]:
        adj = c["adjustments"]
        if adj.get("bounty_util_bb"):
            assert "pko_context_bb" not in adj or \
                adj["pko_context_bb"] != adj["bounty_util_bb"]


# ------------------------------------------------------------ 三模式隔离
def test_three_modes_same_state_isolated():
    """同一基础 state 三模式分别运行：objective/能力矩阵互不污染。"""
    base = make_state("flop", ["9c", "4c"], ["Ah", "Kd", "7c"], "IP", 10.0,
                      "check")
    objs = {}
    for mode in ("cash", "mtt", "pko"):
        st = copy.deepcopy(base)
        st["game_type"] = mode
        if mode in ("mtt", "pko"):
            st["tournament"] = {"payouts": [50, 30, 20]}
        if mode == "pko":
            st["tournament"]["bounty_type"] = "pko"
            for p in st["players"]:
                p["bounty"] = 3000.0
        r = evaluate_intel(st, {"hero_equity": 0.45},
                           range_src={"villains": {"v2": {
                               "distribution": {"AA": 0.1, "QQ": 0.2,
                                                "AQo": 0.3, "KQo": 0.4}}}})
        objs[mode] = r["objective"]
    assert objs == {"cash": "chip_ev", "mtt": "chip_ev+icm",
                    "pko": "chip_ev+icm+bounty"}


# ------------------------------------------------------------ UNKNOWN 闭环
def test_unknown_closure_all_modes():
    """game_type 缺失/未知 → intel available=False（UNKNOWN-first 闭环）。"""
    for gt in (None, "badmode"):
        st = make_state("flop", ["9c", "4c"], ["Ah", "Kd", "7c"], "IP",
                        10.0, "check", game_type="cash")
        if gt is None:
            del st["game_type"]
        else:
            st["game_type"] = gt
        r = evaluate_intel(st, {"hero_equity": 0.45})
        # P65 mode switch 层：未知 game_type 在 resolved_mode 中诚实
        # 记录为 unknown（compute_ev 的 mtt fallback 是 P41 冻结语义）
        assert r["mode_isolation"]["resolved_mode"] == "unknown"
        assert r["mode_isolation"]["capabilities"] is None
        assert r["unknown_first"] is True


def test_vision_missing_fields_honest_unavailable():
    """视觉缺关键字段 → UNAVAILABLE 状态（不猜值）。"""
    pytest.importorskip("cash_game_ui.runner")
    from cash_game_ui.runner import CashGameRunner
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    ev = Path("data2")
    frame = ev / "BUTTON_CHANGED_167_1787992682749_full.png"
    if not frame.exists():
        pytest.skip("evidence frame not present")
    w, h, rgba = decode_png(frame.read_bytes())
    res = CashGameRunner().analyze(rgba, w, h).to_dict()
    assert res["status"] in ("UNAVAILABLE", "DECISION_UNAVAILABLE")
    assert res["decision_action"] is None
    # V-1 FM-1：street 层结构性拒绝——reason 可审计。
    # V-2 更新（test-side only）：召回修复后该帧（目视 GT 4 卡：3c 7h Qc 5c）
    # 可计数控 4 / FULL 牌面 3 → 拒绝原因 street_board_contradiction
    # （诚实拒绝语义不变，绝不伪造决策）。
    assert ("hero_cards" in res.get("blocking_fields", [])
            or "street_new_hand_unconfirmed" in (res.get("reason") or "")
            or "street_board_contradiction" in (res.get("reason") or ""))


# ------------------------------------------------------------ 双计终审
def test_final_double_counting_audit():
    """全层分账：saca_cv（SACA）/ a3v2（替换记录）/ exploit / mtt /
    pko / bounty_util（P44）——字段名互不重叠，恒等式用全部分量。"""
    st = _pko_state()
    r = evaluate_intel(st, {"hero_equity": 0.45},
                       range_src={"villains": {"v2": {"distribution": {
                           "AA": 0.1, "QQ": 0.2, "AQo": 0.3, "KQo": 0.4}}}},
                       exploit={"available": True, "confidence": 1.0,
                                "affected_actions": {"bet": 0.2},
                                "streams_used": ["entry_rate"],
                                "provenance": {}, "fallback": {}},
                       pko_context={"module": "pko_context",
                                    "bounty_type": "pko",
                                    "growth_rate": 0.5, "is_bounty": True,
                                    "factors": {
                                        "effective_bounty": {
                                            "status": "usable",
                                            "max_bounty_bb": 15.0},
                                        "bounty_pressure": {
                                            "status": "usable",
                                            "pressure": 0.15},
                                        "cover_relation": {
                                            "status": "usable",
                                            "relation": "covered"},
                                        "icm_x_bounty": {
                                            "status": "context_only"},
                                        "late_stage_bounty_pressure": {
                                            "status": "UNKNOWN"}}})
    fields_seen = set()
    for c in r["candidates"]:
        adj = c["adjustments"]
        fields_seen.update(adj.keys())
        # 模式感知恒等式（P64 冻结语义）：全模式 chip 恒等式
        # base+saca==final_ev；CASH 另有全加性 utility 恒等式（k=1）；
        # MTT/PKO utility 经 k± 缩放 → final−before==exploit+mtt+pko
        # 加项（bounty_util 已在 k± 内，属 P44 分量）。
        assert c["base_ev_bb"] + adj["saca_cv_bb"] == \
            pytest.approx(c["final_ev_bb"], abs=3e-6)
        if r["mode"] == "cash":
            total = (adj["saca_cv_bb"] + (adj.get("exploit_bb") or 0.0)
                     + (adj.get("mtt_context_bb") or 0.0)
                     + (adj.get("pko_context_bb") or 0.0)
                     + (adj.get("bounty_util_bb") or 0.0))
            assert c["base_ev_bb"] + total == \
                pytest.approx(c["final_utility_bb"], abs=3e-6)
        else:
            if c.get("utility_before_exploit_bb") is not None:
                delta = (c["final_utility_bb"]
                         - c["utility_before_exploit_bb"])
                expect = ((adj.get("exploit_bb") or 0.0)
                          + (adj.get("mtt_context_bb") or 0.0)
                          + (adj.get("pko_context_bb") or 0.0))
                assert delta == pytest.approx(expect, abs=3e-6)
    # 分账字段出现性：本 state 无 mtt_context → mtt 分量不应出现
    assert {"saca_cv_bb", "exploit_bb", "pko_context_bb",
            "bounty_util_bb"} <= fields_seen
    assert "mtt_context_bb" not in fields_seen              # 隔离审计


def test_deterministic_closure():
    a = evaluate_intel(_pko_state(), {"hero_equity": 0.45},
                       range_src={"villains": {"v2": {"distribution": {
                           "AA": 0.1, "QQ": 0.2, "AQo": 0.3, "KQo": 0.4}}}},
                       exploit={"available": True, "confidence": 1.0,
                                "affected_actions": {"bet": 0.2},
                                "streams_used": ["entry_rate"],
                                "provenance": {}, "fallback": {}})
    b = evaluate_intel(_pko_state(), {"hero_equity": 0.45},
                       range_src={"villains": {"v2": {"distribution": {
                           "AA": 0.1, "QQ": 0.2, "AQo": 0.3, "KQo": 0.4}}}},
                       exploit={"available": True, "confidence": 1.0,
                                "affected_actions": {"bet": 0.2},
                                "streams_used": ["entry_rate"],
                                "provenance": {}, "fallback": {}})
    assert a == b
